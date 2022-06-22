"""
Module for helping compute boresight offsets.
Steps for computing boresight offsets:
1. [x] Measure PSF centroids around the subarray
   a. [x] Plot centroids on cutouts of the PSF for confirmation
2. [x] Mark which are from the reference filters and which are not
3. [x] Compute difference between the reference and non-reference filters
4. [x] Plot the offsets, per filter, as a function of x and y position on the subarray
5. [x] Fit the offset vs x/y with a line
6. [x] Use the line to compute the offset at the center (or at any other location in the subarray)
7. [x] Write the central offset to a file and send it off to be included in SIAF
8. [ ] compare to the CDP boresights from miricoord
"""

import numpy as np
import pandas as pd

from .. import read as jwread
from .. import utils as jwutils

# list the filters used by the coronagraphs
filters = {'TA': ['F560W', 'F1000W', 'F1500W', 'FND'],
           'Sci': ['F1065C', 'F1140C', 'F1550C', 'F2300C']}

def generate_centroid_file_template(files, outfile):
    """
    From the list of files, make a template for recording the PSF centroids
    Right now, the centroids are measured using IDP3 and copy-pasted into the
    x, dx, y, dy columns

    Parameters
    ----------
    files : list of strings or pathlib.Paths
      the data files used for centroiding
    outfile : string or path
      where to write the csv file

    Output
    ------
    writes csv file to indicated path

    """
    centroids_file_template = jwread.organize_mast_files(files,
                                                         extra_keys={'FILTER': 0, 'SUBARRAY': 0})
    drop_columns = "prog_id,vis_num,vis_grp,pll_seq,exp_num,seg_num,detector,prod_type,filestem"
    centroids_file_template.drop(columns=drop_columns.split(','), inplace=True)
    for col in ['x','dx','y','dy']:
        centroids_file_template[col] = ''
    centroids_file_template.to_csv(str(outfile), index=False)
    print("Centroid file template written to ", str(outfile))


def determine_ta_region(x, y, aper):
    """
    Determine which TA region a point belongs to (UR, UL, LL, LR, inner vs outer)

    Parameters
    ----------
    x: 1-indexed subarray x position, in pixels
    y: 1-indexed subarray y position, in pixels
    aper: pysiaf aperture object

    Output
    ------
    roi: string representing the TA roi region

    """
    x_idl, y_idl = aper.sci_to_idl(x, y)
    roi = ''
    if x_idl > 0 and y_idl > 0: 
        roi = 'UR'
    elif x_idl > 0 and y_idl < 0: 
        roi = 'LR'
    elif x_idl < 0 and y_idl < 0: 
        roi = 'LL'
    elif x_idl < 0 and y_idl > 0: 
        roi = 'UL'
    if (x_idl**2 + y_idl**2)**0.5 < 5:
        roi = 'C'+roi
    else:
        pass
    return roi


def load_psf_centroid_file(filepath):
    """
    load the file that contains the centroid information, and return the info
    as a dataframe with some extra processing

    Parameters
    ----------
    filepath : str or pathlib.Path
      path to the centroid file

    Output
    ------
    centroids_df: pd.DataFrame
      dataframe with the centroids

    """
    centroids_df = pd.read_csv(filepath)
    # mark the science filters as the references
    is_ref = lambda filt: 'y' if filt in filters['Sci'] else 'n'
    centroids_df['reference'] = centroids_df['filter'].apply(is_ref)
    # get the nearest TA ROI region
    get_roi = lambda row: determine_ta_region(*row[['x','y']],
                                              jwutils.miri_siaf['MIRIM_'+row['subarray']])
    centroids_df['roi'] = centroids_df.apply(get_roi, axis=1)
    return centroids_df


def compute_filter_offsets(centroids_df):
    """
    Compute the offsets between the reference filters and the rest, in pixel space.
    The dataframe is organized such that each observation contains one reference
    filter and one or more other filters, so the computations should be grouped
    by the `obs_num` column.

    Parameters
    ----------
    centroids_df : pd.DataFrame
      dataframe in the format of generate_centroid_file_template() and read in
      by load_psf_centroid_file()

    Output
    ------
    filter_offsets: pd.DataFrame
      dataframe of offsets in pixel counts between the reference and
      non-reference filters. Also contains all columns of the original dataframe

    """
    gb = centroids_df.groupby("obs_num")
    # compute offsets and uncertainties
    def relative_position(group):
        ref_pos =  group.query("reference == 'y'").squeeze()[['x','y', 'dx', 'dy']]
        pos = group[['x','y']] - ref_pos[['x','y']]
        unc = (group[['dx','dy']]**2 + ref_pos[['dx','dy']]**2)**0.5
        vals = {'off_x': pos['x'].astype(float), 'off_y': pos['y'].astype(float),
                'off_dx': unc['dx'].astype(float), 'off_dy': unc['dy'].astype(float)}
        return pd.concat(vals, axis=1)
    filter_offsets = gb.apply(relative_position)

    return centroids_df.join(filter_offsets)


def fit_offsets_v_position(offsets_df):
    """
    Fit the offsets against the position on the detector with a line

    Parameters
    ----------
    offsets_df: centroids dataframe with the offsets

    Output
    ------
    lines_df: dataframe of line parameters (y = p[0] + p[1]*x)
      the index of the parameter is its order (e.g. p[0]*x^0 + p[1]*x^1)

    """
    # fit a line to the Lyot data, using all, inner only, and outer only
    def fit_lines(filter_group):
        # fit x
        x, dx = filter_group[['x', 'dx']].values.T
        y, dy = filter_group[['off_x', 'off_dx']].values.T
        x_line = np.polyfit(x, y, 1, w=1/dy)[::-1] # returns b, m (y = b + m*x)

        # fit y
        x, dx = filter_group[['y', 'dy']].values.T
        y, dy = filter_group[['off_y', 'off_dy']].values.T
        y_line = np.polyfit(x, y, 1, w=1/dy)[::-1]

        df = pd.concat({'off_x': pd.Series(x_line, index=pd.Index(range(len(x_line)), name='param')), 
                        'off_y': pd.Series(y_line, index=pd.Index(range(len(y_line)), name='param'))},
                       axis=1)
        return df
    lines_df = offsets_df.query(f"reference == 'n'").groupby(['subarray', 'filter']).apply(fit_lines)
    return lines_df.unstack('param')


def compute_offset_at_center(offsets_df, lines_df, centers_df):
    """
    For a given subarray and filter pair, use the line fitted to that data to
    predict the boresight offset in the center of the subarray

    Parameters
    ----------
    offsets_df : pd.DataFrame
      dataframe containing the offsets and other metadata
    lines_df : pd.DataFrame
      dataframe with the line parameters stored as y = p[0] + p[1]*x
    centers_df : pd.DataFrame
      dataframe or dict with the x,y centers for each subarray, in pixels

    Output
    ------
    center_offsets_df : pd.DataFrame of offsets at the center of the
      subarrays, per filter combination

    """

    center_offsets_df = {}
    for subarray, filt in lines_df.index:
        line = lines_df.loc[(subarray, filt)]
        x_center = line['off_x'][0] + line['off_x'][1]*centers_df.loc[subarray[-4:], 'x']
        y_center = line['off_y'][0] + line['off_y'][1]*centers_df.loc[subarray[-4:], 'y']
        center_offsets_df[(subarray, filt)] = pd.Series({'x': x_center, 'y': y_center})
    return pd.concat(center_offsets_df, axis=1).T
