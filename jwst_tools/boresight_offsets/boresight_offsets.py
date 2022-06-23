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
   a. [ ] Plot lines and centers to confirm
7. [x] Write the central offset to a file and send it off to be included in SIAF
8. [ ] Compare to the CDP boresights from miricoord
9. [ ] Convert the offsets to arcsec taking into account distortion
"""

from pathlib import Path
import numpy as np
import pandas as pd

from . import read 
from .. import utils as jwutils

# list the filters used by the coronagraphs
filters = {'TA': ['F560W', 'F1000W', 'F1500W', 'FND'],
           'Sci': {'1065': 'F1065C',
                   '1140': 'F1140C',
                   '1550': 'F1550C',
                   'LYOT': 'F2300C'}
           }

def load_subarray_centers(centers_file, frame_from='det', frame_to='sci', pix_shift=0):
    """
    Load a list of the subarray centers and transform them to the frame you need

    Parameters
    ----------
    centers_file : str or pathlib.Path
      csv file of the centroids. First column is subarray, then x, dx, y, dy
    frame_from : str
      default: 'det']
      the siaf coordinate frame that the center file values are in, e.g. 'det'
    frame_to : str
      default: 'sci'
      the frame you want to transform the centers to
    pix_shift : int
      default: 0
      add this amount to the pixel indices, e.g. for going between python and siaf

    Output
    ------
    centers_df : pd.DataFrame of subarray centers. index is subarray, cols are x,y
    """
    centers = pd.read_csv(centers_file)
    def conversion_wrapper(row):
        """Wrapper for converting between any coordinate systems"""
        funcname = f"{frame_from}_to_{frame_to}"
        func = getattr(jwutils.miri_siaf['MIRIM_MASK'+row.name], funcname)
        center = pd.Series(func(row['x'], row['y']), index=['x','y'])
        return center
    transf_centers = centers.set_index('subarray').apply(conversion_wrapper, axis=1)
    # apply shift, if necessary
    transf_centers = transf_centers + pix_shift
    return transf_centers


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
    centroids_file_template = read.organize_mast_files(files,
                                                       extra_keys={'FILTER': 0, 'SUBARRAY': 0})
    # keep: id,,path,filename,filter,prog_id,subarray,obs_num,act_num,reference,x,dx,y,dy,roi
    drop_columns = "vis_num,vis_grp,pll_seq,exp_num,seg_num,detector,prod_type,filestem"
    for dc in drop_columns:
        try:
            centroids_file_template.drop(columns=[dc], inplace=True)
        except KeyError:
            # the column wasn't found, so don't worry about dropping it
            pass
    # centroids_file_template.drop(columns=drop_columns.split(','), inplace=True)
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
    is_ref = lambda filt: 'y' if filt in filters['Sci'].values() else 'n'
    centroids_df['reference'] = centroids_df['filter'].apply(is_ref)
    # get the nearest TA ROI region
    get_roi = lambda row: determine_ta_region(*row[['x','y']],
                                              jwutils.miri_siaf['MIRIM_'+row['subarray']])
    centroids_df['roi'] = centroids_df.apply(get_roi, axis=1)
    # check that at least these columns are present
    for col in ["id","path","filename","filter","prog_id","subarray",
                "obs_num","act_num","reference","x","dx","y","dy","roi"]:
        try:
            assert(col in centroids_df.columns)
        except AssertionError:
            print(f"Column `{col}` missing from columns")
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
    # make sure you're only including observations taken in the same program in the same sequence
    gb = centroids_df.groupby(["prog_id", "obs_num"])
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
    predict the boresight offset in the center of the subarray. Prints the
    offsets for each filter in each subarray

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

    center_offsets_df = pd.DataFrame(np.nan, columns=['dx', 'dy'], index=lines_df.index)
    for subarray, filt in lines_df.index:
        line = lines_df.loc[(subarray, filt)]
        x_center = line['off_x'][0] + line['off_x'][1]*centers_df.loc[subarray[-4:], 'x']
        y_center = line['off_y'][0] + line['off_y'][1]*centers_df.loc[subarray[-4:], 'y']
        center_offsets_df.loc[(subarray, filt)] = pd.Series({'dx': x_center, 'dy': y_center})

        # center_offsets_df = pd.concat(center_offsets_df, axis=1).T
    # print the values
    subarray_gb = center_offsets_df.groupby("subarray")
    for subarray, group in subarray_gb:
        print(f"{subarray[-4:]} boresight offsets (relative to {filters['Sci'][subarray[-4:]]})")
        for ind in group.index:
            row = group.loc[ind]
            print(f"\t{row.name[1]:6s}: x, y = ({row['dx']:0.3f}, {row['dy']:0.3f})")

    return center_offsets_df
    

def write_offset_to_file(center_offsets, saveto):
    """
    Write the boresight offsets at the array centers to a CSV file, after
    adding/changing necessary information

    Parameters
    ----------
    center_offsets: pd.DataFrame
      dataframe of x, y offsets in pixels at the subarray centers, with
      a two-level index of (subarray, filter)
    saveto: str or pathlib.Path
      where to save the CSV file

    Output
    ------
    writes csv file to location specified by `saveto`

    """
    # move the index into columns
    df = center_offsets.reset_index()
    # add the subarray filter
    ref_filter = df['subarray'].apply(lambda el: filters['Sci'][el[-4:]])
    df.insert(1, 'ref. filter', ref_filter)
    df.to_csv(saveto, index=False)
