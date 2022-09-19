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
<<<<<<< HEAD
8. [x] Compare to the CDP boresights from miricoord
9. [ ] Convert the offsets to arcsec taking into account distortion
"""

# python
=======
8. [ ] Compare to the CDP boresights from miricoord
9. [ ] Convert the offsets to arcsec taking into account distortion
"""

>>>>>>> 78bd53f (boresight offsets computations work great)
from pathlib import Path
import numpy as np
import pandas as pd

# astro
from astropy.io import fits

# JWST
from pysiaf import Siaf
import miricoord
import miricoord.imager.mirim_tools as mt

# local
from . import read 

# miri Siaf aperture
miri = Siaf('MIRI')

# list of subarrays
subarrays = {'4QPM': ['1065','1140','1550'],
             'LYOT': ['LYOT']}

# list the filters used by the coronagraphs
filters = {'TA': ['F560W', 'F1000W', 'F1500W', 'FND'],
<<<<<<< HEAD
           'Sci': {'MASK1065': 'F1065C',
                   'MASK1140': 'F1140C',
                   'MASK1550': 'F1550C',
                   'MASKLYOT': 'F2300C'}
           }
=======
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

>>>>>>> 78bd53f (boresight offsets computations work great)

def load_subarray_centers(centers_file, frame_from='det', frame_to='sci', pix_shift=0, csv_args={}):
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
      *add* this amount to the pixel indices, e.g. for going between python and siaf
      provide a negative number to shift the other way
    csv_args : dict
      dict of arguments to pass to pandas.read_csv()

    Output
    ------
    centers_df : pd.DataFrame of subarray centers. index is subarray, cols are x,y
    """
    centers = pd.read_csv(centers_file, **csv_args)
    def conversion_wrapper(row):
        """Wrapper for converting between any coordinate systems"""
        funcname = f"{frame_from}_to_{frame_to}"
        func = getattr(miri['MIRIM_'+row.name], funcname)
        center = pd.Series(func(row['x'], row['y']), index=['x','y'])
        return center
    transf_centers = centers.set_index('subarray').apply(conversion_wrapper, axis=1)
    # apply shift, if necessary
    transf_centers = transf_centers + pix_shift
    return transf_centers


def generate_centroid_file_template(files, outfile, drop_columns=[]):
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
    drop_columns : list
      Default: []
      columns generated by organize_mast_files that you want to drop

    Output
    ------
    writes csv file to indicated path

    """
<<<<<<< HEAD
    centroids_file_template = read.organize_mast_files(files,
                                                       extra_keys={'FILTER': 0, 'SUBARRAY': 0})
    # keep: id,,path,filename,filter,prog_id,subarray,obs_num,act_num,reference,x,dx,y,dy,roi
    for dc in drop_columns:
        try:
            centroids_file_template.drop(columns=dc, inplace=True)
            print(dc, ' dropped')
        except KeyError:
            # the column wasn't found, so don't worry about dropping it
            pass
    # centroids_file_template.drop(columns=drop_columns.split(','), inplace=True)
=======
    centroids_file_template = jwread.organize_mast_files(files,
                                                         extra_keys={'FILTER': 0, 'SUBARRAY': 0})
    drop_columns = "vis_num,vis_grp,pll_seq,exp_num,seg_num,detector,prod_type,filestem"
    centroids_file_template.drop(columns=drop_columns.split(','), inplace=True)
>>>>>>> 78bd53f (boresight offsets computations work great)
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
                                              miri['MIRIM_'+str(row['subarray'])])
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
    # lines_df = offsets_df.query(f"reference == 'n'").groupby(['subarray', 'filter']).apply(fit_lines)
    lines_df = offsets_df.groupby(['subarray', 'filter']).apply(fit_lines)
    return lines_df.unstack('param')


def compute_offset_at_center(offsets_df, lines_df, centers_df, use_1140=False):
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
    use_1140 [False]: bool
      if True, use the 1140 slopes to find the 1065 and 1550 centers (because of a
      lack of data in those subarrays)

    Output
    ------
    center_offsets_df : pd.DataFrame of offsets at the center of the
      subarrays, per filter combination

    """

    center_offsets_df = pd.DataFrame(np.nan, columns=['dx', 'dy'], index=lines_df.index)
    for subarray, filt in lines_df.index:
        line = lines_df.loc[(subarray, filt)]
<<<<<<< HEAD
        # Since we don't  have complete coverage of the 1065 and 1550 arrays
        # and the slope of the offsets as a function of position seems to
        # diverge near the center, we might want to use the 1140 slopes instead
        if (use_1140 == True) and (subarray in ['MASK1065', 'MASK1550'] and (filt in filters['TA'])):
            print(f"Using 1140 slopes for ({subarray}, {filt}), and updating the lines dataframe")
            offsets = offsets_df.query(f"filter == '{filt}' and subarray == '{subarray}'")
            # use the 1140 slope instead
            # adjust the intercept by transforming the data to fit only the intercept
            slopes = lines_df.loc[('MASK1140', filt), (slice(None), 1)]
            # get the intercepts and overwrite them
            intercepts = lines_df.loc[(subarray, filt), (slice(None), 0)]
            x, off_x = offsets[['x', 'off_x']].values.T
            y, off_y = offsets[['y', 'off_y']].values.T
            intercepts.loc[('off_x', 0)] = np.polyfit(x, off_x - slopes[('off_x', 1)]*x, 0)[0]
            intercepts.loc[('off_y', 0)] = np.polyfit(y, off_y - slopes[('off_y', 1)]*y, 0)[0]
            # adjust parameters
            line[slopes.index] = slopes
            line[intercepts.index] = intercepts
        x_center = line['off_x'][0] + line['off_x'][1]*centers_df.loc[subarray, 'x']
        y_center = line['off_y'][0] + line['off_y'][1]*centers_df.loc[subarray, 'y']
        center_offsets_df.loc[(subarray, filt)] = pd.Series({'dx': x_center, 'dy': y_center})

    # add the reference filter
    ref_filter = center_offsets_df.apply(lambda row: filters['Sci'][row.name[0]], axis=1)
    center_offsets_df.insert(0, 'ref_filter', ref_filter)

    # print the values
    subarray_gb = center_offsets_df.groupby("subarray")
    for subarray, group in subarray_gb:
        print(f"{subarray} boresight offsets (relative to {filters['Sci'][subarray]})")
        for ind in group.index:
            row = group.loc[ind]
            print(f"\t{row.name[1]:6s}: x, y = ({row['dx']:+0.3f}, {row['dy']:+0.3f})")
=======
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
>>>>>>> 78bd53f (boresight offsets computations work great)

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
<<<<<<< HEAD
    df.to_csv(saveto, index=False)

def load_cdp_offsets():
    """
    Load the boresight offsets currently in CDP

    Parameters
    ----------
    None

    Output
    ------
    cdp_offsets: pd.DataFrame
      dataframe of offsets relative to the F770W filter

    """
    cdpfile=mt.get_fitsreffile()

    with fits.open(cdpfile) as hdul:
        # this is in pixels
        tbdata = pd.DataFrame(hdul['Boresight offsets'].data)
    tbdata = tbdata.rename(columns={'FILTER': 'filter',
                                    'COL_OFFSET': 'dx',
                                    'ROW_OFFSET': 'dy'})
    tbdata = tbdata.set_index('filter')
    tbdata = tbdata.astype({'dx': float, 'dy': float})
    cdp = pd.concat({filt: tbdata.loc[filters['TA']] - tbdata.loc[filt]
                     for subarray, filt in filters['Sci'].items()},
              names=['ref_filter'])
    return cdp


def compute_offset_idl(x, y, ref_x, ref_y, aperture):
    """
    (author: C. Cossou)
    Convert from detector to ideal coordinate then return the
    offset in arcsec (ideal coordinates)

    :param float x: coordinate in detector pixel (1-indexed)
    :param float y: coordinate in detector pixel (1-indexed)
    :param float ref_x: coordinate in detector pixel (1-indexed)
    :param float ref_y: coordinate in detector pixel (1-indexed)
    :param Siaf.aperture aperture: Siaf aperture object

    :return: (x, y) ideal coordinates offset in arcsec
    """
    xidl, yidl = aperture.det_to_idl(x, y)
    ref_xidl, ref_yidl = aperture.det_to_idl(ref_x, ref_y)
    dxidl, dyidl = xidl - ref_xidl, yidl - ref_yidl

    return dxidl, dyidl


def compute_apt_offsets(center_offsets, cdp_offsets, gnd_centers):
    """
    Compute the offsets you need to apply to APT to adjust for the difference
    between measured and CDP offsets
    """
    bso_shift = center_offsets.reset_index().set_index(['ref_filter', 'filter']).join(cdp_offsets, lsuffix='_meas', rsuffix='_cdp')
    bso_shift.drop(bso_shift.index[[i[0]==i[1] for i in bso_shift.index]], inplace=True)
    bso_shift['dx'] = bso_shift['dx_meas']-bso_shift['dx_cdp']
    bso_shift['dy'] = bso_shift['dy_meas']-bso_shift['dy_cdp']

    # define a function to run on each row's filter combination
    def get_shift_arcsec(row):
        aper = miri['MIRIM_'+row['subarray']]
        # use the SIAF values as the reference
        x_ref, y_ref = aper.reference_point('sci') # 1-indexed
        x_gnd, y_gnd = gnd_centers.loc[row['subarray']][['x','y']] #+ 1 # shift to 1-indexed
        dx_bso, dy_bso = row[['dx','dy']]
        # to compensate for the SIAF and boresight differences, this is where we have to aim
        x_aim, y_aim = x_gnd + dx_bso, y_gnd + dy_bso
        # convert this pixel value to IDL coordinates
        dx_idl, dy_idl = compute_offset_idl(x_aim, y_aim, x_ref, y_ref, aper)
        return pd.Series([dx_idl, dy_idl], index=['dx_idl', 'dy_idl'])
    bso_shift[['dx_idl', 'dy_idl']] = bso_shift.apply(get_shift_arcsec, axis=1)

    # print the boresight offsets
    for name, group in bso_shift.groupby("subarray"):
        print(f"APT shifts for {name}:")
        for i, row in group.iterrows():
            dx, dy = row['dx_idl'], row['dy_idl']
            print(f"\t{i[1]:>6s} -> {i[0]:<6s}: dx, dy = ({dx:0.3f}, {dy:0.3f}) arcsec")
    return bso_shift


def make_uncal_flat(filename, outfile, make_ta_image):
    """
    use the JWST TA tools function to make a flat image out of the uncal image
    so that the TA algorithm will work. flat_func is the function from TA_crossinst,
    but I had to code it up in a way that this library doesn't import TA_crossinst

    Parameters
    ----------
    filename: str or pathlib.Path
      path to an uncal file
    outfile: str or pathlib.Path
      Where to save the output file.
    make_ta_image: the function jwst_ta.make_ta_image

    Output
    ------
    writes uncalflat to file
    """
    new_filename = Path(outfile)#Path(filename.parent) / (filename.stem + 'flat' + '.fits')
    ta_img = make_ta_image(str(filename), ext=1)
    pri_hdu = fits.PrimaryHDU(header=fits.getheader(filename, 0))
    sci_hdu = fits.ImageHDU(header=fits.getheader(filename, 1), data=ta_img)
    hdulist = fits.HDUList([pri_hdu, sci_hdu])
    hdulist.writeto(str(new_filename))
    print(new_filename, "written")
=======
    # add the subarray filter
    ref_filter = df['subarray'].apply(lambda el: filters['Sci'][el[-4:]])
    df.insert(1, 'ref. filter', ref_filter)
    df.to_csv(saveto, index=False)
>>>>>>> 78bd53f (boresight offsets computations work great)
