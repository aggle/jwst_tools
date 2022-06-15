"""
Helpers used by the StarFinder notebooks
"""
from pathlib import Path
import numpy as np

import pandas as pd

# for computing boundaries of apertures
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LinearRing

import pysiaf
from pysiaf import Siaf
from jwst.datamodels import dqflags

# MIRI Siaf object
miri_siaf = Siaf("MIRI")

# hard-code the ROI types here, in order, to reuse for consistency
roi_types = ('OUTER', 'INNER', 'OCC')
# these are the options for coronagraph names
coron_names = ('1065', '1140', '1550', 'LYOT')

def siaf_python_coords(coords, to='py'):
    """
    Convert coordinates from SIAF to Python, or from Python to SIAF
    coords: array
      coordinates to convert
    to: 'py' or 'siaf'
      direction of conversion
    """
    # make sure coords are a type that can be added and subtracted to
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    # set the direction
    shift = 1
    if to == 'py':
        shift = shift * -1
    # apply the shift
    return coords + shift

def create_visit_df(obs_df):
    """
    Create a dataframe for processing a visit, starting from the observation
    group from the master program dataframe

    Parameters
    ----------
    obs_df: dataframe containing the information for all the observations in a visit
        must not be a view

    Output
    ------
    visit_df: expanded version of the observation df with more metadata

    """
    if obs_df._is_view == True:
        obs_df = obs_df.copy()
    def get_roi_name(fname):
        fname = Path(fname).stem
        roi_name = '_'.join(fname.split('-')[1].split('_')[:-1])
        return roi_name
    obs_df['ROI'] = obs_df['File'].apply(get_roi_name)

    def get_roi_type(roi_name):
        if 'CORON' in roi_name.upper():
            return 'OCC'
        elif 'C' in roi_name.split('_')[-1]:
            return 'INNER' # TA-INNER
        else:
            return 'OUTER' # TA-OUTER
    obs_df['ROItype'] = obs_df['ROI'].apply(get_roi_type)

    return obs_df

def list_remaining_crossrefs(crossrefs, visit_cat):
    """
    List the sources in the catalog that haven't been assigned an object ID yet
    """
    found_sources = set([i for v in crossrefs.values() for i in v[1]])
    print("Sources left to assign:")
    for source_id in visit_cat['source_id']:
        if int(source_id) not in found_sources:
            print(source_id)
    return

def filter_catalog(full_cat):
    """
    Filter the catalog to only retain the target star and also stars that appear in the `coron` exposure plus at least one more.

    Parameters
    ----------
    full_cat : the full catalog

    Output
    ------
    filt_cat : the filtered catalog

    """
    gb_obj = full_cat.groupby("obj_id")
    keep_refs = gb_obj.apply(lambda group: (len(group['exp_id']) >= 2) and ("coron" in group['exp_type'].values))
    qstr = f"obj_id in {list(keep_refs[keep_refs].index.values)} or obj_type == 'target'"
    filt_cat = full_cat.query(qstr).copy()
    return filt_cat

# Mask functions, for masking out regions and pixels
def mask_subaperture(subarray_aper, inner_aper, buffer=-10):
    """
    Create a binary mask the size of the subarray aperture, that is 0
    outside the inner aperture and 1 inside the inner aperture.

    Parameters
    ----------
    img : an image read out from the subarray corresponding to subarray_aper
    subarray_aper: the SIAF aperture for the subarray
    inner_aper: the subaperture (TA, coron, etc)
    buffer [-10]: pad the border by this number of pixels (negative in, positive out)

    Output
    ------
    binary array

    """
    # get the detector coordinates of all the pixels in the subarray
    x_llim, y_llim = np.min(subarray_aper.corners("det"), axis=1)
    x_ulim, y_ulim = np.max(subarray_aper.corners("det"), axis=1)
    img_shape = np.array([y_ulim-y_llim, x_ulim-x_llim]).astype(int)
    y_coords = np.tile(np.arange(y_llim, y_ulim), (img_shape[1], 1)).T
    x_coords = np.tile(np.arange(x_llim, x_ulim), (img_shape[0], 1))

    # We're going to use Shapely to decide if a point is contained in the
    # inner aperture or not. Let's make a polygon defined by its boundaries
    inner_pol = inner_aper.closed_polygon_points('det')
    inner_aper_boundary = LinearRing(list(zip(*inner_pol)))
    inner_aper_img = Polygon(inner_aper_boundary).buffer(buffer)

    # now create a Point object for every coordinate pair.
    # Use numpy ravel and unravel to handle the ordering for the final array
    points = [Point(x, y) for y, x in zip(y_coords.ravel(), x_coords.ravel())]
    mask = [True if inner_aper_img.contains(point) else False for point in points]
    mask = np.reshape(mask, img_shape)
    return mask

def aper_index(siaf_aper, parent_aper='MIRIM_FULL'):
    """
    Get the indices of a subaperture so you can pull it out of an array

    Parameters
    ----------
    siaf_aper : str or siaf aperture object
      the siaf aperture you want to pull out of the array. If it's a string, get the aperture
    parent_aper : str or siaf aperture object [MIRIM_FULL]
      the aperture object that represents the array you want to index

    Output
    ------
    indices: tuple of 2-D arrays of (row, col) indices

    """
    if isinstance(siaf_aper, str):
        siaf_aper = miri_siaf[siaf_aper]
    if isinstance(parent_aper, str):
        parent_aper = miri_siaf[parent_aper]

    corners = siaf_aper.corners('det')
    array_corners = parent_aper.det_to_sci(*corners)
    xrange, yrange = (np.stack(array_corners)[:, [0, 2]] - 0.5).astype(int)
    xdim = xrange[1]-xrange[0]
    ydim = yrange[1]-yrange[0]
    indices = [np.tile(np.arange(yrange[0], yrange[1]), (xdim, 1)).T,
               np.tile(np.arange(xrange[0], xrange[1]), (ydim, 1))]
    return indices


def img_cutout(img, center, dims, return_ind=False):
    """
    Make a cutout of an image centered at `center` with size length `dims`

    Parameters
    ----------
    img: 2-D numpy array in row, col order
    center: tuple of (row, col) center of box. If float, gets `np.floor`'ed. 
    dims: tuple or integer of the box side length. Should be odd.
    return_ind [False]: if True, return indices to plot with plt.pcolor

    Output
    ------
    cutout: 2-D numpy array of the cutout
    ind: tuple of row, col 1-D index arrays for pcolor (from corner to corner)
    """
    row_max, col_max = img.shape
    # center = [np.floor(c).astype(int) for c in center]
    if np.ndim(dims) == 0:
        dims = np.tile(dims, 2)
    dims = np.array(dims).astype(int)
    box_rad = np.array(dims)/2
    # row and column limit math - make sure it doesn't go out of bounds
    row_lims = [np.max((center[0] - box_rad[0], 0)),      # row llim
                np.min((center[0] + box_rad[0], row_max)) # row ulim
                ]
    col_lims = [np.max((center[1] - box_rad[1], 0)),       # col llim
                np.min((center[1] + box_rad[1], col_max))  # col ulim
                ]
    # correct row and column limit format.
    # Use `floor` because that will index the correct pixel
    row_lims = [np.floor(row_lims[0]).astype(int),
                np.floor(row_lims[1]).astype(int),
                ]
    col_lims = [np.floor(col_lims[0]).astype(int),
                np.floor(col_lims[1]).astype(int),
                ]
    # now you can index the image
    cutout = img[row_lims[0]:row_lims[1], col_lims[0]:col_lims[1]]
    try:
        assert(all(cutout.shape == dims))
    except AssertionError:
        print(f"Error: cutout shape {cutout.shape} does not match requested {dims}")
    if return_ind == True:
        ind = [np.linspace(row_lims[0], row_lims[1], row_lims[1]-row_lims[0] + 1),
               np.linspace(col_lims[0], col_lims[1], col_lims[1]-col_lims[0] + 1)]
        return cutout, ind
    return cutout

