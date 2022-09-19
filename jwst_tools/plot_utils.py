"""
Wrappers for quickly making plots
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.io import fits
from photutils.aperture import CircularAperture

from pysiaf import Siaf

from . import utils


def get_vlims(img, llim_std=2, ulim_std=2):
    """
    Returns a tuple of (vmin, vmax). llim_std and ulim_std are integers that get
    multiplied by the image std, and centered on the image median.
    """
    return np.nanmedian(img) + np.nanstd(img) * np.array([llim_std, ulim_std])


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


def preview_visit(visit_df):
    """
    Plot the exposures and apertures from a visit sequence

    Parameters
    ----------
    visit_df : one row for each observation in a visit
    ax : plt.axis [None]
      optional: provide the axis to draw on

    Output
    ------
    fig : plt.Figure instance
    """
    ncols = 3
    nrows = np.ceil(len(visit_df)/ncols).astype(int)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 7*nrows))

    for i, row in visit_df.iterrows():
        ax = axes.flat[i]
        ax.set_title(f"Visit {row['SeqID']}, Obs {row['ObsID']}\n{row['ROI']}")

        img = np.median(fits.getdata(row['File']), axis=0)
        roi_name = row['ROI']
        roi_aper = utils.miri_siaf[roi_name]
        subarray_name = "MIRIM_MASK" + [c for c in utils.coron_names if c in row['ROI']][0]
        subarray_aper = utils.miri_siaf[subarray_name]

        # compute the right to match the image with the subarray detector coordinates
        det_corner = np.min(subarray_aper.corners('det'), axis=1)[::-1] # put in row, col order
        img_det_ind = np.indices([i+1 for i in img.shape]) + det_corner[:, None, None]

        vmin, vmax = get_vlims(img, -2, 2)
        ax.pcolor(*(img_det_ind[::-1]), img, cmap=mpl.cm.cividis, vmin=vmin, vmax=vmax)

        subarray_aper.plot(frame='det', ax=ax, mark_ref=True, fill=False)
        roi_aper.plot(frame='det', ax=ax, mark_ref=True, fill=False)

        ax.set_aspect('equal')

    for ax in axes.flat[i+1:]:
        ax.set_visible(False)

    return fig

def preview_visit_with_sources(visit_df, visit_cat):
    """
    Plot the exposures and apertures from a visit sequence, and plot the detected sources too

    Parameters
    ----------
    visit_df : one row for each observation in a visit
    source_cat : catalog of sources in each observation in the visit

    Output
    ------
    fig : plt.Figure instance
    """
    ncols = 4
    nrows = np.ceil(len(visit_df)/ncols).astype(int)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 7*nrows))

    for i, row in visit_df.iterrows():
        ax = axes.flat[i]
        ax.set_title(f"Visit {row['SeqID']}, Obs {row['ObsID']}\n{row['ROI']}")

        img = np.median(fits.getdata(row['File']), axis=0)
        roi_name = row['ROI']
        roi_aper = utils.miri_siaf[roi_name]
        subarray_name = "MIRIM_MASK" + [c for c in utils.coron_names if c in row['ROI']][0]
        subarray_aper = utils.miri_siaf[subarray_name]

        # compute the right to match the image with the subarray detector coordinates
        img_ind = np.indices([i+1 for i in img.shape])  

        vmin, vmax = get_vlims(img, -2, 2)
        ax.pcolor(*(img_ind[::-1]), img, cmap=mpl.cm.cividis, vmin=vmin, vmax=vmax)

        # plot sources
        sources = visit_cat.query(f"ObsID == {row['ObsID']}")
        if not isinstance(sources, type(None)):
            for j, source_row in sources.iterrows():
                pos = source_row[['xcentroid', 'ycentroid']].values
                aperture = CircularAperture(pos, r=6)
                aperture.plot(axes=ax, color='red', lw=1.5, alpha=0.5);
                ax.text(pos[0], pos[1], int(source_row['source_id']),
                        va='bottom',
                        ha='right')

        # axis configurations
        ax.set_aspect('equal')

    for ax in axes.flat[i+1:]:
        ax.set_visible(False)

    return fig


def visual_ta_check(filename, siaf_aper, pad=10):
    """
    Given a filename and a SIAF aperture covering the data from that filename,
    show zoom-ins on the data from that aperture from all the different headers in the file

    Parameters
    ----------
    filename : str or Path
      path to fits file with the data
    siaf_aper : str or pySiaf aperture object
      the aperture to plot
    buffer : int
      how much should you add to the edge of the cutout? (pixels)

    Output
    ------
    fig : matplotlib Figure
      multi-axis figure with the plots

    """
    filename = Path(filename)
    if isinstance(siaf_aper, str):
        siaf_aper = utils.miri_siaf[siaf_aper]
    # you need the subarray object for coordinate conversion
    subarray_name = fits.getval(str(filename), "SUBARRAY", 0)
    subarray = utils.miri_siaf["MIRIM_" + subarray_name]
    # get the center and size of the box in x, y (not row, col)
    center = siaf_aper.reference_point('det')
    center = utils.siaf_python_coords(subarray.det_to_sci(*center), to='py')
    cutout_dims = np.max(np.diff(np.stack(siaf_aper.corners('sci')), axis=1), axis=1) + 2*pad

    # calculate aperture indices with padding, making sure it doesn't extend
    # beyond the subarray
    aper_orig = np.stack(siaf_aper.corners("det")).T[0]
    subarray_orig = np.stack(subarray.corners("det")).T[0]
    aper_dims = np.max(np.diff(np.stack(siaf_aper.corners("det")), axis=1), axis=1)
    subarray_dims = np.max(np.diff(np.stack(subarray.corners("det")), axis=1), axis=1)
    aper_indices = [np.arange(max(a_o-pad, 0.5), min(a_d+a_o+pad, s_d+s_o)+1)
                    for a_d, a_o, s_d, s_o
                    in zip(aper_dims, aper_orig, subarray_dims, subarray_orig)]
    
    # get the sci, err, and dq cutouts and plot them
    # pull the cutout
    imgs = {'sci': None,
            'err': None,
            'dq': None}
    for k in imgs.keys():
        img = fits.getdata(str(filename), k.upper())
        # reduce dimensionality in this very dumb way
        while np.ndim(img) > 2:
            if k == 'dq':
                # produce a simple summary that you can drill into if you want
                img = img.astype(bool)
                img = np.all(img.astype(int) == 0, axis=0)
            else:
                img = np.nanmedian(img, axis=0)
        cutout = img_cutout(img, center[::-1], cutout_dims[::-1])
        imgs[k] = cutout

    ncols = len(imgs.keys())
    nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5*ncols, 5*nrows))
    for ax, (name, img) in zip(axes.ravel(), imgs.items()):
        ax.set_title(name.upper())
        vmin, vmax = get_vlims(img, -2, 2)
        cmap = 'cividis'
        if name == 'dq':
            img = img.astype(int)
            cmap = 'Greys'
        # print(name, siaf_aper.AperName, aper_indices[0].size, aper_indices[1].size, img.shape)
        ax.pcolor(*aper_indices, img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.scatter(*siaf_aper.reference_point('det'), marker='+')
        siaf_aper.plot(frame='det', ax=ax, mark_ref=True, fill=False)
    fig.suptitle(siaf_aper.AperName)
    return fig

def set_up_multifig(nimgs, ncols, scalesize, kwargs={}):
    """
    Wrapper for the usual things I do to set up a multiple axes figure

    Parameters
    ----------
    nimgs: int
      the number of images
    ncols: int
      the number of columns (the number of rows will be computed)
    scalesize: float
      use this to scale the figure size; will be proportional to fig dimensions
    kwargs: dict (default: {})
      any other arguments to pass to plt.subplots()

    Output
    ------
    tuple of (fig, axes)

    """
    nrows = np.ceil(nimgs/ncols).astype(int)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(scalesize*ncols, scalesize*nrows),
                             squeeze=False,
                             **kwargs)
    for ax in axes.ravel()[nimgs:]:
        ax.set_visible(False)
    return (fig, axes)
