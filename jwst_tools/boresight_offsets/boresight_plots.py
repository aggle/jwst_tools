from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from astropy.io import fits

# local to jwst_tools
from .. import plot_utils as jwplots
from .. import utils as jwutils
from . import boresight_offsets as bso


def confirm_psf_centroids(centroids_df, saveto=None):
    """
    Plot the centroids on top of cutouts of the psfs

    Parameters
    ----------
    centroid_dfs: pd.DataFrame
      dataframe of centroids and metadata
    saveto [None]: str or pathlib.Path
      path to location for saving the figure. If None, do not save

    Output
    ------
    centroids_fig: mpl.plt.Figure object

    """
    centroids_gb = centroids_df.groupby(['prog_id', 'obs_num'])
    # one row for each filter, one column for each observing sequence
    col_names = {i: j for j, i in enumerate(['ref'] + bso.filters['TA'])}
    row_names = {i[0]: j for j, i in enumerate(centroids_gb)}
    nrows = len(row_names) # 1 row for each set of observations
    ncols = len(col_names) # 1 reference and 4 TA filters 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows),
                             sharex='row', sharey='row')

    for name, group in centroids_gb:
        for i, row in group.iterrows():
            # select the axis, and label it
            # row is the observation
            ax_row = row_names[name]
            # column is the filter
            ax_col = 0 if row['reference'] == 'y' else col_names[row['filter']]
            ax = axes[ax_row, ax_col]
            ax.set_ylabel(row['filter'])
            ax.set_title(f"{name[0]}, Obs {name[1]}")
            # get the stamp
            filename = Path(row['path']) / row['filename']
            img, coords = jwplots.img_cutout(fits.getdata(filename, 1), 
                                                row[['y', 'x']],
                                                31, 
                                                True)
            coords = [c-0.5 for c in coords]
            vmin, vmax = jwplots.get_vlims(img, -1, 3)


            ax.pcolor(coords[1], coords[0], img, vmin=vmin, vmax=vmax)
            ax.set_aspect('equal')
            ax.errorbar(row['x'], row['y'],
                        xerr=row['dx'], yerr=row['dy'],
                        marker='x', color='k', ms=10)

    if saveto is not None:
       fig.savefig(saveto, dpi=150)
    return fig


def plot_centroids_v_position_2d(offsets_df, saveto=None):
    """
    Plot the x and y offsets 

    Parameters
    ----------
    offsets_df: pd.DataFrame
      column of offsets from the reference filter, along with metadata
      like subarray, filter, x and y position
    saveto [None]: str or pathlib.Path
      if not None, save figure to this path

    Output
    ------
    fig [None]: plt.Figure instance
    """
    subarrays = offsets_df['subarray'].unique()
    ncols = len(subarrays)
    nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 6*nrows),
                             sharex=True, sharey=True,
                             squeeze=False)

    for subarray, ax in zip(subarrays, axes.ravel()):
        ax.set_title(subarray, size='x-large')
        ax.set_xlabel("x offset [pix]")
        ax.set_ylabel("y offset [pix]")
        ax.errorbar(0, 0, marker='*', ms=20, label='reference position')
        # plot each filter in a different color
        gb =  offsets_df.query(f"reference == 'n' and subarray == '{subarray}'").groupby('filter')
        for filt in gb.groups:
            group = gb.get_group(filt)
            ax.errorbar(group['off_x'], group['off_y'],
                        xerr=group['off_dx'], yerr=group['off_dy'],
                        ls='-', marker='o', label=f"{filt}")
        
        ax.legend(ncol=1, loc='best')
        ax.grid(True, alpha=1)

    if saveto is not None:
       fig.savefig(saveto, dpi=150)
    return fig


def plot_centroids_v_position_1d(offsets_df, saveto=None,
                                 lines=None,
                                 center_offsets=None,
                                 subarray_centers=None):
    """
    Plot the offset against the position on the detector, x and y

    Parameters
    ----------
    offsets_df : pd.DataFrame
      dataframe of offsets from the reference filter, and also metadata like
      subarray and filter
    saveto [None] : str or pathlib.Path
      where to save the figure. If None, not saved.
    lines [None] : None or pd.DataFrame
      a dataframe of line fit parameters (y = p[0] + p[1]*x). if None, skip.
    centers [None] : dataframe or dict
      dataframe or dict of (x, y) coronagraph centers. if None, skip.


    Output
    ------
    fig: matplotlib Figure (and saves to disk if desired)

    """

    subarrays = offsets_df['subarray'].unique()
    ncols = len(subarrays)
    nrows = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(8*ncols, 8*nrows),
                             sharex='row', sharey='row',
                             squeeze=False)

    for coord, ax_row in zip(['x', 'y'], [0, 1]):
        for subarray, ax in zip(subarrays, axes[ax_row]):
            aper = jwutils.miri_siaf['MIRIM_'+subarray]
            # descriptions
            ax.set_title(subarray, size='x-large')
            ax.set_xlabel(f"{coord} [pix]")
            ax.set_ylabel(f"{coord} offset [pix]")
            # plot a line at 0 offset
            ax.axhline(0, ls='-', color='k', label='No offset')

            coron_filter = bso.filters['Sci'][subarray[-4:]]
            # plot each filter in a different color
            subset = offsets_df.query(f"reference == 'n' and subarray == '{subarray}'")
            gb = subset.groupby('filter')
            for filt in gb.groups:
                group = gb.get_group(filt)
                errorbar = ax.errorbar(group[f'{coord}'], group[f'off_{coord}'],
                                       xerr=group[f'd{coord}'], yerr=group[f'off_d{coord}'],
                                       ls='', marker='o', 
                                       label=f"{filt} - {coron_filter}")
                color = errorbar.get_children()[0].get_color()
                if lines is not None:
                    line = lines.loc[(subarray, filt), 'off_'+coord]
                    line_x = np.array([0, aper.XSciSize]) #np.array([group[coord].max(), group[coord].min()])
                    dx = line_x[1]-line_x[0]
                    # stretch the line by 10% past the ends
                    line_x = np.array(line_x) + 0.10*dx*np.array([-1, 1])
                    line_y = line[0] + line[1]*line_x
                    ax.plot(line_x, line_y, ls='-.', color=color)
                if center_offsets is not None:
                    center_offset = center_offsets.loc[(subarray, filt)]
                    subarray_center = subarray_centers.loc[subarray[-4:]]
                    ax.scatter(subarray_center[coord], center_offset['d'+coord], marker='x', color=color)

            # dummy plots, for the legend
            if lines is not None:
                ax.plot([], [], color='k', ls='-.', label='linear fit')
            if center_offsets is not None:
                ax.scatter([], [], color='k', marker='x', label='center offset')
            ax.legend(ncol=2, loc='best', framealpha=0.2)
            ax.grid(True, alpha=1)

    if saveto is not None:
       fig.savefig(saveto, dpi=150)
    return fig

