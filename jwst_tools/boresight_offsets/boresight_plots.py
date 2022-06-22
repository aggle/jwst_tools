from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from astropy.io import fits

# local to jwst_tools
from .. import plot_utils


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
    nrows = len(centroids_df['filter'].unique())
    ncols = int(np.ceil(len(centroids_df)/nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows),
                            sharex='col', sharey='col')

    for col_ind, (name, group_ind) in enumerate(centroids_df.groupby('obs_num').groups.items()):
        group = centroids_df.loc[group_ind]
        for row_ind, ind in enumerate(group.index):
            ax = axes[row_ind, col_ind]

            row = group.loc[ind]
            filename = Path(row['path']) / row['filename']
            img, coords = plot_utils.img_cutout(fits.getdata(filename, 1), 
                                                row[['y', 'x']],
                                                31, 
                                                True)
            coords = [c-0.5 for c in coords]
            vmin, vmax = plot_utils.get_vlims(img, -1, 3)
            # full image plotting
        #     img = fits.getdata(mast_data_path / row['file'])
        #     coords = [np.arange(i+1)-0.5 for in img.shape[::-1]]

            if row_ind == 0:
                ax.set_title(f"Obs {row['obs_num']}", size='xx-large')
            if col_ind == 0:
                ax.set_ylabel(f"{row['filter']}", size='xx-large')


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
        
        ax.legend(ncol=1, loc='center left')
        ax.grid(True, alpha=1)

    if saveto is not None:
       fig.savefig(saveto, dpi=150)
    return fig


def plot_centroids_v_position_1d(offsets_df, saveto=None, lines=None, centers=None):
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
            # descriptions
            ax.set_title(subarray, size='x-large')
            ax.set_xlabel(f"{coord} [pix]")
            ax.set_ylabel(f"{coord} offset [pix]")
            # plot a line at 0 offset
            ax.axhline(0, ls='--', color='k', label='No offset')
            # plot each filter in a different color
            subset = offsets_df.query(f"reference == 'n' and subarray == '{subarray}'")
            gb = subset.groupby('filter')
            for filt in gb.groups:
                group = gb.get_group(filt)
                errorbar = ax.errorbar(group[f'{coord}'], group[f'off_{coord}'],
                                       xerr=group[f'd{coord}'], yerr=group[f'off_d{coord}'],
                                       ls='', marker='o', 
                                       label=f"{filt} - F{subarray}C")
                color = errorbar.get_children()[0].get_color()

            # dummy plots, for the legend
            ax.legend(ncol=2, loc='best', framealpha=0.2)
            ax.grid(True, alpha=1)

    if saveto is not None:
       fig.savefig(saveto, dpi=150)
    return fig

