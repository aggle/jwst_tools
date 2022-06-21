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


def plot_centroids_v_position(offsets_df):
    """
    Docstring goes here

    Parameters
    ----------
    any other parameters?
    ax : plt.axis [None]
      optional: provide the axis to draw on

    Output
    ------
    fig : plt.Figure instance
      if ax is given, returns a reference to the parent figure
    """
    if ax == None:
      fig, ax = plt.subplots(1, 1)
    else:
      fig = ax.get_figure()

    ## insert body here ##

    return fig
