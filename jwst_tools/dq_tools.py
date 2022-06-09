"""
Tools for manipulating and parsing DQ images
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits

from pysiaf import Siaf
from jwst import datamodels
from jwst.datamodels import dqflags

def separate_dq_flags(dq_img):
    """
    Separate a DQ image into a separate image for each DQ flag

    Parameters
    ----------
    dq_img: np.ndarray
      image (or cube) of DQ flags

    Output
    ------
    dq_flags: dict
      a dictionary whose entries are {dq_flag : image/cube of pixels with this flag}

    """
    dq_flags = {}
    for flag, bad_bitvalue in dqflags.pixel.items():
        if flag == 'GOOD':  # comparison to '0' is a special case
            img = ~np.bitwise_or(dq_img, bad_bitvalue).astype(bool)
        else:
            img = np.bitwise_and(dq_img, bad_bitvalue).astype(bool)
        # convert the image to boolean (True if flag present, False if not
        dq_flags[flag] = img
    return dq_flags


def plot_dq_flags(flag_dict):
    """Plot the distribution of each DQ flag in its own image"""
    ncols = 4
    nrows = np.ceil(len(flag_dict)/ncols).astype(int)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    for ax, (flag, img) in zip(axes.ravel(), flag_dict.items()):
        ax.set_title(flag, size='x-large')
        ax.pcolor(img, cmap='cividis')
    for ax in axes.ravel()[len(flag_dict):]:
        ax.set_visible(False)
    return fig
