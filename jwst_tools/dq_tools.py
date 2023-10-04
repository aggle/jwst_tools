"""
Tools for manipulating and parsing DQ images

Usage:
- To get a DQ image, use `dq_img = get_dq_img(path/to/fits/file.fits)`.
  This is just a wrapper for fits.getdata(file, extname='DQ')
- To get a dictionary of flagged pixels, where the keys are the flags, use
  `dq_dictionary = separate_dq_flags(dq image)`.
- To plot the positions of the flagged pixels, separated by flag, use
  `plot_dq_flags(dq_dictionary)`
Each uses the output of the previous.

"""
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


from jwst.datamodels import dqflags


def separate_dq_flags(dq_img, flags=None):
    """
    Separate a DQ image into a separate image for each DQ flag

    Parameters
    ----------
    dq_img: np.ndarray
      image (or cube) of DQ flags
    flags: str or list
      (optional) a list of flag names to check for, if you only want to check
      for some flags. If None, checks for all flags

    Output
    ------
    dq_flags: dict
      a dictionary whose entries are {dq_flag : image/cube of pixels with this flag}

    """
    # if requested, limit the flags checked
    check_flags = list(dqflags.pixel.keys())
    if flags is not None:
        if isinstance(flags, str):
            flags = [flags]
        check_flags = flags
    dq_flags = {}
    for flag in check_flags:
        bad_bitvalue = dqflags.pixel[flag]
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


def get_dq_img(filename):
    """
    Pull the DQ image from a fits file

    Parameters
    ----------
    filename : str
      valid path to the file

    Output
    ------
    dq_img : np.array
      the DQ image. Flags can be parsed with `separate_dq_flags(dq_img)`

    """
    dq_img = fits.getdata(str(filename), extname='DQ')
    return dq_img


def usage():
    """Usage:
    - To get a DQ image, use `dq_img = get_dq_img(path/to/fits/file.fits)`.
      This is just a wrapper for fits.getdata(file, extname='DQ')
    - To get a dictionary of flagged pixels, where the keys are the flags, use
      `dq_dictionary = separate_dq_flags(dq image)`
    - To plot the positions of the flagged pixels, separated by flag, use
      `plot_dq_flags(dq_dictionary)`
    Each uses the output of the previous.
    """
    usage_str = usage.__doc__.replace("\t","")
    print(usage_str)
