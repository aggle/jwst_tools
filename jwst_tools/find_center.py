"""
find_center.py

Various tools for finding the centers of the coronagraphs
"""

from functools import wraps
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage

from astropy.io import fits

import radonCenter as rc

def radon_center(img, init_x, init_y, nangles=41, searchCenter_args={}):
    """
    Wrapper for radonCenter.searchCenter

    Parameters
    ----------
    img : np.array
        2-D image whose center you want to find
    init_x : float
        initial x (col) value
    init_y : float
        initial y (row) value
    nangles : int
        the number of angles to search for in the angular search region, which is 4 deg wide
    searchCenter_args : dict
        dictionary of arguments to pass to radonCenter.searchCenter()

    Output
    ------
    center : (float, float)
        (x, y) center as computed by radonCenter.searchCenter
    """
    # set the search angles
    h_angle = -4.5
    v_angle = 85.5
    thetas = np.concatenate([np.linspace(a-2, a+2, nangles) for a in [h_angle, v_angle]])
    args = {"m": 0.2,
            "theta": thetas,
            "decimals": 3}
    # override default args with the passed values
    args.update(searchCenter_args)
    center = rc.searchCenter(img, init_x, init_y, min(img.shape)/2,
                             **args)
    return center


def iterative_center_of_mass(image, fwhm=2, atol=1e-5):
    """
    Find the location of the maximum. This function is slow, so only used once we've identified where the center is,
    we redo the estimation with this function

    We use a iterative center of mass method.

    (0,0) is assumed to be the center of the lower leftmost pixel

    :param image: The image should be small (around 50px size)
    :type image: np.ndarray(y, x)
    :param float fwhm: [optional] Expected size of the PSF for the given image (2 by default)
    :param float atol: Absolute tolerance: convergence criterium for the iteration process

    :return: (y, x)
    :rtype: tuple(float, float)
    """
    ny, nx = image.shape

    # Renormalization between 0 and 1
    im_min = image.min()
    im_max = image.max()
    image = (image - im_min) / (im_max - im_min)

    # Threshold
    image[image < 0.3] = 0
    # P. Baudoz
    image -= 0.3
    image[image < 0] = 0

    #  compute the weigths for the ponderation
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2)))
    mult_pond = 1. / (2 * np.pi * sigma ** 2)

    #  compute the center of the image
    x_range = np.arange(nx)
    y_range = np.arange(ny)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)

    (y_center, x_center) = ndimage.center_of_mass(image)
    n_iter = 0
    dr = 2 * atol
    while dr > atol:
        n_iter += 1
        old_y, old_x = y_center, x_center
        pond_ref = mult_pond * np.exp(-((x_mesh - x_center) ** 2 + (y_mesh - y_center) ** 2) / (2 * sigma ** 2))
        (y_center, x_center) = ndimage.center_of_mass(image * pond_ref)

        dr = np.sqrt((old_x - x_center) ** 2 + (old_y - y_center) ** 2)

    print(f"cdg Converged in {n_iter} iterations, (x, y) = ({x_center}, {y_center})")

    return y_center, x_center
