from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import numpy as np
from scipy.ndimage.filters import correlate1d
from pims import Frame, pipeline
from .utils import validate_tuple
from trackpy.preprocessing import bandpass, scalefactor_to_gamut, scale_to_gamut
from trackpy.masks import gaussian_kernel

@pipeline
def lowpass(image, lshort, threshold=None):
    """Convolve with a Gaussian to remove short-wavelength noise.

    Parameters
    ----------
    image : ndarray
    lshort : small-scale cutoff (noise)
        give a tuple value for different sizes per dimension
        give int value for same value for all dimensions
    threshold : float or integer
        By default, 1 for integer images and 1/256. for float images.

    Returns
    -------
    result : array
        the filtered image

    See Also
    --------
    bandpass
    """
    lshort = validate_tuple(lshort, image.ndim)
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):
            threshold = 1
        else:
            threshold = 1/256.
    result = np.array(image, dtype=np.float)
    for (axis, size) in enumerate(lshort):
        if size > 0:
            correlate1d(result, gaussian_kernel(size, 4), axis,
                        output=result, mode='constant', cval=0.0)
    try:
        frame_no = image.frame_no
    except AttributeError:
        frame_no = None
    return Frame(np.where(result > threshold, result, 0), frame_no)

@pipeline
def preprocess(raw_image, noise_size=None, smoothing_size=None, threshold=None):
    if noise_size is not None:
        image = bandpass(raw_image, noise_size, smoothing_size, threshold)
        # Coerce the image into integer type. Rescale to fill dynamic range.
        if np.issubdtype(raw_image.dtype, np.integer):
            dtype = raw_image.dtype
        else:
            dtype = np.uint8
        scale_factor = scalefactor_to_gamut(image, dtype)
        image = scale_to_gamut(image, dtype, scale_factor)
    elif np.issubdtype(raw_image.dtype, np.integer):
        # Do nothing when image is already of integer type
        scale_factor = 1.
        image = raw_image
    else:
        # Coerce the image into uint8 type. Rescale to fill dynamic range.
        scale_factor = scalefactor_to_gamut(raw_image, np.uint8)
        image = scale_to_gamut(raw_image, np.uint8, scale_factor)
    try:
        frame_no = raw_image.frame_no
    except AttributeError:
        frame_no = None
    return Frame(image, frame_no,
                 metadata=dict(scale_factor=scale_factor))
