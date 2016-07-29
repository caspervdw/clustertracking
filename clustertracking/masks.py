from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np

from .utils import validate_tuple


def slice_pad(image, center, radius):
    """Slice a single feature from an image, pad when appropriate."""
    radius = validate_tuple(radius, image.ndim)
    center = validate_tuple(center, image.ndim)

    corner = [int(round(c - r)) for (c, r) in zip(center, radius)]
    shape = [int(round(2*r + 1)) for r in radius]

    padding = [(max(-c, 0), max(c + s - o, 0))
               for (c, s, o) in zip(corner, shape, image.shape)]

    if np.any(padding):
        image = np.pad(image, padding, mode='constant')
        padded_corner = [c - min(c, 0) for c in corner]
        cropped = image[[slice(c, c+s) for (c, s) in zip(padded_corner, shape)]]
    else:
        cropped = image[[slice(c, c+s) for (c, s) in zip(corner, shape)]]

    return cropped, corner


def _slice(lower, upper, shape):
    ndim = len(shape)
    origin = [None] * ndim
    slices = [None] * ndim
    for i, sh, low, up in zip(range(ndim), shape, lower, upper):
        lower_bound_trunc = max(0, low)
        upper_bound_trunc = min(sh, up)
        slices[i] = slice(lower_bound_trunc, upper_bound_trunc)
        origin[i] = lower_bound_trunc
    return slices, origin


def _in_bounds(coords, shape, radius):
    ndim = len(shape)
    in_bounds = np.array([(coords[:, i] >= -r) & (coords[:, i] < sh + r)
                         for i, sh, r in zip(range(ndim), shape, radius)])
    return coords[np.all(in_bounds, axis=0)]


def slices_multiple(coords, shape, radius):
    """Creates the smallest box so that every coord in `coords` is in the box
    up to `radius` from the coordinate."""
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    coords = np.atleast_2d(np.round(coords).astype(np.int))
    coords = _in_bounds(coords, shape, radius)

    if len(coords) == 0:
        return [slice(None, 0)] * ndim, None

    return _slice(coords.min(axis=0) - radius,
                  coords.max(axis=0) + radius + 1, shape)


def slice_image(coords, image, radius):
    """Creates the smallest box so that every coord in `coords` is in the box
    up to `radius` from the coordinate."""
    slices, origin = slices_multiple(coords, image.shape, radius)
    return image[slices], origin  # mask origin


def binary_mask_multiple(coords_rel, shape, radius, include_edge=True,
                         return_masks=False):
    """Creates multiple elliptical masks.

    Parameters
    ----------
    coords_rel : ndarray (N x 2 or N x 3)
        coordinates
    shape : tuple
        shape of the image
    radius : number or tuple of number
        size of the masks
    """
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    coords_rel = np.atleast_2d(coords_rel)

    if include_edge:
        dist = [np.sum(((np.indices(shape).T - coord) / radius)**2, -1) <= 1
                for coord in coords_rel]
    else:
        dist = [np.sum(((np.indices(shape).T - coord) / radius)**2, -1) < 1
                for coord in coords_rel]
    mask_total = np.any(dist, axis=0).T
    masks_single = np.empty((len(coords_rel), mask_total.sum()), dtype=np.bool)
    if return_masks:
        for i, _dist in enumerate(dist):
            masks_single[i] = _dist.T[mask_total]
        return mask_total, masks_single
    else:
        return mask_total


def mask_image(coords, image, radius, origin=None, invert=False):
    """Masks an image with elliptical masks with size `radius`. At every coord
    in `coords`, the mask is applied to the image. When invert=True, the coords
    instead of the background will be made 0.
    Optionally, specify the topleft coordinate (origin) of the image."""
    if origin is not None:
        coords_rel = coords - np.array(origin)[np.newaxis, :]
    else:
        coords_rel = coords

    mask_cluster = binary_mask_multiple(coords_rel, image.shape, radius,
                                        include_edge=(not invert))

    if invert:
        return image * ~mask_cluster
    else:
        return image * mask_cluster
