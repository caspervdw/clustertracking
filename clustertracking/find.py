from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from .utils import validate_tuple, guess_pos_columns


class Clusters(object):
    @classmethod
    def from_pairs(cls, pairs, length):
        clusters = cls(range(length))
        for (a, b) in pairs:
            clusters.add(a, b)
        return clusters

    @classmethod
    def from_kdtree(cls, kdtree, separation):
        pairs = kdtree.query_pairs(separation)
        return cls.from_pairs(pairs, len(kdtree.data))

    @classmethod
    def from_coords(cls, coords, separation):
        pairs = cKDTree(coords / separation).query_pairs(1)
        return cls.from_pairs(pairs, len(coords))

    def __init__(self, indices):
        self._cl = {i: {i} for i in indices}
        self._f = list(indices)

    @property
    def clusters(self):
        return self._cl

    def __iter__(self):
        return (list(self._cl[k]) for k in self._cl)

    def add(self, a, b):
        i1 = self._f[a]
        i2 = self._f[b]
        if i1 != i2:  # if a and b are already clustered, do nothing
            self._cl[i1] = self._cl[i1].union(self._cl[i2])
            for f in self._cl[i2]:
                self._f[f] = i1
            del self._cl[i2]

    @property
    def cluster_id(self):
        return self._f

    @property
    def cluster_size(self):
        result = [None] * len(self._f)
        for cluster in self:
            for f in cluster:
                result[f] = len(cluster)
        return result


def iter_clusters(pos, separation):
    pos = np.atleast_2d(pos)
    pairs = cKDTree(pos / separation).query_pairs(1)
    clusters = Clusters(range(len(pos)))
    for (a, b) in pairs:
        clusters.add(a, b)
    return iter(clusters)


def _find(f, separation):
    """ Find clusters in a list or ndarray of coordinates.

    Parameters
    ----------
    f: iterable of coordinates
    separation: tuple of numbers
        Separation distance below which particles are considered inside cluster

    Returns
    ----------
    ids : ndarray of cluster IDs per particle
    sizes : ndarray of cluster sizes
    """
    # kdt setup
    pairs = cKDTree(np.array(f) / separation).query_pairs(1)

    clusters = Clusters(range(len(f)))
    for (a, b) in pairs:
        clusters.add(a, b)

    return clusters.cluster_id, clusters.cluster_size


def find_iter(f, separation, pos_columns=None, t_column='frame'):
    """ Find clusters in a DataFrame, returns generator iterating over frames.

    Parameters
    ----------
    f: DataFrame
        pandas DataFrame containing pos_columns and t_column
    separation: number or tuple
        Separation distance below which particles are considered inside cluster
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    t_column: string
        Column name containing the frame number (Default: 'frame')

    Returns
    ----------
    generator of:
        frame_no
        DataFrame with added cluster and cluster_size column
    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)

    next_id = 0

    for frame_no, f_frame in f.groupby(t_column):
        ids, sizes = _find(f_frame[pos_columns].values, separation)
        result = f_frame.copy()
        result['cluster'] = ids
        result['cluster_size'] = sizes
        result['cluster'] += next_id
        next_id = result['cluster'].max() + 1
        yield frame_no, result


def find_clusters(f, separation, pos_columns=None, t_column='frame'):
    """ Find clusters in a DataFrame of points from several frames.

    Parameters
    ----------
    f: DataFrame
        pandas DataFrame containing pos_columns and t_column
    separation: number or tuple
        Separation distance below which particles are considered inside cluster
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ['y', 'x'] (or ['z', 'y', 'x'] if 'z' exists)
    t_column: string
        Column name containing the frame number (Default: 'frame')

    Returns
    ----------
    DataFrame
    """
    if t_column not in f:
        f[t_column] = 0
        remove_t_column = True
    else:
        remove_t_column = False

    result = pd.concat((x[1] for x in find_iter(f, separation,
                                                pos_columns, t_column)))

    if remove_t_column:
        del f[t_column]

    return result


def where_close(pos, separation, intensity=None):
    """ Returns indices of features that are closer than separation from other
    features. When intensity is given, the one with the lowest intensity is
    returned: else the most topleft is returned (to avoid randomness)"""
    intensity = np.asarray(intensity)
    if len(pos) == 0:
        return []
    separation = validate_tuple(separation, pos.shape[1])
    if any([s == 0 for s in separation]):
        return []
    # Rescale positions, so that pairs are identified below a distance
    # of 1.
    pos_rescaled = pos / separation
    duplicates = cKDTree(pos_rescaled, 30).query_pairs(1 - 1e-7)
    if len(duplicates) == 0:
        return []
    index_0 = np.fromiter((x[0] for x in duplicates), dtype=int)
    index_1 = np.fromiter((x[1] for x in duplicates), dtype=int)
    if intensity is None:
        to_drop = np.where(np.sum(pos_rescaled[index_0], 1) >
                           np.sum(pos_rescaled[index_1], 1),
                           index_1, index_0)
    else:
        intensity_0 = intensity[index_0]
        intensity_1 = intensity[index_1]
        to_drop = np.where(intensity_0 > intensity_1, index_1, index_0)
        edge_cases = intensity_0 == intensity_1
        if np.any(edge_cases):
            index_0 = index_0[edge_cases]
            index_1 = index_1[edge_cases]
            to_drop[edge_cases] = np.where(np.sum(pos_rescaled[index_0], 1) >
                                           np.sum(pos_rescaled[index_1], 1),
                                           index_1, index_0)
    return np.unique(to_drop)


def drop_close(pos, separation, intensity=None):
    """ Removes features that are closer than separation from other features.
    When intensity is given, the one with the lowest intensity is dropped:
    else the most topleft is dropped (to avoid randomness)"""
    to_drop = where_close(pos, separation, intensity)
    return np.delete(pos, to_drop, axis=0)


def percentile_threshold(image, percentile):
    """Find grayscale threshold based on distribution in image."""

    not_black = image[np.nonzero(image)]
    if len(not_black) == 0:
        return np.nan
    return np.percentile(not_black, percentile)


def grey_dilation(image, separation, percentile=64, margin=None, precise=True):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    image : ndarray
        The algorithm works fastest when image is of integer type.
    separation : number or tuple of numbers
        Minimum separation between maxima. See precise for more information.
    percentile : float in range of [0,100], optional
        Features must have a peak brighter than pixels in this percentile.
        This helps eliminate spurious peaks. Default 64.
    margin : integer or tuple of integers, optional
        Zone of exclusion at edges of image. Default is ``separation / 2``.
    precise : boolean, optional
        Determines whether there will be an extra filtering step (``drop_close``)
        discarding features that are too close. Degrades performance.
        Because of the square kernel used, too many features are returned when
        precise=False. Default True.

    See Also
    --------
    drop_close : removes features that are too close to brighter features
    grey_dilation_legacy : local maxima finding routine used until trackpy v0.3
    """
    ndim = image.ndim
    separation = validate_tuple(separation, ndim)
    if margin is None:
        margin = tuple([int(s / 2) for s in separation])

    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)
    if np.isnan(threshold):
        return np.empty((0, ndim))

    # Find the largest box that fits inside the ellipse given by separation
    size = [int(2 * s / np.sqrt(ndim)) for s in separation]

    # The intersection of the image with its dilation gives local maxima.
    dilation = ndimage.grey_dilation(image, size, mode='constant')
    maxima = (image == dilation) & (image > threshold)
    if np.sum(maxima) == 0:
        return np.empty((0, ndim))

    pos = np.vstack(np.where(maxima)).T

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]

    if len(pos) == 0:
        return np.empty((0, ndim))

    # Remove local maxima that are too close to each other
    if precise:
        pos = drop_close(pos, separation, image[maxima][~near_edge])

    return pos
