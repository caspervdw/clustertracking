from __future__ import (division, print_function, unicode_literals,
                        absolute_import)
import six
import numpy as np

from trackpy.utils import validate_tuple


def guess_pos_columns(f):
    if 'z' in f:
        pos_columns = ['z', 'y', 'x']
    else:
        pos_columns = ['y', 'x']
    return pos_columns


def obtain_size_columns(isotropic, pos_columns=None):
    if isotropic:
        size_columns = ['size']
    else:
        size_columns = ['size_{}'.format(p) for p in pos_columns]
    return size_columns


def default_pos_columns(ndim):
    return ['z', 'y', 'x'][-ndim:]


def default_size_columns(ndim, isotropic):
    if isotropic:
        size_columns = ['size']
    else:
        size_columns = ['size_z', 'size_y', 'size_x'][-ndim:]
    return size_columns


def is_isotropic(value):
    if hasattr(value, '__iter__'):
        return np.all(value[1:] == value[:-1])
    else:
        return True


class ReaderCached(object):
    def __init__(self, reader):
        self.reader = reader
        self._cache = None
        self._cache_i = None

    def __getitem__(self, i):
        if self._cache_i == i:
            return self._cache.copy()
        else:
            value = self.reader[i]
            self._cache = value.copy()
            return value

    def __repr__(self):
        return repr(self.reader) + "\nWrapped in ReaderCached"

    def __getattr__(self, attr):
        return getattr(self.reader, attr)


def mass_to_max(mass, size, ndim):
    if hasattr(size, '__iter__'):
        assert len(size) == ndim
        return mass / (np.pi * np.prod(size))
    else:
        return mass / (np.pi * size**ndim)


def max_to_mass(max_value, size, ndim):
    if hasattr(size, '__iter__'):
        assert len(size) == ndim
        return max_value * (np.pi * np.prod(size))
    else:
        return max_value * (np.pi * size**ndim)


class RefineException(Exception):
    pass
