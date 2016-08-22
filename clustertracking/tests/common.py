from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from pandas.util.testing import assert_frame_equal

from clustertracking.utils import cKDTree

def sort_positions(actual, expected):
    tree = cKDTree(actual)
    deviations, argsort = tree.query([expected])
    if len(set(range(len(actual))) - set(argsort[0])) > 0:
        raise AssertionError("Position sorting failed. At least one feature is "
                             "very far from where it should be.")
    return deviations, actual[argsort][0]


def assert_coordinates_close(actual, expected, atol):
    assert_equal(len(actual), len(expected))
    _, sorted_actual = sort_positions(actual, expected)
    assert_allclose(sorted_actual, expected, atol=atol)


def assert_traj_equal(actual, expected):
    assert_frame_equal(actual.drop('particle', 1), actual.drop('particle', 1))
    actual = actual.sort_values(['frame', 'x', 'y']).reset_index(drop=True)
    expected = expected.sort_values(['frame', 'x', 'y']).reset_index(drop=True)
    for p_actual in actual.particle.unique():
        actual_ind = actual.index[actual['particle'] == p_actual]
        p_expected = expected.loc[actual_ind[0], 'particle']
        expected_ind = expected.index[expected['particle'] == p_expected]
        assert_array_equal(actual_ind, expected_ind)
