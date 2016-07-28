from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import numpy as np

from numpy.testing import assert_equal
import pandas as pd
import clustertracking as ct


def dummy_cluster(N, center, separation, ndim=2):
    devs = (np.random.random((N, ndim)) - 0.5) * separation / np.sqrt(ndim)
    return np.array(center)[np.newaxis, :] + devs


def dummy_clusters(N, max_size, separation, ndim=2):
    center = [separation] * ndim
    sizes = np.random.randint(1, max_size, N)
    displ = (np.random.random((N, ndim)) + 2) * separation
    res = []
    for i, size in enumerate(sizes):
        center += displ[i]
        res.append(dummy_cluster(size, center, separation, ndim))
    return res


def pos_to_df(pos):
    pos_a = np.concatenate(pos)
    ndim = pos_a.shape[1]
    pos_columns = ['z', 'y', 'x'][-ndim:]
    return pd.DataFrame(pos_a, columns=pos_columns)


class TestFindClusters(unittest.TestCase):
    def setUp(self):
        self.N = 10

    def test_single_cluster_2D(self):
        separation = np.random.random(self.N) * 10
        for sep in separation:
            pos = dummy_clusters(1, 10, sep)
            df = pos_to_df(pos)
            df = ct.find_clusters(df, sep)
            assert_equal(df['cluster_size'].values, len(pos[0]))

    def test_multiple_clusters_2D(self):
        numbers = np.random.randint(1, 10, self.N)
        for number in numbers:
            pos = dummy_clusters(number, 10, 1)
            df = pos_to_df(pos)
            df = ct.find_clusters(df, 1)
            assert_equal(df['cluster'].nunique(), number)

    def test_single_cluster_3D(self):
        separation = np.random.random(self.N) * 10
        for sep in separation:
            pos = dummy_clusters(1, 10, sep, 3)
            df = pos_to_df(pos)
            df = ct.find_clusters(df, sep)
            assert_equal(df['cluster_size'].values, len(pos[0]))

    def test_multiple_clusters_3D(self):
        numbers = np.random.randint(1, 10, self.N)
        for number in numbers:
            pos = dummy_clusters(number, 10, 1, 3)
            df = pos_to_df(pos)
            df = ct.find_clusters(df, 1)
            assert_equal(df['cluster'].nunique(), number)

    def test_line_cluster(self):
        separation = np.random.random(self.N) * 10
        angle = np.random.random(self.N) * 2 * np.pi
        ds = np.array([np.cos(angle), np.sin(angle)]).T
        for vec, sep in zip(ds, separation):
            pos = np.arange(10)[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = ct.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = ct.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            df = pos_to_df([pos[::-1]])
            df = ct.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = ct.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            ind = np.arange(10)
            np.random.shuffle(ind)
            pos = ind[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = ct.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = ct.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

    def test_line_cluster_3D(self):
        separation = np.random.random(self.N) * 10
        phi = np.random.random(self.N) * 2 * np.pi
        theta = np.random.random(self.N) * np.pi
        ds = np.array([np.cos(theta),
                       np.cos(phi)*np.sin(theta),
                       np.sin(phi)*np.sin(theta)]).T
        for vec, sep in zip(ds, separation):
            pos = np.arange(10)[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = ct.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = ct.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            df = pos_to_df([pos[::-1]])
            df = ct.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = ct.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            ind = np.arange(10)
            np.random.shuffle(ind)
            pos = ind[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = ct.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = ct.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
