from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import numpy as np

from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import clustertracking as ct
from clustertracking.motion import orientation_df, diffusion_tensor

def random_walk(N):
    return np.cumsum(np.random.randn(N))

class TestDiffTensor2D(unittest.TestCase):
    def setUp(self):
        self.N = 2000

    def test_single(self):
        f = pd.DataFrame.from_dict(dict(x=random_walk(self.N),
                                        y=random_walk(self.N),
                                        frame=np.arange(self.N),
                                        particle=np.zeros(self.N),
                                        cluster=np.zeros(self.N)))
        pos, oren = orientation_df(f, 1, ndim=2)
        tensor = diffusion_tensor(pos, oren, ndim=2)

        expected = np.array([[0.5, 0], [0, 0.5]])
        assert_allclose(tensor[:2, :2], expected, rtol=0.1, atol=0.01)

    def test_dimer(self):
        # dimer fixed in y direction
        spacing = 1.
        f1 = pd.DataFrame.from_dict(dict(x=random_walk(self.N),
                                         y=random_walk(self.N),
                                         frame=np.arange(self.N),
                                         particle=np.zeros(self.N),
                                         cluster=np.zeros(self.N)))
        f2 = f1.copy()
        f2['y'] += spacing
        f2['particle'] = 1
        f = pd.concat([f1, f2])

        pos, oren = orientation_df(f, 2, ndim=2)
        tensor = diffusion_tensor(pos, oren, ndim=2)

        expected = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0]])
        assert_allclose(tensor, expected, rtol=0.1, atol=0.01)


class TestDiffTensor3D(unittest.TestCase):
    def setUp(self):
        self.N = 2000

    def test_single(self):
        f = pd.DataFrame.from_dict(dict(x=random_walk(self.N),
                                        y=random_walk(self.N),
                                        z=random_walk(self.N),
                                        frame=np.arange(self.N),
                                        particle=np.zeros(self.N),
                                        cluster=np.zeros(self.N)))
        pos, oren = orientation_df(f, 1, ndim=3)
        tensor = diffusion_tensor(pos, oren, ndim=3)

        expected = np.array([[0.5, 0, 0], [0, 0.5, 0],
                             [0, 0, 0.5]])
        assert_allclose(tensor[:3, :3], expected, rtol=0.1, atol=0.01)

    def test_dimer_fixed(self):
        # dimer fixed in y direction
        spacing = 1.
        f1 = pd.DataFrame.from_dict(dict(x=random_walk(self.N),
                                         y=random_walk(self.N),
                                         z=random_walk(self.N),
                                         frame=np.arange(self.N),
                                         particle=np.zeros(self.N),
                                         cluster=np.zeros(self.N)))
        f2 = f1.copy()
        f2['y'] += spacing
        f2['particle'] = 1
        f = pd.concat([f1, f2])

        pos, oren = orientation_df(f, 2, ndim=3)
        tensor = diffusion_tensor(pos, oren, ndim=3)

        expected = np.array([[0.5, 0, 0, 0, 0], [0, 0.5, 0, 0, 0],
                             [0, 0, 0.5, 0, 0], [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])
        assert_allclose(tensor[:5, :5], expected, rtol=0.1, atol=0.01)
    #
    # def test_dimer_rotating(self):
    #     spacing = 1.
    #     f1 = pd.DataFrame.from_dict(dict(x=random_walk(self.N),
    #                                      y=random_walk(self.N),
    #                                      z=random_walk(self.N),
    #                                      frame=np.arange(self.N),
    #                                      particle=np.zeros(self.N),
    #                                      cluster=np.zeros(self.N)))
    #     f2 = f1.copy()
    #
    #     angle = random_walk(self.N) * 0.1
    #     f2['y'] += np.sin(angle) * spacing
    #     f2['x'] += np.cos(angle) * spacing
    #     f2['particle'] = 1
    #     f = pd.concat([f1, f2])
    #
    #     pos, oren = orientation_df(f, 2, ndim=3)
    #     tensor = diffusion_tensor(pos, oren, ndim=3)
    #     tensor[3:, 3:] *= 100
    #     print(np.diag(tensor))
    #
    #     expected = np.array([[0.5, 0, 0, 0, 0], [0, 0.5, 0, 0, 0],
    #                          [0, 0, 0.5, 0, 0], [0, 0, 0, 0.1, 0],
    #                          [0, 0, 0, 0, 0.1]])
    #     assert_allclose(tensor[:5, :5], expected, rtol=0.1, atol=0.01)

