from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from trackpy.utils import validate_tuple
from trackpy import refine as refine_com
from clustertracking import refine_leastsq, train_leastsq
from clustertracking.preprocessing import lowpass, preprocess
from clustertracking.artificial import (SimulatedImage, feat_gauss, rot_2d,
                                        rot_3d, draw_feature, draw_cluster)
from clustertracking.constraints import dimer, trimer, tetramer, dimer_global
from clustertracking.tests.common import assert_coordinates_close
from nose import SkipTest

SIGNAL = 160
NOISE_IMPERFECT = 16  # S/N of 10
NOISE_NOISY = 48      # S/N of 3
DISC_SIZE = 0.5
RING_THICKNESS = 0.2
SIZE_2D = 4.
SIZE_3D = 4.
SIZE_2D_ANISOTROPIC = (5., 3.)
SIZE_3D_ANISOTROPIC = (3., 5., 5.)

class RefineTsts(object):
    skip = False
    dtype = np.uint8
    signal = SIGNAL
    repeats = 20
    noise = NOISE_IMPERFECT
    pos_diff = 0.5  # in fraction of size
    train_params = []
    signal_dev = 0.2
    size_dev = 0.2
    const_rtol = 1E-7
    com_pos_atol = 0.1  # tolerance for center-of-mass position test
    precision_perfect = 0.01      # in pixels
    precision_imperfect = 0.05    # in pixels
    precision_noisy = 0.1         # in pixels
    accuracy_perfect = 0.01       # in units of size
    accuracy_imperfect = 0.01     # in units of size
    accuracy_noisy = 0.01         # in units of size
    accuracy_constrained = 0.001  # in units of size
    signal_rtol_perfect = 0.01
    signal_rtol_imperfect = 0.5  # accounting for 10% of noise
    size_rtol_perfect = 0.01
    size_rtol_imperfect = 0.5    # accounting for 10% of noise
    train_rtol = 0.05
    bounds = dict(signal=(20, 2000), size=(.9, 9))

    @classmethod
    def setUpClass(cls):
        cls.size = validate_tuple(cls.size, cls.ndim)
        if not hasattr(cls, 'diameter'):
            cls.diameter = tuple([int(s * 4) for s in cls.size])
        else:
            cls.diameter = validate_tuple(cls.diameter, cls.ndim)
        cls.separation = tuple([d * 2 for d in cls.diameter])
        cls.radius = tuple([int(d // 2) for d in cls.diameter])
        cls.isotropic = np.all([cls.diameter[1:] == cls.diameter[:-1]])
        cls.pos_columns = ['z', 'y', 'x'][-cls.ndim:]
        cls.pos_columns_err = [col + '_std' for col in cls.pos_columns]
        if cls.isotropic:
            cls.size_columns = ['size']
        else:
            cls.size_columns = ['size_z', 'size_y', 'size_x'][-cls.ndim:]
        cls.names = cls.pos_columns + ['signal'] + cls.size_columns
        # cls.rtol = [None] * len(cls.im.pos_columns)
        # cls.atol = [0.1] * len(cls.im.pos_columns)
        cls.bounds = dict()
        if not hasattr(cls, 'param_mode'):
            cls.param_mode = dict(signal='var', size='const')
        if not hasattr(cls, 'feat_kwargs'):
            cls.feat_kwargs = dict()
        if not hasattr(cls, 'param_val'):
            cls.param_val = dict()

    def setUp(self):
        if self.skip:
            raise SkipTest()

    def get_image(self, noise=0, signal_dev=0., size_dev=0., separation=None,
                  denoise_size=None, smoothing_size=None, N=None):
        if N is None:
            N = self.repeats
        if separation is None:
            separation = self.separation
        margin = self.separation
        Nsqrt = int(N**(1/self.ndim) + 0.9999)
        pos = np.meshgrid(*[np.arange(0, s * Nsqrt, s) for s in separation],
                          indexing='ij')
        pos = np.array([p.ravel() for p in pos], dtype=np.float).T[:N] + margin
        pos += (np.random.random(pos.shape) - 0.5)  #randomize subpixel location
        shape = tuple(np.max(pos, axis=0).astype(np.int) + margin)
        if signal_dev > 0:
            signal = self.signal * np.random.uniform(1-signal_dev, 1+signal_dev,
                                                     N)
        else:
            signal = np.repeat(self.signal, N)
        if size_dev > 0:
            size = np.array([self.size]) * np.random.uniform(1-size_dev,
                                                             1+size_dev,
                                                             (N, 1))
        else:
            size = np.repeat([self.size], N, axis=0)

        image = np.zeros(shape, dtype=self.dtype)
        for _pos, _signal, _size in zip(pos, signal, size):
            draw_feature(image, _pos, _size, _signal,
                         self.feat_func, **self.feat_kwargs)
        if self.isotropic:
            size = size[:, 0]

        if noise > 0:
            image = image + np.random.poisson(noise, shape)
            if image.max() <= 255:
                image = image.astype(np.uint8)
            if smoothing_size is not None and denoise_size is None:
                raise ValueError('Cannot do smoothing without denoising')
            if smoothing_size is not None:
                image = preprocess(image, denoise_size, smoothing_size)
            elif denoise_size is not None:
                image = lowpass(image, denoise_size)

        return image, (pos, signal, size)

    def get_image_clusters(self, cluster_size, hard_radius=1., noise=0,
                           signal_dev=0, size_dev=0, denoise_size=None,
                           smoothing_size=None, angle=None):
        N = self.repeats
        separation = [int(sep + 2 * hard_radius * s)
                      for (sep, s) in zip(self.separation, self.size)]
        margin = self.separation
        Nsqrt = int(N**(1/self.ndim) + 0.9999)
        pos = np.meshgrid(*[np.arange(0, s * Nsqrt, s) for s in separation],
                          indexing='ij')
        pos = np.array([p.ravel() for p in pos], dtype=np.float).T[:N] + margin
        pos += (np.random.random(pos.shape) - 0.5)  #randomize subpixel location
        if self.ndim == 2 and angle is None:
            angles = np.random.uniform(0, 2*np.pi, N)
        elif self.ndim == 3 and angle is None:
            angles = np.random.uniform(0, 2*np.pi, (N, 3))
        elif self.ndim == 2 and angle is not None:
            angles = np.repeat(angle, N)
        elif self.ndim == 3 and angle is not None:
            angles = np.repeat([angle], N, axis=0)

        shape = tuple(np.max(pos, axis=0).astype(np.int) + margin)
        if signal_dev > 0:
            signal = self.signal * np.random.uniform(1-signal_dev, 1+signal_dev,
                                                     N)
        else:
            signal = np.repeat(self.signal, N)
        if size_dev > 0:
            size = np.array([self.size]) * np.random.uniform(1-size_dev,
                                                             1+size_dev,
                                                             (N, 1))
        else:
            size = np.repeat([self.size], N, axis=0)

        image = np.zeros(shape, dtype=self.dtype)
        coords = []
        for _pos, _signal, _size, _angle in zip(pos, signal, size, angles):
            coords.extend(draw_cluster(image, _pos, _size, cluster_size,
                                       hard_radius, _angle, max_value=_signal,
                                       feat_func=self.feat_func,
                                       **self.feat_kwargs))
        coords = np.array(coords)
        signal = np.repeat(signal, cluster_size)
        size = np.repeat(size, cluster_size, axis=0)
        if self.isotropic:
            size = size[:, 0]

        if noise > 0:
            image = image + np.random.poisson(noise, shape)
            if image.max() <= 255:
                image = image.astype(np.uint8)
            if smoothing_size is not None and denoise_size is None:
                raise ValueError('Cannot do smoothing without denoising')
            if smoothing_size is not None:
                image = preprocess(image, denoise_size, smoothing_size)
            elif denoise_size is not None:
                image = lowpass(image, denoise_size)

        return image, (coords, signal, size), (pos, angles)

    def to_dataframe(self, coords, signal, size, cluster_size=1):
        f0 = pd.DataFrame(coords, columns=self.pos_columns)
        f0['signal'] = signal
        f0['signal'] = f0['signal'].astype(np.float)
        if self.isotropic:
            f0[self.size_columns[0]] = size[0]
            f0[self.size_columns[0]] = f0[self.size_columns[0]].astype(np.float)
        else:
            for col, s in zip(self.size_columns, size):
                f0[col] = s
                f0[col] = f0[col].astype(np.float)
        f0['cluster'] = np.repeat(np.arange(len(coords) // cluster_size),
                                  cluster_size)
        f0['cluster_size'] = cluster_size
        return f0

    def from_dataframe(self, df):
        pos = df[self.pos_columns].values
        try:
            pos_err = df[self.pos_columns_err].values
        except KeyError:
            pos_err = None
        signal = df['signal'].values
        if self.isotropic:
            size = df[self.size_columns[0]].values
        else:
            size = df[self.size_columns].values
        return pos, signal, size, pos_err

    def from_tp_ndarray(self, ndarray):
        ndim = self.ndim
        pos = ndarray[:, ndim-1:None:-1]
        if self.isotropic:
            size = ndarray[:, ndim + 1]
            signal = ndarray[:, ndim + 3]
        else:
            size = ndarray[:, 2*ndim:ndim:-1]
            signal = ndarray[:, 2*ndim + 2]
        return pos, signal, size, None

    def get_deviations(self, actual_pos, expected_pos, cluster_size,
                       expected_center, expected_angle):
        """Obtain deviations in cluster coordinate system"""
        pos_diff_rot = np.full((self.repeats, cluster_size * self.ndim), np.nan)
        if self.ndim == 2:
            rot_func = lambda x, y: np.dot(x, rot_2d(y))
        elif self.ndim == 3:
            rot_func = lambda x, y: np.dot(x, rot_3d(y))
        for n, (center, angle) in enumerate(zip(expected_center,
                                                expected_angle)):
            cluster = slice(n * cluster_size, (n + 1) * cluster_size)
            # rotate actual positions around expected center
            actual_rot = rot_func(actual_pos[cluster] - center, angle) + center
            # rotate expected positions around expected center
            expected_rot = rot_func(expected_pos[cluster] - center, angle) + center
            pos_diff_rot[n] = (actual_rot - expected_rot).ravel()
        return pos_diff_rot
    #
    # def sort(self, actual, expected_pos):
    #     return actual
    #     pos, signal, size, pos_err = actual
    #     tree = cKDTree(pos)
    #     deviations, argsort = tree.query(expected_pos)
    #     if len(set(range(len(pos))) - set(argsort)) > 0:
    #         raise AssertionError("Position sorting failed. At least one feature is "
    #                              "very far from where it should be.")
    #     if pos_err is not None:
    #         pos_err = pos_err[argsort]
    #     return pos[argsort], signal[argsort], size[argsort], pos_err

    def gen_p0_coords(self, expected_pos, pos_diff):
        # generate random points in a box
        OVERSAMPLING = 10
        N = expected_pos.shape[0]
        p0_pos_diff = np.array([self.size]) * pos_diff
        deviations = (np.random.random((OVERSAMPLING*N, self.ndim)) - 0.5) * p0_pos_diff * 2
        # calculate algebraic distance of each deviation
        dist = np.sum((deviations / p0_pos_diff)**2, axis=1)
        # drop the ones that are too far away
        deviations = deviations[dist <= 1]
        deviations = deviations[:N]
        return expected_pos + deviations

    def compute_deviations(self, actual, expected):
        actual_pos, actual_signal, actual_size, pos_err = actual
        expected_pos, expected_signal, expected_size = expected
        # values to be tested
        result = dict()
        deviations = expected_pos - actual_pos
        # rms absolute positional deviation
        result['pos'] = np.mean(deviations**2)**0.5
        # per direction
        moment1 = np.mean(deviations, axis=0)
        moment2 = np.mean(deviations**2, axis=0)
        for col, _m1, _m2, _pos in zip(self.pos_columns, moment1, moment2,
                                       actual_pos.T):
            result[col + '_subpx'] = np.histogram(_pos % 1,
                                                  bins=np.arange(0, 1.1, 0.1))[0]
            result[col] = np.sqrt(_m2 - _m1**2)
            result[col + '_rms'] = np.sqrt(_m2)
            result[col + '_mean'] = _m1
        if pos_err is not None:
            result['pos_err_mean'] = np.mean(pos_err)
            print(self._testMethodName, result['pos_err_mean'] - result['pos'],
                  100 * (result['pos_err_mean'] / result['pos'] - 1), '%')
            # for col, _err in zip(self.pos_columns, pos_err.T):
            #     result[col + '_err_mean'] = np.mean(_err)
            #     result[col + '_err_accuracy'] = np.mean(_err) - result[col + '_rms']
            #     result[col + '_err_precision'] = np.mean((_err - result[col + '_rms'])**2)**0.5
        # rms relative signal deviation
        result['signal'] = np.mean((1 - actual_signal / expected_signal)**2)**0.5
        # rms relative size deviation
        # averaged over all directions
        result['size'] = np.mean((1 - actual_size / expected_size)**2)**0.5
        if not self.isotropic:
            err = np.mean((1 - actual_size / expected_size)**2, axis=0)**0.5
            for col, _err in zip(self.size_columns, err):
                result[col] = _err
        return result

    def compute_deviations_cluster(self, actual, expected, deviations,
                                   pos_err=None):
        actual_pos, actual_signal, actual_size, _ = actual
        expected_pos, expected_signal, expected_size = expected
        cluster_size = deviations.shape[1] // self.ndim
        # values to be tested
        result = dict()
        # rms absolute positional deviation
        result['pos'] = np.mean(deviations**2)**0.5

        cluster_pos_cols = [pos + str(i) for i in range(cluster_size)
                            for pos in self.pos_columns]
        for col, _pos in zip(self.pos_columns, actual_pos.T):
            result[col + '_subpx'] = np.histogram(_pos % 1,
                                                  bins=np.arange(0, 1.1, 0.1))[0]
        # per direction in cluster axes system
        moment1 = np.mean(deviations, axis=0)
        moment2 = np.mean(deviations**2, axis=0)
        for col, _moment1, _moment2 in zip(cluster_pos_cols, moment1, moment2):
            result[col] = np.sqrt(_moment2 - _moment1**2)
            result[col + '_mean'] = _moment1
            result[col + '_rms'] = np.sqrt(_moment2)
        if pos_err is not None:
            for col, _err in zip(cluster_pos_cols, pos_err):
                result[col + '_err_mean'] = np.mean(_err)
                result[col + '_err_accuracy'] = np.mean(_err) - result[col + '_rms']
                result[col + '_err_precision'] = np.mean((_err - result[col + '_rms'])**2)**0.5
        if cluster_size == 2:
            result['parr'] = np.sqrt(0.5*(result['x0']**2 + result['x1']**2))
            result['parr_mean'] = (result['x1_mean'] - result['x0_mean']) / 2
            result['parr_rms'] = np.sqrt(0.5*(result['x0_rms']**2 + result['x1_rms']**2))
            result['perp'] = np.sqrt(0.5*(result['y0']**2 + result['y1']**2))
            result['perp_mean'] = (result['y1_mean'] - result['y0_mean']) / 2
            result['perp_rms'] = np.sqrt(0.5*(result['y0_rms']**2 + result['y1_rms']**2))
            if pos_err is not None:
                result['parr_err_mean'] = (result['x1_err_mean'] - result['x0_err_mean']) / 2
                result['perp_err_mean'] = (result['y1_err_mean'] - result['y0_err_mean']) / 2
                result['parr_err_accuracy'] = (result['x1_err_accuracy'] - result['x0_err_accuracy']) / 2
                result['perp_err_accuracy'] = (result['y1_err_accuracy'] - result['y0_err_accuracy']) / 2
                result['parr_err_precision'] = np.sqrt((result['x0_err_precision']**2 + result['x0_err_precision']**2))
                result['perp_err_precision'] = np.sqrt((result['y0_err_precision']**2 + result['y0_err_precision']**2))
        # rms relative signal deviation
        result['signal'] = np.mean((1 - actual_signal / expected_signal)**2)**0.5
        # rms relative size deviation
        # averaged over all directions
        result['size'] = np.mean((1 - actual_size / expected_size)**2)**0.5
        if not self.isotropic:
            err = np.mean((1 - actual_size / expected_size)**2, axis=0)**0.5
            for col, _err in zip(self.size_columns, err):
                result[col] = _err
        return result

    # def train(self, train_N=20):
    #     pos_diff = 0.1
    #
    #     image, expected = self.get_image(noise=NOISE_IMPERFECT,
    #                                      signal_dev=self.signal_dev,
    #                                      size_dev=0, N=train_N,
    #                                      separation=[d*4 for d in self.diameter])
    #     expected_pos, expected_signal, expected_size = expected
    #
    #     p0_coords = self.gen_p0_coords(expected_pos, pos_diff)
    #     # generate noisy size initial conditions
    #     p0_size = np.array([self.size]) * np.random.uniform(1-self.size_dev,
    #                                                         1+self.size_dev,
    #                                                         (len(expected_pos), 1))
    #
    #     f0 = self.to_dataframe(p0_coords, self.signal, p0_size.T)
    #
    #     # the fit function should be updated such that it is defined everywhere
    #     # in the ROI with arbitrary center location. this means that we need to
    #     # double the fit mask size here
    #     fit_diameter = tuple([d * 2 for d in self.diameter])
    #
    #     param_val = train_leastsq(f0, image, fit_diameter, self.diameter,
    #                               self.fit_func, bounds=self.bounds)
    #     return param_val


    def refine(self, pos_diff=None, signal_dev=None, size_dev=None, noise=None,
               param_mode=None, denoise_size=None, smoothing_size=None,
               **kwargs):
        """
        Parameters
        ----------
        noise : integer
            noise level
        pos_diff :
            pixels deviation of p0 from true feature location
        signal_dev :
            deviation of feature signal with respect to p0
        size_dev :
            deviation of feature size with respect to p0
        """
        if param_mode is None:
            param_mode = self.param_mode
        else:
            param_mode = dict(self.param_mode, **param_mode)
        if pos_diff is None:
            pos_diff = self.pos_diff
        if signal_dev is None:
            signal_dev = self.signal_dev
        if size_dev is None:
            size_dev = self.size_dev
        if noise is None:
            noise = self.noise
        # generate image with array of features and deviating signal and size
        image, expected = self.get_image(noise, signal_dev, size_dev,
                                         denoise_size=denoise_size,
                                         smoothing_size=smoothing_size)
        expected_pos, expected_signal, expected_size = expected
        p0_pos = self.gen_p0_coords(expected_pos, pos_diff)
        f0 = self.to_dataframe(p0_pos, self.signal, self.size)
        # Set an estimate for the background value. Helps convergence,
        # especially for 3D tests with high noise (tested here: S/N = 3)
        f0['background'] = noise / 2

        actual = refine_leastsq(f0, image, self.diameter, separation=None,
                                param_mode=dict(self.param_mode, **param_mode),
                                param_val=self.param_val,
                                pos_columns=self.pos_columns,
                                t_column='frame',
                                fit_function=self.fit_func,
                                bounds=self.bounds,
                                **kwargs)

        assert not np.any(np.isnan(actual['cost']))

        actual = self.from_dataframe(actual)
       # actual = self.sort(actual, expected_pos)
        return self.compute_deviations(actual, expected)


    def refine_com(self, pos_diff=None, signal_dev=None, size_dev=None,
                   noise=None, denoise_size=1, smoothing_size=None, **kwargs):
        """
        Parameters
        ----------
        noise : integer
            noise level
        pos_diff :
            pixels deviation of p0 from true feature location
        signal_dev :
            deviation of feature signal with respect to p0
        size_dev :
            deviation of feature size with respect to p0
        """
        if pos_diff is None:
            pos_diff = self.pos_diff
        if signal_dev is None:
            signal_dev = self.signal_dev
        if size_dev is None:
            size_dev = self.size_dev
        if noise is None:
            noise = self.noise
        # generate image with array of features and deviating signal and size
        image, expected = self.get_image(noise, signal_dev, size_dev,
                                         denoise_size=denoise_size,
                                         smoothing_size=smoothing_size)
        expected_pos, expected_signal, expected_size = expected
        p0_pos = self.gen_p0_coords(expected_pos, pos_diff)

        actual = refine_com(image, image, self.radius, p0_pos, **kwargs)

        actual = self.from_tp_ndarray(actual)
        # actual = self.sort(actual, expected_pos)
        return self.compute_deviations(actual, expected)


    def refine_cluster(self, cluster_size, hard_radius, pos_diff=None,
                       signal_dev=None, size_dev=None, noise=None,
                       param_mode=None, denoise_size=None, smoothing_size=None,
                       angle=None, **kwargs):
        """
        Parameters
        ----------
        noise : integer
            noise level
        pos_diff :
            pixels deviation of p0 from true feature location
        signal_dev :
            deviation of feature signal with respect to p0
        size_dev :
            deviation of feature size with respect to p0
        """
        if param_mode is None:
            param_mode = self.param_mode
        else:
            param_mode = dict(self.param_mode, **param_mode)
        if pos_diff is None:
            pos_diff = self.pos_diff
        if signal_dev is None:
            signal_dev = self.signal_dev
        if size_dev is None:
            size_dev = self.size_dev
        if noise is None:
            noise = self.noise
        # generate image with array of features and deviating signal and size
        image, expected, clusters = self.get_image_clusters(cluster_size,
                                                            hard_radius,
                                                            noise, signal_dev,
                                                            size_dev,
                                                            denoise_size,
                                                            smoothing_size,
                                                            angle)
        expected_pos, expected_signal, expected_size = expected
        expected_center, expected_angle = clusters
        p0_pos = self.gen_p0_coords(expected_pos, pos_diff)
        f0 = self.to_dataframe(p0_pos, self.signal, self.size, cluster_size)
        # Set an estimate for the background value. Helps convergence,
        # especially for 3D tests with high noise (tested here: S/N = 3)
        f0['background'] = noise / 2

        actual = refine_leastsq(f0, image, self.diameter, separation=None,
                                param_mode=dict(self.param_mode, **param_mode),
                                param_val=self.param_val,
                                pos_columns=self.pos_columns,
                                t_column='frame',
                                fit_function=self.fit_func,
                                bounds=self.bounds, **kwargs)

        assert not np.any(np.isnan(actual['cost']))
        assert np.all(actual['cluster_size'] <= cluster_size)

        actual = self.from_dataframe(actual)
        # actual_pos, actual_signal, actual_size, pos_err = self.sort(actual, expected_pos)
        actual_pos, actual_signal, actual_size, pos_err = actual
        deviations = self.get_deviations(actual_pos, expected_pos, cluster_size,
                                         expected_center, expected_angle)
        return self.compute_deviations_cluster(actual, expected, deviations)


    def refine_cluster_com(self, cluster_size, hard_radius, pos_diff=None,
                           signal_dev=None, size_dev=None, noise=None,
                           denoise_size=1, smoothing_size=None, angle=None,
                           **kwargs):
        """
        Parameters
        ----------
        noise : integer
            noise level
        pos_diff :
            pixels deviation of p0 from true feature location
        signal_dev :
            deviation of feature signal with respect to p0
        size_dev :
            deviation of feature size with respect to p0
        """
        if pos_diff is None:
            pos_diff = self.pos_diff
        if signal_dev is None:
            signal_dev = self.signal_dev
        if size_dev is None:
            size_dev = self.size_dev
        if noise is None:
            noise = self.noise
        # generate image with array of features and deviating signal and size
        image, expected, clusters = self.get_image_clusters(cluster_size,
                                                            hard_radius,
                                                            noise, signal_dev,
                                                            size_dev,
                                                            denoise_size,
                                                            smoothing_size,
                                                            angle)
        expected_pos, expected_signal, expected_size = expected
        expected_center, expected_angle = clusters
        p0_pos = self.gen_p0_coords(expected_pos, pos_diff)

        actual = refine_com(image, image, self.radius, p0_pos, **kwargs)

        actual = self.from_tp_ndarray(actual)
        # actual_pos, actual_signal, actual_size, pos_err = self.sort(actual, expected_pos)
        actual_pos, actual_signal, actual_size, pos_err = actual
        deviations = self.get_deviations(actual_pos, expected_pos, cluster_size,
                                         expected_center, expected_angle)
        return self.compute_deviations_cluster(actual, expected, deviations)

    # def test_train(self):
    #     param_val = self.train()
    #     print(param_val)
    #     for p in self.param_val:
    #         self.assertLess(abs(1 - self.param_val[p] / param_val[p]),
    #                         self.train_rtol)

    def test_perfect_com(self):
        # sanity check for test
        devs = self.refine_com(signal_dev=0, size_dev=0, noise=0)
        self.assertLess(devs['pos'], self.com_pos_atol)

    def test_perfect_const(self):
        # const signal and size
        devs = self.refine(param_mode=dict(signal='const', size='const'),
                           signal_dev=0, size_dev=0, noise=0)
        if 'signal' not in self.param_val:
            # test this once: constant means constant
            # unless we are in trained mode
            self.assertLess(devs['signal'], self.const_rtol)
        if 'size' not in self.param_val:
            self.assertLess(devs['size'], self.const_rtol)
        self.assertLess(devs['pos'], self.precision_perfect)

    def test_perfect_var_signal(self):
        # var signal, const size
        devs = self.refine(param_mode=dict(signal='var', size='const'), noise=0,
                           signal_dev=self.signal_dev, size_dev=0)
        self.assertLess(devs['signal'], self.signal_rtol_perfect)
        self.assertLess(devs['pos'], self.precision_perfect)

    def test_perfect_var_size(self):
        # const signal, var size
        devs = self.refine(param_mode=dict(signal='const', size='var'), noise=0,
                           signal_dev=0, size_dev=self.size_dev)
        self.assertLess(devs['size'], self.size_rtol_perfect)
        self.assertLess(devs['pos'], self.precision_perfect)

    def test_perfect_var(self):
        # var signal, var size
        devs = self.refine(param_mode=dict(signal='var', size='var'),
                           noise=0, signal_dev=self.signal_dev,
                           size_dev=self.size_dev)
        self.assertLess(devs['signal'], self.signal_rtol_perfect)
        self.assertLess(devs['size'], self.size_rtol_perfect)
        self.assertLess(devs['pos'], self.precision_perfect)

    def test_imperfect_const(self):
        # const signal and size
        devs = self.refine(param_mode=dict(signal='const', size='const'),
                           signal_dev=0, size_dev=0, noise=NOISE_IMPERFECT)
        self.assertLess(devs['pos'], self.precision_imperfect)

    def test_imperfect_var_signal(self):
        # var signal, const size
        devs = self.refine(param_mode=dict(signal='var', size='const'),
                           noise=NOISE_IMPERFECT,
                           signal_dev=self.signal_dev, size_dev=0)
        self.assertLess(devs['signal'], self.signal_rtol_imperfect)
        self.assertLess(devs['pos'], self.precision_imperfect)

    def test_imperfect_var_size(self):
        # const signal, var size
        devs = self.refine(param_mode=dict(signal='const', size='var'),
                           noise=NOISE_IMPERFECT,
                           signal_dev=0, size_dev=self.size_dev)
        self.assertLess(devs['size'], self.size_rtol_imperfect)
        self.assertLess(devs['pos'], self.precision_imperfect)

    def test_imperfect_var(self):
        # var signal, var size
        devs = self.refine(param_mode=dict(signal='var', size='var'),
                           noise=NOISE_IMPERFECT, signal_dev=self.signal_dev,
                           size_dev=self.size_dev)
        self.assertLess(devs['signal'], self.signal_rtol_imperfect)
        self.assertLess(devs['size'], self.size_rtol_imperfect)
        self.assertLess(devs['pos'], self.precision_imperfect)

    def test_noisy_const(self):
        # const signal and size
        devs = self.refine(param_mode=dict(signal='const', size='const'),
                           signal_dev=0, size_dev=0, noise=NOISE_NOISY)
        self.assertLess(devs['pos'], self.precision_noisy)

    def test_noisy_var_signal(self):
        # var signal, const size
        devs = self.refine(param_mode=dict(signal='var', size='const'),
                           noise=NOISE_NOISY,
                           signal_dev=self.signal_dev, size_dev=0)
        self.assertLess(devs['pos'], self.precision_noisy)

    def test_noisy_var_size(self):
        devs = self.refine(param_mode=dict(signal='const', size='var'),
                           noise=NOISE_NOISY,
                           signal_dev=0, size_dev=self.size_dev)
        self.assertLess(devs['pos'], self.precision_noisy)

    def test_noisy_var(self):
        # var signal, var size
        devs = self.refine(param_mode=dict(signal='var', size='var'),
                           noise=NOISE_NOISY, signal_dev=self.signal_dev,
                           size_dev=self.size_dev)
        self.assertLess(devs['pos'], self.precision_noisy)

    def test_dimer_perfect(self):
        # dimer is defined as such: np.array([[0, -1], [0, 1]]
        devs = self.refine_cluster(2, hard_radius=1., noise=0,
                                   param_mode=dict(signal='var', size='const'),
                                   signal_dev=self.signal_dev, size_dev=0)
        self.assertLess(devs['parr_mean'],
                        self.accuracy_perfect * max(self.size))
        self.assertLess(devs['perp_rms'], self.precision_perfect)

    def test_var_single(self):
        # dimer is defined as such: np.array([[0, -1], [0, 1]]
        devs = self.refine_cluster(2, hard_radius=1., noise=0,
                                   param_mode=dict(signal='cluster',
                                                   size='const'),
                                   signal_dev=self.signal_dev, size_dev=0)
        self.assertLess(devs['parr_mean'],
                        self.accuracy_perfect * max(self.size))
        self.assertLess(devs['perp_rms'], self.precision_perfect)

    def test_dimer_imperfect(self):
        # dimer is defined as such: np.array([[0, -1], [0, 1]]
        devs = self.refine_cluster(2, hard_radius=1., noise=NOISE_IMPERFECT,
                                   param_mode=dict(signal='var',
                                                   size='const'),
                                   signal_dev=self.signal_dev, size_dev=0)
        self.assertLess(devs['parr_mean'],
                        self.accuracy_imperfect * max(self.size))
        self.assertLess(devs['perp_rms'], self.precision_imperfect)

    def test_dimer_noisy(self):
        # dimer is defined as such: np.array([[0, -1], [0, 1]]
        devs = self.refine_cluster(2, hard_radius=1., noise=NOISE_NOISY,
                                   param_mode=dict(signal='var',
                                                   size='const'),
                                   signal_dev=self.signal_dev, size_dev=0)
        self.assertLess(devs['parr_mean'],
                        self.accuracy_noisy * max(self.size))
        self.assertLess(devs['perp_rms'], self.precision_noisy)

    def test_dimer_constrained(self):
        hard_radius = 1.
        constraints = dimer(2*np.array(self.size)*hard_radius, self.ndim)

        devs = self.refine_cluster(2, hard_radius=hard_radius, noise=0,
                                   param_mode=dict(signal='var',
                                                   size='const'),
                                   signal_dev=self.signal_dev, size_dev=0,
                                   constraints=constraints)
        self.assertLess(devs['parr_mean'], self.accuracy_constrained * max(self.size))
        self.assertLess(devs['perp_rms'], self.precision_perfect)

    def test_trimer_constrained(self):
        hard_radius = 1.
        constraints = trimer(2*np.array(self.size)*hard_radius, self.ndim)

        devs = self.refine_cluster(3, hard_radius=hard_radius, noise=0,
                                   param_mode=dict(signal='var',
                                                   size='const'),
                                   signal_dev=self.signal_dev, size_dev=0,
                                   constraints=constraints)

        self.assertLess(devs['pos'], self.precision_perfect)

    def test_tetramer_constrained(self):
        if self.ndim != 3:
            raise SkipTest('Tetramers are only tested in 3D')
        hard_radius = 1.
        constraints = tetramer(2*np.array(self.size)*hard_radius, self.ndim)

        devs = self.refine_cluster(4, hard_radius=hard_radius, noise=0,
                                   param_mode=dict(signal='var',
                                                   size='const'),
                                   signal_dev=self.signal_dev, size_dev=0,
                                   constraints=constraints)

        self.assertLess(devs['pos'], self.precision_perfect)


class TestFit_gauss2D(RefineTsts, unittest.TestCase):
    size = SIZE_2D
    ndim = 2
    feat_func = 'gauss'
    fit_func = 'gauss'


class TestFit_gauss2D_a(RefineTsts, unittest.TestCase):
     size = SIZE_2D_ANISOTROPIC
     ndim = 2
     feat_func = 'gauss'
     fit_func = 'gauss'


class TestFit_gauss3D(RefineTsts, unittest.TestCase):
     size = SIZE_3D
     ndim = 3
     feat_func = 'gauss'
     fit_func = 'gauss'


class TestFit_gauss3D_a(RefineTsts, unittest.TestCase):
     size = SIZE_3D_ANISOTROPIC
     ndim = 3
     feat_func = 'gauss'
     fit_func = 'gauss'


class TestFit_disc2D(RefineTsts, unittest.TestCase):
    size = SIZE_2D
    ndim = 2
    feat_func = 'disc'
    feat_kwargs = dict(disc_size=DISC_SIZE)
    param_val = feat_kwargs.copy()
    fit_func = 'disc'


class TestFit_disc2D_a(RefineTsts, unittest.TestCase):
    size = SIZE_2D_ANISOTROPIC
    ndim = 2
    feat_func = 'disc'
    feat_kwargs = dict(disc_size=DISC_SIZE)
    param_val = feat_kwargs.copy()
    fit_func = 'disc'


class TestFit_disc3D(RefineTsts, unittest.TestCase):
    size = SIZE_3D
    ndim = 3
    feat_func = 'disc'
    feat_kwargs = dict(disc_size=DISC_SIZE)
    param_val = feat_kwargs.copy()
    fit_func = 'disc'


class TestFit_disc3D_a(RefineTsts, unittest.TestCase):
    size = SIZE_3D_ANISOTROPIC
    ndim = 3
    feat_func = 'disc'
    feat_kwargs = dict(disc_size=DISC_SIZE)
    param_val = feat_kwargs.copy()
    fit_func = 'disc'


class TestFit_ring2D(RefineTsts, unittest.TestCase):
    size = SIZE_2D
    ndim = 2
    feat_func = 'ring'
    feat_kwargs = dict(thickness=RING_THICKNESS)
    param_val = feat_kwargs.copy()
    fit_func = 'ring'
    # as the rings are sharp features, large initial deviations in p0 may result
    # in a failed convergence
    pos_diff = 0.25  # 25% of feature radius
    size_dev = 0.05  #  5% of feature size


class TestFit_ring2D_a(RefineTsts, unittest.TestCase):
    size = SIZE_2D_ANISOTROPIC
    ndim = 2
    feat_func = 'ring'
    feat_kwargs = dict(thickness=RING_THICKNESS)
    param_val = feat_kwargs.copy()
    fit_func = 'ring'
    # as the rings are sharp features, large initial deviations in p0 may result
    # in a failed convergence
    pos_diff = 0.25  # 25% of feature radius
    size_dev = 0.05  #  5% of feature size


class TestFit_ring3D(RefineTsts, unittest.TestCase):
    size = SIZE_3D
    ndim = 3
    feat_func = 'ring'
    feat_kwargs = dict(thickness=RING_THICKNESS)
    param_val = feat_kwargs.copy()
    fit_func = 'ring'
    # as the rings are sharp features, large initial deviations in p0 may result
    # in a failed convergence
    pos_diff = 0.25  # 25% of feature radius
    size_dev = 0.05  #  5% of feature size


class TestFit_ring3D_a(RefineTsts, unittest.TestCase):
    size = SIZE_3D_ANISOTROPIC
    ndim = 3
    feat_func = 'ring'
    feat_kwargs = dict(thickness=RING_THICKNESS)
    param_val = feat_kwargs.copy()
    fit_func = 'ring'
    # as the rings are sharp features, large initial deviations in p0 may result
    # in a failed convergence
    pos_diff = 0.25  # 25% of feature radius
    size_dev = 0.05  #  5% of feature size


class TestMultiple(unittest.TestCase):
    shape = (256, 256)
    pos_err = 7
    diameter = 21
    separation = 24
    def setUp(self):
        self.signal = 200
        if not hasattr(self, 'size'):
            if hasattr(self.diameter, '__iter__'):
                self.size = tuple([d / 4 for d in self.diameter])
            else:
                self.size = self.diameter / 4
        self.im = SimulatedImage(self.shape, self.size, dtype=np.uint8,
                                 signal=self.signal, feat_func=feat_gauss,
                                 noise=0)
        self.names = self.im.pos_columns + ['signal'] + self.im.size_columns
        self.bounds = dict()

    def test_multiple_simple_sparse(self):
        self.im.clear()
        self.im.draw_features(10, self.separation, self.diameter)

        f0 = self.im.f(noise=self.pos_err)

        result = refine_leastsq(f0, self.im(), self.diameter, self.separation)

        assert_coordinates_close(result[self.im.pos_columns].values,
                                 self.im.coords, 0.1)

    def test_multiple_overlapping(self):
        self.im.clear()
        self.im.draw_features(100, 15, self.diameter)

        f0 = self.im.f(noise=self.pos_err)

        result = refine_leastsq(f0, self.im(), self.diameter, self.separation)

        assert_coordinates_close(result[self.im.pos_columns].values,
                                 self.im.coords, 0.1)

    def test_var_global(self):
        self.im.clear()
        self.im.draw_features(100, 15, self.diameter)

        f0 = self.im.f(noise=self.pos_err)
        f0['signal'] = 180

        result = refine_leastsq(f0, self.im(), self.diameter, self.separation,
                                param_mode=dict(signal='global'),
                                options=dict(maxiter=10000))

        assert_coordinates_close(result[self.im.pos_columns].values,
                                 self.im.coords, 0.1)
        assert (result['signal'].values[1:] == result['signal'].values[:-1]).all()
        assert_allclose(result['signal'].values, 200, atol=5)

    def test_constraint_global_dimer(self):
        hard_radius = 1.
        constraints = dimer_global(1, self.im.ndim)
        self.im.clear()
        self.im.draw_clusters(10, 2, hard_radius, 2*self.separation,
                              2*self.diameter)

        f0 = self.im.f(noise=self.pos_err)
        result = refine_leastsq(f0, self.im(), self.diameter,
                                self.separation, constraints=constraints,
                                options=dict(maxiter=1000))

        dists = []
        for _, f_cl in result.groupby('cluster'):
            pos = result[['y', 'x']].values
            dists.append(np.sqrt(np.sum(((pos[0] - pos[1])/np.array(self.im.size))**2)))

        assert_allclose(dists, 2*hard_radius, atol=0.01)

        assert_coordinates_close(result[self.im.pos_columns].values,
                                 self.im.coords, 0.1)

    def test_constraint_global_noisy(self):
        hard_radius = 1.
        constraints = dimer_global(1, self.im.ndim)
        self.im.clear()
        self.im.draw_clusters(10, 2, hard_radius, 2*self.separation,
                              2*self.diameter)

        f0 = self.im.f(noise=self.pos_err)
        f0['signal'] = 180
        f0['size'] = 6.

        result = refine_leastsq(f0, self.im.noisy_image(0.2*self.signal),
                                self.diameter, self.separation,
                                constraints=constraints,
                                param_mode=dict(signal='global',
                                                size='global'),
                                options=dict(maxiter=1000))

        dists = []
        for _, f_cl in result.groupby('cluster'):
            pos = result[['y', 'x']].values
            dists.append(np.sqrt(np.sum(((pos[0] - pos[1])/np.array(self.im.size))**2)))

        assert_allclose(dists, 2*hard_radius, atol=0.1)

        assert_coordinates_close(result[self.im.pos_columns].values,
                                 self.im.coords, 0.1)
        assert_allclose(result['signal'].values, 200, atol=2)
        assert_allclose(result['size'].values, 5.25, atol=1)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
