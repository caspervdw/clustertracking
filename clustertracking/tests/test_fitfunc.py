from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import numpy as np

from numpy.testing import assert_allclose
from clustertracking.fitfunc import FitFunctions, vect_from_params
from scipy.optimize.slsqp import approx_jacobian


EPSILON = 1E-7
ATOL = 0.001
RTOL = 0.01

class TestFitFunctions(unittest.TestCase):
    repeats = 100
    def get_residual(self, ff, n, params, groups=None):
        if groups is None:
            # just assume that they are all clustered
            groups = [[list(range(n))]]
        images = np.random.random((len(groups[0]), self.repeats)) * 200
        meshes = np.random.random((len(groups[0]), ff.ndim, self.repeats)) * 10
        masks = []
        for group in groups[0]:
            masks.append(np.random.random((len(group), self.repeats)) > 0.5)

        return ff.get_residual(images, meshes, masks, params, groups)

    def compare_jacobian(self, fit_function, ndim, isotropic, n, groups=None,
                         custom_param_mode=None):
        ff = FitFunctions(fit_function, ndim, isotropic)
        param_mode = {param: 'var' for param in ff.params}
        param_mode['background'] = 'cluster'
        if custom_param_mode is not None:
            param_mode.update(custom_param_mode)
        ff = FitFunctions(fit_function, ndim, isotropic, param_mode=param_mode)
        params = np.random.random((n, len(ff.params))) * 10
        residual, jacobian = self.get_residual(ff, n, params, groups)
        vect = vect_from_params(params, ff.modes, groups, operation=np.mean)
        actual = jacobian(vect)
        expected = approx_jacobian(vect, residual, EPSILON)[0]
        assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    def test_custom_no_jac(self):
        fit_function = dict(name='parabola', params=['a'],
                            func=lambda r2, p: p[0]*r2)
        ff = FitFunctions(fit_function, ndim=2, isotropic=True)
        params = np.random.random((1, len(ff.params))) * 10
        residual, jacobian = self.get_residual(ff, 1, params)
        assert jacobian is None

    def test_custom_jac(self):
        fit_function = dict(name='parabola', params=['a'],
                            func=lambda r2, p, ndim: p[0]*r2,
                            dfunc=lambda r2, p, ndim: (p[0]*r2, [p[0], r2]))
        self.compare_jacobian(fit_function, ndim=2, isotropic=True, n=1)

    def test_custom_jac2(self):
        fit_function = dict(name='parabola', params=['a', 'b', 'c'],
                            func=lambda r2, p, ndim: p[0]*r2 + p[1]*r2**2 + p[2]*r2**3,
                            dfunc=lambda r2, p, ndim: (p[0]*r2 + p[1]*r2**2 + p[2]*r2**3,
                                                       [p[0] + 2*p[1]*r2 + 3*p[2]*r2**2, r2, r2**2, r2**3]))
        self.compare_jacobian(fit_function, ndim=2, isotropic=True, n=1)

    def test_2D_gauss(self):
        def gauss(p, y, x):
            background, signal, yc, xc, size = p
            return background + signal*np.exp(-((y - yc)**2/size**2 + (x - xc)**2/size**2))
        ff = FitFunctions('gauss', ndim=2, isotropic=True)
        params = np.array([[5, 200, 4, 5, 6]], dtype=np.float)

        image = np.random.random(self.repeats) * 200
        mesh = np.random.random((ff.ndim, self.repeats)) * 10
        masks = np.full((1, self.repeats), True, dtype=np.bool)

        residual, jacobian = ff.get_residual([image], [mesh], [masks], params)

        vect = vect_from_params(params, ff.modes, groups=None, operation=np.mean)

        actual = residual(vect)
        expected = np.sum((image - gauss(params[0], *mesh))**2) / self.repeats

        assert_allclose(actual, expected, atol=1E-7)

    def test_2D_isotropic(self):
        self.compare_jacobian('gauss', ndim=2, isotropic=True, n=1)

    def test_2D_anisotropic(self):
        self.compare_jacobian('gauss', ndim=2, isotropic=False, n=1)

    def test_3D_isotropic(self):
        self.compare_jacobian('gauss', ndim=3, isotropic=True, n=1)

    def test_3D_anisotropic(self):
        self.compare_jacobian('gauss', ndim=3, isotropic=False, n=1)

    def test_ring(self):
        self.compare_jacobian('ring', ndim=2, isotropic=True, n=1)

    def test_dimer(self):
        self.compare_jacobian('gauss', ndim=2, isotropic=True, n=2)

    def test_dimer_var_cluster(self):
        self.compare_jacobian('gauss', ndim=2, isotropic=True, n=2,
                              custom_param_mode=dict(signal='cluster'))

    def test_var_global_in_cluster(self):
        self.compare_jacobian('gauss', ndim=2, isotropic=True, n=2,
                              custom_param_mode=dict(signal='global'),
                              groups=[[[0, 1]]])

    def test_var_global_out_cluster(self):
        self.compare_jacobian('gauss', ndim=2, isotropic=True, n=2,
                              custom_param_mode=dict(signal='global'),
                              groups=[[[0], [1]]])

    def test_ring(self):
        self.compare_jacobian('ring', ndim=2, isotropic=True, n=1)

    # def test_inv_series(self):
    #     self.compare_jacobian('inv_series_6', ndim=2, isotropic=True, n=1)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
