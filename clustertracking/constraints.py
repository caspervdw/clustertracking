from __future__ import division, print_function, absolute_import

import six
import numpy as np
from .utils import validate_tuple
from .fitfunc import vect_to_params
from warnings import warn


def _wrap_fun(func, params_const, modes, ids=None):
    def wrapped(vect, *args, **kwargs):
        params = vect_to_params(vect, params_const, modes, ids)
        return func(params, *args, **kwargs)
    return wrapped


def wrap_constraints(constraints, params_const, modes, groups=None):
    if constraints is None:
        return []

    if groups is not None:
        cl_sizes = np.array([len(params_const)], dtype=np.int)

    result = []
    for cons in constraints:
        cluster_size = cons.get('cluster_size', None)
        if cluster_size is None:
            # provide all parameters to the constraint
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                return cons['fun'](params[np.newaxis, :, :], *args, **kwargs)
        elif groups is None:
            if len(params_const) != cluster_size:
                continue
            # provide all parameters to the constraint
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                return cons['fun'](params[np.newaxis, :, :], *args, **kwargs)
        elif cluster_size in cl_sizes:
            groups_this = groups[0][cl_sizes == cluster_size]
            if len(groups_this) == 0:
                continue
            # group the appropriate clusters together and return multiple values
            def wrapped(vect, *args, **kwargs):
                params = vect_to_params(vect, params_const, modes, groups)
                params_grouped = np.array([params[g] for g in groups_this])
                return cons['fun'](params_grouped, *args, **kwargs)
        else:
            continue
        cons_wrapped = cons.copy()
        cons_wrapped['fun'] = wrapped
        result.append(cons_wrapped)
        if 'jac' in cons_wrapped:
            warn('Constraint jacobians are not implemented')
            del cons_wrapped['jac']
    return result


def _dimer_fun(x, dist, ndim):
    pos = x[..., 2:2+ndim]  # get positions only
    return 1 - np.sum(((pos[:, 0] - pos[:, 1])/dist)**2, axis=1)


# def _dimer_jac(x, dist, ndim):
#     result = np.zeros_like(x)
#     x = x[:, 2:2+ndim]  # get positions only
#     result[:, 2:2+ndim] = -2 * (x - x[[1, 0]])/dist[np.newaxis, :]**2
#     return result


def dimer(dist, ndim=2):
    dist = np.array(validate_tuple(dist, ndim))
    return (dict(type='eq', cluster_size=2, fun=_dimer_fun, args=(dist, ndim)),)


def _trimer_fun(x, dist, ndim):
    x = x[..., 2:2+ndim]  # get positions only
    return np.concatenate((1 - np.sum(((x[:, 0] - x[:, 1])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 1] - x[:, 2])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 0] - x[:, 2])/dist)**2, axis=1)))


# def _trimer_jac(x, dist, ndim, indices):
#     result = np.zeros_like(x)
#     x = x[:, -ndim:]  # get positions only
#     result[indices, -ndim:] = -2 * (x[indices] - x[indices[::-1]])/dist[np.newaxis, :]**2
#     return result


def trimer(dist, ndim=2):
    dist = np.array(validate_tuple(dist, ndim))
    return (dict(type='eq', cluster_size=3, fun=_trimer_fun, args=(dist, ndim)),)


def _tetramer_fun_2d(x, dist):
    x = x[..., 2:4]  # get positions only
    dists = np.vstack((np.sum(((x[:, 0] - x[:, 1])/dist)**2, axis=1),
                       np.sum(((x[:, 1] - x[:, 2])/dist)**2, axis=1),
                       np.sum(((x[:, 0] - x[:, 2])/dist)**2, axis=1),
                       np.sum(((x[:, 1] - x[:, 3])/dist)**2, axis=1),
                       np.sum(((x[:, 0] - x[:, 3])/dist)**2, axis=1),
                       np.sum(((x[:, 2] - x[:, 3])/dist)**2, axis=1)))
    # take the 4 smallest: they should be 1
    # do not test the other 2: they are fixed by the 4 first constraints.
    dists = np.sort(dists, axis=0)[:4]
    return np.ravel(1 - dists)


def _tetramer_fun_3d(x, dist):
    x = x[..., 2:5]  # get positions only
    return np.concatenate((1 - np.sum(((x[:, 0] - x[:, 1])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 1] - x[:, 2])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 0] - x[:, 2])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 1] - x[:, 3])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 0] - x[:, 3])/dist)**2, axis=1),
                           1 - np.sum(((x[:, 2] - x[:, 3])/dist)**2, axis=1)))

def tetramer(dist, ndim=2):
    dist = np.array(validate_tuple(dist, ndim))
    if ndim == 2:
        return (dict(type='eq', cluster_size=4, fun=_tetramer_fun_2d, args=(dist,)),)
    elif ndim == 3:
        return (dict(type='eq', cluster_size=4, fun=_tetramer_fun_3d, args=(dist,)),)
    else:
        raise NotImplementedError


def _dimer_fun_global(x, mpp, ndim):
    if x.ndim == 2 or len(x) <= 1:
        return []
    pos = x[..., 2:2+ndim]  # get positions only, shape (n_clusters, 2, ndim)
    dist_squared = np.sum(((pos[:, 0] - pos[:, 1])*mpp)**2, axis=1)**2
    return np.diff(dist_squared)


# def _dimer_jac_global(x, mpp, ndim):
#     if x.ndim == 2 or len(x) <= 1:
#         return []
#     result = np.zeros((x.shape[0] - 1,) + x.shape)
#     x = x[..., -ndim:]  # get positions only, shape (n_clusters, 2, ndim)
#     _jac = -2 * (x - x[:, [1, 0]]) * mpp**2
#     # result[:, :-1, :, -ndim] = _jac
#     # result[:, 1:, :, -ndim] = -1*_jac
#
#     for i in range(x.shape[0] - 1):
#          result[i, i, :, -ndim:] = _jac[i]
#          result[i, i + 1, :, -ndim:] = -_jac[i + 1]
#     return result


def dimer_global(mpp, ndim=2):
    # the jacobian seems to slow things down.
    # in tests: 26 iterations without, 198 with
    mpp = np.array(validate_tuple(mpp, ndim))
    return (dict(type='eq', fun=_dimer_fun_global, args=(mpp, ndim,)),)
