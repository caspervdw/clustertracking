from __future__ import (division, print_function, unicode_literals,
                        absolute_import)

import numpy as np


def show_feature(im, mesh, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    _kwargs = dict(color='b')
    _kwargs.update(kwargs)
    ax.plot_wireframe(mesh[1], mesh[0], im, **_kwargs)
    return ax


def show_feature3d(im, mesh, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh[2], mesh[1], mesh[0], c=im, **kwargs)
    return ax


def show_fit(residual, res, fit_args, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    _kwargs = dict(color='r')
    _kwargs.update(kwargs)
    y, x = fit_args[1:3]
    z = residual(res['x'], 0, *fit_args[1:])
    ax.plot_wireframe(x, y, z, **_kwargs)
    return ax


def display3d(stack, pos=None, origin=None):
    _plot_style = dict(markersize=15, markeredgewidth=2,
                       markerfacecolor='none', markeredgecolor='r',
                       marker='o', linestyle='none')
    from matplotlib.pyplot import imshow
    from pandas import DataFrame
    from trackpy import annotate
    n, tile_y, tile_x = stack.shape[-3:]
    n_rows = int(np.sqrt(stack.size) // tile_y)
    n_cols = int(np.ceil(n / n_rows))

    result = np.zeros((n_rows * tile_y, n_cols * tile_x),
                      dtype=stack.dtype)

    if pos is not None and origin is not None:
        pos_rel = np.array(pos) - np.array(origin)[np.newaxis, :]
    elif pos is not None:
        pos_rel = np.array(pos)

    i_row = 0
    i_col = 0
    if pos is not None:
        plot_pos = np.empty((0, 2), dtype=np.float)
    for i in range(n):
        tile = stack.take(i, axis=stack.ndim - 3)
        result[i_row * tile_y: (i_row + 1) * tile_y,
               i_col * tile_x: (i_col + 1) * tile_x] = tile
        if i_col < n_cols - 1:
            i_col += 1
        else:
            i_col = 0
            i_row += 1
        if pos is not None:
            mask = (pos_rel[:, 0] > (i - 0.5)) & (pos_rel[:, 0] < (i + 0.5))
            if mask.sum() > 0:
                plot_pos = np.append(plot_pos, pos_rel[mask, 1:] +
                                               np.array([[i_row * tile_y,
                                                          i_col * tile_x]]),
                                               axis=0)
    if pos is None:
        imshow(result)
    else:
        f = DataFrame(plot_pos, columns=['y', 'x'])
        annotate(f, result)
