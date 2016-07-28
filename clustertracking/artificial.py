from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import pandas as pd
from pims import Frame, FramesSequence
from trackpy.utils import validate_tuple
from scipy.spatial import cKDTree
from .preprocessing import preprocess


def feat_gauss(r, ndim):
    """ Gaussian at r = 0. """
    return np.exp(r**2 * ndim/-2)


def feat_ring(r, ndim, thickness):
    """ Ring feature with a gaussian profile with a certain thickness."""
    return np.exp(((r-1+thickness)/thickness)**2 * ndim/-2)


def feat_hat(r, ndim, disc_size):
    """ Solid disc of size disc_size, with Gaussian smoothed borders. """
    result = np.ones_like(r)
    mask = r > disc_size
    result[mask] = np.exp(((r[mask] - disc_size)/(1 - disc_size))**2 *
                          ndim/-2)
    return result


def feat_step(r, ndim):
    """ Solid disc. """
    return (r <= 1).astype(np.float)


feat_disc = feat_hat
feat_dict = dict(gauss=feat_gauss, disc=feat_disc, ring=feat_ring,
                 hat=feat_hat, step=feat_step)

def gen_random_locations(shape, count, margin=0):
    """ Generates `count` number of positions within `shape`. If a `margin` is
    given, positions will be inside this margin. Margin may be tuple-valued.
    """
    margin = validate_tuple(margin, len(shape))
    np.random.seed(0)
    pos = [np.random.uniform(round(m), round(s - m), count)
           for (s, m) in zip(shape, margin)]
    return np.array(pos).T


def eliminate_overlapping_locations(f, separation):
    """ Makes sure that no position is within `separation` from each other, by
    deleting one of the that are to close to each other.
    """
    separation = validate_tuple(separation, f.shape[1])
    assert np.greater(separation, 0).all()
    # Rescale positions, so that pairs are identified below a distance of 1.
    f = f / separation
    while True:
        duplicates = cKDTree(f, 30).query_pairs(1)
        if len(duplicates) == 0:
            break
        to_drop = []
        for pair in duplicates:
            to_drop.append(pair[1])
        f = np.delete(f, to_drop, 0)
    return f * separation


def gen_nonoverlapping_locations(shape, count, separation, margin=0):
    """ Generates `count` number of positions within `shape`, that have minimum
    distance `separation` from each other. The number of positions returned may
    be lower than `count`, because positions too close to each other will be
    deleted. If a `margin` is given, positions will be inside this margin.
    Margin may be tuple-valued.
    """
    positions = gen_random_locations(shape, count, margin)
    return eliminate_overlapping_locations(positions, separation)


def draw_feature(image, position, size, max_value=None, feat_func='gauss',
                 ecc=None, mask_diameter=None, **kwargs):
    """ Draws a radial symmetric feature and adds it to the image at given
    position. The given function will be evaluated at each pixel coordinate,
    no averaging or convolution is done.

    Parameters
    ----------
    image : ndarray
        image to draw features on
    position : iterable
        coordinates of feature position
    size : number
        the size of the feature (meaning depends on feature, for feat_gauss,
        it is the radius of gyration)
    max_value : number
        maximum feature value. should be much less than the max value of the
        image dtype, to avoid pixel wrapping at overlapping features
    feat_func : {'gauss', 'disc', 'ring'} or callable
        Default: 'gauss'
        When callable is given, it should take an ndarray of radius values
        and it should return intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    mask_diameter :
        defines the box that will be drawn on. Default 8 * size.
    kwargs : keyword arguments are passed to feat_func
    """
    if len(position) != image.ndim:
        raise ValueError("Number of position coordinates should match image"
                         " dimensionality.")
    if not hasattr(feat_func, '__call__'):
        feat_func = feat_dict[feat_func]
    size = validate_tuple(size, image.ndim)
    if ecc is not None:
        if len(size) != 2:
            raise ValueError("Eccentricity is only defined in 2 dimensions")
        if size[0] != size[1]:
            raise ValueError("Diameter is already anisotropic; eccentricity is"
                             " not defined.")
        size = (size[0] / (1 - ecc), size[1] * (1 - ecc))
    if mask_diameter is None:
        mask_diameter = tuple([s * 8 for s in size])
    else:
        mask_diameter = validate_tuple(mask_diameter, image.ndim)
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    rect = []
    vectors = []
    for (c, s, m, lim) in zip(position, size, mask_diameter, image.shape):
        if (c >= lim) or (c < 0):
            raise ValueError("Position outside of image.")
        lower_bound = max(int(np.floor(c - m / 2)), 0)
        upper_bound = min(int(np.ceil(c + m / 2 + 1)), lim)
        rect.append(slice(lower_bound, upper_bound))
        vectors.append(np.arange(lower_bound - c, upper_bound - c) / s)
    coords = np.meshgrid(*vectors, indexing='ij', sparse=True)
    r = np.sqrt(np.sum(np.array(coords)**2, axis=0))
    spot = max_value * feat_func(r, ndim=image.ndim, **kwargs)
    image[rect] += spot.astype(image.dtype)


def draw_cluster(image, position, size, cluster_size, hard_radius=1., angle=0,
                 **kwargs):
    """Draws a cluster of size `n` at `pos` with angle `angle`. The distance
    between particles is determined by `hard_radius`."""
    if image.ndim == 2:
        rot = rot_2d(angle)
        coord = clusters_2d[cluster_size]
    elif image.ndim == 3:
        rot = rot_3d(angle)
        coord = clusters_3d[cluster_size]
    coord = np.dot(coord, rot.T)  # rotate
    coord *= hard_radius * np.array(size)[np.newaxis, :]  # scale
    coord += np.array(position)[np.newaxis, :]  # translate
    for pos in coord:
        draw_feature(image, pos, size, **kwargs)
    return coord


def rot_2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], float)


def rot_3d(angles):
    # Tait-Bryan angles in ZYX convention
    if not hasattr(angles, '__iter__'):
        angles = (angles, 0, 0)
    if len(angles) == 2:
        angles = (angles[0], angles[1], 0)
    s1, s2, s3 = [np.sin(x) for x in angles]
    c1, c2, c3 = [np.cos(x) for x in angles]
    return np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                     [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2 - c1*s3],
                     [-s2, c2*s3, c2*c3]], float)


clusters_2d = {1: np.array([[0, 0]], float),
               2: np.array([[0, -1], [0, 1]], float),
               3: np.array([[0, 1],
                            [-0.5 * np.sqrt(3), -0.5],
                            [0.5 * np.sqrt(3), -0.5]], float)*2/3*np.sqrt(3),
               4: np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], float)}
clusters_3d = {1: np.array([[0, 0, 0]], float),
               2: np.array([[0, 0, -1], [0, 0, 1]], float),
               3: np.array([[0, 0, 2/np.sqrt(3)],
                            [-1, 0, -1/np.sqrt(3)],
                            [1, 0,  -1/np.sqrt(3)]], float),
               4: np.array([[0, 0, (1/2)*np.sqrt(6)],
                            [0, -(2/3)*np.sqrt(3), -(1/6)*np.sqrt(6)],
                            [1, (1/3)*np.sqrt(3), -(1/6)*np.sqrt(6)],
                            [-1, (1/3)*np.sqrt(3), -(1/6)*np.sqrt(6)]], float)}


class SimulatedImage(object):
    """ This class makes it easy to generate artificial pictures.

    Parameters
    ----------
    shape : tuple of int
    dtype : numpy.dtype, default np.uint8
    saturation : maximum value in image
    hard_radius : default radius of particles, used for determining the
                  distance between particles in clusters
    feat_dict : dictionary of arguments passed to tp.artificial.draw_feature

    Attributes
    ----------
    image : ndarray containing pixel values
    center : the center [y, x] to use for radial coordinates

    Examples
    --------
    image = SimulatedImage(shape=(50, 50), dtype=np.uint8, hard_radius=7,
                           feat_dict={'diameter': 20, 'max_value': 100,
                                      'feat_func': SimulatedImage.feat_hat,
                                      'disc_size': 0.2})
    image.draw_feature((10, 10))
    image.draw_dimer((32, 35), angle=75)
    image.add_noise(5)
    image()
    """
    def __init__(self, shape, size, dtype=np.uint8, saturation=None,
                 hard_radius=None, signal=None, noise=0,
                 feat_func=feat_gauss, **feat_kwargs):
        self.ndim = len(shape)
        self.shape = shape
        self.dtype = dtype
        self.image = Frame(np.zeros(shape, dtype=dtype))
        self.size = validate_tuple(size, self.ndim)
        self.isotropic = np.all([self.size[1:] == self.size[:-1]])
        self.feat_func = feat_func
        self.feat_kwargs = feat_kwargs
        self.noise = noise
        if saturation is None and np.issubdtype(dtype, np.integer):
            self.saturation = np.iinfo(dtype).max
        elif saturation is None and np.issubdtype(dtype, np.float):
            self.saturation = 1
        else:
            self.saturation = saturation
        if signal is None:
            self.signal = self.saturation
        else:
            self.signal = signal
        self.center = tuple([s // 2 for s in shape])
        self.hard_radius = hard_radius
        self._coords = []
        self.pos_columns = ['z', 'y', 'x'][-self.ndim:]
        if self.isotropic:
            self.size_columns = ['size']
        else:
            self.size_columns = ['size_z', 'size_y', 'size_x'][-self.ndim:]

    def __call__(self):
        # so that you can checkout the image with image() instead of image.image
        return self.noisy_image(self.noise)

    def clear(self):
        """Clears the current image"""
        self._coords = []
        self.image = np.zeros_like(self.image)

    def draw_feature(self, pos):
        """Draws a feature at `pos`."""
        pos = [float(p) for p in pos]
        self._coords.append(pos)
        draw_feature(self.image, pos, self.size, self.signal,
                     self.feat_func, **self.feat_kwargs)

    def draw_features(self, N, separation=0, margin=None):
        """Draws N features at random locations, using minimum separation
        and a margin. If separation > 0, less than N features may be drawn."""
        if margin is None:
            margin = self.hard_radius
        if margin is None:
            margin = 0
        pos = gen_random_locations(self.shape, N, margin)
        if separation > 0:
            pos = eliminate_overlapping_locations(pos, separation)
        for p in pos:
            self.draw_feature(p)
        return pos

    def draw_feature_radial(self, r, angle, center=None):
        """Draws a feature at radial coordinates `r`, `angle`. The center
        of the radial coordinates system is determined by `center`. If this
        is not given, self.center is used.

        For 3D, angle has to be a tuple of length 2: (phi, theta), in which
        theta is the angle with the positive z axis."""
        if center is None:
            center = self.center
        if self.ndim == 2:
            pos = (center[0] + self.size[0]*r*np.sin(angle*(np.pi/180)),
                   center[1] + self.size[1]*r*np.cos(angle*(np.pi/180)))
        elif self.ndim == 3:
            if not hasattr(angle, '__iter__'):
                angle = (angle, 0)
            r_sin_theta = r*np.sin(angle[1]*(np.pi/180))
            pos = (center[0] + self.size[0]*r*np.cos(angle[1]*(np.pi/180)),
                   center[1] + self.size[1]*r_sin_theta*np.sin(angle[0]*(np.pi/180)),
                   center[2] + self.size[2]*r_sin_theta*np.cos(angle[0]*(np.pi/180)))
        else:
            raise ValueError("Don't know how to draw in {} dimensions".format(self.ndim))
        self.draw_feature(pos)
        return pos

    def draw_dimer(self, pos, angle, hard_radius=None):
        """Draws a dimer at `pos` with angle `angle`. The distance
        between particles is determined by 2*`hard_radius`. If this is not
        given, self.separation is used."""
        return self.draw_cluster(2, pos, angle, hard_radius)
    draw_dumbell = draw_dimer

    def draw_trimer(self, pos, angle, hard_radius=None):
        """Draws a trimer at `pos` with angle `angle`. The distance
        between particles is determined by `separation`. If this is not
        given, self.separation is used."""
        return self.draw_cluster(3, pos, angle, hard_radius)
    draw_triangle = draw_trimer

    def draw_cluster(self, cluster_size, center=None, angle=0, hard_radius=None):
        """Draws a cluster of size `n` at `pos` with angle `angle`. The distance
        between particles is determined by `separation`. If this is not
        given, self.separation is used."""
        if hard_radius is None:
            hard_radius = self.hard_radius
        if center is None:
            center = self.center
        if self.ndim == 2:
            rot = rot_2d(angle)
            coord = clusters_2d[cluster_size]
        elif self.ndim == 3:
            rot = rot_3d(angle)
            coord = clusters_3d[cluster_size]
        coord = np.dot(coord, rot.T)  # rotate
        coord *= hard_radius * np.array(self.size)[np.newaxis, :]  # scale
        coord += np.array(center)[np.newaxis, :]  # translate
        for pos in coord:
            self.draw_feature(pos)
        return coord

    def draw_clusters(self, N, cluster_size, hard_radius=None, separation=0,
                      margin=None):
        """Draws N clusters at random locations, using minimum separation
        and a margin. If separation > 0, less than N features may be drawn."""
        if hard_radius is None:
            hard_radius = self.hard_radius
        if margin is None:
            margin = self.hard_radius
        if margin is None:
            margin = 0
        pos = gen_random_locations(self.shape, N, margin)
        if separation > 0:
            pos = eliminate_overlapping_locations(pos, separation)

        if self.ndim == 2:
            angles = np.random.uniform(0, 2*np.pi, N)
        elif self.ndim == 3:
            angles = np.random.uniform(0, 2*np.pi, (N, 3))

        for p, a in zip(pos, angles):
            self.draw_cluster(cluster_size, p, a, hard_radius)
        return pos

    def noisy_image(self, noise_level):
        """Adds noise to the current image, uniformly distributed
        between 0 and `noise_level`, not including noise_level."""
        if noise_level <= 0:
            return self.image
        if np.issubdtype(self.dtype, np.integer):
            noise = np.random.poisson(noise_level, self.shape)
        else:
            noise = np.clip(np.random.normal(noise_level, noise_level/2, self.shape), 0, self.saturation)
        noisy_image = np.clip(self.image + noise, 0, self.saturation)
        return Frame(np.array(noisy_image, dtype=self.dtype))

    def denoised(self, noise_level, noise_size, smoothing_size=None,
                 threshold=None):
        image = self.noisy_image(noise_level)
        return preprocess(image, noise_size, smoothing_size, threshold)

    @property
    def coords(self):
        if len(self._coords) == 0:
            return np.zeros((0, self.ndim), dtype=np.float)
        return np.array(self._coords)

    def f(self, noise=0):
        result = self.coords + np.random.random(self.coords.shape) * noise
        result = pd.DataFrame(result, columns=self.pos_columns)
        result['signal'] = float(self.signal)
        if self.isotropic:
            result[self.size_columns[0]] = float(self.size[0])
        else:
            for col, s in zip(self.size_columns, self.size):
                result[col] = float(s)
        return result


class CoordinateReader(FramesSequence):
    def __init__(self, f, shape, size, **kwargs):
        self._f = f.copy()
        self.pos_columns = ['z', 'y', 'x'][-len(shape):]
        self.shape = shape
        self.size = size
        self.kwargs = kwargs
        self.im = SimulatedImage(shape, size, **self.kwargs)
        self._len = int(f['frame'].max() + 1)

    def __len__(self):
        return self._len

    def get_frame(self, ind):
        self.im.clear()
        pos = self._f.loc[self._f['frame'] == ind, self.pos_columns].values
        for _pos in pos:
            self.im.draw_feature(_pos)
        return Frame(self.im(), frame_no=ind)

    @property
    def pixel_type(self):
        return self.im.dtype

    @property
    def frame_shape(self):
        return self.im.shape


def get_single(shape, size=4, offset=0, feat_func=feat_hat, signal=100,
               **kwargs):
    if feat_func is feat_hat and 'disc_size' not in kwargs:
        kwargs['disc_size'] = 0.5
    elif feat_func is feat_ring and 'thickness' not in kwargs:
        kwargs['thickness'] = 0.5
    im = SimulatedImage(shape, size, dtype=np.uint8, signal=signal,
                        feat_func=feat_func, **kwargs)
    offset=np.atleast_1d(offset)
    im.draw_feature(im.center + offset)
    return im


def get_dimer(shape, size=4, hard_radius=2.5, angle=0, offset=0,
              feat_func=feat_hat, signal=100, **kwargs):
    if feat_func is feat_hat and 'disc_size' not in kwargs:
        kwargs['disc_size'] = 0.5
    elif feat_func is feat_ring and 'thickness' not in kwargs:
        kwargs['thickness'] = 0.5
    im = SimulatedImage(shape, size, dtype=np.uint8, signal=signal,
                        feat_func=feat_func, **kwargs)
    offset=np.atleast_1d(offset)
    im.draw_cluster(2, im.center + offset, angle, hard_radius)
    return im


def get_multiple(N, signal_range=(100, 255), size=4, separation=None, offset=0,
                 feat_func=feat_hat, signal=100, **kwargs):
    if feat_func is feat_hat and 'disc_size' not in kwargs:
        kwargs['disc_size'] = 0.5
    elif feat_func is feat_ring and 'thickness' not in kwargs:
        kwargs['thickness'] = 0.5
    if separation is None:
        separation = size * 4
    margin = separation

    Nsqrt = int(np.sqrt(N) + 0.9999)
    pos = np.meshgrid(np.arange(0, separation * Nsqrt, separation),
                      np.arange(0, separation * Nsqrt, separation), indexing='ij')
    pos = np.array([pos[0].ravel(), pos[1].ravel()], dtype=np.float).T[:N] + margin
    shape = tuple((np.max(pos, axis=0) + margin).astype(np.int))
    pos += (np.random.random(pos.shape) - 0.5)  #randomize subpixel location
    signal = np.random.uniform(signal_range[0], signal_range[1], N)
    im = SimulatedImage(shape, size, dtype=np.uint8, signal=signal,
                        feat_func=feat_func, **kwargs)

    for _pos, _signal in zip(pos, signal):
        im.signal = _signal
        im.draw_feature(_pos)

    return im

