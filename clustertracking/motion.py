import numpy as np

def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def rot_2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], float)



def center_of_mass(pos, sizes):
    """Returns the center of mass of a list of positions weighted by
    sizes**ndim"""
    ndim = pos.shape[1]
    return np.sum(pos * sizes**ndim, 0) / np.sum(sizes**ndim)


def check_orthonormality(u1, u2, u3):
    """Check wether u1, u2, u3 form an orthonormal base."""
    np.testing.assert_allclose([np.dot(u1, u1),
                                np.dot(u2, u2),
                                np.dot(u3, u3)], 1, atol=0.001)
    np.testing.assert_allclose([np.dot(u1, u2),
                                np.dot(u1, u3),
                                np.dot(u2, u3)], 0, atol=0.001)
    np.testing.assert_allclose([np.cross(u1, u2),
                                np.cross(u3, u1),
                                np.cross(u2, u3)],
                               [u3, u2, u1], atol=0.001)


def _orientation_2d(pos, sizes):
    pos = np.atleast_2d(pos)
    com = center_of_mass(pos, sizes)
    if len(pos) == 1:
        raise NotImplemented
    elif len(pos) == 2:
        x = pos[0] - com
        x /= np.linalg.norm(x)
        x = np.concatenate([x, [0]])
        z = np.array([0, 0, 1], dtype=np.float)
    elif len(pos) == 3:
        raise NotImplemented
    elif len(pos) == 4:
        raise NotImplemented

    y = np.cross(z, x)  # right-handed
    y /= np.linalg.norm(y)

    check_orthonormality(x, y, z)
    return com, np.array([x, y, z])


def _orientation_3d(pos, sizes):
    pos = np.atleast_2d(pos)
    com = center_of_mass(pos, sizes)
    if len(pos) == 1:
        z = np.random.random(3)
        z /= np.linalg.norm(z)
        # generate a random vector orthogonal to z
        angle = np.random.random() * 2 * np.pi
        x = np.dot(rotation_matrix(z, angle), np.cross([1, 0, 0], z))
        x /= np.linalg.norm(x)
    elif len(pos) == 2:
        z = pos[0] - com
        z /= np.linalg.norm(z)
        # generate a random vector orthogonal to z
        angle = np.random.random() * 2 * np.pi
        x = np.dot(rotation_matrix(z, angle), np.cross([1, 0, 0], z))
        x /= np.linalg.norm(x)
    elif len(pos) == 3:
        z = pos[0] - com
        z /= np.linalg.norm(z)
        x = np.cross(z, pos[1] - com)
        x /= np.linalg.norm(x)
    elif len(pos) == 4:
        z = pos[0] - com
        z /= np.linalg.norm(z)
        x = np.cross(z, pos[2] - pos[1])
        x /= np.linalg.norm(x)

    y = np.cross(z, x)  # right-handed
    y /= np.linalg.norm(y)

    check_orthonormality(x, y, z)
    return com, np.array([x, y, z])


def orientation_df(f, cluster_size=2, mpp=1., ndim=3, sizes=None):
    """Calculate the orientation of a dataframe of clusters, given by three
    orthonormal unit vectors.

    The input and output coordinates are given in x, y, z order.
    All permutations of the feature coordinates are done.

    Returns positions and orientations.

    Definition of coordinate systems:

    For a single particle:
        - random coordinate system
    For the dimer (features 0 and 1):
        - z axis is from center to feature 0
        - xy axis are chosen randomly (with uniformly distributed angle)
    For the trimer (features 0, 1, and 2):
        - z axis is from center to feature 0
        - x axis is perpendicular to vector from feature 0 to 1 and to z
    For the tetramer (features 0, 1, 2, and 3)
        - z axis is from center to feature 0
        - x axis is perpendicular to vector from feature 1 to 2 and to z
    """
    if ndim == 2:
        orientation_func = _orientation_2d
    elif ndim == 3:
        orientation_func = _orientation_3d

    if cluster_size == 1:
        permutations = [[0]]
    elif cluster_size == 2:
        permutations = [[0, 1], [1, 0]] # must be lists for correct numpy indexing
    elif cluster_size == 3:
        permutations = [[0, 1, 2], [2, 0, 1], [1, 2, 0],
                        [2, 1, 0], [0, 2, 1], [1, 0, 2]]
    elif cluster_size == 4:
        permutations = [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2], [1, 0, 2, 3],
                        [1, 2, 3, 0], [1, 3, 0, 2], [2, 0, 1, 3], [2, 1, 3, 0],
                        [2, 3, 0, 1], [3, 0, 1, 2], [3, 1, 2, 0], [3, 2, 0, 1]]
    start_i = int(f['frame'].min())
    if sizes is None:
        sizes = np.ones((cluster_size, 1))
    else:
        sizes = np.array(sizes)[:, np.newaxis]
    length = int(f['frame'].max() - start_i) + 1
    com = np.full((length, 3), np.nan)
    bases = np.full((len(permutations), length, 3, 3), np.nan)
    f = f.sort_values(['frame', 'cluster', 'particle'], inplace=False)
    for (frame_no, cluster_id), cluster in f.groupby(['frame', 'cluster']):
        frame_no = int(frame_no) - start_i
        if len(cluster) == cluster_size:
            coords = (cluster[['y', 'x']].values * mpp)[:, ::-1]
            for i, perm in enumerate(permutations):
                _com, bases[i, frame_no] = orientation_func(coords[perm], sizes)
            com[frame_no] = _com
    return com, bases


def diffusion_tensor(positions, orientations, lagtime=1, fps=1., ndim=3):
    """Calculate the diffusion tensor."""
    if orientations.ndim == 3:
        orientations = orientations[np.newaxis]
    all_xjn = []
    delta_tjn = lagtime / fps
    for bases in orientations:
        delta_xjn = np.einsum('...ij,...j',
                              bases[:-lagtime],
                              positions[lagtime:] - positions[:-lagtime])
        unit_vectors = np.identity(3)
        delta_ujn = [0.5 * np.sum([np.cross(unit_vectors[i],
                                            np.dot(bases[b],
                                                   bases[b + lagtime, i]))
                                   for i in range(3)], axis=0)
                     for b in range(len(bases) - lagtime)]
        x_jn = np.concatenate([delta_xjn, delta_ujn], axis=1)
        x_jn = x_jn[np.isfinite(x_jn).all(axis=1)]
        all_xjn.append(x_jn)

    all_xjn = np.concatenate(all_xjn, axis=0)
    tensor = (all_xjn[:, :, np.newaxis] * all_xjn[:, np.newaxis, :]).mean(0) * 0.5 / delta_tjn

    if ndim == 2:
        result = np.empty((3, 3))
        result[:2, :2] = tensor[:2, :2]
        result[:2, 2] = tensor[:2, 5]
        result[2, :2] = tensor[5, :2]
        result[2, 2] = tensor[5, 5]
        return result

    return tensor


def diffusion_tensor_ci(positions, orientations, lagtime=1, fps=1., **kwargs):
    """Calculate the diffusion tensor and the confidence interval using bootstrap."""
    from scikits import bootstrap

    if orientations.ndim == 3:
        orientations = orientations[np.newaxis]
    all_xjn = []
    delta_tjn = lagtime / fps
    for bases in orientations:
        delta_xjn = np.einsum('...ij,...j',
                              bases[:-lagtime],
                              positions[lagtime:] - positions[:-lagtime])
        unit_vectors = np.identity(3)
        delta_ujn = [0.5 * np.sum([np.cross(unit_vectors[i],
                                            np.dot(bases[b],
                                                   bases[b + lagtime, i]))
                                   for i in range(3)], axis=0)
                     for b in range(len(bases) - lagtime)]
        x_jn = np.concatenate([delta_xjn, delta_ujn], axis=1)
        x_jn = x_jn[np.isfinite(x_jn).all(axis=1)]
        all_xjn.append(x_jn)

    all_xjn = np.concatenate(all_xjn, axis=0)
    statfunc = lambda x: (x[:, :, np.newaxis] * x[:, np.newaxis, :]).mean(0).ravel() * 0.5 / delta_tjn
    return bootstrap.ci(all_xjn, statfunc, **kwargs).reshape((2, 6, 6))


def friction_tensor(diff_tens):
    """Convert diffusion tensor to friction tensor.

    For physical units, multiply the result with  eta / (kB T)"""
    return np.linalg.inv(diff_tens.reshape((6, 6)))
