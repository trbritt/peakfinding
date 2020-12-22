import numpy as np
import matplotlib.pyplot as plt


def reflect(arr, axis=0, sign=1):
    """Reflect the elements of a numpy array along a specified axis about the first element.
    Parameters
    ----------
    arr : ndarray of coordiantes to be symmetrized
    axis: direction along which to symmetrize
    sign: relative sign between original and symmetrized values.

    Returns
    -------
        : ndarray of total collection of points (orig+reflected)

    """
    refl_idx = axis * [slice(None)] + [slice(None, 0, -1), Ellipsis]
    return np.concatenate((arr[tuple(refl_idx)], arr), axis=axis)


def generate_lattice(min, max, lattice_vectors):
    """Return some points from a 2-d lattice (with the origin first).
    Parameters
    ----------
    min             : iterable representing (min(x),min(y)) coordinate in cartesian grid
    max             : iterable representing (max(x),max(y)) coordinate in cartesian grid
    lattice_vectors : vectors used to transform the cartesian grid to reciprocal grid.

    Returns
    -------
    lattice         : ndarray containing pairs of (x,y) cordinates representing
                      the reciprocal lattice
    indices         : iterable telling which reflection each peak is
    """
    # roll is necessary to make sure final list of pairs of points is in order
    # (0,0),(0,1),(0,-1),etc
    # Generates all possible x coords
    xs = np.roll(np.arange(min[0], max[0]), max[0])
    # generates y coords
    ys = np.roll(np.arange(min[1], max[1]), max[1])
    # fancy way of putting all combinations of points into
    # column vector
    lattice = np.dstack(np.meshgrid(xs, ys)).reshape(-1, 2)
    lattice = lattice[1:]
    indices = []
    for m in range(lattice.shape[0]):
        i = str(lattice[m, 0])
        j = str(lattice[m, 1])
        indices.append(i + j + "0")
    # transforms uniform grid to new reciprocal grid
    lattice = np.matmul(lattice, lattice_vectors)
    return lattice, indices


def rank_of_first(xs, axis=0):
    """Return the rank of the first item in a collection of items when sorted.
    Parameters
    ----------
    xs      : ndarray of items to be sorted
    axis    : int representing direction along which elements are sorted

    Returns
    -------
            : ndarray of indices telling how to sort the elements of xs
    """
    return np.argpartition(np.argsort(xs, axis=axis), 0, axis=axis)[0, :]


def brillouin_zone_index(x, lattice):
    """Determine the index of the Brillouin zone in which a given point
    (or collection of points) lies.
    Parameters
    ----------
    x       : ndarray whose last dimension represents spatial coordinates
    lattice : ndarray of whose first dimension indexes over
              lattice points, with the origin given as the first lattice
              point.

    Returns
    -------
            : ndarray of indicies identifying which BZ the lattice coordinates
              belongs to
    """
    # calculate the distances from each lattice point to each point in x
    lat_norms = np.apply_along_axis(
        np.linalg.norm, -1, lattice[:, np.newaxis, np.newaxis, :] - x
    )
    # return the rank of the origin
    return rank_of_first(lat_norms)


def reciprocal_vectors3D(a, b, c, alpha, beta, gamma):
    """Takes real space unit cell side lengths (a,b,c) and relative angles
    (alpha, beta, gamma), and determines the real space vectors and
    reciprocal vectors.
    Parameters
    ----------
    a       : float32(64) representing seperation of atoms in unit cell
    b       : float32(64) representing seperation of atoms in unit cell
    c       : float32(64) representing seperation of atoms in unit cell
    alpha   : float32(64) denoting angle between b and c
    beta    : float32(64) denoting angle between a and c
    gamma   : float32(64) denoting angle between b and a

    Returns
    -------
    u       : ndarray representing basis vector in reciprocal lattice
    v       : ndarray representing basis vector in reciprocal lattice
    w       : ndarray representing basis vector in reciprocal lattice

    Notes
    -----
    http://gisaxs.com/index.php/Unit_cell
    """
    vecA = np.array([a, 0, 0])
    vecB = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
    vecC = np.array(
        [
            c * np.cos(beta),
            (c / np.sin(gamma)) * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)),
            c
            * np.sqrt(
                1
                - np.cos(beta) ** 2
                - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)) ** 2
            ),
        ]
    )
    # new numpy cross is faster than einsum, so wahoo!
    vol = np.abs(np.einsum("i,i", vecA, np.cross(vecB, vecC)))
    u = (2 * np.pi / vol) * np.cross(vecB, vecC)
    v = (2 * np.pi / vol) * np.cross(vecC, vecA)
    w = (2 * np.pi / vol) * np.cross(vecA, vecB)
    return u, v, w


def test_BZ():
    # ML-MoS2 specs
    a = 3.199
    b = 3.199
    c = 6.494
    alpha = 90.00 * np.pi / 180
    beta = 90.00 * np.pi / 180
    gamma = 120.00 * np.pi / 180

    u, v, w = reciprocal_vectors3D(a, b, c, alpha, beta, gamma)

    img_max = (2.0, 2.0)
    res = 0.001

    img_xs_full = np.arange(-img_max[0], img_max[0], res)
    img_ys_full = np.arange(-img_max[1], img_max[1], res)
    img_xs_symm = np.arange(0, img_max[0], res)
    img_ys_symm = np.arange(0, img_max[1], res)
    image_pts_symm = np.dstack(np.meshgrid(img_xs_symm, img_ys_symm))

    # square_lattice = generate_lattice(min=(-3,-3), max=(5,5),
    #                               lattice_vectors=np.eye(2))
    # sq_bril_zones = bz.brillouin_zone_index(image_pts_symm, square_lattice)
    # sq_bril_zones = reflect(reflect(sq_bril_zones, axis=0), axis=1)
    # plt.pcolormesh(img_xs_full, img_ys_full,
    #                sq_bril_zones, cmap=plt.get_cmap('Paired'))
    # plt.axes().set_aspect('equal')
    # plt.show()

    # lattice_vectors = np.array([[1.0, 0.0],
    #                                        [np.cos(np.pi/3.0), np.sin(np.pi/3.0)]])
    lattice_vectors = np.vstack((u[:2], v[:2]))

    # lattice_vectors = np.array([u,v])
    hexagonal_lattice_pts, idx = generate_lattice(
        min=(-4, -3), max=(6, 6), lattice_vectors=lattice_vectors
    )
    hex_bril_zones = brillouin_zone_index(image_pts_symm, hexagonal_lattice_pts)
    hex_bril_zones = reflect(reflect(hex_bril_zones, axis=0), axis=1)
    plt.pcolormesh(
        img_xs_full, img_ys_full, hex_bril_zones, cmap=plt.get_cmap("Paired")
    )
    plt.axes().set_aspect("equal")
    plt.show()
