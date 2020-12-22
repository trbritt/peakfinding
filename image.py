import numpy as np
import copy
import itertools
from scipy.spatial.ckdtree import cKDTree as spsp
from skimage import img_as_float, exposure
import brillouin as bz
import matplotlib.pyplot as plt

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    Parameters
    ----------
    image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
    axes  : matplotlib.pyplo.axes objects to be used to plot results 
    bins  : int of bins used in numpy histogram call for plotting on 
            matplotlib axis object

    Returns
    -------
    ax_img  : axes object to plot image
    ax_hist : axes object to plot histogram
    ax_cdf  : axes object to plot cumulative distribution function

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype="step", color="black")
    ax_hist.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_hist.set_xlabel("Pixel intensity")
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, "r")
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def rgb2gamma(rgb, gamma=2.40):
    """Gamma corrects RGB image 
    Parameters
    ----------
    rgb : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats

    Returns
    -------
    nonlinear : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
    """
    normed = rgb / 255
    linear = np.dot(normed, [0.2126, 0.7152, 0.0722])
    cutoff = 0.0031308
    nonlinear = np.where(linear <= cutoff, 12.92 * linear, linear)
    nonlinear = np.where(
        linear > cutoff, 1.055 * pow(linear, 1 / gamma) - 0.055, linear
    )
    return nonlinear


def grey2gamma(grey):
    """Gamma corrects greyscale image 
    Parameters
    ----------
    grey : ndarray ([M[, N[, ...P]]) of ints, uints or floats

    Returns
    -------
    nonlinear : ndarray ([M[, N[, ...P]]) of ints, uints or floats
    """
    cutoff = 0.0031308 / 2
    nonlinear = np.where(grey <= cutoff, 12.92 * grey, grey)
    nonlinear = np.where(grey > cutoff, 1.055 * pow(grey, 1 / 2.40) - 0.055, grey)
    return nonlinear


def get_std(image):
    """Returns standard deviation along default axis of an array of floats (derived from image)
    Parameters
    ----------
    image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats

    Returns
    -------
    float  : standard deviation
    """
    return np.std(image)


def close_points(px, py, mindist, remove=True):
    """Uses method of quadtrees to remove points less than mindist away from each other point
    Parameters
    ----------
    px      : iterable containing x coordinates 
    py      : iterable containing y coordinates 
    mindist : maximum side length of a square in the quadtree
              directly corresponding to the minimum distance 
              allowed between any two pairs of points in (px,py)

    Returns
    -------
    x       : list (can be any iterable) containing x coordinates
                of scrubbed list of points, where neighbors less than mindist away
                from each other have been removed
    y       : list (can be any iterable) containing y coordinates
                of scrubbed list of points, where neighbors less than mindist away
                from each other have been removed
    
    Notes
    -----
    https://en.wikipedia.org/wiki/Quadtree
    """

    xmin, xmax = np.min(px), np.max(px)
    ymin, ymax = np.min(py), np.max(py)

    gx_count = np.ceil(np.abs(xmax - xmin) / mindist).astype(int)
    gy_count = np.ceil(np.abs(ymax - ymin) / mindist).astype(int)

    gx = np.linspace(xmin, xmax, gx_count)
    gy = np.linspace(ymin, ymax, gy_count)
    grid = list(itertools.product(gx, gy))

    points = np.array(list(zip(px, py)))
    kdtree = spsp(points)

    gd, idx = kdtree.query(grid, k=1, distance_upper_bound=mindist, n_jobs=-1)

    idx = idx[gd != np.inf]
    rare_points = points[idx]

    kdtree = spsp(rare_points)
    exclude = {}

    for i, pt in enumerate(rare_points):
        if i in exclude:
            continue

        nhoods = kdtree.query_ball_point(pt, mindist)
        exclude.update({n: None for n in nhoods if n != i})

    exclude = list(exclude.keys())
    if (remove):
        res_points = np.delete(rare_points, exclude, axis=0)
        x = res_points[:, 0]
        y = res_points[:, 1]

        return x, y


def get_max(image, sigma, alpha=20, size=10):
    """Determines peaks in 2D images
    Parameters
    ----------
    image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
    sigma : standard deviation of a given array (image)
    alpha : number of standard deviations to use to determine peak true 
            or false of a given coordinate in image
    
    Returns
    -------
    i_out : list - x coordinates of peaks
    j_out : list - y coordinates of peaks

    """
    i_out = []
    j_out = []
    image_temp = copy.deepcopy(image)
    width, height = image_temp.shape
    while True:
        k = np.argmax(image_temp)
        j, i = np.unravel_index(k, image_temp.shape)
        if image_temp[j, i] >= alpha * sigma:
            i_out.append(i)
            j_out.append(j)
            x = np.arange(i - size, i + size)
            y = np.arange(j - size, j + size)
            xv, yv = np.meshgrid(x, y)
            image_temp[
                yv.clip(0, image_temp.shape[0] - 1), xv.clip(0, image_temp.shape[1] - 1)
            ] = 0
        else:
            break
    ni_out, nj_out = [i / width for i in i_out], [j / height for j in j_out]
    ni_out, nj_out = close_points(ni_out, nj_out, 0.1)
    i_out, j_out = [int(i * width) for i in ni_out], [int(j * height) for j in nj_out]
    return i_out, j_out


def get_center(x, y):
    """Returns centroid of given points in (xy) plane
    Parameters
    ----------
    x   : ndarray of x coordinates 
    y   : ndarray of y coordinates

    Returns
    -------
        : numpy array of (x_c,y_c)
    """
    nPeaks = len(x)
    return np.array([np.sum(x) / nPeaks, np.sum(y) / nPeaks])


def sort_peaks(x, y):
    """Sorts coordinates (xy) into order of increasing radius
    from the centroid
    Parameters
    ----------
    x   : ndarray of x coordinates 
    y   : ndarray of y coordinates

    Returns
    -------
    x   : ndarray of sorted x coordinates 
    y   : ndarray of sorted y coordinates

    Notes
    -----
    For computational expense:
    TODO rewrite this function with sorted() and lambda functions
    """
    assert len(x) == len(y)
    nPeaks = len(x)
    centroid = get_center(x, y)
    nodes = np.empty((nPeaks, 2))
    peaks = np.ndarray(shape=(nPeaks, 2))
    nRemaining = nPeaks
    for m in range(nPeaks):
        nodes[m] = np.array([x[m], y[m]])
    for s in range(nPeaks):
        deltas = nodes - centroid
        dist_2 = np.einsum("ij,ij->i", deltas, deltas)
        min_idx = np.argmin(dist_2)
        peaks[s] = nodes[min_idx]
        nRemaining -= 1
        nodes = nodes[nodes != peaks[s]].reshape((nRemaining, 2))
    return peaks


def labeled_peaks(x, y, width, height):
    """Takes peaks and returns lattice coordinates and iterable of strings
    corresponding to the reflection (hk) of each lattice points
    Parameters
    ----------
    x       : ndarray of x coordinates 
    y       : ndarray of y 
    width   : width in pixels of the image (for trimming)
    height  : height in pixels of the image (for trimming)

    Returns
    -------
    lattice_points : np array of reflections
    idx            : iterable containing what reflection each element of `lattice_points` is
    """
    sorted_peaks = sort_peaks(x, y)                     
    lat_vec = np.vstack(
        (np.array([sorted_peaks[0,0], sorted_peaks[0,1]]), np.array([sorted_peaks[1,0], sorted_peaks[1,1]]))
    )

    centroid = get_center(sorted_peaks[:,0], sorted_peaks[:,1])
    lat_vec = lat_vec - centroid
    lattice_points, idx = bz.generate_lattice(
        min=(-4, -3), max=(6, 6), lattice_vectors=lat_vec
    )    

    lattice_points += centroid
    lattice_points, idx =  trimmed(lattice_points, idx, width, height)
    sorted_lattice = sort_peaks(lattice_points[:,0], lattice_points[:,1])

    unrolled_peaks = sorted_peaks.reshape(-1,2)
    #get lattice points corresponding to reflections determined by peak detection
    unrolled_lattice = sorted_lattice.reshape(-1,2)[:unrolled_peaks.shape[0],:]
    #SVD to transform lattice points to peaks detected
    ret_R, ret_t = rigid_transform_3D(unrolled_lattice,unrolled_peaks)
    trans_unrolled_lattice = (ret_R@unrolled_lattice) + ret_t
    #for the lattice points picked up by peak detection, take
    #lattice_point --> transformed_lattice_point 
    for m in range(lattice_points.shape[0]):
        tmp = lattice_points[m]
        if tmp in unrolled_lattice:
            deltas = trans_unrolled_lattice - tmp
            dist_2 = np.einsum("ij,ij->i", deltas, deltas)
            min_idx = np.argmin(dist_2)
            lattice_points[m] = trans_unrolled_lattice[min_idx]

    return lattice_points, idx



def rigid_transform_3D(A, B):
    """Perform SVD decomp of two sets of points `A` and `B`
    Parameters
    ----------

    A   : ndarray representing primary points
    B   : ndarray representing target points

    Returns
    -------
    R   : ndarray - transformation (rotation, scaling, rotation) matrix 
    t   : ndarary - translation vector
    """
    # assert A.shape == B.shape

    # num_rows, num_cols = A.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    # num_rows, num_cols = B.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def trimmed(lat_pts, index, width, height):
    """Takes peaks and returns only reflections inside target image and iterable of strings
    corresponding to the reflection (hk) of each of said lattice points 
    Parameters
    ----------
    x       : ndarray of x coordinates 
    y       : ndarray of y 
    width   : width in pixels of the image (for trimming)
    height  : height in pixels of the image (for trimming)

    Returns
    -------
    lat_pts : np array of reflections inside image
    idx            : iterable containing what reflection each element of `lat_pts` is
    """
    nPts = len(index)
    mask = np.zeros((nPts, 1), dtype=int)
    for m in range(nPts):
        b1, b2, b3, b4 = (
            lat_pts[m, 0] > 0,
            lat_pts[m, 1] > 0,
            lat_pts[m, 0] < width,
            lat_pts[m, 1] < height,
        )
        if b1 and b2 and b3 and b4:
            mask[m] = 1
    return lat_pts[np.where(mask)[0]], [index[i] for i in np.where(mask)[0]]

