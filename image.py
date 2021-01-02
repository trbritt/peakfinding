import numpy as np
import copy
import itertools
from scipy.spatial.ckdtree import cKDTree as spsp
from skimage import img_as_float, exposure
import brillouin as bz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings


class labeled_image:
    def __init__(self, image, gamma=2.40):
        self.__mImage = image
        self.__mData = img_as_float(mpimg.imread(image))
        self.set_channel(gamma)
        self.__ma = -1.0
        self.__mb = -1.0
        self.__mc = -1.0
        self.__malpha = -1.0
        self.__mbeta = -1.0
        self.__mgamma = -1.0

    @property
    def a(self):
        return self.__ma

    @a.setter
    def a(self, value):
        self.__ma = value

    @property
    def b(self):
        return self.__mb

    @b.setter
    def b(self, value):
        self.__mb = value

    @property
    def c(self):
        return self.__mc

    @c.setter
    def c(self, value):
        self.__mc = value

    @property
    def alpha(self):
        return self.__malpha

    @alpha.setter
    def alpha(self, value):
        self.__malpha = value

    @property
    def beta(self):
        return self.__mbeta

    @beta.setter
    def beta(self, value):
        self.__mbeta = value

    @property
    def gamma(self):
        return self.__mgamma

    @gamma.setter
    def gamma(self, value):
        self.__mgamma = value

    @property
    def width(self):
        return self.__mWidth

    @property
    def height(self):
        return self.__mHeight

    @property
    def image(self):
        return self.__mImage

    @property
    def corrected_data(self):
        return self.__mDataCorrected

    @property
    def data(self):
        return self.__mData

    def set_channel(self, gamma):
        if len(self.__mData.shape) == 3:
            self.__mDataCorrected = self.rgb2gamma(self.__mData, gamma)
        else:
            self.__mDataCorrected = self.grey2gamma(self.__mData, gamma)
        self.__mWidth, self.__mHeight = self.__mDataCorrected.shape

    def rgb2gamma(self, rgb, gamma=2.40):
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

    def grey2gamma(self, grey, gamma=2.40):
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
        nonlinear = np.where(grey > cutoff, 1.055 * pow(grey, 1 / gamma) - 0.055, grey)
        return nonlinear

    def get_std(self, image):
        """Returns standard deviation along default axis of an array of floats (derived from image)
        Parameters
        ----------
        image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats

        Returns
        -------
        float  : standard deviation
        """
        return np.std(image)

    def close_points(self, px, py, mindist, remove=True):
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
        if remove:
            res_points = np.delete(rare_points, exclude, axis=0)
            x = res_points[:, 0]
            y = res_points[:, 1]

            return x, y

    def peaks(self, alpha=20, size=10):
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
        sigma = self.get_std(self.__mDataCorrected)
        i_out = []
        j_out = []
        image_temp = copy.deepcopy(self.__mDataCorrected)
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
                    yv.clip(0, image_temp.shape[0] - 1),
                    xv.clip(0, image_temp.shape[1] - 1),
                ] = 0
            else:
                break
        ni_out, nj_out = [i / width for i in i_out], [j / height for j in j_out]
        ni_out, nj_out = self.close_points(ni_out, nj_out, 0.1)
        i_out, j_out = (
            [int(i * width) for i in ni_out],
            [int(j * height) for j in nj_out],
        )
        return i_out, j_out

    def get_center(self, x, y):
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

    def sort_peaks(self, x, y):
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
        centroid = self.get_center(x, y)
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

    def labeled_peaks(self, x, y, max_bz = 5, register_lattice=True):
        """Takes peaks and returns lattice coordinates and iterable of strings
        corresponding to the reflection (hk) of each lattice points
        Parameters
        ----------
        x       : ndarray of x coordinates
        y       : ndarray of y

        Returns
        -------
        lattice_points : np array of reflections
        idx            : iterable containing what reflection each element of `lattice_points` is

        Notes
        -----
        Idea behind this function:
            (i)     generate lattice determined by the two closest detected peaks
            (ii)    keep only points inside image dimensions
            (iii)   sort both detected peaks and lattice points by increasing
                        radius from centroid
            (iv)    the sets of peaks should be close to each other but won't be
                        exact, so we do SVD to determine how to translate
                        `n` detected peaks to `n` lattice points
                        (a) translation will be small but necessary to correctly
                            identify the exact detected peak with the reflection
            (v)     for the lattice points identified with detected peaks, their
                    value will now be the translated value so the peaks and
                    generated lattice points will be as close to each other
                    as can be. Lattice points corresponding to relfections not
                    seen in the image will be left untouched and are thus pred-
                    ictions of where the reflections should be based on the
                    Bragg peaks automatically determined.
        """

        sorted_peaks = self.sort_peaks(x, y)
        # self.__ma = 3.199
        # self.__mb = 3.199
        # self.__mc = 6.494
        # self.__malpha = 90.00 * np.pi / 180
        # self.__mbeta = 90.00 * np.pi / 180
        # self.__mgamma = 120.00 * np.pi / 180

        if np.any(
            np.array(
                [
                    self.__ma,
                    self.__mb,
                    self.__mc,
                    self.__malpha,
                    self.__mbeta,
                    self.__mgamma,
                ]
            )
            == -1.0
        ):
            print(
                f"Unit cell lengths (a,b,c) and/or relative angles (alpha, beta, gamma) for `{self.__mImage}` not set..."
            )
            print(
                "Using two peaks closest to centroid as basis of reciprocal lattice ..."
            )
            lat_vec = sorted_peaks[:2].reshape(-1,2)
            print(lat_vec)
        else:
            u, v, _ = bz.reciprocal_vectors3D(
                self.__ma,
                self.__mb,
                self.__mc,
                self.__malpha,
                self.__mbeta,
                self.__mgamma,
            )
            lat_vec = np.vstack((u[:2], v[:2]))

        centroid = self.get_center(sorted_peaks[:, 0], sorted_peaks[:, 1])
        lat_vec = lat_vec - centroid
        lattice_points, idx = bz.generate_lattice(
            min=(-max_bz, -max_bz), max=(max_bz, max_bz), lattice_vectors=lat_vec
        )

        lattice_points += centroid
        lattice_points, idx = self.trimmed(lattice_points, idx)
        sorted_lattice = self.sort_peaks(lattice_points[:, 0], lattice_points[:, 1])

        unrolled_peaks = sorted_peaks.reshape(-1, 2)
        # get lattice points corresponding to reflections determined by peak detection
        unrolled_lattice = sorted_lattice.reshape(-1, 2)[: unrolled_peaks.shape[0], :]

        # SVD to transform lattice points to peaks detected
        ret_R, ret_t = self.rigid_transform_3D(unrolled_lattice, unrolled_peaks)
        trans_unrolled_lattice = (ret_R @ unrolled_lattice) + ret_t
        # for the lattice points picked up by peak detection, take
        # lattice_point --> transformed_lattice_point
        if register_lattice:
            for m in range(lattice_points.shape[0]):
                tmp = lattice_points[m]
                if tmp in unrolled_lattice:
                    deltas = trans_unrolled_lattice - tmp
                    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
                    min_idx = np.argmin(dist_2)
                    lattice_points[m] = trans_unrolled_lattice[min_idx]
            self.__mLatticePoints = lattice_points
            self.__mIDX = idx
        return lattice_points, idx

    def reflection(self, h, k, l):
        """Returns coordinates in lattice of (`h`,`k`,`l`) reflection
        Parameters
        ----------
        h   :   int, miller index in `u` direction
        k   :   int, miller index in `v` direction
        l   :   int, miller index in `w` direction, which, since this class only
                handles 2D images, should always be zero

        Returns
        -------
            :   nparray of coordinates corresponding to the above reflection

        """
        assert l == 0, "Image is 2D, only reflections of form (h, k, l=0) allowed"
        return self.__mLatticePoints[self.__mIDX.index(str(k) + str(k) + str(l))]

    def rigid_transform_3D(self, A, B):
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
        # if linalg.matrix_rank(H) < 3:
        #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print("det(R) < R, reflection detected!, correcting for it ...")
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = -R @ centroid_A + centroid_B

        return R, t

    def trimmed(self, lat_pts, index):
        """Takes peaks and returns only reflections inside target image and iterable of strings
        corresponding to the reflection (hk) of each of said lattice points
        Parameters
        ----------
        x       : ndarray of x coordinates
        y       : ndarray of y

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
                lat_pts[m, 0] < self.__mWidth,
                lat_pts[m, 1] < self.__mHeight,
            )
            if b1 and b2 and b3 and b4:
                mask[m] = 1
        return lat_pts[np.where(mask)[0]], [index[i] for i in np.where(mask)[0]]
