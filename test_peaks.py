# utf-8 encoding
#

# =====================================================================
# File  : test_peaks.py
# Author: Tristan Britt
# Email : tristan.britt@mail.mcgill.ca
# Date  : 18/12/2020
# Notes :
#   This script will take any preconverted jpg image and compute the
#   peaks. Quadtrees are used to ensure the peaks are unique within
#   a given radius of each other.
#   TODO: Implement preconversion from other image formats (tif,png,..)
#   to be more inclusive of data sources
# Dependencies:
#   This is just a test script that requires both an RGB and BW test
#   image to function properly
#   As coded, these are 'test_RGB.jpg' and 'test_BW.jpg' respectively
# =====================================================================

from image import *
from skimage import img_as_float


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
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


if __name__ == "__main__":
    # ============================================
    # Below tests peak finding on color RGB image
    # Plots image, pixel CDF, and histograms
    # before and after gamma corr. + peak finding
    # ============================================
    test_color = True
    if test_color:
        RGB_image = "test_RGB.jpg"
        test_color = labeled_image(RGB_image)
        i, j = test_color.peaks(alpha=20, size=10)
        lattice_points, idx = test_color.labeled_peaks(i, j)
        # let's see the results
        # Display results
        fig = plt.figure(figsize=(12, 8))
        axes = np.zeros((2, 2), dtype=np.object)
        axes[0, 0] = plt.subplot(2, 2, 1)
        axes[0, 1] = plt.subplot(2, 2, 2)
        axes[1, 0] = plt.subplot(2, 2, 3)
        axes[1, 1] = plt.subplot(2, 2, 4)

        ax_img, ax_hist, ax_cdf = plot_img_and_hist(test_color.data, axes[:, 0])
        ax_img.set_title("Low contrast image")

        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_ylabel("Number of pixels")
        ax_hist.set_yticks(np.linspace(0, y_max, 5))

        ax_img, ax_hist, ax_cdf = plot_img_and_hist(
            test_color.corrected_data, axes[:, 1]
        )

        ax_img.plot(lattice_points[:, 0], lattice_points[:, 1], "gx", ms=10)

        [
            ax_img.text(
                lattice_points[m, 0], lattice_points[m, 1], idx[m], color="white"
            )
            for m in range(len(idx))
        ]

        ax_img.set_title("Gamma correction")

        ax_cdf.set_ylabel("Fraction of total intensity")
        ax_cdf.set_yticks(np.linspace(0, 1, 5))

        # prevent overlap of y-axis labels
        fig.tight_layout()
        plt.show()
    # ============================================
    # Below tests peak finding on greyscale image
    # ============================================
    test_bw = False
    if test_bw:
        BW_image = "test_BW.jpg"

        test_BW = labeled_image(BW_image)
        i, j = test_BW.peaks(alpha=10, size=10)
        lattice_points, idx = test_BW.labeled_peaks(i, j)

        fig, ax = plt.subplots(1, 2, figsize=(8, 6))
        ax[0].imshow(test_BW.data, cmap=plt.cm.gray)
        ax[0].set_axis_off()
        ax[0].set_title("Original")

        ax[1].plot(lattice_points[:, 0], lattice_points[:, 1], "gx", ms=10)

        [
            ax[1].text(
                lattice_points[m, 0], lattice_points[m, 1], idx[m], color="white"
            )
            for m in range(len(idx))
        ]

        ax[1].imshow(test_BW.corrected_data, cmap=plt.cm.gray)
        ax[1].set_axis_off()
        ax[1].plot(i, j, "ro", markersize=10, alpha=0.5)
        ax[1].set_title("Greyscale + Gamma Corrected")
        fig.tight_layout()

        plt.show()

# =====================================================================
# END OF test_peaks.py
# =====================================================================
