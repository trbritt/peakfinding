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
import matplotlib.image as mpimg
from skimage import img_as_float

if __name__ == "__main__":
    # ============================================
    # Below tests peak finding on color RGB image
    # Plots image, pixel CDF, and histograms
    # before and after gamma corr. + peak finding
    # ============================================
    test_color = True
    if test_color:
        RGB_image = "test_RGB.jpg"

        # reading the image
        orig = img_as_float(mpimg.imread(RGB_image))
        gamma_corrected = rgb2gamma(orig)
        width, height = gamma_corrected.shape
        # computing the standard deviation of the image
        sigma = get_std(gamma_corrected)
        # getting the peaks
        i, j = get_max(gamma_corrected, sigma, alpha=20, size=10)
        lattice_points, idx = labeled_peaks(i,j, width, height)
        # let's see the results
        # Display results

        fig = plt.figure(figsize=(12, 8))
        axes = np.zeros((2, 2), dtype=np.object)
        axes[0, 0] = plt.subplot(2, 2, 1)
        axes[0, 1] = plt.subplot(2, 2, 2)
        axes[1, 0] = plt.subplot(2, 2, 3)
        axes[1, 1] = plt.subplot(2, 2, 4)

        ax_img, ax_hist, ax_cdf = plot_img_and_hist(orig, axes[:, 0])
        ax_img.set_title("Low contrast image")

        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_ylabel("Number of pixels")
        ax_hist.set_yticks(np.linspace(0, y_max, 5))

        ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])
        

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
    test_bw = True
    if test_bw:
        BW_image = "test_BW.jpg"

        test = img_as_float(mpimg.imread(BW_image))
        # computing the standard deviation of the image
        width, height = grey2gamma(test).shape

        sigma = get_std(grey2gamma(test))
        # getting the peaks
        i, j = get_max(grey2gamma(test), sigma, alpha=10, size=10)
        lattice_points, idx = labeled_peaks(i,j, width, height)

        fig, ax = plt.subplots(1, 2, figsize=(8, 6))
        ax[0].imshow(test, cmap=plt.cm.gray)
        ax[0].set_axis_off()
        ax[0].set_title("Original")
        
        ax[1].plot(lattice_points[:, 0], lattice_points[:, 1], "gx", ms=10)

        [
            ax[1].text(
                lattice_points[m, 0], lattice_points[m, 1], idx[m], color="white"
            )
            for m in range(len(idx))
        ]

        ax[1].imshow(grey2gamma(test), cmap=plt.cm.gray)
        ax[1].set_axis_off()
        ax[1].plot(i, j, "ro", markersize=10, alpha=0.5)
        ax[1].set_title("Greyscale + Gamma Corrected")
        fig.tight_layout()

        plt.show()

# =====================================================================
# END OF test_peaks.py
# =====================================================================
