import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import time

from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.ndimage.filters import *

def compute_psnr(img1, img2):
    """
      @param: img1 first input Image
      @param: img2 second input Image

      @return: Peak signal-to-noise ratio between the first and second image
    """

    # Compute mse
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf

    maxValue = 1.0
    return 10 * np.log2(maxValue / mse)


def compute_mean(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the mean
      of this window.

      @return: image containing the mean for each pixel
    """

    # Add padding to complete image
    radius = (filter_size - 1) // 2
    image_padded = np.pad(image, ((radius, radius), (radius, radius)), 'symmetric')

    # Build the sliding window
    cimgs = []

    for fi in range(filter_size):
        for fj in range(filter_size):

            if -(filter_size - fi) + 1 == 0 and -(filter_size - fj) + 1 == 0:
                cimg = image_padded[fi:, fj:]
            elif -(filter_size - fi) + 1 == 0:
                cimg = image_padded[fi:, fj: -(filter_size - fj) + 1]
            elif -(filter_size - fj) + 1 == 0:
                cimg = image_padded[fi: -(filter_size - fi) + 1, fj:]
            else:
                cimg = image_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1]

            cimgs.append(cimg)

    # Convert list of images to 3D array
    cimgs_as_array = np.asarray(cimgs)

    # Perform mean for each channel and merge
    image_mean = np.mean(cimgs_as_array, axis=0)

    return image_mean


def compute_variance(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the variance
      of this window.

      @return: image containing the variance (\sigma^2) for each pixel
    """

    # Add padding to complete image
    radius = (filter_size - 1) // 2
    img = np.pad(image, ((radius, radius), (radius, radius)), 'symmetric')

    # Build the sliding windows
    cimgs = []

    for fi in range(filter_size):
        for fj in range(filter_size):
            if -(filter_size - fi) + 1 == 0 and -(filter_size - fj) + 1 == 0:
                cimg = img[fi:, fj:]
            elif -(filter_size - fi) + 1 == 0:
                cimg = img[fi:, fj: -(filter_size - fj) + 1]
            elif -(filter_size - fj) + 1 == 0:
                cimg = img[fi: -(filter_size - fi) + 1, fj:]
            else:
                cimg = img[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1]

            cimgs.append(cimg)

    # Convert list of images to 3D array
    cimgs_as_array = np.asarray(cimgs)

    # Perform var for each channel and merge
    image_var = np.var(cimgs_as_array, axis=0)

    return image_var


def compute_a(F, I, m, mu, variance, filter_size, epsilon):
    """
      Compute the intermediate result 'a' as described in the task (equation 4)

      @param: F input image
      @param: I guidance image
      @param: m mean of input image
      @param: mu mean of guidance image
      @param: variance of guidance image
      @param: filter_size
      @param: epsilon smoothing parameter

      @return: image containing a_k for each pixel
    """

    # Add padding to complete images
    radius = (filter_size - 1) // 2
    I_padded = np.pad(I, ((radius, radius), (radius, radius)), 'symmetric')
    F_padded = np.pad(F, ((radius, radius), (radius, radius)), 'symmetric')

    # Build the windows for each channel and jointly multiply F by I
    cimgs = []

    for fi in range(filter_size):
        for fj in range(filter_size):

            if -(filter_size - fi) + 1 == 0 and -(filter_size - fj) + 1 == 0:
                cimg = F_padded[fi:, fj:] * I_padded[fi:, fj:]
            elif -(filter_size - fi) + 1 == 0:
                cimg = F_padded[fi:, fj: -(filter_size - fj) + 1] * I_padded[fi:, fj: -(filter_size - fj) + 1]
            elif -(filter_size - fj) + 1 == 0:
                cimg = F_padded[fi: -(filter_size - fi) + 1, fj:] * I_padded[fi: -(filter_size - fi) + 1, fj:]
            else:
                cimg = F_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1] * I_padded[fi: -(filter_size - fi) + 1,fj: -(filter_size - fj) + 1]

            cimgs.append(cimg)

    # Convert list of images to 3D array
    cimgs_as_array = np.asarray(cimgs)

    # Perform mean for the F*I channels and merge
    mean_fi = np.mean(cimgs_as_array, axis=0)

    # Compute a
    a = (mean_fi - m * mu) / (variance + epsilon)

    return a


def compute_b(m, a, mu):
    """
      Compute the intermediate result 'b' as described in the task (equation 5)

      @param: m mean of input image
      @param: a
      @param: mu mean of guidance image

      @return: image containing b_k for each pixel
    """

    # Use the formula in the pdf
    b = m - a * mu

    return b


def compute_q(mean_a, mean_b, I):
    """
      Compute the final filtered result 'q' as described in the task (equation 6)
      @return: filtered image
    """

    # Use the formula in the pdf
    q = mean_a * I + mean_b
    q = np.clip(q, 0.0, 1.0)

    return q


def calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon):
    """
      Apply the guided filter to an image

      @param: input_img Image to be filtered
      @param: guidance_img Image used as guidance
      @param: filter_size
      @param: epsilon Smoothing parameter

      @returns:
        a_mean: the mean value of a according to (6)
        b_mean: the mean value of b according to (6)
        q: filtered image
    """

    # Operations on guidance image
    guidance_img_mean = compute_mean(guidance_img, filter_size)
    guidance_img_var = compute_variance(guidance_img, filter_size)

    # Operations of input image
    input_img_mean = compute_mean(input_img, filter_size)

    # Compute a and and b
    a = compute_a(input_img, guidance_img, input_img_mean, guidance_img_mean, guidance_img_var, filter_size, epsilon)
    b = compute_b(input_img_mean, a, guidance_img_mean)

    # Compute mean of a and b
    a_mean = compute_mean(a, filter_size)
    b_mean = compute_mean(b, filter_size)

    # Compute final filtered image q and normalize
    q = compute_q(a_mean, b_mean, input_img)

    return a_mean, b_mean, q


def guided_upsampling(input_img, guidance_img, filter_size, epsilon):
    """
      Perform an upsampling of a lower res color image using a higher res grayscale guidance image

      @param: input_img The image to be upsampled
      @param: guidance_img The upsampled version of the same image but grayscale
      @param: filter_size
      @param: epsilon The smoothing factor of the guided filter

      @returns:
        a_mean: the mean value of a according to (6)
        b_mean: the mean value of b according to (6)
        upsampled_img A higher resolution version of the input image, with the resolution of the guidance image
    """

    # Filter the image using the guidance image
    if input_img.ndim == 2:
        a_mean, b_mean, upsampled_img = calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon)
    else:
        a_mean_r, b_mean_r, upsampled_img_r = calculate_guided_image_filter(input_img[:, :, 0], guidance_img, filter_size, epsilon)
        a_mean_g, b_mean_g, upsampled_img_g = calculate_guided_image_filter(input_img[:, :, 1], guidance_img, filter_size, epsilon)
        a_mean_b, b_mean_b, upsampled_img_b = calculate_guided_image_filter(input_img[:, :, 2], guidance_img, filter_size, epsilon)
        a_mean = np.dstack((a_mean_r, a_mean_g, a_mean_b))
        b_mean = np.dstack((b_mean_r, b_mean_g, b_mean_b))
        upsampled_img = np.dstack((upsampled_img_r, upsampled_img_g, upsampled_img_b))

    return a_mean, b_mean, upsampled_img


def prepare_imgs(input_filename, downsample_ratio):
    """
      Prepare the images for the guided upsample filtering

      @param: input_filename Filename of the input image
      @param: upsample_ratio ratio between the filter input resolution and the guidance image resolution

      @returns:
        input_img: the input image of the filter
        guidance_img: the guidance image of the filter
        reference_img: the high resolution reference image, this should only be used for calculation of the PSNR and plots for comparison
    """

    # Load original image
    initial_img = io.imread(input_filename, as_gray=False)

    # Calculate grayscale for guidance image
    guidance_img = rgb2gray(initial_img)

    # Downsample original image
    input_img = resize(initial_img, (initial_img.shape[0] // downsample_ratio, initial_img.shape[1] // downsample_ratio), anti_aliasing=True)

    return input_img, guidance_img, initial_img


def plot_result(input_img, guidance_img, filtered_img):

    # Prepare the figure
    fig, axes = plt.subplots(nrows=1, ncols=3)

    ax = axes.ravel()
    ax[0].imshow(input_img)
    ax[1].imshow(guidance_img, cmap='gray')
    ax[2].imshow(filtered_img)

    ax[0].set_title("Input image")
    ax[1].set_title("Guidance image")
    ax[2].set_title("Upsampled image")

    plt.tight_layout()
    plt.show()

    return


def approach_1(input_img, guidance_img, filter_size, epsilon):

    upsample_size = (guidance_img.shape[0], guidance_img.shape[1])
    a_mean_1, b_mean_1, filtered_img_1 = guided_upsampling(resize(input_img, upsample_size), guidance_img, filter_size, epsilon)

    return filtered_img_1


def approach_2(input_img, guidance_img, filter_size, epsilon):

    downsample_size = (input_img.shape[0], input_img.shape[1])
    a_mean_2, b_mean_2, filtered_img_2 = guided_upsampling(input_img, resize(guidance_img, downsample_size), filter_size, epsilon)

    a_mean_2_resized = resize(a_mean_2, guidance_img.shape)
    b_mean_2_resized = resize(b_mean_2, guidance_img.shape)
    filtered_img_2_r = compute_q(a_mean_2_resized[:, :, 0], b_mean_2_resized[:, :, 0], guidance_img)
    filtered_img_2_g = compute_q(a_mean_2_resized[:, :, 1], b_mean_2_resized[:, :, 1], guidance_img)
    filtered_img_2_b = compute_q(a_mean_2_resized[:, :, 2], b_mean_2_resized[:, :, 2], guidance_img)
    filtered_img_2 = np.dstack((filtered_img_2_r, filtered_img_2_g, filtered_img_2_b))

    return filtered_img_2

def test_suite():

    # Define parameters to test
    input_images = ['monarch.png']
    epsilons = [0.001, 1, 1000]
    filters_sizes = [3, 7, 15]
    downsampling_rates = [2, 8, 16]

    # Do tests and fill arrays
    for input_filename in input_images:

        id_test = 0
        result_figures = []
        result_runs = []

        # Create figure for approach1 (eps vs filter_size)
        fig_img1, axes_img1 = plt.subplots(nrows=3, ncols=3)

        # Create figure for approach2 (eps vs filter_size)
        fig_img2, axes_img2 = plt.subplots(nrows=3, ncols=3)

        # Create figure for downsampling comparison of both approaches
        fig_img_ds, axes_img_ds = plt.subplots(nrows=2, ncols=3)

        for epsilon_index in range(len(epsilons)):
            for filter_size_index in range(len(filters_sizes)):
                for downsample_rate_index in range(len(downsampling_rates)):

                    # Get the array values
                    epsilon = epsilons[epsilon_index]
                    filter_size = filters_sizes[filter_size_index]
                    downsample_rate = downsampling_rates[downsample_rate_index]

                    print("Running " + input_filename + " with params: eps = " + str(epsilon) + ", size = " + str(filter_size) + ", downsample = " + str(downsample_rate))
                    input_img, guidance_img, initial_img = prepare_imgs(input_filename, downsample_rate)

                    time_approach_1 = time.time()
                    filtered_img_1 = approach_1(input_img, guidance_img, filter_size, epsilon)
                    time_approach_1 = time.time() - time_approach_1

                    time_approach_2 = time.time()
                    filtered_img_2 = approach_2(input_img, guidance_img, filter_size, epsilon)
                    time_approach_2 = time.time() - time_approach_2

                    psnr_filtered_1 = compute_psnr(filtered_img_1, initial_img)
                    psnr_filtered_2 = compute_psnr(filtered_img_2, initial_img)
                    psnr_upsampled_1 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)
                    psnr_upsampled_2 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)

                    # If csv is empty add header
                    if len(result_runs) == 0:
                        result_header = ['epsilon', 'filter_size', 'downsample_rate', 'psnr_filtered_1', 'psnr_filtered_2', 'psnr_upsampled_1', 'psnr_upsampled_2', 'time_approach_1', 'time_approach_2']
                        result_runs.append(result_header)

                    # Append iteration result to csv
                    result_run = [epsilon, filter_size, downsample_rate, '%.4f' % psnr_filtered_1, '%.4f' % psnr_filtered_2, '%.4f' % psnr_upsampled_1, '%.4f' %psnr_upsampled_2, '%.4f' % time_approach_1, '%.4f' % time_approach_2]
                    result_runs.append(result_run)
                    id_test += 1

                    if epsilon_index == 1 and filter_size_index == 0:
                        axes_img_ds[0, downsample_rate_index].set_title("approach_1: eps = " + str(epsilon) + ", fs = " + str(filter_size) + ", ds = " + str(downsample_rate))
                        axes_img_ds[1, downsample_rate_index].set_title("approach_2: eps = " + str(epsilon) + ", fs = " + str(filter_size) + ", ds = " + str(downsample_rate))
                        axes_img_ds[0, downsample_rate_index].imshow(filtered_img_1)
                        axes_img_ds[1, downsample_rate_index].imshow(filtered_img_2)

                    if downsample_rate_index == 0:
                        # Fill the figure with the images
                        axes_img1[epsilon_index, filter_size_index].set_title("eps = " + str(epsilon) + ", fs = " + str(filter_size))
                        axes_img2[epsilon_index, filter_size_index].set_title("eps = " + str(epsilon) + ", fs = " + str(filter_size))
                        axes_img1[epsilon_index, filter_size_index].imshow(filtered_img_1)
                        axes_img2[epsilon_index, filter_size_index].imshow(filtered_img_2)

        # Write figures
        plt.tight_layout()
        plt.show()
        fig_img1.savefig('approach-1.png')
        fig_img2.savefig('approach-2.png')
        fig_img_ds.savefig('downsampling-ap1-ap2.png')

        # Write CSV file with the PSNRS and timestamps
        #csv_filename = input_filename + ".csv"
        #pd.DataFrame(result_runs).to_csv(csv_filename, header=False, index=False, float_format='%.3f')

    return


if __name__ == "__main__":

    # If no parameters then run complete test suite
    if len(sys.argv) != 2:
        print('Running test suite...')
        test_suite()
        exit(0)

    # Set some default parameters
    start_time = time.time()
    half_size = 3
    downsample_ratio = 3
    filter_size = 3
    epsilon = 1
    input_filename = sys.argv[1]

    # Prepare Images
    input_img, guidance_img, initial_img = prepare_imgs(input_filename, downsample_ratio)

    # Perform Guided Upsampling

    # Approach (1):
    filtered_img_1 = approach_1(input_img, guidance_img, filter_size, epsilon)

    # Approach (2):
    filtered_img_2 = approach_2(input_img, guidance_img, filter_size, epsilon)

    # Calculate PSNR
    psnr_filtered_1 = compute_psnr(filtered_img_1, initial_img)
    psnr_filtered_2 = compute_psnr(filtered_img_2, initial_img)
    psnr_upsampled_1 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)
    psnr_upsampled_2 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)

    # Print information
    print('Runtime: {} - [Approach 1: PSNR filtered: {:.4f} - PSNR upsampled: {:.4f}] [Approach 2: PSNR filtered: {:.4f} - PSNR upsampled: {:.4f}]'.format(time.time() - start_time, psnr_filtered_2, psnr_upsampled_2, psnr_filtered_1, psnr_upsampled_1))

    # Plot result
    plot_result(input_img, guidance_img, filtered_img_1)
    plot_result(input_img, guidance_img, filtered_img_2)

