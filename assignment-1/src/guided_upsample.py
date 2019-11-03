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

    # check images have same size
    if img1.shape != img2.shape:
        return -1

    # Compute mse
    mse = 0
    for y in img1.shape[0]:
        for x in img1.shape[1]:
            diff = img1[y, x] - img2[y,x]
            mse += diff * diff

    # Normalize mse
    mse /= img1.shape[0] * img2.shape[1]
    if mse == 0:
        return np.inf

    # Compute psnr
    max_value = 1.0
    psnr = 10 * np.log10(max_value / np.sqrt(mse))

    return psnr


def compute_mean(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the mean
      of this window.

      @return: image containing the mean for each pixel
    """

    # Add padding to complete image
    image_padded = np.pad(image, ((filter_size, filter_size), (filter_size, filter_size), (0, 0)), 'symmetric')

    # Build the windows for each channel
    cimgs_r = []
    cimgs_g = []
    cimgs_b = []

    for fi in range(filter_size):
        for fj in range(filter_size):

            if -(filter_size - fi) + 1 == 0 and -(filter_size - fj) + 1 == 0:
                cimg_r = image_padded[fi:, fj:, 0]
                cimg_g = image_padded[fi:, fj:, 1]
                cimg_b = image_padded[fi:, fj:, 2]
            elif -(filter_size - fi) + 1 == 0:
                cimg_r = image_padded[fi:, fj: -(filter_size - fj) + 1, 0]
                cimg_g = image_padded[fi:, fj: -(filter_size - fj) + 1, 1]
                cimg_b = image_padded[fi:, fj: -(filter_size - fj) + 1, 2]
            elif -(filter_size - fj) + 1 == 0:
                cimg_r = image_padded[fi: -(filter_size - fi) + 1, fj:, 0]
                cimg_g = image_padded[fi: -(filter_size - fi) + 1, fj:, 1]
                cimg_b = image_padded[fi: -(filter_size - fi) + 1, fj:, 2]
            else:
                cimg_r = image_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 0]
                cimg_g = image_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 1]
                cimg_b = image_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 2]

            cimgs_r.append(cimg_r)
            cimgs_g.append(cimg_g)
            cimgs_b.append(cimg_b)

    # Convert list of images to 3D array
    cimgs_r_asarray = np.asarray(cimgs_r)
    cimgs_g_asarray = np.asarray(cimgs_g)
    cimgs_b_asarray = np.asarray(cimgs_b)

    # Perform mean for each window
    mean_r = np.mean(cimgs_r_asarray, axis=0)
    mean_g = np.mean(cimgs_g_asarray, axis=0)
    mean_b = np.mean(cimgs_b_asarray, axis=0)

    # Build mean image
    image_mean = np.dstack((mean_r, mean_g, mean_b))

    return image_mean


def compute_variance(image, filter_size):
    """
      For each pixel in the image image consider a window of size
      filter_size x filter_size around the pixel and compute the variance
      of this window.

      @return: image containing the variance (\sigma^2) for each pixel
    """

    # Add padding to complete image
    img = np.pad(image, ((filter_size, filter_size), (filter_size, filter_size), (0, 0)), 'symmetric')

    # Build the windows for each channel
    cimgs_r = []
    cimgs_g = []
    cimgs_b = []

    for fi in range(filter_size):
        for fj in range(filter_size):

            if -(filter_size - fi) + 1 == 0 and -(filter_size - fj) + 1 == 0:
                cimg_r = img[fi:, fj:, 0]
                cimg_g = img[fi:, fj:, 1]
                cimg_b = img[fi:, fj:, 2]
            elif -(filter_size - fi) + 1 == 0:
                cimg_r = img[fi:, fj: -(filter_size - fj) + 1, 0]
                cimg_g = img[fi:, fj: -(filter_size - fj) + 1, 1]
                cimg_b = img[fi:, fj: -(filter_size - fj) + 1, 2]
            elif -(filter_size - fj) + 1 == 0:
                cimg_r = img[fi: -(filter_size - fi) + 1, fj:, 0]
                cimg_g = img[fi: -(filter_size - fi) + 1, fj:, 1]
                cimg_b = img[fi: -(filter_size - fi) + 1, fj:, 2]
            else:
                cimg_r = img[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 0]
                cimg_g = img[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 1]
                cimg_b = img[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 2]

            cimgs_r.append(cimg_r)
            cimgs_g.append(cimg_g)
            cimgs_b.append(cimg_b)

    # Convert list of images to 3D array
    cimgs_r_asarray = np.asarray(cimgs_r)
    cimgs_g_asarray = np.asarray(cimgs_g)
    cimgs_b_asarray = np.asarray(cimgs_b)

    # Perform var for each window
    var_r = np.var(cimgs_r_asarray, axis=0)
    var_g = np.var(cimgs_g_asarray, axis=0)
    var_b = np.var(cimgs_b_asarray, axis=0)

    # Build var image
    image_var = np.dstack((var_r, var_g, var_b))

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
    I_padded = np.pad(I, ((filter_size, filter_size), (filter_size, filter_size), (0, 0)), 'symmetric')
    F_padded = np.pad(F, ((filter_size, filter_size), (filter_size, filter_size), (0, 0)), 'symmetric')

    # Build the windows for each channel and jointly multiply F by I
    cimgs_r = []
    cimgs_g = []
    cimgs_b = []

    for fi in range(filter_size):
        for fj in range(filter_size):

            if -(filter_size - fi) + 1 == 0 and -(filter_size - fj) + 1 == 0:
                cimg_r = F_padded[fi:, fj:, 0] * I_padded[fi:, fj:, 0]
                cimg_g = F_padded[fi:, fj:, 1] * I_padded[fi:, fj:, 1]
                cimg_b = F_padded[fi:, fj:, 2] * I_padded[fi:, fj:, 2]
            elif -(filter_size - fi) + 1 == 0:
                cimg_r = F_padded[fi:, fj: -(filter_size - fj) + 1, 0] * I_padded[fi:, fj: -(filter_size - fj) + 1, 0]
                cimg_g = F_padded[fi:, fj: -(filter_size - fj) + 1, 1] * I_padded[fi:, fj: -(filter_size - fj) + 1, 1]
                cimg_b = F_padded[fi:, fj: -(filter_size - fj) + 1, 2] * I_padded[fi:, fj: -(filter_size - fj) + 1, 2]
            elif -(filter_size - fj) + 1 == 0:
                cimg_r = F_padded[fi: -(filter_size - fi) + 1, fj:, 0] * I_padded[fi: -(filter_size - fi) + 1, fj:, 0]
                cimg_g = F_padded[fi: -(filter_size - fi) + 1, fj:, 1] * I_padded[fi: -(filter_size - fi) + 1, fj:, 1]
                cimg_b = F_padded[fi: -(filter_size - fi) + 1, fj:, 2] * I_padded[fi: -(filter_size - fi) + 1, fj:, 2]
            else:
                cimg_r = F_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 0] * I_padded[fi: -(filter_size - fi) + 1,fj: -(filter_size - fj) + 1,0]
                cimg_g = F_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 1] * I_padded[fi: -(filter_size - fi) + 1,fj: -(filter_size - fj) + 1,1]
                cimg_b = F_padded[fi: -(filter_size - fi) + 1, fj: -(filter_size - fj) + 1, 2] * I_padded[fi: -(filter_size - fi) + 1,fj: -(filter_size - fj) + 1,2]

            cimgs_r.append(cimg_r)
            cimgs_g.append(cimg_g)
            cimgs_b.append(cimg_b)

    # Convert list of images to 3D array
    cimgs_r_asarray = np.asarray(cimgs_r)
    cimgs_g_asarray = np.asarray(cimgs_g)
    cimgs_b_asarray = np.asarray(cimgs_b)

    # Perform mean for the F*I windows
    mean_FI_r = np.mean(cimgs_r_asarray, axis=0)
    mean_FI_g = np.mean(cimgs_g_asarray, axis=0)
    mean_FI_b = np.mean(cimgs_b_asarray, axis=0)

    # Build the image from the means
    mean_FI = np.dstack((mean_FI_r, mean_FI_g, mean_FI_b))

    # Compute a
    a = (mean_FI - m * mu) / (variance + epsilon)

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

    return q


def calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon):
    """
      Apply the guided filter to an image

      @param: input_img Image to be filtered
      @param: guidance_img Image used as guidance
      @param: filter_size
      @param: epsilon Smoothing parameter

      @returns:
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
    q = np.clip(q, 0.0, 1.0)

    return q


def guided_upsampling(input_img, guidance_img, filter_size, epsilon):
    """
      Perform an upsampling of a lower res color image using a higher res grayscale guidance image

      @param: input_img The image to be upsampled
      @param: guidance_img The upsampled version of the same image but grayscale
      @param: filter_size
      @param: epsilon The smoothing factor of the guided filter

      @returns:
        upsampled_img A higher resolution version of the input image, with the resolution of the guidance image
    """

    # Resize input image to guidance image res
    upsampled_img = resize(input_img, (guidance_img.shape[0], guidance_img.shape[1]), anti_aliasing=True)

    # Filter the image using the guidance image
    upsampled_img = calculate_guided_image_filter(input_img, guidance_img, filter_size, epsilon)

    return upsampled_img


def prepare_imgs(input_filename, upsample_ratio):
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
    initial_img = io.imread(input_filename)

    # Calculate grayscale for guidance image
    guidance_img = rgb2gray(initial_img)

    # Downsample original image
    input_img = resize(initial_img, (initial_img.shape[0] // upsample_ratio, initial_img.shape[1] // upsample_ratio), anti_aliasing=True)

    return input_img, guidance_img, initial_img


def plot_result(input_img, guidance_img, filtered_img, output_filename):

    # Prepare the figure
    fig, axes = plt.subplots(nrows=1, ncols=3)

    ax = axes.ravel()
    ax[0].imshow(input_img)
    ax[1].imshow(guidance_img)
    ax[2].imshow(filtered_img)

    ax[0].set_title("Input image")
    ax[1].set_title("Guidance image")
    ax[2].set_title("Upsampled image")

    plt.tight_layout()
    plt.show()
    plt.savefig(output_filename)
    return


if __name__ == "__main__":
    start_time = time.time()

    # Set Parameters
    upsample_ratio = 5.0
    filter_size = 10
    epsilon = 1

    # Parse Parameter
    if len(sys.argv) != 2:
        raise ValueError('Wrong arguments')
    input_filename = sys.argv[1]

    # Prepare Images
    input_img, guidance_img, initial_img = prepare_imgs(input_filename, upsample_ratio)

    # Perform Guided Upsampling

    # Approach (1):
    filtered_img_1 = guided_upsampling(resize(input_img, guidance_img.shape), guidance_img, filter_size, epsilon)

    # Approach (2):
    filtered_img_2 = guided_upsampling(input_img, guidance_img, filter_size, epsilon)

    # Calculate PSNR
    psnr_filtered_1 = compute_psnr(filtered_img_1, initial_img)
    psnr_upsampled_1 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)

    psnr_filtered_2 = compute_psnr(filtered_img_2, initial_img)
    psnr_upsampled_2 = compute_psnr(resize(input_img, (guidance_img.shape[0], guidance_img.shape[1])).astype(np.float32), initial_img)

    print('Runtime: {} - [Approach 1: PSNR filtered: {:.2f} - PSNR upsampled: {:.2f}] [Approach 2: PSNR filtered: {:.2f} - PSNR upsampled: {:.2f}]'.format(time.time() - start_time, psnr_filtered_2, psnr_upsampled_2, psnr_filtered_1, psnr_upsampled_1))

    # Plot result
    plot_result(input_img, guidance_img, filtered_img_2, "method2.png")
    plot_result(input_img, guidance_img, filtered_img_1, "method1.png")
