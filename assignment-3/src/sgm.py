import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import view_as_windows


def add_padding(image, radius):
    '''
        Adds border to the image i by mirroring
        :param image: image to add padding to
        :param radius: how much padding to add
        :return: i with the added padding
    '''
    return np.pad(image, ((radius, radius), (radius, radius)), 'symmetric')


def compute_cost_volume_sad(left_image, right_image, D, radius):
    """
    Sum of Absolute Differences (SAD) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """

    H, W = left_image.shape
    cost_volume = np.full((H, W, D), np.inf)

    padded_left = add_padding(left_image, radius)
    padded_right = add_padding(right_image, radius)

    window_size = 2 * radius + 1
    padded_windows_left = view_as_windows(padded_left, (window_size, window_size))
    padded_windows_right = view_as_windows(padded_right, (window_size, window_size))

    for y in range(H):
        for x in range(W):
            window_left = padded_windows_left[y, x]
            for d in range(D):
                if x - d > 0:
                    window_right = padded_windows_right[y, x - d]
                    cost_volume[y, x, d] = np.sum((window_left - window_right)**2)

    return cost_volume


def compute_cost_volume_ssd(left_image, right_image, D, radius):
    """
    Sum of Squared Differences (SSD) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """
    H, W = left_image.shape
    cost_volume = np.full((H, W, D), np.inf)

    padded_left = add_padding(left_image, radius)
    padded_right = add_padding(right_image, radius)

    window_size = 2 * radius + 1
    padded_windows_left = view_as_windows(padded_left, (window_size, window_size))
    padded_windows_right = view_as_windows(padded_right, (window_size, window_size))

    for y in range(H):
        for x in range(W):
            window_left = padded_windows_left[y, x]
            for d in range(D):
                if x - d >= 0:
                    window_right = padded_windows_right[y, x - d]
                    cost_volume[y, x, d] = np.sum(np.abs(window_left - window_right))

    return cost_volume


def compute_cost_volume_ncc(left_image, right_image, D, radius):
    """
    Normalized Cross Correlation (NCC) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """
    H, W = left_image.shape
    cost_volume = np.full((H, W, D), np.inf)

    padded_left = add_padding(left_image, radius)
    padded_right = add_padding(right_image, radius)

    window_size = 2 * radius + 1

    padded_windows_left = view_as_windows(padded_left, (window_size, window_size))
    padded_windows_right = view_as_windows(padded_right, (window_size, window_size))

    window_norm = 1.0 / (window_size**2)

    for y in range(H):
        for x in range(W):
            window_left = padded_windows_left[y, x]
            window_left_mean = window_norm * np.sum(window_left)
            window_left_no_mean = window_left - window_left_mean
            denominator_left = np.sum(window_left_no_mean ** 2)
            for d in range(D):
                if x - d > 0:
                    window_right = padded_windows_right[y, x - d]
                    window_right_mean = window_norm * np.sum(window_right)
                    window_right_no_mean = window_right - window_right_mean
                    numerator = np.sum(np.multiply(window_left_no_mean, window_right_no_mean))
                    denominator_right = np.sum(window_right_no_mean**2)
                    denominator = np.sqrt(denominator_left * denominator_right)
                    cost_volume[y, x, d] = 1 - numerator / denominator

    return cost_volume


def compute_wta(cost_volume):
    """
    Compute the disparity map using WTA scheme
    :param cost_volume: cost volume of shape (H,W,D)
    :return: pixel wise disparity map of shape (H,W)
    """
    H, W, D = cost_volume.shape
    disparity_map = cost_volume.argmin(axis=2) / D
    return disparity_map


def compute_accuracy(disparity_map, ground_truth, occlusion_mask, disparity_threshold):
    '''
    Counts number of accurate points below a certain disparity threshold
    param disparity_map: the disparity map that is being analyzed (H, W)
    param ground_truth: the corresponding ground truth disparity for the previous map (H, W)
    param occlusion_mask: points that don't have ground truth, should be ignored (H, W)
    param disparity_threshold: scalar to threshold disparities
    return percentage of accurate points
    '''
    H, W = disparity_map.shape
    correct_points = np.zeros((H, W))

    diff_wrt_gt = np.abs(disparity_map - ground_truth)
    diff_wrt_gt_thr = np.where(diff_wrt_gt <= disparity_threshold)
    correct_points[diff_wrt_gt_thr] = 1.0

    unocc_points = np.where(occlusion_mask == 1.0)
    total_points = len(unocc_points[0]) - 1

    return np.sum(correct_points * occlusion_mask) / total_points

def dp_chain(g, f, m):
    '''
    g: unary costs with shape (H,W,D)
    f: pairwise costs with shape (H,W,D,D)
    m: messages with shape (H,W,D)
    '''
    # TODO
    return


def get_pairwise_costs(H, W, D, weights=None):
    """
    :param H: height of input image
    :param W: width of input image
    :param D: maximal disparity
    :param weights: edge-dependent weights (necessary to implement the bonus task)
    :return: pairwise_costs of shape (H,W,D,D)
             Note: If weight=None, then each spatial position gets exactly the same pairwise costs.
             In this case the array of shape (D,D) can be broadcasted to (H,W,D,D) by using np.broadcast_to(..).
    """
    # TODO
    return


def compute_sgm(cost_volume, f):
    """
    Compute the SGM
    :param cost_volume: cost volume of shape (H,W,D)
    :param f: Pairwise costs of shape (H,W,D,D)
    :return: pixel wise disparity map of shape (H,W)
    """
    # TODO
    return


def main():
    # Set parameters
    D = 65
    radius = 2

    # Load input images
    im0 = imread("data/Adirondack_left.png")
    im1 = imread("data/Adirondack_right.png")

    im0g = rgb2gray(im0)
    im1g = rgb2gray(im1)

    # Plot input images
    plt.figure(figsize=(8,4))
    plt.subplot(121), plt.imshow(im0g, cmap='gray'), plt.title('Left')
    plt.subplot(122), plt.imshow(im1g, cmap='gray'), plt.title('Right')
    plt.tight_layout()

    # Use either SAD, NCC or SSD to compute the cost volume
    cost_volume = compute_cost_volume_ncc(im0g, im1g, D, radius)
    cost_volume_wta = compute_wta(cost_volume)

    # Plot the resulting disparity map
    plt.figure()
    plt.imshow(cost_volume_wta, cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

    # Compute pairwise costs
    H, W, D = cost_volume.shape
    f = get_pairwise_costs(H, W, D)

    # Compute SGM
    disp = compute_sgm(cost_volume, f)

    # Plot result
    plt.figure()
    plt.imshow(disp)
    plt.show()


if __name__== "__main__":
    main()
