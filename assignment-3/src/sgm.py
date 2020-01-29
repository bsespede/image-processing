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
    cost_volume = np.zeros((H, W, D))

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
    cost_volume = np.zeros((H, W, D))

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
    cost_volume = np.full((H, W, D), 1.0)

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
    return percentage of accurate pointsx`
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
    :return: updated messages
    '''
    H, W, D = g.shape

    # Use the message passing scheme seen in the KU
    for y in range(H):
        for x in range(1, W):
            cur_gm = g[y, x - 1, :] + m[y, x - 1, :]
            cur_gm = np.repeat(cur_gm, D).reshape((D, D)).T
            cur_f = f[y, x, :, :]
            m[y, x, :] = np.amin(cur_gm + cur_f, axis=1)

    return m


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
    pairwise_costs = np.zeros((D, D))
    L0 = 0
    L1 = 1
    L2 = 2

    for d1 in range(D):
        for d2 in range(D):
            if d1 == d2:
                pairwise_costs[d1, d2] = L0
            elif abs(d1 - d2) == 1:
                pairwise_costs[d1, d2] = L1
            else:
                pairwise_costs[d1, d2] = L2

    return np.broadcast_to(pairwise_costs, (H, W, D, D))


def compute_sgm(cost_volume, pairwise_costs):
    """
    Compute the SGM
    :param cost_volume: cost volume of shape (H,W,D)
    :param f: Pairwise costs of shape (H,W,D,D)
    :return: pixel wise disparity map of shape (H,W)
    """
    H, W, D = cost_volume.shape

    # Direction: left-to-right
    message_ltr = np.zeros((H, W, D))
    cost_volume_ltr = cost_volume
    message_ltr = dp_chain(cost_volume_ltr, pairwise_costs, message_ltr)

    # Direction: right-to-left
    message_rtl = np.zeros((H, W, D))
    cost_volume_rtl = np.flip(cost_volume, axis=1)
    message_rtl = np.flip(dp_chain(cost_volume_rtl, np.flip(pairwise_costs), message_rtl), axis=1)

    # Direction: up-to-down
    message_utd = np.zeros((W, H, D))
    cost_volume_utd = np.swapaxes(cost_volume, 0, 1)
    message_utd = np.swapaxes(dp_chain(cost_volume_utd, np.swapaxes(pairwise_costs, 0, 1), message_utd), 0, 1)

    # Direction: up-to-down
    message_dtu = np.zeros((W, H, D))
    cost_volume_dtu = np.flip(np.swapaxes(cost_volume, 0, 1), axis=1)
    message_dtu = np.flip(np.swapaxes(dp_chain(cost_volume_dtu, np.flip(np.swapaxes(pairwise_costs, 0, 1), axis=1), message_dtu), 0, 1), axis=1)

    # Belief propagation and WTA
    cost_volume += message_ltr
    cost_volume += message_rtl
    cost_volume += message_utd
    cost_volume += message_dtu
    disparity_map = compute_wta(cost_volume)

    return disparity_map


def main():
    # Set parameters
    D = 64
    radius = 2

    # Load input images
    im0 = imread("data/cones_left.png")
    im1 = imread("data/cones_right.png")

    im0g = rgb2gray(im0)
    im1g = rgb2gray(im1)

    # Plot input images
    plt.figure(figsize=(8,4))
    plt.subplot(121), plt.imshow(im0g, cmap='gray'), plt.title('Left')
    plt.subplot(122), plt.imshow(im1g, cmap='gray'), plt.title('Right')
    plt.tight_layout()

    # Use either SAD, NCC or SSD to compute the cost volume
    print('Computing initial cost volume')
    cost_volume = compute_cost_volume_ncc(im0g, im1g, D, radius)

    # Compute pairwise costs
    H, W, D = cost_volume.shape
    f = get_pairwise_costs(H, W, D)

    # Compute SGM
    print('Computing aggregated cost volume')
    disp = compute_sgm(cost_volume, f)

    # Plot result
    plt.figure()
    plt.imshow(disp)
    plt.show()


if __name__== "__main__":
    main()
