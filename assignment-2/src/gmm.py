import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.util import view_as_windows
from skimage.color import rgb2gray
from numpy import linalg as LA
import numba
import pickle
import collections
import glob

rng = np.random.RandomState(seed=42)

def compute_psnr(img1, img2):
    """
    :param img1:
    :param img2:
    :return: the PSNR between img1 and img2
    """
    mse = np.mean((img1 - img2)**2)
    return (10 * np.log10(1.0 / mse))


def reconstruct_average(P):
    """
    :param P: (MM,NN,W,W)
    :return: (M,N)
    """
    MM, NN, w, _ = P.shape
    M = MM + w - 1
    N = NN + w - 1
    p = np.zeros((M, N))
    c = np.zeros((M, N))
    for x in range(0, w):
        for y in range(0, w):
            p[y:MM + y, x:NN + x] += P[:, :, y, x]
            c[y:MM + y, x:NN + x] += 1
    p /= c
    return p


def wiener_filter(U, F, E, precisions, means, lamb, max_k):
    """
    Applies the wiener filter to N patches each having K pixels.
    The parameters of a learned GMM with C kernels are passed as an argument.

    :param U: (N,K) denoised patches from previous step
    :param F: (N,K) noisy patches
    :param E: (K,K) matrix that projects patches onto a set of zero-mean patches
    :param precisions: (C,K,K) precisions of the GMM
    :param means: (C,K) mean values of the GMM
    :param lamb: lambda parameter of the Wiener filter
    :param max_k: the most suitable kernel for the projected patch
    :return: (N,K) result of the wiener filter, equivalent to x_i^~ in Algorithm 1
    """
    N, K = F.shape
    A = np.linalg.inv(np.matmul(np.matmul(E.T, precisions[max_k]), E + lamb * np.eye(K)))
    b = np.matmul(E.T, np.matmul(precisions[max_k], means[max_k]))

    for i in range(0, N):
        U[i, :] = np.matmul(A, (b + lamb * F[i, :]))

    return U


def get_noisy_img(clean_img):
    """
    Adds noise on the given input image

    :param clean_img:
    :return:
    """
    assert(clean_img.min()>=0.0)
    assert(clean_img.max()<=1.0)
    assert(len(clean_img.shape)==2)

    sigma = 25.0 / 255.0
    noisy_img = clean_img + rng.randn(*clean_img.shape) * sigma

    return noisy_img


def get_e_matrix(K):
    """
    Returns a matrix that projects a patch onto the set of zero-mean patches

    :param K: total number of pixels in a patch
    :return: (K,K) projection matrix
    """
    e = np.ones((K, 1))
    E = np.eye(K) - e @ e.T / K
    return E


def train_gmm(X, C, max_iter, plot=False):
    """
    Trains a GMM with the EM algorithm
    :param X: (N,K) N image patches each having K pixels that are used for training the GMM
    :param C: Number of kernels in the GMM
    :param max_iter: maximum number of iterations
    :param plot: set to true to plot steps of the algorithm
    :return: alpha: (C) weight for each kernel
             mu: (C,K) mean for each kernel
             sigma: (C,K,K) covariance matrix of the learned model
    """
    # general setup
    N, K = X.shape
    alpha = np.zeros((C))
    mu = np.zeros((C, K))
    sigma = np.zeros((C, K, K))
    for k in range(C):
        alpha[k] = 1.0 / C
        sigma[k] = np.random.normal(0, 0.01, (K, K))
        sigma[k] = np.matmul(sigma[k].T, sigma[k])

    for j in range(max_iter):
        print('E-STEP iter ' + str(j))
        # E-step, compute gammas
        gamma = logsum_gamma(X, mu, sigma, alpha)

        # M-step, update params of gaussian k
        print('M-STEP iter ' + str(j))
        for k in range(C):
            # pre-compute some stuff for: alphas, mean
            gammas_sum = 0
            gammas_sum_scaled = np.zeros(K)
            for i in range(N):
                gammas_sum += gamma[k][i]
                gammas_sum_scaled += gamma[k][i] * X[i]
            # now update params: alphas, mean
            alpha[k] = gammas_sum / N
            mu[k] = gammas_sum_scaled / gammas_sum

            # pre-compute some stuff for: cov
            cov_numerator = np.zeros((K, K))
            for i in range(N):
                zero_mean_X = X[i] - mu[k]
                cov_numerator += (gamma[k][i] * np.matmul(zero_mean_X, zero_mean_X.T))
            # now update params: cov
            sigma[k] = cov_numerator / gammas_sum

            if plot:
                plot(mu[k], sigma[k], np.sqrt(K))

    return alpha, mu, sigma

def logsum_gamma(X, mu, sigma, alpha):
    """
    Computes the prob of all the patches belonging to one of the gaussian mixture classes using the log sum exp trick
    :param X: (N,K) N image patches each having K pixels that are used for training the GMM
    :param mu: (C,K) mean for each kernel
    :param sigma: (C,K,K) covariance matrix of the learned model
    :param alpha: (C) current weight of each kernel
    :return: gamma: (C,N) prob of multivar gaussian for each patch
    """
    N, K = X.shape
    C, K = mu.shape
    z = np.zeros(C)
    z_max = -np.inf
    c = np.zeros(C)
    log_c = np.zeros(C)
    gamma = np.zeros((C, N))

    for i in range(N):
        for k in range(C):
            # compute z
            zero_mean_X = X[i] - mu[k]
            sigma_inv = np.linalg.inv(sigma[k])
            z[k] = -0.5 * np.matmul(np.matmul(zero_mean_X.T, sigma_inv), zero_mean_X)
            if (z[k] > z_max):
                z_max = z[k]
            # compute c
            signs, logdet = LA.slogdet(sigma[k])
            log_c[k] = np.log(alpha[k]) - 0.5 * K * np.log(2 * np.pi) - 0.5 * logdet
            c[k] = np.exp(log_c[k])

        sum_exp = 0
        for k in range(C):
            # compute sums of exps
            sum_exp += (c[k] * np.exp(z[k] - z_max))

        for k in range(C):
            # compute a, b, and gamma
            a = log_c[k] + z[k]
            b = z_max + np.log(sum_exp)
            gamma[k][i] = np.exp(a - b)

    return gamma

def logsum_max_k(X_proj, mu, sigma, alpha):
    """
    Uses the log sum trick to find the maximum k given a projected patch and a GMM model
    :param X_proj: (K) projected patch
    :param mu: (C,K) mean for each kernel of the GMM
    :param sigma: (C,K,K) covariance matrix of the learned model
    :param alpha: (C) current weight of each kernel
    :return: gamma: (C,N) prob of multivar gaussian for each patch
    """
    N, K = X_proj.shape
    C, K = mu.shape
    z = np.zeros(C)
    log_c = np.zeros(C)

    max_k = -1
    max_a = -np.inf

    for i in range(K):
        for k in range(C):
            # compute z
            zero_mean_X = X_proj[i] - mu[k]
            sigma_inv = np.linalg.inv(sigma[k])
            z[k] = -0.5 * np.matmul(np.matmul(zero_mean_X.T, sigma_inv), zero_mean_X)

            # compute log_c efficiently
            signs, logdet = LA.slogdet(sigma[k])
            log_c[k] = np.log(alpha[k]) - 0.5 * K * np.log(2 * np.pi) - 0.5 * logdet

        for k in range(C):
            # compute a
            a = log_c[k] + z[k]
            if a > max_a:
                max_a = a
                max_k = k

    return max_k

def load_imgs(dir):
    files = glob.glob('{}/*.png'.format(dir))
    imgs = [ski.img_as_float(ski.io.imread(fname)) for fname in files]
    return imgs


def make_dictionary(d):
    N,M,M = d.shape
    NH = np.int32(np.sqrt(N))
    dict = np.zeros((NH*M, NH*M))
    ii = 0
    idx = 0
    for i in range(0,NH):
        jj = 0
        for j in range(0,NH):
            dd = np.copy(d[idx,:,:])
            dd -= dd.min()
            dd /= dd.max()
            dict[ii:ii+M, jj:jj+M] = dd
            jj += M
            idx += 1
        ii +=M
    return dict


def plot(mu, precisions, w):
    plt.figure(2, figsize=(5,5))
    plt.subplot(121)
    plt.imshow(mu.reshape(w,w), cmap="gray")
    plt.subplot(122)

    eigval, eigvec = LA.eig(precisions)
    filters = np.zeros((w*w, w, w))
    for i in range(0,w*w):
        filters[i,:,:] = eigvec[:,i].reshape(w,w)
    dict = make_dictionary(filters)
    plt.imshow(dict, cmap="gray")
    plt.show()


def denoise():
    C = 2  # Number of mixture components
    W = 5  # Window size
    K = W**2  # Number of pixels in each patch

    train_imgs = load_imgs("train_set")
    val_imgs = load_imgs("valid_set")
    test_imgs = np.load("test_set.npy", allow_pickle=True).item()

    # train
    X = np.zeros((0, K))
    for img in train_imgs:
        train_img = img/255
        X = np.vstack((X, view_as_windows(train_img, (W, W), 1).reshape(-1, K)))
    # remove means
    mean_X = np.mean(X, axis=1)
    X = X - mean_X[:, np.newaxis]

    gmm = {}
    gmm['alpha'], gmm['mu'], gmm['sigma'] = train_gmm(X, C=C, max_iter=3)
    gmm['precisions'] = np.linalg.inv(gmm['sigma'] + np.eye(K) * 1e-6)
    plot(gmm['mu'][0], gmm['precisions'][0], W)

    # validate
    F = np.zeros((0, K))
    clean_img = val_imgs[0] / 255
    noisy_img = get_noisy_img(clean_img)
    F = np.vstack((F, view_as_windows(noisy_img, (W, W), 1).reshape(-1, K)))
    MM, NN, _, _ = F.shape

    lamb = 100.0
    alpha = 0.5
    maxiter = 5
    E = get_e_matrix(K)

    # Use Algorithm 1 for Patch-Denoising
    U = F.copy()  # Initialize with the noisy image patches
    for iter in range(0, maxiter):
        X_proj = np.matmul(E, U)
        max_k = logsum_max_k(X_proj, gmm['mu'], gmm['sigma'], gmm['alpha'])
        U = alpha * U + (1 - alpha) * wiener_filter(U, F, E, gmm['precisions'], gmm['mu'], lamb, max_k)
        u = reconstruct_average(U.reshape(MM, NN, W, W))
        psnr_denoised = compute_psnr(u, clean_img)
        print("Iter: {} - PSNR: {}".format(iter, psnr_denoised))

    psnr_noisy = compute_psnr(noisy_img, clean_img)
    psnr_denoised = compute_psnr(u, clean_img)

    print("PSNR noisy: {} - PSNR denoised: {}".format(psnr_noisy, psnr_denoised))


if __name__ == "__main__":
    denoise()
