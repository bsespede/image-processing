import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp


def _make_nabla(M, N):
    row = np.arange(0, M * N)
    dat = np.ones(M * N)
    col = np.arange(0, M * N).reshape(M, N)
    col_xp = np.hstack([col[:, 1:], col[:, -1:]])
    col_yp = np.vstack([col[1:, :], col[-1:, :]])

    nabla_x = scipy.sparse.coo_matrix((dat, (row, col_xp.flatten())), shape=(M * N, M * N)) - \
              scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M * N, M * N))

    nabla_y = scipy.sparse.coo_matrix((dat, (row, col_yp.flatten())), shape=(M * N, M * N)) - \
              scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M * N, M * N))

    nabla = scipy.sparse.vstack([nabla_x, nabla_y])

    return nabla, nabla_x, nabla_y


def compute_Wi(W, i):
    """
    Used for calculation of the dataterm projection

    can be used for confidences or set to zero if datapoint is not available
    @param W:
    @param i: index of the observation
    @return:
    """
    Wi = -np.sum(W[:, :, :i], axis=-1) + np.sum(W[:, :, i:], axis=-1)
    return Wi


def prox_sum_l1(u, f, tau, Wis):
    """
    Used for calculation of the dataterm projection

    compute pi with pi = \bar x + tau * W_i
    @param u: MN
    @param tau: scalar
    @param Wis: MN x K
    @param f: MN x K
    """
    pis = u[..., np.newaxis] + tau * Wis

    var = np.concatenate((f, pis), axis=-1)

    prox = np.median(var, axis=-1)

    return prox


def make_K(M, N):
    """
    @param M:
    @param N:
    @return: the K operator as described in Equation (5)
    """
    # TODO
    return


def proj_ball(Y, lamb):
    """
    Projection to a ball as described in Equation (6)
    @param Y: either 2xMN or 4xMN
    @param lamb: scalar hyperparameter lambda
    @return: projection result either 2xMN or 4xMN
    """
    # TODO
    pass


def compute_accX(x, y, X=1):
    # TODO
    pass


def tgv2_pd(f, alpha, maxit):
    """
    @param f: the K observations of shape MxNxK
    @param alpha: tuple containing alpha1 and alpha2
    @param maxit: maximum number of iterations
    @return: tuple of u with shape MxN and v with shape 2xMxN
    """
    M, N, K = f.shape
    f = f.reshape(M*N, K)

    # make operators
    k = make_K(M,N)

    # Used for calculation of the dataterm projection
    W = np.ones((M, N, K))
    Wis = np.asarray([compute_Wi(W, i) for i in range(K)])
    Wis = Wis.transpose(1, 2, 0)
    Wis = Wis.reshape(M * N, K)

    # Lipschitz constant of K
    L = np.sqrt(12)

    # initialize primal variables
    # TODO

    # initialize dual variables
    # TODO

    # primal and dual step size
    tau = 0.0  # TODO
    sigma = 0.0  # TODO

    for it in range(0, maxit):
        # TODO calculate iterates as described in Equation (4)
        # To calculate the data term projection you can use:
        # prox_sum_l1(x, f, tau, Wis)
        # where x is the parameter of the projection function i.e. u^(n+(1/2))

    return


# Load Observations
samples = np.array([np.load('observation{}.npy'.format(i)) for i in range(0,9)])
f = samples.transpose(1,2,0)

# Perform TGV-Fusion
res, v = tgv2_pd(f, alpha=(0.0, 0.0), maxit=0)  # TODO: set appropriate parameters

# Plot result
# TODO

# Calculate Accuracy
# TODO