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
    I = scipy.sparse.identity(M * N)
    nabla, nabla_x, nabla_y = _make_nabla(M, N)

    K = sp.bmat([[nabla_x, -I, None],
                 [nabla_y, None, -I],
                 [None, nabla_x, None],
                 [None, nabla_y, None],
                 [None, None, nabla_x],
                 [None, None, nabla_y]])
    return K


def proj_ball(Y, lamb):
    """
    Projection to a ball as described in Equation (6)
    @param Y: either 2xMN or 4xMN
    @param lamb: scalar hyperparameter lambda
    @return: projection result either 2xMN or 4xMN
    """
    S, MN = Y.shape
    ones = np.ones(MN)
    norm = np.sqrt(np.sum(Y**2, axis=0))
    max_proj = np.maximum(ones,  (1/lamb) * norm)
    return Y / max_proj


def compute_accX(x, y, X=1):
    """
    Computation of the accuracy as described in eq (9)
    @param x: fused disparity map of size MxN
    @param y: ground-truth disparity map MxN
    @param X: scalar threshold used for accuracy
    @return: accuracy scalar
    """
    # compute difference with gt
    difference_mask = np.abs(x - y)

    # count the number of valid elements
    valids = np.where(difference_mask < X)[0]
    accurate_pixels = len(valids) - 1

    # compute total number of pixels
    M, N = x.shape
    total_pixels = M * N

    return accurate_pixels / total_pixels


def tgv2_pd(f, alpha, maxit):
    """
    @param f: the K observations of shape MxNxK
    @param alpha: tuple containing alpha1 and alpha2
    @param maxit: maximum number of iterations
    @return: tuple of u with shape MxN, v with shape 2xMxN, and an array of size maxit with the energy of each step
    """
    print('Precomputing stuff')
    M, N, K = f.shape
    f_flattened = f.reshape(M * N, K)

    # make operators
    print('Computing K')
    k = make_K(M, N)
    alpha1, alpha2 = alpha

    # used for calculation of the dataterm projection
    print('Computing dataterm projection stuff')
    W = np.ones((M, N, K))
    Wis = np.asarray([compute_Wi(W, i) for i in range(K)])
    Wis = Wis.transpose(1, 2, 0)
    Wis = Wis.reshape(M * N, K)

    # initialize primal and dual variables to zero
    u = np.zeros((M * N))
    v = np.zeros((2 * M * N))
    p = np.zeros((2 * M * N))
    q = np.zeros((4 * M * N))
    energy = np.zeros((maxit))

    # other algorithm parameters
    L = np.sqrt(12)
    tau = 0.1
    sigma = 1 / tau / (L ** 2)

    # compute equation (4)
    for it in range(0, maxit):
        print('Executing iteration ' + str(it))
        # make sure to concatenate flattened arrays beforehand so that sizes are correct
        u_v = np.concatenate((u, v))
        p_q = np.concatenate((p, q))

        # half step update for u and v
        u_v_half = u_v - tau * (k.T @ p_q)
        u_half = u_v_half[0: len(u)]
        v_half = u_v_half[len(u): len(u) + len(v)]

        # next step for u and v
        u_next = prox_sum_l1(u_half, f_flattened, tau, Wis)
        v_next = v_half
        u_v_next = np.concatenate((u_next, v_next))

        # half step update for p and q
        p_q_half = p_q + sigma * (k @ (2 * u_v_next - u_v))
        p_half = p_q_half[0: len(p)]
        q_half = p_q_half[len(p): len(p) + len(q)]

        # next step for p and q, make sure to reshape before and afterwards so that its treated in a pixelwise manner
        p_half_reshaped = np.reshape(p_half, (2, M * N))
        q_half_reshaped = np.reshape(q_half, (4, M * N))
        p_next = np.reshape(proj_ball(p_half_reshaped, alpha1), (2 * M * N))
        q_next = np.reshape(proj_ball(q_half_reshaped, alpha2), (4 * M * N))

        # update values for the next iteration
        u = u_next
        v = v_next
        p = p_next
        q = q_next

        # reshape to original matrix form
        U = np.reshape(u, (M, N))
        V = np.reshape(v, (2, M, N))

        # compute the energy of this iteration. eq (3)
        energy[it] = compute_energy(u, v, f, alpha)

    return U, V, energy


def compute_energy(u, v, f, alpha):
    """
    @param u: MN flattened vector
    @param v: tuple containing MN, MN flattened gradient vectors
    @param f: the K observations of shape MxNxK
    @param alpha: tuple containing alpha1 and alpha2
    @return: the energy value for the iteration
    """
    alpha1, alpha2 = alpha
    M, N, K = f.shape

    # compute the regularization term
    nabla, nabla_x, nabla_y = _make_nabla(M, N)
    stacked_nablas = sp.bmat([[nabla, None], [None, nabla]], format='csr')
    diag_nabla = sp.diags(sp.csr_matrix.diagonal(stacked_nablas), shape=(4 * M * N, 2 * M * N))
    tgv_1 = nabla @ u - v
    tgv_2 = diag_nabla @ v

    # reshape to matrix and compute 21 norm
    tgv_1 = np.reshape(tgv_1, (2, M, N))
    tgv_2 = np.reshape(tgv_2, (4, M, N))
    norm21_tgv_1 = np.sum(np.abs(np.sqrt(np.sum(tgv_1 ** 2, axis=1))))
    norm21_tgv_2 = np.sum(np.abs(np.sqrt(np.sum(tgv_2 ** 2, axis=1))))

    # compute the data term -> couldn't broadcast it properly so the code looks disgusting and inefficient, SORRY :(
    U = np.reshape(u, (M, N))
    norm1_data_diff = 0
    for k in range(K):
        data_diff = f[:, :, k] - U
        norm1_data_diff += np.sum(np.abs(data_diff))

    # now compute the energy
    energy = alpha1 * norm21_tgv_1 + alpha2 * norm21_tgv_2 + norm1_data_diff

    return energy


# load Observations
samples = np.array([np.load('data/observation{}.npy'.format(i)) for i in range(0,9)])
f = samples.transpose(1,2,0)

# perform TGV-Fusion
U, V, energy = tgv2_pd(f, alpha=(0.8, 0.3), maxit=300)

# plot fusion
plt.imshow(U)
plt.suptitle('Fused image')
plt.colorbar()
plt.savefig('fused.png')
plt.close()

# plot energy
plt.plot(energy)
plt.suptitle('Energy over time')
plt.ylabel('Energy')
plt.xlabel('Iteration')
plt.savefig('energy.png')
plt.close()

# calculate accuracy
gt = np.load('data/gt.npy')
accuracy = compute_accX(U, gt)
print('Accuracy: ' + str(accuracy))

