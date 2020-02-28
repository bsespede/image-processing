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
    I = np.eye(M * N, M * N)
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
    max_proj =  np.maximum(1,  (1/lamb) * np.sqrt(np.sum(Y**2, axis = 0)))
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
    f = f.reshape(M * N, K)

    # make operators
    K = make_K(M, N)
    alpha1, alpha2 = alpha

    # used for calculation of the dataterm projection
    W = np.ones((M, N, K))
    Wis = np.asarray([compute_Wi(W, i) for i in range(K)])
    Wis = Wis.transpose(1, 2, 0)
    Wis = Wis.reshape(M * N, K)

    # initialize primal and dual variables to zero
    u = np.zeros((M * N))
    v = np.zeros((2 * M * N))
    p = np.zeros((2 * M * N))
    q = np.zeros((4 * M * N))
    energy = np.zeros((1, maxit))

    # other algorithm parameters
    L = np.sqrt(12)
    tau = 0.1
    sigma = 1 / tau / (L ** 2)

    # compute equation (4)
    for it in range(0, maxit):
        print('Executing iteration ' + str(it))
        # make sure to copy previous results
        u_prev = np.copy(u)
        v_prev = np.copy(v)
        p_prev = np.copy(p)
        q_prev = np.copy(q)

        # half step update for u and v
        u_half = u_prev - tau * (K.T @ p_prev)
        v_half = v_prev - tau * (K.T @ q_prev)

        # next step for u and v
        u_next = prox_sum_l1(u_half, f, tau, Wis)
        v_next = v_half

        # half step update for p and q
        p_half = p_prev + sigma * (K @ (2 * u_next - u_prev))
        q_half = q_prev + sigma * (K @ (2 * v_next - v_prev))

        # next step for p and q
        p_next = proj_ball(p_half, alpha1)
        q_next = proj_ball(q_half, alpha2)

        # update values for the next iteration
        u = u_next
        v = v_next
        p = p_next
        q = q_next

        # reshape to original matrix form
        U = np.reshape(u, (M, N))
        V = np.reshape(v, (2, M, N))

        # compute the energy of this iteration. eq (3) TODO
        energy[0, it] = it

    return U, V, energy


# Load Observations
samples = np.array([np.load('data/observation{}.npy'.format(i)) for i in range(0,9)])
f = samples.transpose(1,2,0)

# Perform TGV-Fusion
U, V, energy = tgv2_pd(f, alpha=(0.8, 0.3), maxit=300)  # TODO: set appropriate parameters

# Plot fusion
plt.imshow(U)
plt.suptitle('Fused image')
plt.colorbar()
plt.savefig('fused.png')

# Plot energy
plt.plot(energy)
plt.suptitle('Energy over time')
plt.ylabel('Energy')
plt.xlabel('Iteration')
plt.savefig('energy.png')

# Calculate Accuracy
gt = np.load('data/gt.npy')
accuracy = compute_accX(res, gt)
print('Accuracy: ' + str(accuracy))

