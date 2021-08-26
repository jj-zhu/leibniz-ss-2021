'''
code by: j.zhu

'''
# %% imports

import numpy as np

# for kernel computation
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel, check_pairwise_arrays, pairwise_kernels

def rkhs_dist(x, wx, y, wy, kernel, **param):
    # compute gram matrix
    Kxx, Kxxp, Kxpxp = gram_mat(x, y, kernel=kernel, **param)

    # compute rkhs norm squared
    cost_rkhs = wy.T @ Kxpxp @ wy - 2 * (
            wx.T @ Kxxp @ wy) + wx.T @ Kxx @ wx

    return cost_rkhs # return squared norm


def mmd_sqr(x, y, kernel, is_biased=True, **param):
    # compute gram matrices
    # J. Zhu
    Kxx, Kxy, Kyy = gram_mat(x, y, kernel=kernel, **param)

    # coeff for MMD
    nx = x.shape[0]
    ny = y.shape[0]

    if is_biased: # whether to use the U or V estimator; cf Gretton12
        mmd_squared = np.sum(Kxx) / (nx * nx) + np.sum(Kyy) / (ny * ny) - 2 * np.sum(Kxy) / (nx * ny)
    else:
        mmd_squared = sum_off_diag(Kxx) / (nx * (nx - 1)) + sum_off_diag(Kyy) / (ny * (ny - 1)) - 2 * np.sum(Kxy) / (nx * ny)

    return mmd_squared # return norm squared

def sum_off_diag(K):
    return np.sum(K) - np.trace(K)

def gram_mat(x, y, kernel=None, **param):
    '''compute gram matrix'''
    if kernel is None:
        kernel = rbf_kernel

    if len(np.asarray(x).shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(np.asarray(y).shape) < 2:
        y = np.expand_dims(y, axis=1)

    # first
    K11 = kernel(x,x, **param)

    # second
    K22 = kernel(y,y, **param)

    # cross term
    K12 = kernel(x,y, **param)

    return K11, K12, K22

'''exponential kernel'''
def exp_kernel(X, Y=None, sigma=None):
    """Compute the exp kernel between X and Y.
    """
    X, Y = check_pairwise_arrays(X, Y)
    if sigma is None:
#         sigma = 1.0
        sigma = 1.0 / X.shape[1]

    K = sigma * pairwise_kernels(X, Y, metric='linear')
    np.exp(K, K)  # exponentiate K in-place
    return K

# %% run
if __name__ == '__main__':
    pass
