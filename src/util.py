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


def medium_heuristic(X, Y):
    '''
    the famous kernel medium heuristic
    :param X:
    :param Y:
    :return:
    '''
    distsqr = euclidean_distances(X, Y, squared=True)
    kernel_width = np.sqrt(0.5 * np.median(distsqr))

    '''in sklearn, kernel is done by K(x, y) = exp(-gamma ||x-y||^2)'''
    kernel_gamma = 1.0 / (2 * kernel_width ** 2)

    return kernel_width, kernel_gamma


def sum_of_kernel(X, Y=None, gamma=None, scale_sum=[0.01, 0.1, 1.0, 10, 100]):
    '''sum of kernels, same as the casadi version, but use sklearn/numpy'''

    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)

    # make sum of kernels
    sok = 0 # sum of kernels
    for s in scale_sum:
        Kk = - s * gamma * K # scaled gram mat
        sok += np.exp(Kk, Kk) # exponentiate K in-place
    return sok


def matDecomp(K):
    # import scipy
    # decompose matrix
    try:
        L = np.linalg.cholesky(K)
    except:
        print('warning, K is singular')
        d, v = np.linalg.eigh(K) #L == U*diag(d)*U'. the scipy function forces real eigs
        d[np.where(d < 0)] = 0 # get rid of small eigs
        L = v @ np.diag(np.sqrt(d))
    return L

# %% run
if __name__ == '__main__':
    pass
