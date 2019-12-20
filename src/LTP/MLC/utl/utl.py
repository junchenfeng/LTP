import math
import numpy as np


def B(v, q):
    if v == 0:
        return 1 - q
    else:
        return q


def L_assembly(response_list, learning_curve, p):
    """
    # Input:
    (1) [Y1,Y2, ..., Yt] where t is the number of practice opportunity and Yt 0/1
    (2) learning curve: T*1 array where t element is the prob(Y=0) at t th practice opportunity.
    (3) p is the mixture density for this curve
    """
    l = math.log(p)
    for j in range(len(response_list)):
        l += math.log(B(response_list[j], learning_curve[j]))
    return l


# def Z_assembly(response_list, learning_curve_matrix, mixture_density):
# '''
# Input:
# (1) [Y1,Y2, ..., Yt] where t is the number of practice opportunity and Yt 0/1
# (2) learning curve matrix: T*j array, where T is the largest practice opportunity
# (3) response_list: {t:Y} where t is the number of practice opportunity and Y 0/1 is the binary responses

# all inputs are numpy array

# Output
# level z
# '''
# J = mixture_density.shape[0]
# T, J1 = learning_curve_matrix.shape
# if J != J1:
# raise ValueError('Mixture density and learning curve matrix have different lengths.')

# ls = np.zeros(J)
# for j in range(J):
# ls[j] = np.exp(L_assembly(response_list, learning_curve_matrix[:, j], mixture_density[j]))

# z = ls/ls.sum()

# return z


def Z_assembly(response_list, learning_curve_matrix, mixture_density):
    M = len(response_list)
    N = len(mixture_density)

    respM = np.array(response_list).reshape((M, 1)).repeat(N, axis=1)
    lc = learning_curve_matrix[:M, :]

    respM_c = 1 - respM
    lc_c = 1 - lc

    logPosterior = np.sum(
        np.log(np.multiply(respM, lc) + np.multiply(respM_c, lc_c)), axis=0
    ) + np.log(mixture_density)
    posterior = np.exp(logPosterior)
    return posterior / posterior.sum()


def update_mixture_density(response_lists, learning_curve_matrix, mixture_density):
    """
    # Input:
    (1) response_lists: [[Y1,Y2,...,Yt],[]] where t is the number of practice opportunity and Yt 0/1 is the binary responses
    (2) learning curve matrix: T*K array, where T is the largest practice opportunity
    (3) mixture density: J*1 array, where J is the number of learning curves
    all inputs are numpy array
    """
    # input checks are done at the Z_assembly level
    N = len(response_lists)
    J = mixture_density.shape[0]
    z = np.zeros((J, N), order="F")
    for i in range(N):
        z[:, i] = Z_assembly(response_lists[i], learning_curve_matrix, mixture_density)

    return z.sum(axis=1) / z.sum()


def predict_response(learning_curve_matrix, mixture_density, t):
    """
    # Input:
    (1) learning_curve_matrix, T*K
    (2) mixture_density, K*1
    (3) t, the time of practice, from 1..T
    """
    if t > learning_curve_matrix.shape[0] - 1:
        raise ValueError("Exceeds the model specification.")
    return np.dot(learning_curve_matrix[t, :], mixture_density)


def predict_delta_response(learning_curve_matrix, mixture_density, t):
    if t > learning_curve_matrix.shape[0]:
        raise ValueError("Exceeds the model specification.")
    delta_learning_curve = learning_curve_matrix[t, :] - learning_curve_matrix[t - 1, :]
    return np.dot(delta_learning_curve, mixture_density)


def list2array(lst, M, N):
    result = np.zeros((N, M), dtype=np.int8, order="F")
    mask = np.copy(result, order="F")
    for j, col in enumerate(lst):
        for i, val in enumerate(col):
            result[i][j] = val
            mask[i][j] = 1
    return result, mask
