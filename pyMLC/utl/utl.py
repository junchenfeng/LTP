import math
import numpy as np

def B(v, q):
    if v == 0:
        return 1-q
    else:
        return q

def L_assembly(response_list, learning_curve, p):
    '''
    # Input:
    (1) [Y1,Y2, ..., Yt] where t is the number of practice opportunity and Yt 0/1
    (2) learning curve: T*1 array where t element is the prob(Y=0) at t th practice opportunity.
    (3) p is the mixture density for this curve
    '''
    l = math.log(p)
    for j in range(len(response_list)):
        l += math.log(B(response_list[j], learning_curve[j]))  
    return l


def Z_assembly(response_list, learning_curve_matrix, mixture_density):
    '''
    # Input:
    (1) [Y1,Y2, ..., Yt] where t is the number of practice opportunity and Yt 0/1
    (2) learning curve matrix: T*j array, where T is the largest practice opportunity
    (3) response_list: {t:Y} where t is the number of practice opportunity and Y 0/1 is the binary responses

    all inputs are numpy array

    # Output
    level z
    '''
    J = mixture_density.shape[0]
    T, J1 = learning_curve_matrix.shape
    if J != J1:
        raise ValueError('Mixture density and learning curve matrix have different lengths.')

    ls = np.zeros((J, 1))
    for j in range(J):
        ls[j] = np.exp(L_assembly(response_list, learning_curve_matrix[:, j], mixture_density[j]))

    z = ls/ls.sum()

    return z



def update_mixture_density(response_lists, learning_curve_matrix, mixture_density):
    '''
    # Input:
    (1) response_lists: [[Y1,Y2,...,Yt],[]] where t is the number of practice opportunity and Yt 0/1 is the binary responses
    (2) learning curve matrix: T*K array, where T is the largest practice opportunity
    (3) mixture density: K*1 array, where K is the number of learning curves
    all inputs are numpy array
    '''
    # input checks are done at the Z_assembly level
    N = len(response_lists)
    J = mixture_density.shape[0]
    z = np.zeros((J, N))
    for i in range(N):
        z[:,i] = Z_assembly(response_lists[i], learning_curve_matrix, mixture_density).reshape(J)

    return z.sum(axis=1)/z.sum()



def predict_response(learning_curve_matrix, mixture_density, t):
    '''
    # Input:
    (1) learning_curve_matrix, T*K
    (2) mixture_density, K*1
    (3) t, the time of practice, from 1..T
    '''
    if t > learning_curve_matrix.shape[0]+1:
        raise ValueError('Exceeds the model specification.')
    return np.dot(learning_curve_matrix[t-1,:].T, mixture_density)


def predict_delta_response(learning_curve_matrix, mixture_density, t):
    if t > learning_curve_matrix.shape[0]:
        raise ValueError('Exceeds the model specification.')
    delta_learning_curve = learning_curve_matrix[t,:]-learning_curve_matrix[t-1,:]
    return np.dot(delta_learning_curve.T, mixture_density)




