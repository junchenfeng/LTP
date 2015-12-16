import math
import numpy as np

def B(v, q):
    if v == 0:
        return 1-q
    else:
        return q
    
def L_assembly(response_dict, learning_curve, p):
    '''
    # Input:
    (1) response_dict:{t:Y} where t is the number of practice opportunity and Y 0/1 is the binary responses
    (2) learning curve: T*1 array where t element is the prob(Y=0) at t th practice opportunity.
    (3) p is the mixture density for this curve
    '''
    # TODO:standardize the time code.
    if response_dict.keys()[0] != 1:
        raise ValueError('Practice time start code from 1.')
    l = math.log(p)
    for j, v_j in response_dict.iteritems():
        l += math.log(B(v_j, learning_curve[j-1]))  # j-1 because the times are coded from 1-N
    return l


def Z_assembly(response_dict, learning_curve_matrix, mixture_density):
    '''
    # Input:
    (1) mixture density: K*1 array, where K is the number of learning curves
    (2) learning curve matrix: T*j array, where T is the largest practice opportunity
    (3) response_dict: {t:Y} where t is the number of practice opportunity and Y 0/1 is the binary responses

    all inputs are numpy array
    '''
    J = mixture_density.shape[0]
    T, J1 = learning_curve_matrix.shape
    if J != J1:
        raise ValueError('Mixture density and learning curve matrix have different lengths.')

    ls = np.zeros((J, 1))
    for j in range(J):
        ls[j] = np.exp(L_assembly(response_dict, learning_curve_matrix[:,j], mixture_density[j]))

    z = ls/ls.sum()

    return z

  



def update_mixture_density(response_dict, learning_curve_matrix, mixture_density):
    '''
    # Input:
    (1) mixture density: K*1 array, where K is the number of learning curves
    (2) learning curve matrix: T*K array, where T is the largest practice opportunity
    (3) response_dict: {t:Y} where t is the number of practice opportunity and Y 0/1 is the binary responses

    all inputs are numpy array
    '''

    J = mixture_density.shape[0]
    T, J1 = learning_curve_matrix.shape
    if J != J1:
        raise ValueError('Mixture density and learning curve matrix have different lengths.')

    


