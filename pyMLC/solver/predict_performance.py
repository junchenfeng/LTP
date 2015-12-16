import os, sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)

import numpy as np


from utl.utl import update_mixture_density, predict_response

def forecast_spell_performance(response_list, learning_curve_matrix, prior_mixture_density=None):
    #TODO: this may not be the right interpretation
    '''
    # Input:
    (1) response_list: [Y1,Y2,...,Yt]
    (2) learning_curve_matrix: learning curves, T*J
    (3) prior_mixture_density: the prior guess of the user type, J*1, sum to 1
    '''
    
    T, J = learning_curve_matrix.shape

    if prior_mixture_density is None:
        mixture_density = np.ones((J, 1))/J
    else:
        mixture_density = prior_mixture_density

    Yhats = []
    user_T = len(response_list)
    for t in range(user_T):
        Yhat = predict_response(learning_curve_matrix, mixture_density, t)
        Yhats.append(Yhat)
        if t < user_T:
            # wrap in a list
            mixture_density = update_mixture_density([response_list[:(t+1)]],
                                                    learning_curve_matrix,
                                                    mixture_density)

    return Yhats


def get_predict_performance(response_lists, learning_curve_matrix, prior_mixture_density=None):

    max_T = max([len(x) for x in response_lists])
    forecast_tabs = np.zeros((max_T, 2))

    for response_list in response_lists:
        item_T = len(response_list)
        yHats = forecast_spell_performance(response_list, learning_curve_matrix, prior_mixture_density)
        for t in range(item_T):
            forecast_tabs[t,1] += 1
            forecast_tabs[t,0] += float(int(yHats[t]>0.5) == response_list[t])

    return forecast_tabs[:,0]/forecast_tabs[:,1]







    
