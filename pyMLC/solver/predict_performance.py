import os, sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)

import numpy as np


from utl.utl import update_mixture_density, predict_response

def forecast_spell_performance(response_dict, learning_curve_matrix, prior_mixture_density=None):
    #TODO: this may not be the right interpretation
    '''
    # Input:
    (1) response_dict: {t,Y} where t is the practice time, 1...T, Y is the response, 0/1
    (2) learning_curve_matrix: learning curves, T*J
    (3) prior_mixture_density: the prior guess of the user type, J*1, sum to 1
    '''
    
    T, J = learning_curve_matrix.shape

    if prior_mixture_density is None:
        mixture_density = np.ones((J, 1))/J
    else:
        mixture_density = prior_mixture_density

    Yhats = []
    Ys = [response_dict[x] for x in range(1, T+1)]
    for t, Y in response_dict.items():
        Yhat = predict_response(learning_curve_matrix, mixture_density, t)
        Yhats.append(Yhat)

        Yt = {0:dict(zip(range(1, t+1), Ys[:t]))}  # make shift
        mixture_density = update_mixture_density(Yt, learning_curve_matrix, mixture_density)

    return Yhats


def get_predict_performance(response_dicts, learning_curve_matrix, max_T, prior_mixture_density=None):

    forecast_tabs = np.zeros((max_T, 2))

    for uid, response_dict in response_dicts.items():
        item_T = max(response_dict.keys())
        yHats = forecast_spell_performance(response_dict, learning_curve_matrix, prior_mixture_density)
        ys = [response_dict[x] for x in sorted(list(response_dict.keys()))]
        for t in range(item_T):
            forecast_tabs[t,1] += 1
            forecast_tabs[t,0] += float(int(yHats[t]>0.5) == ys[t])

    return forecast_tabs[:,0]/forecast_tabs[:,1]







    
