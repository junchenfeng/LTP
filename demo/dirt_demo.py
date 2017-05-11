# encoding:utf-8
'''
This file demos the dgp and model fit for Discrete Item Response Model(DIRT) 
'''
import numpy as np
# meta parameters
N = 1000
T = 2
# Mx = 2, My = 2, J=2
state_init_dist = np.array([0.6, 0.4])
observ_matrix = np.array([
    [[0.8,0.2],[0.2,0.8]],
    [[0.5,0.5],[0.1,0.9]]
    ])

data = []
for i in range(N):  
    S = int( np.random.binomial(1, state_init_dist[1]) )
    for t in range(T):
        j = t
        y = int( np.random.binomial(1, observ_matrix[j, S, 1]) )
        data.append(('stu_%d'%i, 'item_%d'%j, y))

import ipdb; ipdb.set_trace() # BREAKPOINT
# estimate
#from LTP import DIRT_MCMC

import sys, os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg_dir = proj_dir + '/src/LTP'
sys.path.insert(0, pkg_dir)

from HMM.dirt import DIRT_MCMC

mcmc_instance = DIRT_MCMC()
mcmc_instance.estimate(data, chain_num=1, max_iter = 2000)
item_param = mcmc_instance.get_item_param()
learner_param = mcmc_instance.get_learner_param()

print(observ_matrix[0,:,1])
print(item_param['item_0']['point'])
print(item_param['item_0']['ci'])
print(observ_matrix[1,:,1])
print(item_param['item_1']['point'])
print(item_param['item_1']['ci'])


print(state_init_dist[1])
print(learner_param['point'])
print(learner_param['ci'])

