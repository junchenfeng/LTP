# encoding:utf-8
'''
This file demos the dgp and model fit for Discrete Item Response Model(DIRT) 
'''
import numpy as np
# meta parameters
N = 500
# Mx = 2, My = 2, J=2
state_init_dist = np.array([0.3, 0.7])

observ_matrix = np.array([
    [[0.8,0.2],[0.2,0.8]],
    [[0.6,0.4],[0.3,0.7]],
    [[0.45,0.55],[0.2,0.8]],
    [[0.4,0.6],[0.1,0.9]],
    [[0.25,0.75],[0.05,0.95]]
    ])
J = observ_matrix.shape[0]
data = []
for i in range(N):  
    S = int( np.random.binomial(1, state_init_dist[1]) )
    for j in range(J):
        y = int( np.random.binomial(1, observ_matrix[j, S, 1]) )
        data.append(('stu_%d'%i, 'item_%d'%j, y))

# estimate
from LTP import DIRT_MCMC

mcmc_instance = DIRT_MCMC()
mcmc_instance.estimate(data, max_iter=1000)
item_param = mcmc_instance.get_item_param()
learner_param = mcmc_instance.get_learner_param()


for j in range(J):
    print(observ_matrix[j,:,1])
    print(item_param['item_%d'%j]['point'])
    print(item_param['item_%d'%j]['ci'])

print(state_init_dist[0])
print(learner_param['point'])
print(learner_param['ci'])

