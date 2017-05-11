# encoding:utf-8
'''
This file demos the dgp and model fit for Discrete Item Response Model(DIRT) 
'''
import numpy as np
from LTP import DIRT_MCMC

# meta parameters
N = 1000
T = 5


# Mx = 2, My = 2, J=1 (BKT)
state_init_dist = np.array([0.6, 0.4])
observ_matrix = np.array([[0.8,0.2],[0.2,0.8]])

data = []
j = 0
for i in range(N):	
	S = int( np.random.binomial(1, state_init_dist[1]) )
	for t in range(T):
		y = int( np.random.binomial(1, observ_matrix[S, 1]) )
		data.append((i, t, j, y))

mcmc_instance = DIRT_MCMC()
est_param = mcmc_instance.estimate(data, chain_num=1, max_iter = 500)

print(observ_matrix[:,1])
print(est_param['c'])
print(state_init_dist[0])
print(est_param['pi'])

