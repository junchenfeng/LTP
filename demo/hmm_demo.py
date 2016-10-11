# encoding:utf-8
import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)

from HMM.mcmc import LTP_HMM_MCMC
import matplotlib.pyplot as plt

import ipdb

# meta parameters
N = 2000
T = 5
'''
# Mx = 2, My = 2(BKT)
state_init_dist = np.array([0.6, 0.4])
state_transit_matrix = np.array([[0.6, 0.4],[0, 1]])
observ_matrix = np.array([[0.8,0.2],[0.5,0.95]])

data = []
j = 0
for i in range(N):	
	for t in range(T):
		if t ==0:
			S = int( np.random.binomial(1, state_init_dist[1]) )
		else:
			S = int( np.random.binomial(1, state_transit_matrix[S, 1]) )
		y = int( np.random.binomial(1, observ_matrix[S, 1]) )
	
		data.append((i, t, j, y))

mcmc_instance = LTP_HMM_MCMC()
est_param = mcmc_instance.estimate(data, max_iter = 500)
print(est_param)
'''
# Mx = 3, My = 3

state_init_dist = np.array([0.25, 0.5, 0.25])
state_transit_matrix = np.array([[0.8,0.2,0],[0, 0.6, 0.4],[0, 0, 1]])
observ_matrix = np.array([[0.7,0.3,0.0],[0.2, 0.6, 0.2],[0.0, 0.1, 0.9]])

data = []
j = 0
for i in range(N):
	end_of_spell = 0
	is_observ = 1
	
	for t in range(T):
		if t ==0:
			S = int( np.random.choice(3, 1, p=state_init_dist) )
		else:
			S = int( np.random.choice(3, 1, p=state_transit_matrix[S, :]) )
		y = int( np.random.choice(3, 1, p=observ_matrix[S,:]) )
					
	
		data.append((i, t, j, y))	

mcmc_instance = LTP_HMM_MCMC()
est_param = mcmc_instance.estimate(data, max_iter = 500)
print(est_param)