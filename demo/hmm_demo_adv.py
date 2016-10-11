# encoding:utf-8
import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from HMM.mcmc import LTP_HMM_MCMC
import matplotlib.pyplot as plt
import ipdb

'''
This file demos the dgp and model fit for general LTP model that accounts for effort and attrition
'''

# meta parameters
N = 2000
T = 5
mcmc_instance = LTP_HMM_MCMC()



'''
# Mx = 2, My = 2, J=1, Exit
state_init_dist = np.array([0.6, 0.4])
state_transit_matrix = np.array([[0.6, 0.4],[0, 1]])
observ_matrix = np.array([[0.8,0.2],[0.2,0.8]])

Lambda = [0.3, 0.2]
betas = [np.log(1.2), np.log(1.1)]
h0_vec = [Lambda[0]*np.exp(betas[0]*t) for t in range(T)]
h1_vec = [Lambda[1]*np.exp(betas[1]*t) for t in range(T)]
hazard_matrix = np.array([h0_vec, h1_vec])

data = []
j = 0
for i in range(N):
	end_of_spell = 0
	
	for t in range(T):
		if t ==0:
			S = int( np.random.binomial(1, state_init_dist[1]) )
		else:
			S = int( np.random.binomial(1, state_transit_matrix[S, 1]) )
		y = int( np.random.binomial(1, observ_matrix[S, 1]) )
		
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[S,t]:
				end_of_spell = 1
	
		data.append((i, t, j, y, end_of_spell))
		if end_of_spell:
			break

est_param_1 = mcmc_instance.estimate(data, max_iter = 500, is_exit=True)			
est_param_0 = mcmc_instance.estimate(data, max_iter = 500, is_exit=False)
print(est_param_0)
print(est_param_1)	
'''

# Mx = 3, My = 3, J=2, Effort
observ_matrix = np.array([[[0.8,0.2,0.0],[0.2, 0.6, 0.2],[0.0, 0.2, 0.8]], 
						  [[0.5,0.5,0.0],[0.3, 0.4, 0.3],[0.0, 0.1, 0.9]]])
state_init_dist = np.array([0.4, 0.3, 0.3])
state_transit_matrix = np.array([[[0.6,0.4,0],[0, 0.4, 0.6],[0, 0, 1]],
								 [[0.2,0.8,0],[0, 0.7, 0.3],[0, 0, 1]]])

valid_prob_matrix = np.array([[[0.8,0.2],[0.5,0.5],[0.1,0.9]],
							  [[0.7,0.3],[0.4,0.6],[0.01,0.99]]])

pj = 0.3
data = []

for i in range(N):
	for t in range(T):
		j =  np.random.binomial(1, pj)
		if t ==0:
			S = int( np.random.choice(3, 1, p=state_init_dist) )
			E = int( np.random.binomial(1, valid_prob_matrix[j,S,1]) )
		else:
			# This is X from last period
			E = int(np.random.binomial(1, valid_prob_matrix[j,S,1]))
			if E ==1:
				# no pain no gain
				S = int( np.random.choice(3, 1, p=state_transit_matrix[j, S, :]) )

		# If E = 1, generate valid data
		# If E = 0, generate 0
		if E == 1:
			y = int( np.random.choice(3, 1, p=observ_matrix[j, S,:]) )
		else:
			y = 0
					
	
		data.append((i, t, j, y, 0, E))
		
est_param = mcmc_instance.estimate(data, max_iter = 250, is_effort=True)			
print(est_param)	
ipdb.set_trace()

