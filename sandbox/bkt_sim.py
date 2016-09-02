import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ipdb


# model parameters
s = 0.05
g = 0.2
pi = 0.7
l = 0.3

h1_vec = [0.3, 0.3, 0.4, 0.4, 0.5]
h0_vec = [0.4, 0.5, 0.6, 0.6, 0.6]

# sim parameters
N = 2000
T = 5

hazard_matrix = np.array([h0_vec, h1_vec])
state_init_dist = np.array([1-pi, pi])
state_transit_matrix = np.array([[1-l, l],[0, 1]])
observ_matrix = np.array([[1-g,g],[s,1-s]])

# The data format is
# id, t, y, is_observed, x
data = []
for i in range(N):
	end_of_spell = 0
	is_observ = 1
	for t in range(T):
		if t ==0:
			S = int( np.random.binomial(1, state_init_dist[1]) )
		else:
			S = int( np.random.binomial(1, state_transit_matrix[S, 1]) )
		
		y = int( np.random.binomial(1, observ_matrix[S, 1]) )
				
		# update if observed
		if end_of_spell == 1:
			is_observ = 0
		# the end of the spell check is later than the is_observ check so that the last spell is observed
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[y,t]:
				end_of_spell = 1
	
		data.append((i, t, y, S, end_of_spell, is_observ))
		

		
with open(proj_dir + '/data/bkt/test/single_sim.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d\n' % log)