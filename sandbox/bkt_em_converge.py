import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import sys
sys.path.append(proj_dir)

from BKT.hmm_em import BKT_HMM_EM

import numpy as np
import ipdb

import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ipdb

N = 1000
T = 5
# model parameters
s = 0.05
g = 0.2
pi = 0.7
l = 0.3

#h1_vec = [0.3, 0.3, 0.4, 0.4, 0.5]
#h0_vec = [0.4, 0.5, 0.6, 0.6, 0.6]
h1_vec = [0.0]*T
h0_vec = [0.5]*T

# sim parameters

hazard_matrix = np.array([h0_vec, h1_vec])
state_init_dist = np.array([1-pi, pi])
state_transit_matrix = np.array([[1-l, l],[0, 1]])
observ_matrix = np.array([[1-g,g],[s,1-s]])

# The data format is
# id, t, y, is_observed, x
l_s = []
g_s = []
s_s = []
pi_s = []
for m  in range(50):

	data_array = []
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
				if np.random.uniform() < hazard_matrix[S,t]:
					end_of_spell = 1
			if is_observ >0 :
				data_array.append((i, t, y, end_of_spell))
				data.append((i,t,y,S,end_of_spell))
	'''		
	# check validity
	y_cnt = np.zeros((2,))
	n_cnt = np.zeros((2,))
	for log in data:
		i,t,y,x,e = log
		if t==0:
			if x == 1:
				y_cnt[e]+=1
			else:
				n_cnt[e]+=1
		
	ipdb.set_trace()
	'''
	
	### (2) Initiate the instance
	em_instance = BKT_HMM_EM()

	### (3) Section 1: Single Factor Full Spell
	y0s = [log[2] for log in data_array if log[1]==0]
	y1s = [log[2] for log in data_array if log[1]==1]
	yTs = [log[2] for log in data_array if log[1]==4]



	init_param = {'s': max(1-np.array(yTs).mean(), 0.01),
				  'g': 0.3, 
				  'pi': min(max(np.array(y0s).mean(), 0.01), 0.99),
				  'l': max(np.array(y1s).mean() - np.array(y0s).mean(), 0.2),
				  'h0': 0,
				  'h1': 0}

	'''		  
	init_param = {'s': s,
				  'g': g, 
				  'pi': pi,
				  'l': l,
				  'h0': 0,
				  'h1': 0}			  
	'''
	em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, data_array, max_iter = 10)

	l_s.append(em_l);s_s.append(em_s);g_s.append(em_g);pi_s.append(em_pi)

print(np.array(l_s).mean())
print(np.array(s_s).mean())
print(np.array(g_s).mean())
print(np.array(pi_s).mean())

ipdb.set_trace()
