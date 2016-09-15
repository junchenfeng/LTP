import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ipdb

########################
#    Single Item Test  #
########################
# model parameters
s = 0.05
g = 0.2
pi = 0.7
l = 0.3

h1_vec = [0.05, 0.1, 0.15, 0.2, 0.25]
h0_vec = [0.1, 0.2, 0.3, 0.4, 0.5]

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
j = 0
for i in range(N):
	end_of_spell = 0
	is_observ = 1
	for t in range(T):
		if t ==0:
			S = int( np.random.binomial(1, state_init_dist[1]) )		
		y = int( np.random.binomial(1, observ_matrix[S, 1]) )
					
		# update if observed
		if end_of_spell == 1:
			is_observ = 0
		# the end of the spell check is later than the is_observ check so that the last spell is observed
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[S,t]:
				end_of_spell = 1
	
		data.append((i, t, j, y, S, end_of_spell, is_observ))
		
		# update the state for the next period
		S = int( np.random.binomial(1, state_transit_matrix[S, 1]) )	
		

		
with open(proj_dir + '/data/bkt/test/single_sim_x_1.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d\n' % log)
		
########################
#  Multiple Item Test  #
########################

s = [0.05,0.3]
g = [0.2, 0.4]
pi = 0.4
l = [0.3, 0.43]
h1_vec = [0.05, 0.1, 0.15, 0.2, 0.25]
h0_vec = [0.1, 0.2, 0.3, 0.4, 0.5]

# pick item 1 40% of the time
p1 = 0.4
nJ = 2

state_init_dist = np.array([1-pi, pi]) 
state_transit_matrix = np.stack([ np.array([[1-l[j], l[j]], [0, 1]]) for j in range(nJ)] )
observ_prob_matrix =  np.stack([ np.array([[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
hazard_matrix = np.array([h0_vec, h1_vec])	


for i in range(N):
	end_of_spell = 0
	is_observ = 1
	for t in range(T):
		# pick j
		if np.random.uniform() < p1:
			j = 0
		else:
			j = 1
		
		if t ==0:
			S = int( np.random.binomial(1, state_init_dist[1]) )
			
		y = int( np.random.binomial(1, observ_prob_matrix[j, S, 1]) )
		
		
		# update if observed
		if end_of_spell == 1:
			is_observ = 0
		# the end of the spell check is later than the is_observ check so that the last spell is observed
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[S,t]:
				end_of_spell = 1
	
		data.append((i, t, j, y, S, end_of_spell, is_observ))
		
		# update the state for the next period
		if S==0:
			p = state_transit_matrix[j, S, 1]
			S = int( np.random.binomial(1, p))
print(crit_trans/tot_trans)
ipdb.set_trace()
# check the consistency of learning rate
tot_trans = np.zeros((2,))
crit_trans = np.zeros((2,))
for m in range(len(data)):
	i,t,j,y,S,e,a = data[m]
	if t>0 and t<4 and S==0:
		tot_trans[j]+=1
		if data[m+1][4] == 1:
			crit_trans[j]+=1
print(crit_trans/tot_trans)
		
		
	
		
		
with open(proj_dir + '/data/bkt/test/single_sim_x_2.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d\n' % log)