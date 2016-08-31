import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model parameters
s = 0.05
g = 0.2
pi = 0.4
l = 0.3

h00 = 0.3
h01 = 0.3
h10 = 0.1
h11 = 0.1

# sim parameters
N = 2000
T = 5

harzard_matrix = np.array([[h00,h01],[h10,h11]])
state_init_dist = np.array([1-pi, pi])
state_transit_matrix = np.array([[1-l,l],[0,1]])
observ_matrix = np.array([[1-g,g],[s,1-s]])

# The data format is
# id, t, y, is_observed, x
data = []
for i in range(N):
	end_of_spell = 0
	is_observ = 1
	for t in range(T):
		if t ==0:
			s = int( np.random.binomial(1, state_init_dist[1]) )
		else:
			s = int( np.random.binomial(1, state_transit_matrix[s,1]) )
		
		y = int( np.random.binomial(1, observ_matrix[s,1]) )
				
		# generate imbalance dataset
		if t>=1:
			# update if observed
			if end_of_spell == 1:
				is_observ = 0
			# the end of the spell check is later than the is_observ check so that the last spell is observed
			if end_of_spell == 0:
				# check if the spell terminates
				if np.random.uniform() < harzard_matrix[s,y]:
					end_of_spell = 1
			
				
	
		data.append((i,t,y,s,end_of_spell,is_observ))
		
with open(proj_dir + '/data/bkt/test/single_sim.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d\n' % log)