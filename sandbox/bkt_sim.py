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
		
# check if the data are correctly simulated
# learn
transit = 0.0
tot = 0
for log in data:
	x = log[3]
	if log[1] > 0 and not prev_x:
		transit += x
		tot += 1
	prev_x = x
print(l, transit/tot)

# pi
x1num = 0.0
for log in data:
	if log[1] == 0:
		x1num += log[3]
print(pi, x1num/N)

# s,g
xynum = [[0.0, 0.0], [0.0, 0.0]]
xnum = [0.0, 0.0]
for log in data:
	y = log[2]
	x = log[3]
	xnum[x] += 1
	xynum[x][y] += 1
print(s, xynum[1][0]/xnum[1])
print(g, xynum[0][1]/xnum[0])

# h1 vec
h_cnt = np.zeros((5,2))
s_cnt = np.zeros((5,2))

for log in data:
	t = log[1]
	y = log[2]
	e = log[4]
	a = log[5]
	if a:
		h_cnt[t,y] += e
		s_cnt[t,y] += 1-e
print(hazard_matrix)
print(h_cnt/(h_cnt+s_cnt))
		
with open(proj_dir + '/data/bkt/test/single_sim.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d\n' % log)