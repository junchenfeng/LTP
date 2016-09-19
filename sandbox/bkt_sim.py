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
N = 5000
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
	
		data.append((i, t, j, y, S, end_of_spell, is_observ))
		
		# update the state for the next period
			
tot_trans = 0
crit_trans = 0
for m in range(len(data)):
	i,t,j,y,S,ex,a = data[m]
	if  t<T-1 and S==0:
		S = data[m+1][4]
		tot_trans += 1
		crit_trans += S
print(crit_trans/tot_trans)

		
with open(proj_dir + '/data/bkt/test/single_sim_x_1.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d\n' % log)

print('single item generated!')	

########################
#        Effort        # 		
########################
N = 5000
T = 8

s = 0.05
g = 0.2
pi = 0.7
l = 0.3
e0 = 0.75
e1 = 0.9

h1_vec = [(t+1)*0.05 for t in range(T)]
h0_vec = [(t+1)*0.1 for t in range(T)]

hazard_matrix = np.array([h0_vec, h1_vec])
state_init_dist = np.array([1-pi, pi])
state_transit_matrix = np.array([[1-l, l],[0, 1]])
observ_matrix = np.array([[1-g,g],[s,1-s]])
effort_matrix = np.array([e0,e1])


# The data format is
# id, t, y, is_observed, x
j = 0
stat = np.zeros((2,1))

y_cnt = 0
n_cnt = 0
data = []

for i in range(N):
	end_of_spell = 0
	is_observ = 1
	
	for t in range(T):
		if t ==0:
			S = int( np.random.binomial(1, state_init_dist[1]) )
			E = int( np.random.binomial(1, effort_matrix[S]) )
		else:
			# This is X from last period
			E = int(np.random.binomial(1, effort_matrix[S]))
			if E ==1:
				# no pain no gain
				Sx = int( np.random.binomial(1, state_transit_matrix[S, 1]) )
				if S==0:
					y_cnt += Sx
					n_cnt += 1-Sx
				S = Sx
		# S=1 & E=1 -> X=1
		y = int( np.random.binomial(1, observ_matrix[S*E, 1]) )
					
		# update if observed
		if end_of_spell == 1:
			is_observ = 0
		# the end of the spell check is later than the is_observ check so that the last spell is observed
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[S,t]:
				end_of_spell = 1
	
		data.append((i, t, j, y, S, end_of_spell, is_observ, E))
		
		# update the state for the next period
# check the consistency of learning rate
effective_tot_trans = 0
crit_trans = 0
for m in range(len(data)):
	i,t,j,y,S,ex,a,E = data[m]
	if t<T-1 and S==0:
		Sx = data[m+1][4]
		E = data[m+1][-1]
		if E == 1:

			effective_tot_trans += 1
			crit_trans += Sx

print(crit_trans/effective_tot_trans)

			
e1_x_cnt = 0
x1_e_cnt = 0
e0_x_cnt = 0
x0_e_cnt = 0

x1_cnt = 0
x0_cnt = 0
y_11_cnt = 0
y_cnt = 0
e1x1_cnt = 0

for m in range(len(data)):
	i,t,j,y,S,ex,a,E = data[m]
	S = data[m][4]
	E = data[m][-1]
	x1_cnt += S
	x0_cnt += (1-S)
	e1x1_cnt += E*S
	if S==1 and E==1:
		y_11_cnt += y
	else:
		y_cnt += y
		
	if t>0:
		S = data[m-1][4]
		x0_e_cnt += 1-S
		x1_e_cnt += S
		e0_x_cnt += E*(1-S)
		e1_x_cnt += E*S	
	

print(e0_x_cnt/x0_e_cnt, e0) #P(E_t=1,X_{t-1}=0)/P(X_{t-0})
print(e1_x_cnt/x1_e_cnt, e1) #P(E_t=1,X_{t-1}=1)/P(X_{t-1})
print(y_11_cnt/e1x1_cnt) #P(Y=1,E=1,X=1)/P(X=1,E=1)
print(y_cnt/(x1_cnt+x0_cnt-e1x1_cnt)) #(P(Y_t=1,E_t=0)+P(Y_t=1,E_t=1,X_t=0))/(P(E_t=0)+P(X_t=0,E_t=1))

with open(proj_dir + '/data/bkt/test/single_sim_x_1_e.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d,%d\n' % log)

print('single item wit effort generated!')	
		
ipdb.set_trace()		

########################
#  Multiple Item Test  #
########################

s_vec = [0.05, 0.3]
g_vec = [0.2, 0.4]
pi = 0.7
l_vec = [0.3, 0.4]
h1_vec = [(t+1)*0.05 for t in range(T)]
h0_vec = [(t+1)*0.1 for t in range(T)]

# pick item 1 40% of the time
p1 = 0.4
nJ = 2

state_init_dist = np.array([1-pi, pi]) 
state_transit_matrix = np.stack([ np.array([[1-l_vec[j], l_vec[j]], [0, 1]]) for j in range(nJ)] )
observ_prob_matrix =  np.stack([ np.array([[1-g_vec[j], g_vec[j]], [s_vec[j], 1-s_vec[j]]])  for j in range(nJ)] ) # index by state, observ
hazard_matrix = np.array([h0_vec, h1_vec])	

data = []
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
		else:
			p = state_transit_matrix[j, S, 1]
			S = int( np.random.binomial(1, p))			
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

ipdb.set_trace()
# check the consistency of learning rate
tot_trans = np.zeros((2,))
crit_trans = np.zeros((2,))
for m in range(len(data)):
	i,t,j,y,S,e,a = data[m]
	if t>0 and t<T-1 and S==0:
		j = data[m+1][2]
		S = data[m+1][4]
		tot_trans[j] += 1
		crit_trans[j] += S
print(crit_trans/tot_trans)
		
		
	
		
		
with open(proj_dir + '/data/bkt/test/single_sim_x_2.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d\n' % log)
		
print('Multiple items without effort are generated.')		
########################
#  Simulate Experiment #
########################
N = 5000

s = [0.05, 0.05, 0.05, 0.05 ]
g = [0.2,  0.2,  0.2,  0.2]
pi = 0.7
l = [0.3,  0.6,  0.6,  0.3]
e0 = [0.5, 0.5,  0.4,  0.5]
e1 = [0.9, 0.9,  0.8,  0.9]
h1_vec = [0.0, 0.0, 0.00]
h0_vec = [0.0, 0.0, 0.0]

# pick item 1 40% of the time
nJ = 4

state_init_dist = np.array([1-pi, pi]) 
state_transit_matrix = np.stack([ np.array([[1-l[j], l[j]], [0, 1]]) for j in range(nJ)] )
observ_prob_matrix =  np.stack([ np.array([[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
effort_matrix =  np.stack([ np.array([e0[j], e1[j]])  for j in range(nJ)] ) # index by state, observ
hazard_matrix = np.array([h0_vec, h1_vec])	

data = []

for i in range(N):
	end_of_spell = 0
	is_observ = 1
	# choose one of the two sequence 
	seq_rand = np.random.random()
	if  seq_rand<= 0.25:
		item_seq = [0,1,3]
	elif seq_rand>0.25 and seq_rand<=0.5:
		item_seq = [0,2,3]
	elif seq_rand>0.5 and seq_rand<=0.75:
		item_seq = [3,1,0]
	else:
		item_seq = [3,2,0]
	
		
	for t in range(3):
		j = item_seq[t]
		if t ==0:
			S = int( np.random.binomial(1, state_init_dist[1]) ) # same initial distribution
			E = int( np.random.binomial(1, effort_matrix[j,S]) )
		else:
			# This is X from last period
			E = int(np.random.binomial(1, effort_matrix[j,S]))
			if E ==1:
				# no pain no gain
				S = int( np.random.binomial(1, state_transit_matrix[j, S, 1]) )
		# S=1 & E=1 -> X=1
		y = int( np.random.binomial(1, observ_prob_matrix[j, S*E, 1]) )
					
		# update if observed
		if end_of_spell == 1:
			is_observ = 0
		# the end of the spell check is later than the is_observ check so that the last spell is observed
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[S,t]:
				end_of_spell = 1
	
		data.append((i, t, j, y, S, end_of_spell, is_observ, E))
		
	
		
		
with open(proj_dir + '/data/bkt/test/single_sim_exp.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d,%d\n' % log)
		
		
print('Multiple item generated.')		
