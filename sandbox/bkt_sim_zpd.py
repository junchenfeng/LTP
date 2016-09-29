import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ipdb


# sim parameters
N = 2000
T = 5


########################
#    Single Item Test  #
########################
# model parameters
s = 0.05
g = 0.2
pi0=0.1
pi = 0.5
l = 0.3
Lambda = 0.3
betas = [np.log(1.2), 0, np.log(0.7), 0, -.04] # 
h1_vec = [Lambda*np.exp(betas[0]*t) for t in range(T)]
h0_vec = [h1_vec[t]*np.exp(betas[1]+betas[3]*t) for t in range(T)]
h2_vec = [h1_vec[t]*np.exp(betas[2]+betas[4]*t) for t in range(T)]

hazard_matrix = np.array([h0_vec, h1_vec, h2_vec])
state_init_dist = np.array([pi0, 1-pi-pi0, pi])
state_transit_matrix = np.array([[1,0,0],[0, 1-l, l],[0, 0, 1]])
observ_matrix = np.array([[1,0],[1-g,g],[s,1-s]])

# The data format is
# id, t, y, is_observed, x
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
	if  t<T-1 and S==1:
		S = data[m+1][4]
		tot_trans += 1
		if S == 2:
			crit_trans += 1
print(crit_trans/tot_trans)

h_cnt = np.zeros((T,3))
s_cnt = np.zeros((T,3))
for m in range(len(data)):
	i,t,j,y,S,ex,a = data[m]
	if a:
		s_cnt[t,S] += 1
		h_cnt[t,S] += ex
hrates = h_cnt/s_cnt
print(hrates)

with open(proj_dir + '/data/bkt/test/single_sim_zpd_x_1.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d\n' % log)

print('single item generated!')	
ipdb.set_trace()
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
Lambda = 0.3
betas = [np.log(2/3),np.log(1.1)]
h0_vec = [Lambda*np.exp(betas[1]*t) for t in range(T)]
h1_vec = [h*np.exp(betas[0]) for h in h0_vec]


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
		

########################
#  Multiple Item Test  #
########################

s_vec = [0.05, 0.3]
g_vec = [0.0, 0.0]
pi = 0.7
l_vec = [0.5, 0.2]
Lambda = 0.3
betas = [np.log(2/3),np.log(1.1)]
h0_vec = [Lambda*np.exp(betas[1]*t) for t in range(T)]
h1_vec = [h*np.exp(betas[0]) for h in h0_vec]
e0_vec = [0.7, 0.8]
e1_vec = [1.0, 1.0]
# pick item 1 40% of the time
p1 = 0.4
nJ = 2

state_init_dist = np.array([1-pi, pi]) 
state_transit_matrix = np.stack([ np.array([[1-l_vec[j], l_vec[j]], [0, 1]]) for j in range(nJ)] )
observ_prob_matrix =  np.stack([ np.array([[1-g_vec[j], g_vec[j]], [s_vec[j], 1-s_vec[j]]])  for j in range(nJ)] ) # index by state, observ
hazard_matrix = np.array([h0_vec, h1_vec])
effort_matrix = np.stack([ np.array([e0_vec[j],e1_vec[j]]) for j in range(nJ)] )


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
			E = int( np.random.binomial(1, effort_matrix[j,S]) )
		else:
			# This is X from last period
			E = int(np.random.binomial(1, effort_matrix[j,S]))
			if E ==1:
				# no pain no gain
				S = int( np.random.binomial(1, state_transit_matrix[j, S, 1]) )

		y = int( np.random.binomial(1, observ_prob_matrix[j, E*S, 1]) )
		
		
		# update if observed
		if end_of_spell == 1:
			is_observ = 0
		# the end of the spell check is later than the is_observ check so that the last spell is observed
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[S,t]:
				end_of_spell = 1
	
		data.append((i, t, j, y, S, end_of_spell, is_observ, E))

			

# check the consistency of learning rate
tot_trans = np.zeros((2,))
crit_trans = np.zeros((2,))
for m in range(len(data)):
	i,t,j,y,S,e,a,E = data[m]
	if t>0 and t<T-1 and S==0:
		if data[m+1][-1] == 1:
			j = data[m+1][2]
			S = data[m+1][4]
			tot_trans[j] += 1
			crit_trans[j] += S
print(crit_trans/tot_trans)

x1_cnt = np.zeros((2,))
x0_cnt = np.zeros((2,))
y_11_cnt = np.zeros((2,))
y_cnt = np.zeros((2,))
e1x1_cnt = np.zeros((2,))
for m in range(len(data)):
	i,t,j,y,S,ex,a,E = data[m]
	
	S = data[m][4]
	E = data[m][-1]
	x1_cnt[j] += S
	x0_cnt[j] += (1-S)
	e1x1_cnt[j] += E*S
	
	if S==1 and E==1:
		y_11_cnt[j] += y
	else:
		y_cnt[j] += y
sHat = 1-y_11_cnt/e1x1_cnt #P(Y=1,E=1,X=1)/P(X=1,E=1)
gHat = y_cnt/(x1_cnt+x0_cnt-e1x1_cnt) #(P(Y_t=1,E_t=0)+P(Y_t=1,E_t=1,X_t=0))/(P(E_t=0)+P(X_t=0,E_t=1))		
		
with open(proj_dir + '/data/bkt/test/single_sim_x_2.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d,%d\n' % log)
		
print('Multiple items with effort are generated.')		
