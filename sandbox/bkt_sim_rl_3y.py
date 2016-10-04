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
c = [[0.7,0.3,0.0],[0.2, 0.6, 0.2],[0.0, 0.1, 0.9]]
pi0 = 0.2
pi = 0.3
l0 = 0.2
l1 = 0.4
Lambda = 0.3
betas = [np.log(1.2), 0, np.log(0.7), 0, -.04] # 
h1_vec = [Lambda*np.exp(betas[0]*t) for t in range(T)]
h0_vec = [h1_vec[t]*np.exp(betas[1]+betas[3]*t) for t in range(T)]
h2_vec = [h1_vec[t]*np.exp(betas[2]+betas[4]*t) for t in range(T)]

hazard_matrix = np.array([h0_vec, h1_vec, h2_vec])
state_init_dist = np.array([pi0, 1-pi-pi0, pi])
state_transit_matrix = np.array([[1-l0,l0,0],[0, 1-l1, l1],[0, 0, 1]])
observ_matrix = np.array(c)

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
		y = int( np.random.choice(3, 1, p=observ_matrix[S,:]) )
					
		# update if observed
		if end_of_spell == 1:
			is_observ = 0
		# the end of the spell check is later than the is_observ check so that the last spell is observed
		if end_of_spell == 0:
			# check if the spell terminates
			if np.random.uniform() < hazard_matrix[S,t]:
				end_of_spell = 1
	
		data.append((i, t, j, y, S, end_of_spell, is_observ))	
			
tot_trans  = np.zeros((2,))
crit_trans = np.zeros((2,))
for m in range(len(data)):
	i,t,j,y,S,ex,a = data[m]
	if  t<T-1 and S!=2:
		S1 = data[m+1][4]
		tot_trans[S] += 1
		if S1-S==1:
			crit_trans[S] += 1
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

c_matrix = np.zeros((3,3))
for m in range(len(data)):
	i,t,j,y,S,ex,a = data[m]
	c_matrix[S,y] += 1

for i in range(3):
	c_matrix[i,:] = c_matrix[i,:]/c_matrix[i,:].sum()	
print(c_matrix)

with open(proj_dir + '/data/bkt/test/single_sim_rl_3y_x_1.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d\n' % log)

print('single item generated!')	
ipdb.set_trace()

########################
#        Effort        # 		
########################
e = [[0.2],[0.7],[1.0]]
effort_matrix = np.array([[1-e[i][0],e[i][0]] for i in range(3)])


# The data format is
# id, t, y, is_observed, x
j = 0
stat = np.zeros((2,1))


data = []

for i in range(N):
	end_of_spell = 0
	is_observ = 1
	
	for t in range(T):
		if t ==0:
			S = int( np.random.choice(3, 1, p=state_init_dist) )
			E = int( np.random.binomial(1, effort_matrix[S,1]) )
		else:
			# This is X from last period
			E = int(np.random.binomial(1, effort_matrix[S,1]))
			if E ==1:
				# no pain no gain
				S = int( np.random.choice(3, 1, p=state_transit_matrix[S, :]) )

		# If E = 1, generate valid data
		# If E = 0, generate 0
		if E == 1:
			y = int( np.random.choice(3, 1, p=observ_matrix[S,:]) )
		else:
			y = 0
					
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
tot_trans  = np.zeros((2,))
crit_trans = np.zeros((2,))
for m in range(len(data)):
	i,t,j,y,S, *rest = data[m]
	if  t<T-1 and S!=2:
		E = data[m+1][-1]
		S1 = data[m+1][4]
		if E==1:
			tot_trans[S] += 1
			if S1-S==1:
				crit_trans[S] += 1
print(crit_trans/tot_trans)

			
e_x_cnt = [0,0,0]
x_e_cnt = [0,0,0]

c_matrix = np.zeros((3,3))
c0_matrix = np.zeros((3,3))


for m in range(len(data)):
	i,t,j,y,S,ex,a,E = data[m]
	S = data[m][4]
	E = data[m][-1]

	if  E==1:
		c_matrix[S,y] += 1
	else:
		c0_matrix[S,y] += 1
		
	
	if t>0:
		S = data[m-1][4]
		x_e_cnt[S] += 1 
		e_x_cnt[S] += E
print('effort percentage')
print([e_x_cnt[i]/x_e_cnt[i] for i in range(3)]) #P(E_t=1,X_{t-1}=0)/P(X_{t-0})

print('correct rate')
for i in range(3):
	c_matrix[i,:] = c_matrix[i,:]/c_matrix[i,:].sum()	
print(c_matrix)

print('invalid effort correct rate')
for i in range(3):
	c0_matrix[i,:] = c0_matrix[i,:]/c0_matrix[i,:].sum()	
print(c0_matrix)

with open(proj_dir + '/data/bkt/test/single_sim_rl_3y_x_1_e.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d,%d,%d,%d,%d\n' % log)

print('single item with effort generated!')	
ipdb.set_trace()		

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
