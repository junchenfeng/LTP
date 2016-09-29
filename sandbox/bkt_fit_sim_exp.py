import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from BKT.hmm_mcmc import BKT_HMM_MCMC
import numpy as np

mcmc_instance = BKT_HMM_MCMC()

########################
#  Simulate Experiment #
########################
N = 2000
L = 1000

s = [0.05, 0.05, 0.3, 0.05 ]
g = [0.1,  0.1,  0.1,  0.1]
pi = 0.7
l = [0.3,  0.6,  0.7,  0.3]
e0 = [0.7, 0.6,  0.5,  0.5]
e1 = [1.0, 1.0,  1.0,  1.0]
h1_vec = [0.0, 0.0, 0.00]
h0_vec = [0.0, 0.0, 0.0]

# pick item 1 40% of the time
nJ = 4
nT = 3
state_init_dist = np.array([1-pi, pi]) 
state_transit_matrix = np.stack([ np.array([[1-l[j], l[j]], [0, 1]]) for j in range(nJ)] )
observ_prob_matrix =  np.stack([ np.array([[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
effort_matrix =  np.stack([ np.array([e0[j], e1[j]])  for j in range(nJ)] ) # index by state, observ
hazard_matrix = np.array([h0_vec, h1_vec])	

data = []
full_data = []
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
	
		data.append((i, t, j, y,  end_of_spell, E))
		full_data.append((i, t, j, y,  end_of_spell, 1))
		
	
	
	

	
init_param = {'s': [0.05]*nJ,
			  'g': [0.2]*nJ, 
			  'e0':[0.5,0.5,0.5,0.5],
			  'e1':[0.5,0.5,0.5,0.5],
			  'pi': 0.4,
			  'l': [0.2,0.2,0.2,0.2],
			  'h0': [0]*nT,
			  'h1': [0]*nT
			  }
	
pi, s, g, e0,e1, l, h0, h1 = mcmc_instance.estimate(init_param, full_data, max_iter = L, is_exit=False)

print('No effort')
print('point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(l)
print(pi)

np.savetxt(proj_dir+'/data/bkt/test/constant_param_chain.txt', mcmc_instance.parameter_chain, delimiter=',')

init_param = {'s': [0.05]*nJ,
			  'g': [0.2]*nJ, 
			  'e0':[0.5,0.5,0.5,0.5],
			  'e1':[0.5,0.5,0.5,0.5],
			  'pi': 0.4,
			  'l': [0.2,0.2,0.2,0.2],
			  'h0': [0]*nT,
			  'h1': [0]*nT
			  }
mcmc_instance = BKT_HMM_MCMC()
pi, s, g, e0,e1, l, h0, h1 = mcmc_instance.estimate(init_param, data, method='FB', max_iter = L, is_exit=False)


print('slack')
print('FB point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
#print(e0)
#print(e1)
print(l)
print(pi)

np.savetxt(proj_dir+'/data/bkt/test/effort_param_chain.txt', mcmc_instance.parameter_chain, delimiter=',')

