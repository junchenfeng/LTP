# encoding: utf-8
import numpy as np
import copy

def generate_possible_states(T):
	# because of the left-right constraints, the number of states is not 3^T
	# generate the baseline
	X_mat = np.ones([T+1,T], dtype=np.int)
	for t in range(1,T+1):
		X_mat[t,:t]=0
	return X_mat

def generate_incremental_states(state_basis, start_lv):
	# assume input is 1*T np array
	# only allow to go one up
	# no jumping between non-consecutive states
	T = state_basis.shape[1]
	if state_basis.sum() == start_lv*T:
		new_states = generate_possible_states(T)[:-1,]+start_lv
	else:
		# because of the now jummping requirement, count at least two initial level 
		# find the first location of the 2nd start level, and do a incremental states
		cnt = 0
		for t in range(T):
			if state_basis[0,t]==start_lv:
				cnt+=1
				if cnt==2:
					break
		if cnt<2:
			new_states = None
		else:
			# because of the constrains before, it is guaranteed that t<T
			new_states = np.zeros([T-t,T], dtype=np.int)
			new_states[:,0:t] = np.vstack(np.tile(state_basis[0,0:t], (T-t, 1)))
			new_states[:,t:] = generate_possible_states(T-t)[:-1,]+start_lv
	return new_states
					
def generate_states(T, max_level=1):
	# generate the basis
	states = generate_possible_states(T)
	if max_level >1:
		for lv in range(1,max_level):
			all_states = copy.deepcopy(states)
			for i in range(states.shape[0]):
				incre_states = generate_incremental_states(states[i,:].reshape(1,T), lv)
				if incre_states is not None:
					all_states = np.vstack((all_states, incre_states))
			states = all_states
	return states
	

def survivial_llk(h, E):
	# h, T*1 hazard rate
	# T, spell length
	# E, whether right censored
	T = len(h)
	if T == 1:
		base_prob = 1
	else:
		# has survived T-1 period
		base_prob = np.product(1-h[:-1])

	prob = base_prob*(E*h[-1]+(1-E)*(1-h[-1]))
	return prob

def state_llk(X, J, V, init_dist, transit_matrix):
	# X: vector of latent state, list
	# transit matrix is np array [t-1,t]
	#if X[0] == 1:
	#	ipdb.set_trace()
	prob = init_dist[X[0]]*np.product([transit_matrix[J[t-1], V[t-1], X[t-1], X[t]] for t in range(1,len(X))])
	return prob
	
def likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort = False, is_exit=False):
	# X:  Latent state
	# O: observation
	# E: binary indicate whether the spell is ended
	T = len(X)
	# P(E|X)
	if is_exit: 
		h = np.array([hazard_matrix[X[t], t] for t in range(T)])
		pa = survivial_llk(h,E)
	else:
		pa = 1
	
	# P(O|X)
	po = 1
	pv = 1
	# P(V|X)
	if is_effort:
		# The effort is generated base on the initial X.
		for t in range(T):
			pv *= valid_prob_matrix[J[t], X[t], V[t]]
	
		for t in range(T):
			if V[t]!=0:
				po *= observ_prob_matrix[J[t], X[t], O[t]]
			else:
				po *= 1.0 if O[t] == 0 else 0.0 # this is a strong built in restriction	
	else:
		V = [1 for x in X]
		for t in range(T):
			po *= observ_prob_matrix[J[t],X[t],O[t]]
		
	# P(X)	
	px = state_llk(X, J, V, state_init_dist, state_transit_matrix)
	
	lk = pa*po*px*pv
	
	if lk<0:
		raise ValueError('Negative likelihood.')
	
	return lk

def get_llk_all_states(X_mat, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=False, is_exit=False):
	N_X = X_mat.shape[0]
	llk_vec = []
	for i in range(N_X):
		X = [int(x) for x in X_mat[i,:].tolist()]
		llk_vec.append( likelihood(X, O, J,V, E, hazard_matrix, observ_prob_matrix, state_init_dist,state_transit_matrix, valid_prob_matrix, is_effort, is_exit) )
		
	return np.array(llk_vec)

def get_single_state_llk(X_mat, llk_vec, t, x):
	res = llk_vec[X_mat[:,t]==x].sum() 
	return res

def get_joint_state_llk(X_mat, llk_vec, t, x1, x2):
	if t==0:
		raise ValueError('t must > 0.')
	res = llk_vec[ (X_mat[:, t-1]==x1) & (X_mat[:, t]==x2) ].sum() 
	return res
