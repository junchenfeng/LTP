import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy
import ipdb
import math
import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)

from BKT.prop_hazard_ars import ars_sampler

# There are three states, uninitiated, initiated, learned.
# uninitiated and learned are both absorbing state
# initiated can move to learned
# This corresponds to the zone of proximal development

# the learning rate depends on the response from time t-1

def survivial_llk(h,E):
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

def state_llk(X, J, V, O, init_dist, transit_matrix):
	# X: vector of latent state, list
	# transit matrix is np array [t-1,t]
	#if X[0] == 1:
	#	ipdb.set_trace()
	prob = init_dist[X[0]]*np.product([transit_matrix[J[t], V[t], O[t-1], X[t-1], X[t]] for t in range(1,len(X))])
	return prob
	
def likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort = False):
	# X:  Latent state
	# O: observation
	# E: binary indicate whether the spell is ended
	T = len(X)
	# P(E|X)
	h = np.array([hazard_matrix[X[t],t] for t in range(T)])
	pa = survivial_llk(h,E)
	
	# P(O|X)
	po = 1
	# P(V|X)
	if is_effort:
		pv = np.product([valid_prob_matrix[J[t],X[t],V[t]] for t in range(T)])
		for t in range(T):
			if X[t]!=0:
				po *= observ_prob_matrix[J[t],(X[t]-1)*V[t]+1,O[t]]
			else:
				po *= observ_prob_matrix[J[t],0,O[t]]		
		# P(X)
		px = state_llk(X, J, V, O, state_init_dist, state_transit_matrix)				
	else:
		pv = 1
		for t in range(T):
			po *= observ_prob_matrix[J[t],X[t],O[t]]
		px = state_llk(X, J, [1 for x in X], O, state_init_dist, state_transit_matrix)

	return pa*po*px*pv
	
def generate_possible_states(T, is_uninit):
	# because of the left-right constraints, the possible state is T+2
	if is_uninit:
		X_mat = np.ones([T+2,T], dtype=np.int)*2
	else:
		X_mat = np.ones([T+1,T], dtype=np.int)*2
	
	for t in range(1,T+1):
		X_mat[t,:t]=1
	
	if is_uninit:
		X_mat[T+1,:] = 0 # the uninitiated
	
	return X_mat

def get_llk_all_states(X_mat, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=False):
	N_X = X_mat.shape[0]
	llk_vec = []
	for i in range(N_X):
		X = [int(x) for x in X_mat[i,:].tolist()]
		llk_vec.append( likelihood(X, O, J,V, E, hazard_matrix, observ_prob_matrix, state_init_dist,state_transit_matrix, valid_prob_matrix, is_effort) )
		
	return np.array(llk_vec)

def get_single_state_llk(X_mat, llk_vec, t, x):
	res = llk_vec[X_mat[:,t]==x].sum() 
	return res

def get_joint_state_llk(X_mat, llk_vec, t, x1, x2):
	if t==0:
		raise ValueException('t must > 0.')
	res = llk_vec[ (X_mat[:, t-1]==x1) & (X_mat[:, t]==x2) ].sum() 
	return res

def get_E(E,t,T):
	if E == 0:
		Et = 0
	else:
		if t ==T:
			Et = 1
		else:
			Et = 0
	return Et
	
def logExpSum(llk_vec):
	llk_max = max(llk_vec)
	llk_sum = llk_max + np.log(np.exp(llk_vec-llk_max).sum())
	return llk_sum
	
class BKT_HMM_MCMC(object):

	def _load_observ(self, data):
		# data = [(i,t,j,y,e)]  
		# i learner id from 0:N-1
		# t sequence id, t starts from 0
		# j	item id, from 0:J-1
		# y response, 0 or 1
		# e(xit) if the spell ends here
		# valid, 0 or 1
		self.K = len(set([x[0] for x in data]))
		self.T = max([x[1] for x in data]) + 1
		self.J = len(set([x[2] for x in data]))
		
		self.E_array = np.empty((self.T, self.K), dtype=np.int)
		self.V_array = np.empty((self.T, self.K), dtype=np.int)
		self.observ_data = np.empty((self.T, self.K), dtype=np.int)
		self.item_data = np.empty((self.T, self.K), dtype=np.int)
		T_array = np.zeros((self.K,))
		
		for log in data:
			if len(log)==4:
				# The spell never ends; multiple item
				i,t,j,y = log
				is_e = 0; is_v = 1
			elif len(log) == 5:
				i,t,j,y,is_e = log
				is_v = 1
			elif len(log) == 6:
				i,t,j,y,is_e,is_v = log
			else:
				raise Exception('The log format is not recognized.')
			self.observ_data[t, i] = y
			self.item_data[t, i] = j
			self.E_array[t, i] = is_e
			self.V_array[t, i] = is_v
			T_array[i] = t
		
		# This section is used to collapse states
		self.T_vec = [int(x)+1 for x in T_array.tolist()] 
		self.O_data = []
		for i in range(self.K):
			self.O_data.append( [x for x in self.observ_data[0:self.T_vec[i],i].tolist()] )
		self.J_data = []
		for i in range(self.K):
			self.J_data.append( [x for x in self.item_data[0:self.T_vec[i],i].tolist()] )		
		self.V_data = []
		for i in range(self.K):
			self.V_data.append( [x for x in self.V_array[0:self.T_vec[i],i].tolist()] )					
		
		self.E_vec = [int(self.E_array[self.T_vec[i]-1, i]) for i in range(self.K)]
				
		# initialize
		self.h1 = [self.Lambda*np.exp(self.betas[0]*t) for t in range(self.T)]
		self.h0 = [self.h1[t]*np.exp(self.betas[1]+self.betas[3]*t) for t in range(self.T)]
		self.h2 = [self.h1[t]*np.exp(self.betas[2]+self.betas[4]*t) for t in range(self.T)]
		
		self._update_derivative_parameter()  # learning spead
		self._collapse_obser_state()

	def __update_pi(self, t, E, V, observ, item_id, pi_vec, P_mat, is_effort=False):
		# pi(i,t) = P(X_t=i|O1,...,O_t,\theta)
		if t == 0:
			if not E:
				pa0 = 1-self.hazard_matrix[0, t]
				pa1 = 1-self.hazard_matrix[1, t]
				pa2 = 1-self.hazard_matrix[2, t]
			else:
				pa0 = self.hazard_matrix[0, t]
				pa1 = self.hazard_matrix[1, t]
				pa2 = self.hazard_matrix[2, t]
			
			if not is_effort:
				V = 1
			po0 = self.observ_prob_matrix[item_id, 0, observ]
			po1 = self.observ_prob_matrix[item_id, 1, observ] # always guess
			po2 = self.observ_prob_matrix[item_id, 1+V, observ] # if no effort, allow for guess
			
			if is_effort:
				pv0 = self.valid_prob_matrix[item_id, 0, V]
				pv1 = self.valid_prob_matrix[item_id, 1, V]
				pv2 = self.valid_prob_matrix[item_id, 2, V]
			else:
				pv0 = 1
				pv1 = 1
				pv2 = 1
			
			# pi(i,0) = P(X_0=i|O0,\theta)
			p0y = self.pi0 * po0  * pa0 * pv0
			p1y = (1-self.pi-self.pi0) * po1 * pa1 * pv1 
			p2y = self.pi * po2  * pa2 * pv2
			py = p0y+p1y+p2y
					
			pi_vec[t,:] = [p0y/py, p1y/py, p2y/py]
			

		else:
			# pi(i,t) = sum_{j} P(j,i,t) where P(j,i,t) is the (j,i)the element of transition matrix P
			pi_vec[t,:] = P_mat[t-1,:,:].sum(axis=0)
		
		return pi_vec
			
	def __update_P(self, t, E, item_id_l, V, observ, observ0, item_id_O, pi_vec, P_mat, is_effort=False):	
	
		#ipdb.set_trace()
		p_raw = np.zeros((3,3))
		if not is_effort:
			V = 1
		
		if not E:
			pa0 = 1-self.hazard_matrix[0, t+1]
			pa1 = 1-self.hazard_matrix[1, t+1]
			pa2 = 1-self.hazard_matrix[2, t+1]
		else:
			pa0 = self.hazard_matrix[0, t+1]
			pa1 = self.hazard_matrix[1, t+1]
			pa2 = self.hazard_matrix[2, t+1]
			
		po0 = self.observ_prob_matrix[item_id_O, 0, observ]
		po1 = self.observ_prob_matrix[item_id_O, 1, observ] # always guess
		po2 = self.observ_prob_matrix[item_id_O, 1+V, observ] # if no effort, allow for guess
		
		# TODO: change the item_id_O to l
		if is_effort:
			pv0 = self.valid_prob_matrix[item_id_O, 0, V]
			pv1 = self.valid_prob_matrix[item_id_O, 1, V]
			pv2 = self.valid_prob_matrix[item_id_O, 2, V]
		else:
			pv0 = 1
			pv1 = 1
			pv2 = 1
		
		p_raw[0,0] = max(pi_vec[t,0] * self.state_transit_matrix[item_id_l,V,observ0,0,0] * po0 * pa0 * pv0, 0.0)
		p_raw[1,1] = max(pi_vec[t,1] * self.state_transit_matrix[item_id_l,V,observ0,1,1] * po1 * pa1 * pv1, 0.0)
		p_raw[1,2] = max(pi_vec[t,1] * self.state_transit_matrix[item_id_l,V,observ0,1,2] * po2 * pa2 * pv2, 0.0)
		p_raw[2,2] = max(pi_vec[t,2] * self.state_transit_matrix[item_id_l,V,observ0,2,2] * po2 * pa2 * pv2, 0.0)
		
		
		P_mat[t,:,:] = p_raw/p_raw.sum()

		
		return P_mat		
		
	def _update_derivative_parameter(self):
		self.state_init_dist = 		np.array([self.pi0, 1-self.pi-self.pi0, self.pi]) # initial distribution is invariant to item	
		self.state_transit_matrix = np.stack([np.array([[[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]]], [[[1,0,0],[0,1-self.l0[j], self.l0[j]], [0, 0, 1]],[[1,0,0],[0,1-self.l1[j], self.l1[j]], [0, 0, 1]]]]) for j in range(self.J)]) # if V=0 P(X_t=X_{t-1}) = 1
		self.observ_prob_matrix = 	np.stack([np.array([[1,0], [1-self.g[j], self.g[j]], [self.s[j], 1-self.s[j]]])  for j in range(self.J)]) # index by state, observ
		self.hazard_matrix = 		np.array([self.h0, self.h1, self.h2]) # hazard rate is invariant to item
		self.valid_prob_matrix = 	np.stack([np.array([[1,0],[1-self.e1[j], self.e1[j]],[1-self.e2[j], self.e2[j]]]) for j in range(self.J)]) # index by state, V
		#TODO: add robust check
	
	def _collapse_obser_state(self):
		self.obs_type_cnt = defaultdict(int)
		self.obs_type_ref = {}
		for k in range(self.K):
			obs_type_key = str(self.E_vec[k]) + '-' + '|'.join(str(y) for y in self.O_data[k]) + '-' + '|'.join(str(j) for j in self.J_data[k]) + '-' + '|'.join(str(v) for v in self.V_data[k])
			self.obs_type_cnt[obs_type_key] += 1
			self.obs_type_ref[k] = obs_type_key
		# construct the space
		self.obs_type_info = {}
		for key in self.obs_type_cnt.keys():
			e_s, O_s, J_s, V_s = key.split('-')
			self.obs_type_info[key] = {'E':int(e_s), 'O':[int(x) for x in O_s.split('|')], 'J':[int(x) for x in J_s.split('|')], 'V':[int(x) for x in V_s.split('|')]}

	def __forward_recursion(self, is_exit=False, is_effort=False):
		for key in self.obs_type_info.keys():
			# get the obseration state			
			Os = self.obs_type_info[key]['O']
			Js = self.obs_type_info[key]['J']
			E = self.obs_type_info[key]['E']
			Vs = self.obs_type_info[key]['V']
			#calculate the exhaustive state probablity
			T = len(Os)
			
			# if there is a only 1 observations, the P matrix does not exist, pi vector will the first observation
			pi_vec = np.zeros((T,3))
			P_mat = np.zeros((T-1,3,3))
			for t in range(T):
				Et = get_E(E,t,T)
				# The learning happens simulateneously with response. Learning in doing.
				pi_vec = self.__update_pi(t, Et, Vs[t], Os[t], Js[t], pi_vec, P_mat, is_effort)
				if t !=T-1 and T!=1:
					Et = get_E(E,t+1,T-1)
					P_mat = self.__update_P(t, Et,  Js[t+1], Vs[t+1], Os[t+1], Os[t], Js[t+1], pi_vec, P_mat, is_effort)
			self.obs_type_info[key]['pi'] = pi_vec
			self.obs_type_info[key]['P'] = P_mat
	
	def __backward_sampling_scheme(self,is_exit=False, is_effort=False):
		for obs_key in self.obs_type_info.keys():
			pi_vec = self.obs_type_info[obs_key]['pi']
			P_mat = self.obs_type_info[obs_key]['P']
			T = pi_vec.shape[0]
			# This hard codes the fact that uninitiated is an obserbing state
			sample_p_vec = np.zeros((T,3))
			for t in range(T-1,-1,-1):
				if t == T-1:
					sample_p_vec[t,:] = [min(pi,1.0) for pi in pi_vec[t,:]] # the initial state needs full specification
				else:
					for x in range(1,3):
						# th problem is really to sample state 1 and 2 if the previous state is not 0
						sample_p_vec[t,x] = min(P_mat[t,2,x]/P_mat[t,:,x].sum(), 1.0)
			self.obs_type_info[obs_key]['sample_p'] = sample_p_vec
				
	def _MCMC(self, max_iter, method, fixVal, is_exit=False, is_effort=False, is_uninit=False):
		if not is_exit and self.hazard_matrix.sum() != 0: 
			raise Exception('Hazard rates are not set to 0 while disabled the update in hazard parameter.')
		
		if is_exit:
			prop_hazard_mdls = {}
			prop_hazard_mdls[0] = ars_sampler(self.Lambda*math.exp(self.betas[1]), [0])
			prop_hazard_mdls[1] = ars_sampler(self.Lambda, [self.betas[x] for x in [0,2,4]])
			
		self.parameter_chain = np.empty((max_iter, 2+self.J*6+6))
		
		# initialize for iteration
		for iter in tqdm(range(max_iter)):
			# Step 1: Data Augmentation

			if method == "DG":
				# calculate the sample prob
				for key in self.obs_type_info.keys():
					# get the obseration state
					O = self.obs_type_info[key]['O']
					E = self.obs_type_info[key]['E']
					J = self.obs_type_info[key]['J']
					V = self.obs_type_info[key]['V']
					
					#calculate the exhaustive state probablity
					Ti = len(O)					
					X_mat = generate_possible_states(Ti, is_uninit)
					llk_vec = get_llk_all_states(X_mat, O, J, V, E, self.hazard_matrix, self.observ_prob_matrix, self.state_init_dist, self.state_transit_matrix, self.valid_prob_matrix, is_effort)
					self.obs_type_info[key]['pi'] = [get_single_state_llk(X_mat, llk_vec, 0, x)/llk_vec.sum() for x in range(3)]
					self.obs_type_info[key]['llk_vec'] = llk_vec
					self.obs_type_info[key]['l_vec'] = [ get_joint_state_llk(X_mat, llk_vec, t, 1, 2) / get_single_state_llk(X_mat, llk_vec, t-1, 1) for t in range(1,Ti)]
				
				# sample states
				X = np.empty((self.T, self.K),dtype=np.int)
				
				for i in range(self.K):
					# check the key
					obs_key = self.obs_type_ref[i]
					pi = self.obs_type_info[obs_key]['pi']
					l_vec = self.obs_type_info[obs_key]['l_vec']
					Vs = self.obs_type_info[obs_key]['V']
										
					X[0,i] = np.random.choice(3,1,p=pi)
					for t in range(1, self.T_vec[i]):
						if X[t-1,i] != 1:
							X[t,i] = X[t-1,i] # both 0 and 2 are absorbing state
						else:
							if Vs[t]==1:
								# X at t is determined by the transition matrix at t-1
								X[t,i] = np.random.binomial(1, l_vec[t-1])+1
							else:
								X[t,i]  = 1 # no effort
					
			elif method == "FB":
				# forward recursion
				
				self.__forward_recursion(is_exit,is_effort)
				
				# backward sampling scheme
				self.__backward_sampling_scheme(is_exit,is_effort)

				# backward sampling
				X = np.empty((self.T, self.K), dtype=np.int)
				#init_pis = np.zeros((self.K, 1))
				for k in range(self.K):
					# check for the observation type
					obs_key = self.obs_type_ref[k]
					sample_p_vec = self.obs_type_info[obs_key]['sample_p']
					for t in range(self.T_vec[k]-1,-1,-1):
						if t == self.T_vec[k]-1:
							X[t,k] = np.random.choice(3,1,p=sample_p_vec[t,:])
						else:
							next_state = int(X[t+1,k])
							if next_state !=2:
								X[t,k] = next_state # 0 can only comes from 0, 1 can only comes from 1 because of the left to right constraint
							else:
								p = sample_p_vec[t,next_state]
								X[t,k] = np.random.binomial(1,p)+1

				
			
			# Step 2: Update Parameter
			critical_trans = np.zeros((self.J,2),dtype=np.int)
			tot_trans = np.zeros((self.J,2),dtype=np.int)
			
			obs_cnt = np.zeros((self.J,3,3)) # state,observ
			valid_cnt = np.zeros((self.J,3),dtype=np.int)
			valid_state_cnt = np.zeros((self.J,3),dtype=np.int)
			
			for k in range(self.K):
				for t in range(0, self.T_vec[k]):
					l_j = self.item_data[t,k]
					is_v = self.V_array[t,k]
					o_j = self.item_data[t,k]
					y = self.observ_data[t-1,k]
					# if the transition happens at t, item in t-1 should take the credit
					# The last item does not contribute the the learning rate
					# update l
					if t>0 and X[t-1,k] == 1 and is_v>0:
						#P(X_t=1,X_{t-1}=0,V_t=1)/P(X_{t-1}=0,V_t=1)
						tot_trans[l_j,y] += 1
						if X[t,k] == 2:
							critical_trans[l_j,y] += 1
					if t>0:
						if X[t-1,k]!=0:
							x = X[t-1,k]
							valid_cnt[o_j,x] += is_v
							valid_state_cnt[o_j,x] += 1			
					# update obs_cnt
					obs_cnt[o_j, (X[t,k]-1)*is_v+1, self.observ_data[t,k]] += 1 #P(Y=0,V=1,X=1)/P(X=1,V=1) = s; P(Y_t=1,V_t=0)+P(Y_t=1,V_t=1,X_t=0))/(P(V_t=0)+P(X_t=0,V_t=1)) =g
					
			for j in range(self.J):
				self.l0[j] =  np.random.beta(self.prior_param['l'][0]+critical_trans[j,0], self.prior_param['l'][1]+tot_trans[j,0]-critical_trans[j,0])
				self.l1[j] =  np.random.beta(self.prior_param['l'][0]+critical_trans[j,1], self.prior_param['l'][1]+tot_trans[j,1]-critical_trans[j,1])
				self.s[j] =  np.random.beta(self.prior_param['s'][0]+obs_cnt[j,2,0], self.prior_param['s'][1]+obs_cnt[j,2,1])
				self.g[j] =  np.random.beta(self.prior_param['g'][0]+obs_cnt[j,1,1], self.prior_param['g'][1]+obs_cnt[j,1,0])
				
			#ipdb.set_trace()
			
			
			nX0 = len([x for x in X[0,:] if x==0])
			nX2 = len([x for x in X[0,:] if x==2])
			self.pi0 = np.random.beta(self.prior_param['pi'][0]+nX0, self.prior_param['pi'][1]+self.K-nX0)
			self.pi =  np.random.beta(self.prior_param['pi'][0]+nX2, self.prior_param['pi'][1]+self.K-nX2)
			
			if is_exit:
				# Separate out the model for X = 0 and X!=0. Under the assumption of no trasnition, this is valid
				# Otherwise, the X=0 has too few data and MCMC draws suffer
				# TODO: checl if the constrain is valid
			
				# generate X,D
				hX = [[],[]]
				hD = [[],[]]
				hIdx = [[],[]]
				for k in range(self.K):
					for t in range(self.T_vec[k]):
						if X[t,k] < 0:
							hX[0].append([t]) # do not estimate the curve
							hD[0].append(self.E_array[t,k])
						else:
							x2 = int(X[t,k] == 2)
							hX[1].append((t, x2, x2*t))
							hD[1].append(self.E_array[t,k])
						
						
				# do a stratified sampling by t
				for i in range(2):
					M = len(hD[i])
					if M==0:
						continue
					nT = np.zeros((self.T,1),dtype=np.int)
					idxT = [[] for t in range(self.T)]
					for hi in range(M):
						t = hX[i][hi][0]
						nT[t] += 1
						idxT[t].append(hi)
					for t in range(self.T):
						sample_size = min(round(5000/self.T),nT[t])
						if sample_size ==0:
							continue
						idxs = np.random.choice(idxT[t], size = sample_size, replace=False)
						hIdx[i]  += idxs.tolist()
				
				# update the proportional hazard model
				#if len(hD[0])>0:
				#	prop_hazard_mdls[0].load(np.array(hX[0])[hIdx[0],:], np.array(hD[0])[hIdx[0]])
				#	prop_hazard_mdls[0].Lambda = prop_hazard_mdls[0].sample_lambda()[-1]

				prop_hazard_mdls[1].load(np.array(hX[1])[hIdx[1],:], np.array(hD[1])[hIdx[1]])
				prop_hazard_mdls[1].Lambda = prop_hazard_mdls[1].sample_lambda()[-1]
				for k in range(3):
					prop_hazard_mdls[1].betas[k] = prop_hazard_mdls[1].sample_beta(k)[-1]	
				
				# reconfig
				self.Lambda = prop_hazard_mdls[1].Lambda
				self.betas = [prop_hazard_mdls[1].betas[0], 
							  0,
							  prop_hazard_mdls[1].betas[1],
							  0,
							  #(prop_hazard_mdls[0].betas[0]-prop_hazard_mdls[1].betas[0]),
							  prop_hazard_mdls[1].betas[2]]
				
				self.h1 = [self.Lambda*np.exp(self.betas[0]*t) for t in range(self.T)]
				self.h0 = [self.h1[t]*np.exp(self.betas[1]+self.betas[3]*t) for t in range(self.T)]
				self.h2 = [self.h1[t]*np.exp(self.betas[2]+self.betas[4]*t) for t in range(self.T)]
				
				# check for sanity
				if any([h>1 for h in self.h0]) or any([h>1 for h in self.h1]) or any([h>1 for h in self.h2]):
					raise ValueException('Hazard rate is larger than 1.')
				
			else:
				self.h0 = [0.0 for t in range(self.T)]
				self.h1 = [0.0 for t in range(self.T)]
				self.h2 = [0.0 for t in range(self.T)]
			
			if is_effort:
				for j in range(self.J):
					self.e1[j] = np.random.beta(self.prior_param['e1'][0]+valid_cnt[j,1], self.prior_param['e1'][1]+valid_state_cnt[j,1]-valid_cnt[j,1])
					self.e2[j] = np.random.beta(self.prior_param['e2'][0]+valid_cnt[j,2], self.prior_param['e2'][1]+valid_state_cnt[j,2]-valid_cnt[j,2])
			
			# imposing constraint on s or g
			if 's' in fixVal:
				for j in range(self.J):
					self.s[j] = fixVal['s']
			if 'g' in fixVal:
				for j in range(self.J):
					self.g[j] = fixVal['g']			
			self.parameter_chain[iter, :] = [self.pi0, self.pi] + self.s + self.g + self.e1 + self.e2 + self.l0 + self.l1 + [self.Lambda] + self.betas
			self._update_derivative_parameter()
			self.X = X

	def _get_point_estimation(self, start, end):
		# calcualte the llk for the parameters
		gap = max(int((end-start)/100), 10)
		parameter_candidates = self.parameter_chain[range(start, end, gap), :]
		avg_parameter = parameter_candidates.mean(axis=0).tolist()
		return avg_parameter
				
	def estimate(self, init_param, data_array, method='FB', max_iter=1000, is_exit=False, is_effort=False, is_uninit=False, fixVal={}):
		param = copy.deepcopy(init_param)
		# ALL items share the same prior for now
		self.g = param['g']  # guess
		self.s = param['s']  # slippage
		self.pi = param['pi']  # initial prob of mastery
		self.l0 = param['l0']  # learn speed
		self.l1 = param['l1']
		self.pi0 = param['pi0']
		
		if (self.pi+self.pi0)>1:
			raise ValueException('pi0 + pi >1')
		
		if is_effort:
			self.e1 = param['e1'] 
			self.e2 = param['e2']
		else:
			self.e1 = [1.0]*len(param['s'])
			self.e2 = [1.0]*len(param['s'])
		
		if is_exit:
			self.Lambda = param['Lambda']  # harzard rate with response 0
			self.betas = param['betas']  # proportional hazard parametes, hard code to [X,t]
		else:
			self.Lambda = 0.0
			self.betas = [0.0]*5
		
		# generate derivative stats
		
		#TODO: allow for a flexible hazard rate function
		if len(self.betas)!=5:
			raise Exception('Wrong specification for proportional hazard model.')
		
		if self.pi == 0:
			raise Exception('Invalid Prior')
		
		# for now, assume flat prior for the hazard rate
		self.prior_param = {'l': [2, 2],
							's': [1, 2],
							'e1': [2, 2],
							'e2':[2, 2],
							'g': [1, 2],
							'pi':[2, 2],
							'pi0':[1,4]}
		
		self._load_observ(data_array)
		self._MCMC(max_iter, method, fixVal, is_exit, is_effort, is_uninit)
		res = self._get_point_estimation(int(max_iter/2), max_iter)
		#ipdb.set_trace()
		self.pi0 = res[0]; 
		self.pi = res[1]
		self.s = res[2+self.J*0:2+self.J]
		self.g = res[2+self.J:2+self.J*2]
		self.e1 = res[2+self.J*2:2+self.J*3]
		self.e2 = res[2+self.J*3:2+self.J*4]
		self.l0 = res[2+self.J*4:2+self.J*5]
		self.l1 = res[2+self.J*5:2+self.J*6]
		self.Lambda = res[2+self.J*6:2+self.J*6+1]
		self.betas =  res[2+self.J*6+1:]	
		
		return self.pi0, self.pi, self.s, self.g, self.e1, self.e2, self.l0, self.l1, self.Lambda, self.betas
		
if __name__=='__main__':
	# UNIT TEST
	
	# check for the marginal
	s = [0.05]
	g = [0.2]
	pi0 = 0.1
	pi = 0.5
	l0 = [0.2]
	l1 = [0.4]

	e1 = [0.85]
	e2 = [0.9]
	J = [0,0,0]
	V = [1,1,1]
	nJ = 1
	# hazard rate is generated by 
	Lambda = 0.1
	betas = [np.log(1.5), np.log(2), np.log(0.5), 0, 0]
	
	h0 = [0.2, 0.3, 0.45]
	h1 = [0.1, 0.15, 0.225]
	h2 = [0.05, 0.075, 0.1125]
	
	init_param = {'s':copy.copy(s),
			  'g':copy.copy(g), 
			  'e1':copy.copy(e1),
			  'e2':copy.copy(e2),
			  'pi':copy.copy(pi),
			  'l0':copy.copy(l0),
			  'l1':copy.copy(l1),
			  'pi0':copy.copy(pi0),
			  'Lambda':copy.copy(Lambda),
			  'betas':copy.copy(betas),
			  }	
	data_array = [(0,0,0,0,0,0),(0,1,0,0,0,0),(0,2,0,0,1,0)] # E=1, O=[0,0,0], J = [0,0,0], V=[0,0,0]
	obs_key = '1-0|0|0-0|0|0-0|0|0'			  
			  
	
	# derive the derative statistics
	state_init_dist = np.array([pi0, 1-pi-pi0, pi]) 
	state_transit_matrix = np.stack([np.array([[[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]]], [[[1,0,0],[0,1-l0[j], l0[j]], [0, 0, 1]],[[1,0,0],[0,1-l1[j], l1[j]], [0, 0, 1]]]]) for j in range(nJ)])
	observ_prob_matrix = np.stack([ np.array([[1,0],[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
	hazard_matrix = np.array([h0, h1, h2])	
	valid_prob_matrix = np.stack([np.array([[1,0],[1-e1[j],e1[j]], [1-e2[j], e2[j]]]) for j in range(nJ)]) 	
	'''
	############ DG|Likelihood
	
	X = [0,0,0]
	O = [0,0,0]
	E = 0
	V = [1,1,1]	
	
	#px = 0.1*1.0*1.0
	#po = 0.0*1.0*0.0
	#pa = (1-0.2)*(1-0.3)*(1-0.45)
	#pv = 0.0*0.0*0.0
	#prob = (1-0.2)*(1-0.3)*(1-0.45)
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=False)	
	print([prob,0.308])	

	O = [1,0,1]
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=False)	
	print([prob,0.0])	
	
	
	X = [1,1,2] 
	#px = 0.4*0.6*0.2
	#po = 0.2*0.8*0.95
	#pa = (1-0.1)*(1-0.15)*(1-0.1125)
	#pv = 0.85*0.85*0.9
	#prob = 0.4*0.6*0.2*0.2*0.8*0.95*(1-0.1)*(1-0.15)*(1-0.1125)*0.85*0.85*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=True)
	print([prob, 0.003221031582])
	
	
	
	E = 1
	#pa = (1-0.1)*(1-0.15)*0.1125
	#prob =0.4*0.6*0.2*0.2*0.8*0.95*(1-0.1)*(1-0.15)*0.1125*0.85*0.85*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=True)
	print([prob,0.000408299778])
	
	E = 0
	X = [2,2,2]
	# px = 0.5*1*1
	# po = 0.95*0.05*0.95
	# pa = (1-0.05)*(1-0.075)*(1-0.1125)
	# pv = 0.9*0.9*0.9
	# prob = 0.5*1*1*0.95*0.05*0.95*(1-0.05)*(1-0.075)*(1-0.1125)*0.9*0.9*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=True)
	print([prob,0.0128276897431641])
	
	
	E = 1
	X = [2,2,2]
	# px = 0.5*1*1
	# po = 0.95*0.05*0.95
	# pa = (1-0.05)*(1-0.075)*0.1125
	# pv = 0.9*0.9*0.9
	# prob = 0.5*1*1*0.95*0.05*0.95*(1-0.05)*(1-0.075)*0.1125*0.9*0.9*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=True)
	print([prob,0.00162604517871094])	
	
	
	E = 1
	X = [1,1,2]
	V = [0,1,1]
	# px = 0.4*0.6*0.2
	# po = 0.2*0.8*0.95
	# pa = (1-0.1)*(1-0.15)*0.1125
	# pv = 0.15*0.85*0.9
	# prob = 0.4*0.6*0.2*0.2*0.8*0.95*(1-0.1)*(1-0.15)*0.1125*0.15*0.85*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=True)
	print([prob, 7.2052902E-05])	
	
	
	V = [0,0,1]
	# px = 0.4*1*0.2
	# pv = 0.15*0.15*0.9
	# prob =  0.4*1*0.2*0.2*0.8*0.95*(1-0.1)*(1-0.15)*0.1125*0.15*0.15*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=True)	
	print([prob, 2.119203E-05])	
	
	V = [0,0,0]
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix, is_effort=True)	
	print([prob, 0.0])	
	############ DG|Marginal Probability
	# data item is i,t,j,y,e,v

	
	x1 = BKT_HMM_MCMC()
	try:
		x1.estimate(init_param, data_array, method = 'DG', max_iter=1, is_effort=True, is_exit=True)
	except:
		pass
	# the pattern is E-y-j-v
	llk_vec = np.array( x1.obs_type_info['1-0|0|0-0|0|0-0|0|0']['llk_vec'] )
	X_mat = generate_possible_states(3)
	
	# all four possible states are 
	# 2,2,2: 0.5*1*1*	0.8*0.8*0.8*	(1-0.05)*(1-0.075)*0.1125 	*0.1*0.1*0.1	= 2.5308E-05
	# 1,1,1: 0.4*1*1*	0.8*0.8*0.8*	(1-0.1)*(1-0.15)*0.225		*0.15*0.15*0.15 = .0001189728
	# 0,0,0: 0.1*1*1* 	1.0*1.0*1.0*	(1-0.2)*(1-0.3)*0.45		*1.0*1.0*1.0 	= 0.0252
	#P(O,E) = 2.5308E-05 + .0001189728 + 0.0252 = 0.0253442808
	
	# single state marginal
	print(get_single_state_llk(X_mat, llk_vec, 0, 0)/llk_vec.sum(), .0252/0.0253442808) 
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), .0001189728/0.0253442808)
	print(get_single_state_llk(X_mat, llk_vec, 0, 2)/llk_vec.sum(), 2.5308E-05/0.0253442808) 
	print(get_single_state_llk(X_mat, llk_vec, 1, 0)/llk_vec.sum(), .0252/0.0253442808) 
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), .0001189728/0.0253442808)
	print(get_single_state_llk(X_mat, llk_vec, 1, 2)/llk_vec.sum(), 2.5308E-05/0.0253442808)
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 0)/llk_vec.sum(), 0.0252/0.0253442808)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 1)/llk_vec.sum(), 0)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 2)/llk_vec.sum(), 0)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), .0001189728/0.0253442808)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 2)/llk_vec.sum(), 0)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 1)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 2)/llk_vec.sum(), 2.5308E-05/0.0253442808)
	
	
	x2 = BKT_HMM_MCMC()
	try:	
		x2.estimate(init_param, data_array, method = 'DG', max_iter=1, is_effort=False, is_exit=True)
	except:
		pass
	# the pattern is E-y-j-v
	llk_vec = np.array( x2.obs_type_info['1-0|0|0-0|0|0-0|0|0']['llk_vec'] )
	X_mat = generate_possible_states(3)
	
	# all four possible states are 
	# 2,2,2: 0.5*1.0*1.0*	0.05*0.05*0.05*	(1-0.05)*(1-0.075)*0.1125 	= 6.1787109375E-06
	# 1,2,2: 0.4*0.2*1.0*	0.8*0.05*0.05*	(1-0.1)*(1-0.075)*0.1125 	= 1.4985E-05
	# 1,1,2: 0.4*0.8*0.2*	0.8*0.8*0.05*	(1-0.1)*(1-0.15)*0.1125 	= 0.000176256	
	# 1,1,1: 0.4*0.8*0.8*	0.8*0.8*0.8*	(1-0.1)*(1-0.15)*0.225		= .022560768
	# 0,0,0: 0.1*1.0*1.0* 	1.0*1.0*1.0*	(1-0.2)*(1-0.3)*0.45		= 0.0252
	#P(O,E) = 6.1787109375E-06 + 1.4985E-05 + 0.000176256 + .022560768 + 0.0252 = .0479581877109375
	
	# single state marginal
	print(get_single_state_llk(X_mat, llk_vec, 0, 0)/llk_vec.sum(), .0252/0.0479581877109375) 
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), (1.4985E-05+.022560768+0.000176256)/0.0479581877109375)
	print(get_single_state_llk(X_mat, llk_vec, 0, 2)/llk_vec.sum(), 6.1787109375E-06/0.0479581877109375) 
	print(get_single_state_llk(X_mat, llk_vec, 1, 0)/llk_vec.sum(), .0252/0.0479581877109375) 
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), (0.000176256+.022560768)/0.0479581877109375)
	print(get_single_state_llk(X_mat, llk_vec, 1, 2)/llk_vec.sum(), (6.1787109375E-06+1.4985E-05)/0.0479581877109375)
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 0)/llk_vec.sum(), 0.0252/0.0479581877109375)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 1)/llk_vec.sum(), 0)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 2)/llk_vec.sum(), 0)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), (.000176256+.022560768)/0.0479581877109375)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 2)/llk_vec.sum(), 1.4985E-05/0.0479581877109375)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 1)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 2)/llk_vec.sum(), 6.1787109375E-06/0.0479581877109375)
	ipdb.set_trace()
	


	test_data_array = [(0,0,0,0,0,1),(0,1,0,1,0,1),(0,2,0,0,0,1)] # E=1, O=[0,1,0], J = [0,0,0], V=[1,1,1]
	obs_key = '0-0|1|0-0|0|0-1|1|1'	
	init_param['l0'] = [0.3]
	init_param['l1'] = [0.3]
	x2t = BKT_HMM_MCMC()
	x2t.estimate(init_param, test_data_array, method = 'DG', max_iter=1)

	# the pattern is E-y-j-v
	llk_vec = np.array( x2t.obs_type_info[obs_key]['llk_vec'] )
	X_mat = generate_possible_states(3)	
	# all four possible states are 
	# 2,2,2: 0.5*1.0*1.0*	0.05*0.95*0.05 	= .0011875
	# 1,2,2: 0.4*0.3*1.0*	0.8*0.95*0.05 	= .00456
	# 1,1,2: 0.4*0.7*0.3*	0.8*0.2*0.05 	= 0.000672	
	# 1,1,1: 0.4*0.7*0.7*	0.8*0.2*0.8		= .025088
	#P(O) = .0011875 + .00456 + 0.000672 + .025088 = .0315075	
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), (.00456+.025088+0.000672)/0.0315075)
	print(get_single_state_llk(X_mat, llk_vec, 0, 2)/llk_vec.sum(), .0011875/0.0315075) 
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), (0.000672+.025088)/0.0315075)
	print(get_single_state_llk(X_mat, llk_vec, 1, 2)/llk_vec.sum(), (.0011875+.00456)/0.0315075)
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), (.000672+.025088)/0.0315075)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 2)/llk_vec.sum(), .00456/0.0315075)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 1)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 2)/llk_vec.sum(), .0011875/0.0315075)

	init_param['l0'] = [0.2]
	init_param['l1'] = [0.4]	
	
	x2t = BKT_HMM_MCMC()
	x2t.estimate(init_param, test_data_array, method = 'DG', max_iter=1)

	# the pattern is E-y-j-v
	llk_vec = np.array( x2t.obs_type_info[obs_key]['llk_vec'] )
	X_mat = generate_possible_states(3)	
	# all four possible states are 
	# 2,2,2: 0.5*1.0*1.0*	0.05*0.95*0.05 	= .0011875
	# 1,2,2: 0.4*0.2*1.0*	0.8*0.95*0.05 	= .00304
	# 1,1,2: 0.4*0.8*0.4*	0.8*0.2*0.05 	= 0.001024	
	# 1,1,1: 0.4*0.8*0.6*	0.8*0.2*0.8		= .024576
	#P(O) = .0011875 + .00304 + .001024 + .024576 = .0298275	
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), (.00304+.024576+.001024)/0.0298275)
	print(get_single_state_llk(X_mat, llk_vec, 0, 2)/llk_vec.sum(), .0011875/0.0298275) 
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), (.001024+.024576)/0.0298275)
	print(get_single_state_llk(X_mat, llk_vec, 1, 2)/llk_vec.sum(), (.0011875+.00304)/0.0298275)
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), (.001024+.024576)/0.0298275)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 2)/llk_vec.sum(), .00304/0.0298275)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 1)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 2, 2)/llk_vec.sum(), .0011875/0.0298275)
	'''
	data_array = [(0,0,0,0,0,0),(0,1,0,0,0,0),(0,2,0,0,1,0)] # E=1, O=[0,0,0], J = [0,0,0], V=[0,0,0]
	obs_key = '1-0|0|0-0|0|0-0|0|0'		

	
	################ FB
	# the last pi vector should be the posterior distribution of state in the last period
	x3 = BKT_HMM_MCMC()
	try:
		x3.estimate(init_param, data_array, method = 'FB', max_iter=1, is_effort=True, is_exit=True)
	except:
		pass
	pi_vec = x3.obs_type_info[obs_key]['pi']
	# P(X_1=0, Y_1, E_1, V_1) = 0.1*1.0*(1-0.2)*1.0 = 0.08
	# P(X_1=1, Y_1, E_1, V_1) = 0.4*0.8*(1-0.1)*0.15 = 0.0432
	# P(X_1=2, Y_1, E_1, V_1) =	0.5*0.8*(1-0.05)*0.1 = 0.038
	# prob = 0.08 + 0.0432 + 0.038 = 0.1612
	print(pi_vec[0,0], 0.08/(0.08+0.0432+0.038))
	
	# First transition 
	P_mat = x3.obs_type_info[obs_key]['P']
	# P(X_1=0,X_2=0,Y_2,E_2=0,V_2=1|Y_1,E_1,V_1) = 0.08/0.1612		*1		*1		*(1-0.3)	*1 	= 0.347394540942928
	# P(X_1=1,X_2=1,Y_2,E_2=0,V_2=1|Y_1,E_1,V_1) = 0.0432/0.1612	*1		*0.8	*(1-0.15)	*0.15 	= 0.027334987593052106
	# P(X_1=2,X_2=2,Y_2,E_2=0,V_2=1|Y_1,E_1,V_1) = 0.038/0.1612		*1		*0.8	*(1-0.075)	*0.1 	= 0.017444168734491318
	print(P_mat[0][0,0], 0.347394540942928/(0.347394540942928+0.027334987593052106+0.017444168734491318))
	
	# marginal period 2
	#P(X1=0, X2=0, Y1=0, Y2=0, E2=0, V_1=0, V_2=0) = 0.1*1	*1.0*1.0	*(1-0.2)*(1-0.3)	*1*1 = .056
	#P(X1=1, X2=1, Y1=0, Y2=0, E2=0, V_1=0, V_2=0) = 0.4*1	*0.8*0.8	*(1-0.1)*(1-0.15)	*0.15*0.15 = .0044064
	#P(X1=2, X2=2, Y1=0, Y2=0, E2=0, V_1=0, V_2=0) = 0.5*1	*0.8*0.8	*(1-0.05)*(1-0.075)	*0.1*0.1 = .002812

	# .056 + .0044064 + .002812 = 0.0632184
	print(pi_vec[1,0], 0.056/0.0632184)
	
	# marginal period 3
	print(pi_vec[2,0], 0.0252/0.0253442808)
	
	
	# the last pi vector should be the posterior distribution of state in the last period
	x4 = BKT_HMM_MCMC_ZPD()
	try:
		x4.estimate(init_param, data_array, method = 'FB', max_iter=1, is_exit=True)
	except:
		pass
	pi_vec = x4.obs_type_info[obs_key]['pi']
	# P(X_1=0, Y_1, E_1, V_1) = 0.1*1.0*(1-0.2) = 0.08
	# P(X_1=1, Y_1, E_1, V_1) = 0.4*0.8*(1-0.1) = 0.288
	# P(X_1=2, Y_1, E_1, V_1) =	0.5*0.05*(1-0.05) = 0.02375
	# prob = 0.08 + 0.288 + 0.02375 = 0.39175
	print(pi_vec[0,0], 0.08/(0.08+0.288+0.02375))
	
	# First transition 
	P_mat = x4.obs_type_info[obs_key]['P']
	# P(X_1=0,X_2=0,Y_2,E_2=0,V_2=1|Y_1,E_1,V_1) = 0.08/0.39175		*1		*1		*(1-0.3)	 	= 0.142948308870453
	# P(X_1=1,X_2=1,Y_2,E_2=0,V_2=1|Y_1,E_1,V_1) = 0.288/0.39175	*0.7	*0.8	*(1-0.15)	 	= 0.349937460114869
	# P(X_1=1,X_2=2,Y_2,E_2=0,V_2=1|Y_1,E_1,V_1) = 0.288/0.39175	*0.3	*0.05	*(1-0.075)	 	= 0.0102003828972559
	# P(X_1=2,X_2=2,Y_2,E_2=0,V_2=1|Y_1,E_1,V_1) = 0.02375/0.39175	*1		*0.05	*(1-0.075)	 	= 0.00280392469687301
	print(P_mat[0][1,1], 0.349937460114869/(0.142948308870453+0.349937460114869+0.0102003828972559+.00280392469687301))
	
	# marginal period 2
	#P(X1=0, X2=0, Y1=0, Y2=0, E2=0, V_1=0, V_2=0) = 0.1*1	*1.0*1.0	*(1-0.2)*(1-0.3)	 = .056
	#P(X1=1, X2=1, Y1=0, Y2=0, E2=0, V_1=0, V_2=0) = 0.4*0.7*0.8*0.8	*(1-0.1)*(1-0.15)	 = .137088
	#P(X1=1, X2=2, Y1=0, Y2=0, E2=0, V_1=0, V_2=0) = 0.4*0.3*0.8*0.05	*(1-0.1)*(1-0.15)	 = .003672
	#P(X1=2, X2=2, Y1=0, Y2=0, E2=0, V_1=0, V_2=0) = 0.5*1	*0.05*0.05	*(1-0.05)*(1-0.075)	 = .0010984375

	# .056 + .137088 + .003672 + .0010984375 = 0.1978584375
	print(pi_vec[1,1], (.137088+.003672)/0.1978584375)
	
	# marginal period 3
	print(pi_vec[2,0], 0.0252/0.0427330802109375)
	ipdb.set_trace()
	
	########################
	#  Multiple Item Test  #
	########################
	s = [0.05,0.3]
	g = [0.2, 0.4]
	pi = 0.4
	l = [0.3, 0.43]
	h0 = [0.1, 0.2, 0.3]
	h1 = [0, 0.05, 0.1]
	
	J = [0,1,0]
	nJ = 2	
	
	state_init_dist = np.array([1-pi, pi]) 
	state_transit_matrix =  np.stack([ np.array([[[1, 0], [0, 1]],]) for j in range(nJ)] )
	observ_prob_matrix =  np.stack([ np.array([[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
	hazard_matrix = np.array([h0, h1])	
	############ DG|Likelihood
	X = [0,1,1] 
	O = [1,0,1]
	E = 0
	#px = 0.6*0.43*1
	#po = 0.2*0.3*0.95
	#pa = (1-0.1)*(1-0.05)*(1-0.1)
	#prob = 0.6*0.43*1*0.2*0.3*0.95*(1-0.1)*(1-0.05)*(1-0.1)
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.011316267])
	
	E = 1
	#pa = (1-0)*(1-0.2)*0.1
	#prob = 0.6*0.43*1*0.2*0.3*0.95*(1-0.1)*(1-0.05)*0.1
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.001257363])
	
	E = 0
	X = [1,1,1]
	# px = 0.4*1*1
	# po = 0.95*0.3*0.95
	# pa = (1-0.0)*(1-0.05)*(1-0.1)
	# prob = 0.4*1*1*0.95*0.3*0.95*(1-0.0)*(1-0.05)*(1-0.1)
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.0925965])
	
	E = 1
	X = [1,1,1]
	# px = 0.4*0.7*1
	# po = 0.95*0.3*0.95
	# pa = 1*0.8*0.1
	# prob = 0.4*1*1*0.95*0.3*0.95*(1-0.0)*(1-0.05)*0.1
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.0102885])	
	
	ipdb.set_trace()
	
	############ DG|Marginal Probability
	data_array = [(0,0,1,1,0),(0,1,0,0,0),(0,2,0,1,1)] # E=1, O=[1,0,1], J = [0,1,0]
	init_param = {'s':s,
			  'g':g, 
			  'pi':pi,
			  'l':l,
			  'h0':h0,
			  'h1':h1
			  }
	
	x1 = BKT_HMM()
	x1.estimate(init_param, data_array, method = 'DG', max_iter=1)
	
	llk_vec = np.array( x1.obs_type_info['1-1|0|1-1|0|0-1|1|1']['llk_vec'] )
	X_mat = generate_possible_states(3)
	
	# all four possible states are 
	# 1,1,1: 0.4*1*1*	0.7*0.05*0.95*		1*0.95*0.1 = 0.0012635
	# 0,1,1: 0.6*0.43*1*   0.4*0.05*0.95*	0.9*0.95*0.1 = 0.000419121
	# 0,0,1: 0.6*0.57*0.3* 0.4*0.8*0.95*	0.9*0.8*0.1 = 0.0022457088
	# 0,0,0: 0.6*0.57*0.7* 0.4*0.8*0.2*		0.9*0.8*0.3 = 0.0033094656
	
	#P(O,E) = 0.0072377954
	
	# single state marginal
	print(get_single_state_llk(X_mat, llk_vec, 0, 0)/llk_vec.sum(), (0.000419121+0.0022457088+0.0033094656)/0.0072377954) # 0.62645
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), 0.0012635/0.0072377954) # 0.37355
	print(get_single_state_llk(X_mat, llk_vec, 1, 0)/llk_vec.sum(), (0.0022457088+0.0033094656)/0.0072377954) # 0.59106
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), (0.0012635+0.000419121)/0.0072377954) # 0.40894
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 0)/llk_vec.sum(), (0.0022457088+0.0033094656)/0.0072377954)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 1)/llk_vec.sum(), 0.000419121/0.0072377954)# 0.0001368/(0.0001368+0.00153216+0.00075264)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 0)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), 0.0012635/0.0072377954)# 1
	
	################ FB
	# the last pi vector should be the posterior distribution of state in the last period
	s = [0.05,0.3]
	g = [0.2, 0.4]
	pi = 0.4
	l = [0.3, 0.43]
	h0 = [0.1, 0.2, 0.3]
	h1 = [0, 0.05, 0.1]
	init_param = {'s':s,
		  'g':g, 
		  'pi':pi,
		  'l':l,
		  'h0':h0,
		  'h1':h1
		  }
	
	x2 = BKT_HMM()
	x2.estimate(init_param, data_array, method = 'FB', max_iter=1)
	
	X_mat = generate_possible_states(3)	
	pi_vec = x2.obs_type_info['1-1|0|1-1|0|0-1|1|1']['pi']
	# P(X_1=1,Y_1=1,E_1=0) = 0.4*0.7*1 = 0.28
	# P(X_1=0,Y_1=1,E_1=0) = 0.6*0.4*0.9 = 0.216
	print(pi_vec[0,0], 0.435483870968)
	
	# First transition 
	P_mat = x2.obs_type_info['1-1|0|1-1|0|0-1|1|1']['P']
	# P(X_2=1,X_1=0,Y_1=1,Y_2=0,E_2=0) = 0.435483870968*0.43*0.05*0.95  = 0.0088947580645214
	# P(X_2=0,X_1=0,Y_1=1,Y_2=0,E_2=0) = 0.435483870968*0.57*0.8*0.8 = 0.158864516129126
	# P(X_2=1,X_1=1,Y_1=1,Y_2=0,E_2=0) =  (1-0.435483870968)*1*0.05*0.95 = 0.02681451612902
	print(P_mat[0][0,0], 0.158864516129126/0.194573790322667) 
	
	# marginal period 2
	#P(X2=1,X1=1,Y1=1,Y2=0,E2=0) = 0.4*	1  *0.7 *0.05*(1-0)*(1-0.05) = 0.0133
	#P(X2=1,X1=0,Y1=1,Y2=0,E2=0) = 0.6*0.43*0.4 *0.05*(1-0.1)*(1-0.05) = .0044118
	#P(X2=0,X1=0,Y1=1,Y2=0,E2=0) = 0.6*0.57*0.4 *0.8*(1-0.1)*(1-0.2) = .0787968
	print(pi_vec[1,0], 0.0787968/0.0965086)
	
	# marginal period 3
	print(pi_vec[2,0], 0.0033094656/0.0072377954)
	
	
	#############################
	#  Multiple Item with Skip  #
	#############################
		