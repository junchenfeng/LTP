import numpy as np
from collections import defaultdict
from tqdm import tqdm
import ipdb
import copy

import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)

from BKT.prop_hazard_ars import ars_sampler

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

def state_llk(X, J, V, init_dist, transit_matrix):
	# X: vector of latent state, list
	# transit matrix is np array [t-1,t]
	prob = init_dist[X[0]]*np.product([transit_matrix[J[t], V[t], X[t-1], X[t]] for t in range(1,len(X))])
	return prob
	
def likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix):
	# X:  Latent state
	# O: observation
	# E: binary indicate whether the spell is ended
	T = len(X)
	# P(E|X)
	h = np.array([hazard_matrix[X[t],t] for t in range(T)])
	pa = survivial_llk(h,E)
	
	# P(O|X)
	po = 1
	for t in range(T):
		if V[t]:
			po *= observ_prob_matrix[J[t],X[t],O[t]]
		else:
			po *= observ_prob_matrix[J[t],0,O[t]]
	# P(V|X)
	pv = np.product([valid_prob_matrix[J[t],X[t],V[t]] for t in range(T)])
	# P(X)
	px = state_llk(X, J, V, state_init_dist, state_transit_matrix)
	return pa*po*px*pv
	
def generate_possible_states(T):
	# because of the left-right constraints, the possible state is T+1
	X_mat = np.ones([T+1,T], dtype=np.int)
	for t in range(1,T+1):
		X_mat[t,:t]=0
	return X_mat

def get_llk_all_states(X_mat, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix):
	N_X = X_mat.shape[0]
	llk_vec = []
	
	for i in range(N_X):
		X = [int(x) for x in X_mat[i,:].tolist()]
		llk_vec.append( likelihood(X, O, J,V, E, hazard_matrix, observ_prob_matrix, state_init_dist,state_transit_matrix, valid_prob_matrix) )
		
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
		self.h0 = [self.Lambda*np.exp(self.betas[1]*t) for t in range(self.T)]
		self.h1 = [h*np.exp(self.betas[0]) for h in self.h0]		
		self._update_derivative_parameter()  # learning spead
		self._collapse_obser_state()

	def __update_pi(self, t, E, V, observ, item_id, pi_vec, P_mat):
		# pi(i,t) = P(X_t=i|O1,...,O_t,\theta)
		if t == 0:
			if not E:
				pa0 = 1-self.hazard_matrix[0, t]
				pa1 = 1-self.hazard_matrix[1, t]
			else:
				pa0 = self.hazard_matrix[0, t]
				pa1 = self.hazard_matrix[1, t]
				
			po1 = self.observ_prob_matrix[item_id, 1*V, observ]
			po0 = self.observ_prob_matrix[item_id, 0, observ]	
			
			# pi(i,0) = P(X_0=i|O0,\theta)
			p0y = (1-self.pi) * po0  * pa0 * self.valid_prob_matrix[item_id, 0,V]
			p1y = self.pi     * po1 * pa1 * self.valid_prob_matrix[item_id, 1,V]
			py = p0y+p1y
			
		
			pi_vec[t,0] = p0y/py
			pi_vec[t,1] = p1y/py

		else:
			# pi(i,t) = sum_{j} P(j,i,t) where P(j,i,t) is the (j,i)the element of transition matrix P
			pi_vec[t,:] = P_mat[t-1,:,:].sum(axis=0)
		
		return pi_vec
			
	def __update_P(self, t, E, item_id_l, V, observ, item_id_O, pi_vec, P_mat):
		p_raw = np.zeros((2,2))
		if not E:
			pa0 = 1-self.hazard_matrix[0, t+1]
			pa1 = 1-self.hazard_matrix[1, t+1]
		else:
			pa0 = self.hazard_matrix[0, t+1]
			pa1 = self.hazard_matrix[1, t+1]

			
		po1 = self.observ_prob_matrix[item_id_O, 1*V, observ]
		po0 = self.observ_prob_matrix[item_id_O, 0  , observ]			
		
		pv0 =  self.valid_prob_matrix[item_id_O, 0, V]
		pv1 =  self.valid_prob_matrix[item_id_O, 1, V]
			
		p_raw[0,0] = max(pi_vec[t,0] * self.state_transit_matrix[item_id_l,V,0,0] * po0 * pa0 * pv0, 0.0)
		p_raw[0,1] = max(pi_vec[t,0] * self.state_transit_matrix[item_id_l,V,0,1] * po1 * pa1 * pv1, 0.0)
		#p_raw[1,0] = max(pi_vec[t,1] * self.state_transit_matrix[item_id_l,V,1,0] * po0 * pa0 * pv0 , 0.0)  # no forget
		p_raw[1,1] = max(pi_vec[t,1] * self.state_transit_matrix[item_id_l,V,1,1] * po1 * pa1 * pv1, 0.0)
		

		P_mat[t,:,:] = p_raw/p_raw.sum()

		
		return P_mat		
		
	def _update_derivative_parameter(self):
		self.state_init_dist = 		np.array([1-self.pi, self.pi]) # initial distribution is invariant to item
		self.state_transit_matrix = np.stack([np.array([[[1,0], [0, 1]], [[1-self.l[j], self.l[j]], [0, 1]]]) for j in range(self.J)]) # if V=0 P(X_t=X_{t-1}) = 1
		self.observ_prob_matrix = 	np.stack([np.array([[1-self.g[j], self.g[j]], [self.s[j], 1-self.s[j]]])  for j in range(self.J)]) # index by state, observ
		self.hazard_matrix = 		np.array([self.h0, self.h1]) # hazard rate is invariant to item
		self.valid_prob_matrix = 	np.stack([np.array([[1-self.e0[j], self.e0[j]],[1-self.e1[j], self.e1[j]]]) for j in range(self.J)]) # index by state, V
	
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

	def __forward_recursion(self):
		for key in self.obs_type_info.keys():
			# get the obseration state			
			Os = self.obs_type_info[key]['O']
			Js = self.obs_type_info[key]['J']
			E = self.obs_type_info[key]['E']
			Vs = self.obs_type_info[key]['V']
			#calculate the exhaustive state probablity
			T = len(Os)
			
			# if there is a only 1 observations, the P matrix does not exist, pi vector will the first observation
			pi_vec = np.zeros((T,2))
			P_mat = np.zeros((T-1,2,2))
			for t in range(T):
				Et = get_E(E,t,T)
				# The learning happens simulateneously with response. Learning in doing.
				pi_vec = self.__update_pi(t, Et, Vs[t], Os[t], Js[t], pi_vec, P_mat)
				if t !=T-1 and T!=1:
					Et = get_E(E,t+1,T-1)
					P_mat = self.__update_P(t, Et,  Js[t+1], Vs[t+1], Os[t+1], Js[t+1], pi_vec, P_mat)
			self.obs_type_info[key]['pi'] = pi_vec
			self.obs_type_info[key]['P'] = P_mat
	
	def __backward_sampling_scheme(self):
		for obs_key in self.obs_type_info.keys():
			pi_vec = self.obs_type_info[obs_key]['pi']
			P_mat = self.obs_type_info[obs_key]['P']
			T = pi_vec.shape[0]
			sample_p_vec = np.zeros((T,2))
			for t in range(T-1,-1,-1):
				if t == T-1:
					sample_p_vec[t,1] = min(pi_vec[t,1],1.0)
				else:
					for x in range(0,2):
						sample_p_vec[t,x] = min(P_mat[t,1,x]/P_mat[t,:,x].sum(), 1.0)
			self.obs_type_info[obs_key]['sample_p'] = sample_p_vec
				
	def _MCMC(self, max_iter, method, fixVal, is_exit=True, is_effort=True):
	
		
		if not is_effort and self.valid_prob_matrix[:,:,0].sum() != 0: 
			raise Exception('Effort rates are not set to 1 while disabled the update in effort parameter.')
		if not is_exit and self.hazard_matrix.sum() != 0: 
			raise Exception('Hazard rates are not set to 0 while disabled the update in hazard parameter.')
		
		if is_exit:
			prop_hazard_mdl = ars_sampler(self.Lambda, self.betas)
			
		self.parameter_chain = np.empty((max_iter, 1+self.J*5+4))
		
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
					X_mat = generate_possible_states(Ti)				
					llk_vec = get_llk_all_states(X_mat, O, J, V, E, self.hazard_matrix, self.observ_prob_matrix, self.state_init_dist, self.state_transit_matrix, self.valid_prob_matrix)
					
					self.obs_type_info[key]['pi'] = get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum()
					self.obs_type_info[key]['llk_vec'] = llk_vec
					self.obs_type_info[key]['l_vec'] = [ get_joint_state_llk(X_mat, llk_vec, t, 0, 1) / get_single_state_llk(X_mat, llk_vec, t-1, 0) for t in range(1,Ti)]
					
				# sample states
				'''
				#diagnostic: Under sim data with correct parameters, the P(O,E) should be equal to the likelihood
				
				for obs_key, cnt in self.obs_type_cnt.items():
					
					print (obs_key, self.obs_type_info[obs_key]['llk_vec'])
				ipdb.set_trace()				
				'''
				X = np.empty((self.T, self.K),dtype=np.int)
				
				for i in range(self.K):
					# check the key
					obs_key = self.obs_type_ref[i]
					pi = self.obs_type_info[obs_key]['pi']
					l_vec = self.obs_type_info[obs_key]['l_vec']
					Vs = self.obs_type_info[obs_key]['V']
					X[0,i] = np.random.binomial(1,pi)
					for t in range(1, self.T_vec[i]):
						if X[t-1,i] == 1:
							X[t,i] = 1
						else:
							if Vs[t]==1:
								# X at t is determined by the transition matrix at t-1
								X[t,i] = np.random.binomial(1,l_vec[t-1])
							else:
								X[t,i]  = 0
					
			elif method == "FB":
				# forward recursion
				self.__forward_recursion()
				
				# backward sampling scheme
				self.__backward_sampling_scheme()

				# backward sampling
				X = np.empty((self.T, self.K), dtype=np.int)
				#init_pis = np.zeros((self.K, 1))
				for k in range(self.K):
					# check for the observation type
					obs_key = self.obs_type_ref[k]
					sample_p_vec = self.obs_type_info[obs_key]['sample_p']
					for t in range(self.T_vec[k]-1,-1,-1):
						if t == self.T_vec[k]-1:
							p = sample_p_vec[t,1]
						else:
							next_state = int(X[t+1,k])
							p = sample_p_vec[t,next_state]
						try:
							X[t,k] = np.random.binomial(1,p)
						except:
							ipdb.set_trace()
				
				
			# Step 2: Update Parameter
			critical_trans = np.zeros((self.J,1),dtype=np.int)
			tot_trans = np.zeros((self.J,1),dtype=np.int)
			obs_cnt = np.zeros((self.J,2,2)) # state,observ

			
			valid_cnt = np.zeros((self.J,2),dtype=np.int)
			valid_state_cnt = np.zeros((self.J,2),dtype=np.int)
			
			for k in range(self.K):
				for t in range(0, self.T_vec[k]):
					l_j = self.item_data[t,k]
					is_v = self.V_array[t,k]
					o_j = self.item_data[t,k]
					# if the transition happens at t, item in t-1 should take the credit
					# The last item does not contribute the the learning rate
					# update l
					if t>0 and X[t-1,k] == 0 and is_v>0:
						#P(X_t=1,X_{t-1}=0,V_t=1)/P(X_{t-1}=0,V_t=1)
						tot_trans[l_j] += 1
						if X[t,k] == 1:
							critical_trans[l_j] += 1
					if t>0:
						#P(V_t=1,X_{t-1}=i)/P(X_{t-1}=i)
						valid_cnt[o_j,0] += is_v*(1-X[t-1,k])
						valid_state_cnt[o_j,0] += (1-X[t-1,k])
						valid_cnt[o_j,1] += is_v*X[t-1,k]
						valid_state_cnt[o_j,1] += X[t-1,k]					
					# update obs_cnt
					obs_cnt[o_j, X[t,k]*is_v, self.observ_data[t,k]] += 1 #P(Y=0,V=1,X=1)/P(X=1,V=1) = s; P(Y_t=1,V_t=0)+P(Y_t=1,V_t=1,X_t=0))/(P(V_t=0)+P(X_t=0,V_t=1)) =g
			
			for j in range(self.J):
				self.l[j] =  np.random.beta(self.prior_param['l'][0]+critical_trans[j], self.prior_param['l'][1]+tot_trans[j]-critical_trans[j])
				self.s[j] =  np.random.beta(self.prior_param['s'][0]+obs_cnt[j,1,0], self.prior_param['s'][1]+obs_cnt[j,1,1])
				self.g[j] =  np.random.beta(self.prior_param['g'][0]+obs_cnt[j,0,1], self.prior_param['g'][1]+obs_cnt[j,0,0])
			
			self.pi = np.random.beta(self.prior_param['pi'][0]+sum(X[0,:]), self.prior_param['pi'][1]+self.K-sum(X[0,:]))
			if is_exit:
				# generate X,D
				hX = []
				hD = []
				for k in range(self.K):
					for t in range(self.T_vec[k]):
						hX.append([X[t,k], t, X[t,k]*t])
						hD.append(self.E_array[t,k])
						
				# do a stratified sampling by t
				M = len(hD)
				nT = np.zeros((self.T,1),dtype=np.int)
				idxT = [[] for t in range(self.T)]
				for hi in range(M):
					t = hX[hi][1]
					nT[t] += 1
					idxT[t].append(hi)
				select_idx = []
				for t in range(self.T):
					idxs = np.random.choice(idxT[t], size = min(round(5000/self.T),nT[t]), replace=False)
					select_idx += idxs.tolist()
					
				
				'''
				# do a quick check on the hazard rate
				h_cnt = np.zeros((self.T,2))
				s_cnt = np.zeros((self.T,2))
				for m in range(len(hX)):
					S,t=hX[m]
					s_cnt[t,S] += 1
					h_cnt[t,S] += hD[m]
				hrates = h_cnt/s_cnt
				'''
				
				# update the proportional hazard model
				prop_hazard_mdl.load(np.array(hX)[select_idx,:], np.array(hD)[select_idx])
				prop_hazard_mdl.Lambda = prop_hazard_mdl.sample_lambda()[-1]
				for k in range(2):
					prop_hazard_mdl.betas[k] = prop_hazard_mdl.sample_beta(k)[-1]	
					
				self.Lambda = prop_hazard_mdl.Lambda
				self.betas = prop_hazard_mdl.betas
				self.h0 = [self.Lambda*np.exp(self.betas[1]*t) for t in range(self.T)]
				self.h1 = [h*np.exp(self.betas[0]) for h in self.h0]
			else:
				self.h0 = [0.0 for t in range(self.T)]
				self.h1 = [0.0 for t in range(self.T)]
			
			
			if is_effort:
				for j in range(self.J):
					self.e0[j] = np.random.beta(self.prior_param['e0'][0]+valid_cnt[j,0], self.prior_param['e0'][1]+valid_state_cnt[j,0]-valid_cnt[j,0])
					self.e1[j] = np.random.beta(self.prior_param['e1'][0]+valid_cnt[j,1], self.prior_param['e1'][1]+valid_state_cnt[j,1]-valid_cnt[j,1])
			
			# imposing constraint on s or g
			if 's' in fixVal:
				for j in range(self.J):
					self.s[j] = fixVal['s']
			if 'g' in fixVal:
				for j in range(self.J):
					self.g[j] = fixVal['g']			
			self.parameter_chain[iter, :] = [self.pi] + self.s + self.g + self.e0 + self.e1 + self.l + [self.Lambda] + self.betas
			self._update_derivative_parameter()

	
	def _get_point_estimation(self, start, end):
		# calcualte the llk for the parameters
		gap = max(int((end-start)/100), 10)
		parameter_candidates = self.parameter_chain[range(start, end, gap), :]
		avg_parameter = parameter_candidates.mean(axis=0).tolist()
		return avg_parameter
				
	def estimate(self, init_param, data_array, method='FB', max_iter=1000, is_exit=True, fixVal={}):
		param = copy.copy(init_param)
		# ALL items share the same prior for now
		self.g = param['g']  # guess
		self.s = param['s']  # slippage
		self.e1 = param['e1'] 
		self.e0 = param['e0']
		self.pi = param['pi']  # initial prob of mastery
		self.l = param['l']  # learn speed
		self.Lambda = param['Lambda']  # harzard rate with response 0
		self.betas = param['betas']  # proportional hazard parametes, hard code to [X,t]
		
		# generate derivative stats
		
		
		if len(self.betas)!=3:
			raise Exception('Wrong specification for proportional hazard model.')
		if not is_exit and self.Lambda !=0:
			raise Exception('Under no exit regime, baseline hazard rate should be zero.')
		
		if self.pi == 0:
			raise Exception('Invalid Prior')
		
		# for now, assume flat prior for the hazard rate
		self.prior_param = {'l': [2, 2],
							's': [1, 2],
							'e0': [2, 2],
							'e1':[2, 2],
							'g': [1, 2],
							'pi':[2, 2]}
		
		self._load_observ(data_array)
		self._MCMC(max_iter, method, fixVal, is_exit)
		res = self._get_point_estimation(int(max_iter/2), max_iter)
		#ipdb.set_trace()
		self.pi = res[0]; 
		self.s = res[1++self.J*0:1+self.J]
		self.g = res[1+self.J:1+self.J*2]
		self.e0 = res[1+self.J*2:1+self.J*3]
		self.e1 = res[1+self.J*3:1+self.J*4]
		self.l = res[1+self.J*4:1+self.J*5]
		self.Lambda = res[1+self.J*5:1+self.J*5+1]
		self.betas = res[1+self.J*5+1:]	
		
		return self.pi, self.s, self.g, self.e0, self.e1, self.l, self.Lambda, self.betas
		
if __name__=='__main__':
	# UNIT TEST
	# check for the marginal
	s = [0.05]
	g = [0.2]
	pi = 0.7
	l = [0.3]
	h0 = [0.1, 0.2, 0.3]
	h1 = [0.05, 0.1, 0.15]
	e0 = [0.75]
	e1 = [0.9]
	J = [0,0,0]
	V = [1,1,1]
	nJ = 1
	h1 = [0.0, 0.0, 0.0]
	h0 = [0.0, 0.0, 0.0]
	state_init_dist = np.array([1-pi, pi]) 
	state_transit_matrix = np.stack([ np.array([[[1, 0], [0, 1]],[[1-l[j], l[j]], [0, 1]]]) for j in range(nJ)] )
	observ_prob_matrix =  np.stack([ np.array([[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
	hazard_matrix = np.array([h0, h1])	
	valid_prob_matrix = 	np.stack([np.array([[1-e0[j],e0[j]], [1-e1[j], e1[j]]]) for j in range(nJ)]) 	
	#0-1|1|1-0|0|0-1|1|1
	E = 0
	O = [1,1,1]
	J = [0,0,0]
	V = [1,1,1]
	# 1,1,1: (1-.3)*1*1*	  0.95*0.95*0.95	*1.0*1.0*1.0 	*0.9*0.9*0.9	= 0.4375184625
	# 0,1,1: .3*0.3*1*   0.2*0.05*0.95*	1.0*1.0*1.0		*0.75*0.9*0.9 	= 0.0005194125
	# 0,0,1: .3*0.7*0.3* 0.2*0.8*0.95*		1.0*1.0*1.0		*0.75*0.75*0.9 	= 0.00484785
	# 0,0,0: .3*0.7*0.7* 0.2*0.8*0.2*		1.0*1.0*1.0		*0.75*0.75*0.75 = 0.0019845
	# 0.4375184625 + 0.0005194125 + 0.00484785 + 0.0019845 = 0.444870225


	
	########################
	#    Single Item Test  #
	########################
	
	# parameter
	s = [0.05]
	g = [0.2]
	pi = 0.4
	l = [0.3]
	h0 = [0.1, 0.2, 0.3]
	h1 = [0, 0.05, 0.1]
	e0 = [0.85]
	e1 = [0.9]
	J = [0,0,0]
	V = [1,1,1]
	nJ = 1

	state_init_dist = np.array([1-pi, pi]) 
	state_transit_matrix = np.stack([ np.array([[[1, 0], [0, 1]],[[1-l[j], l[j]], [0, 1]]]) for j in range(nJ)] )
	observ_prob_matrix =  np.stack([ np.array([[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
	hazard_matrix = np.array([h0, h1])	
	valid_prob_matrix = 	np.stack([np.array([[1-e0[j],e0[j]], [1-e1[j], e1[j]]]) for j in range(nJ)]) 
	############ DG|Likelihood
	X = [0,1,1] 
	O = [1,0,1]
	E = 0
	V = [1,1,1]
	#px = 0.6*0.3*1
	#po = 0.2*0.05*0.95
	#pa = (1-0.1)*(1-0.05)*(1-0.1)
	#pv = 0.85*0.9*0.9
	#prob = 0.6*0.3*1*0.2*0.05*0.95*(1-0.1)*(1-0.05)*(1-0.1)*0.85*0.9*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix)
	print([prob,0.0009059592825])
	
	E = 1
	#pa = (1-0)*(1-0.2)*0.1
	#prob = 0.6*0.3*1*0.2*0.05*0.95*(1-0.1)*(1-0.05)*0.1*0.85*0.9*0.9
	
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix)
	print([prob,0.0001006621425])
	
	E = 0
	X = [1,1,1]
	# px = 0.4*1*1
	# po = 0.95*0.05*0.95
	# pa = (1-0.0)*(1-0.05)*(1-0.1)
	# pv = 0.9*0.9*0.9
	# prob = 0.4*1*1*0.95*0.05*0.95*(1-0.0)*(1-0.05)*(1-0.1)*0.9*0.9*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix)
	print([prob,0.01125047475])
	
	E = 1
	X = [1,1,1]
	# px = 0.4*1*1
	# po = 0.95*0.05*0.95
	# pa = 1*0.95*0.1
	# pv = 0.9*0.9*0.9
	# prob = 0.4*1*1*0.95*0.05*0.95*(1-0.0)*(1-0.05)*0.1*0.9*0.9*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix)
	print([prob,0.00125005275])	
	
	E = 1
	X = [1,1,1]
	V = [0,1,1]
	# px = 0.4*1*1
	# po = 0.2*0.05*0.95
	# pa = 1*0.95*0.1
	# pv = 0.1*0.9*0.9
	# prob = 0.4*1*1*0.2*0.05*0.95*(1-0.0)*(1-0.05)*0.1*0.1*0.9*0.9
	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix)
	print([prob, 2.9241E-05])	
	
	E = 1
	X = [0,0,1]
	V = [0,0,1]
	# px = 0.6*1*0.3
	# po = 0.2*0.8*0.95
	# pa = 0.9*0.8*0.1
	# pv = 0.15*0.15*0.9
	# prob = 0.6*1*0.3*0.2*0.8*0.95*0.9*0.8*0.1*0.15*0.15*0.9

	prob = likelihood(X, O, J, V, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix, valid_prob_matrix)
	print([prob, 3.989088E-05])		
	


	
	
	############ DG|Marginal Probability
	data_array = [(0,0,0,1,0),(0,1,0,0,0),(0,2,0,1,1)] # E=1, O=[1,0,1], J = [0,0,0], V=[1,1,1]
	init_param = {'s':s,
			  'g':g, 
			  'e1':e1,
			  'e0':e0,
			  'pi':pi,
			  'l':l,
			  'h0':h0,
			  'h1':h1
			  }
	
	x1 = BKT_HMM()
	x1.estimate(init_param, data_array, method = 'DG', max_iter=1)
	
	llk_vec = np.array( x1.obs_type_info['1-1|0|1-0|0|0-1|1|1']['llk_vec'] )
	X_mat = generate_possible_states(3)
	
	# all four possible states are 
	# 1,1,1: 0.4*1*1*	0.95*0.05*0.95*		1*0.95*0.1 	*0.9*0.9*0.9= 0.00125005275
	# 0,1,1: 0.6*0.3*1*   0.2*0.05*0.95*	0.9*0.95*0.1*0.85*0.9*0.9 = 0.0001006621425
	# 0,0,1: 0.6*0.7*0.3* 0.2*0.8*0.95*		0.9*0.8*0.1	*0.85*0.85*0.9 = 0.000896658336
	# 0,0,0: 0.6*0.7*0.7* 0.2*0.8*0.2*		0.9*0.8*0.3	*0.85*0.85*0.85 = 0.001247980608
	
	#P(O,E) = .00125005275+.0001006621425+.000896658336+.001247980608=0.0034953538365
	
	# single state marginal
	print(get_single_state_llk(X_mat, llk_vec, 0, 0)/llk_vec.sum(), (0.0001006621425+0.000896658336+0.001247980608)/0.0034953538365) # 0.62645
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), 0.00125005275/0.0034953538365) # 0.37355
	print(get_single_state_llk(X_mat, llk_vec, 1, 0)/llk_vec.sum(), (0.000896658336+0.001247980608)/0.0034953538365) # 0.59106
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), (0.0001006621425+0.00125005275)/0.0034953538365) # 0.40894
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 0)/llk_vec.sum(), (0.000896658336+0.001247980608)/0.0034953538365)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 1)/llk_vec.sum(), 0.0001006621425/0.0034953538365)# 0.0001368/(0.0001368+0.00153216+0.00075264)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 0)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), 0.00125005275/0.0034953538365)# 1
	
	################ FB
	# the last pi vector should be the posterior distribution of state in the last period
	s = [0.05]
	g = [0.2]
	pi = 0.4
	l = [0.3]
	h0 = [0.1, 0.2, 0.3]
	h1 = [0, 0.05, 0.1]
	e1 = [0.9]
	e0 = [0.85]
	init_param = {'s':s,
		  'g':g, 
		  'pi':pi,
		  'l':l,
		  'e1':e1,
		  'e0':e0,
		  'h0':h0,
		  'h1':h1
		  }
	
	x2 = BKT_HMM()
	x2.estimate(init_param, data_array, method = 'FB', max_iter=1)
	
	X_mat = generate_possible_states(3)	
	pi_vec = x2.obs_type_info['1-1|0|1-0|0|0-1|1|1']['pi']
	# P(X_1=1,Y_1,O_1,V_1) = 0.4*0.95*1*0.9 = 0.342
	# P(X_1=0,Y_1,O_1,V_1) = 0.6*0.2*0.9*0.85 = 0.0918
	# 0.0918/(0.0918+0.342)
	print(pi_vec[0,0],0.211618257261411)
	
	# First transition 
	P_mat = x2.obs_type_info['1-1|0|1-0|0|0-1|1|1']['P']
	# P(X_1=0,X_1=1,Y_1,Y_2,E_2=0,V_1=1,V_2=1) = 0.211618257261411		*0.3	*0.05	*0.95	*0.9 	= 0.0027140041493776
	# P(X_1=0,X_2=0,Y_1,Y_2,E_2=0,V_1=1,V_2=1) = 0.211618257261411		*0.7	*0.8	*0.8	*0.85 	= 0.0805842323651453
	# P(X_1=1,X_2=1,Y_1,Y_2,E_2=0,V_1=1,V_2=1) = (1-0.211618257261411)	*1		*0.05	*0.95	*0.9 	= 0.0337033195020747
	print(P_mat[0][0,0], 0.688744962961978) # 0.0805842323651453/(0.0027140041493776+0.0805842323651453+0.0337033195020747)
	
	# marginal period 2
	#P(X1=0,X2=1,Y1=1,Y2=0,E2=0,V_1=1,V_2=1) = 0.6*0.3	*0.2*0.05	*(1-0.1)*(1-0.05)	*0.85*0.9 = .001177335
	#P(X1=0,X2=0,Y1=1,Y2=0,E2=0,V_1=1,V_2=1) = 0.6*0.7	*0.2*0.8	*(1-0.1)*(1-0.2)	*0.85*0.85 = .03495744
	#P(X1=1,X2=1,Y1=1,Y2=0,E2=0,V_1=1,V_2=1) = 0.4*1	*0.95*0.05	*(1-0)*(1-0.05)		*0.9*0.9 = .0146205

	# .0146205 + .001177335 + .03495744 = 0.050755275
	print(pi_vec[1,0], 0.03495744/0.050755275)
	
	# marginal period 3
	print(pi_vec[2,0], 0.001247980608/0.0034953538365)
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
	# px = 0.4*1*1
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
		