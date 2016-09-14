import numpy as np
from collections import defaultdict
from tqdm import tqdm
import ipdb

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

def state_llk(X, J, init_dist, transit_matrix):
	# X: vector of latent state, list
	# transit matrix is np array [t-1,t]
	prob = init_dist[X[0]] * np.product([transit_matrix[J[t], X[t-1], X[t]] for t in range(1, len(X))])

	return prob
	
def likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix):
	# X:  Latent state
	# O: observation
	# E: binary indicate whether the spell is ended
	T = len(X)
	# P(E|X)
	h = np.array([hazard_matrix[X[t],t] for t in range(T)])
	pa = survivial_llk(h,E)
	
	# P(O|X)
	po = np.product([observ_prob_matrix[J[t],X[t],O[t]] for t in range(T)])

	# P(X)
	px = state_llk(X, J, state_init_dist, state_transit_matrix)
	
	return pa*po*px
	
def generate_possible_states(T):
	# because of the left-right constraints, the possible state is T+1
	X_mat = np.ones([T+1,T], dtype=np.int)
	for t in range(1,T+1):
		X_mat[t,:t]=0
	return X_mat

def get_llk_all_states(X_mat, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix):
	N_X = X_mat.shape[0]
	llk_vec = []
	
	for i in range(N_X):
		X = [int(x) for x in X_mat[i,:].tolist()]
		llk_vec.append( likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist,state_transit_matrix) )
		
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
	
class BKT_HMM_SURVIVAL(object):

	def _load_observ(self, data):
		# data = [(i,t,j,y,e)]  
		# i learner id from 0:N-1
		# t sequence id, t starts from 0
		# j	item id, from 0:J-1
		# y response, 0 or 1
		# e if the spell ends here
		self.K = len(set([x[0] for x in data]))
		self.T = max([x[1] for x in data]) + 1
		self.J = len(set([x[2] for x in data]))
		
		self.E_array = np.empty((self.T, self.K))
		self.observ_data = np.empty((self.T, self.K), dtype=np.int)
		self.item_data = np.empty((self.T, self.K), dtype=np.int)
		T_array = np.zeros((self.K,))
		
		for log in data:
			if len(log)==4:
				# The spell never ends; multiple item
				i = log[0]; t = log[1]; j = log[2]; y = log[3]; is_e = 0
			elif len(log)==5:
				i,t,j,y,is_e = log
			else:
				raise Exception('The log format is not recognized.')
			self.observ_data[t, i] = y
			self.item_data[t, i] = j
			self.E_array[t, i] = is_e
			T_array[i] = t
		
		# This section is used to collapse states
		self.T_vec = [int(x)+1 for x in T_array.tolist()] 
		self.O_data = []
		for i in range(self.K):
			self.O_data.append( [x for x in self.observ_data[0:self.T_vec[i],i].tolist()] )
		self.J_data = []
		for i in range(self.K):
			self.J_data.append( [x for x in self.item_data[0:self.T_vec[i],i].tolist()] )		
		self.E_vec = [int(self.E_array[self.T_vec[i]-1, i]) for i in range(self.K)]		
				
		# initialize
		self._update_derivative_parameter()  # learning spead
		self._collapse_obser_state()

	def __update_pi(self, t, E, observ, item_id, pi_vec, P_mat):
		# pi(i,t) = P(X_t=i|O1,...,O_t,\theta)
		if t == 0:
			if not E:
				pa0 = 1-self.hazard_matrix[0, t]
				pa1 = 1-self.hazard_matrix[1, t]
			else:
				pa0 = self.hazard_matrix[0, t]
				pa1 = self.hazard_matrix[1, t]
			# pi(i,0) = P(X_0=i|O0,\theta)
			p0y = (1-self.pi) * self.observ_prob_matrix[item_id, 0, observ] * pa0
			p1y = self.pi     * self.observ_prob_matrix[item_id, 1, observ] * pa1
			py = p0y+p1y
			pi_vec[t,0] = p0y/py
			pi_vec[t,1] = p1y/py
		else:
			# pi(i,t) = sum_{j} P(j,i,t) where P(j,i,t) is the (j,i)the element of transition matrix P
			pi_vec[t,:] = P_mat[t-1,:,:].sum(axis=0)
		
		return pi_vec
			
	def __update_P(self, t, E, observ, item_id, pi_vec, P_mat):
		p_raw = np.zeros((2,2))
		
		if not E:
			pa0 = 1-self.hazard_matrix[0, t+1]
			pa1 = 1-self.hazard_matrix[1, t+1]
		else:
			pa0 = self.hazard_matrix[0, t+1]
			pa1 = self.hazard_matrix[1, t+1]
			
		p_raw[0,0] = max(pi_vec[t,0] * self.state_transit_matrix[item_id,0,0] * self.observ_prob_matrix[item_id,0,observ] * pa0, 0.0)
		p_raw[0,1] = max(pi_vec[t,0] * self.state_transit_matrix[item_id,0,1] * self.observ_prob_matrix[item_id,1,observ] * pa1, 0.0)
		#p_raw[1,0] = max(pi_vec[t,1] * self.state_transit_matrix[item_id,1,0] * self.observ_prob_matrix[item_id,0,observ] * pa0, 0.0)  # no forget
		p_raw[1,1] = max(pi_vec[t,1] * self.state_transit_matrix[item_id,1,1] * self.observ_prob_matrix[item_id,1,observ] * pa1, 0.0)
		
		P_mat[t,:,:] = p_raw/p_raw.sum()
		return P_mat		
		
	def _update_derivative_parameter(self):
		self.state_init_dist = 		np.array([1-self.pi, self.pi]) # initial distribution is invariant to item
		self.state_transit_matrix = np.stack([np.array([[1-self.l[j], self.l[j]], [0, 1]]) for j in range(self.J)])
		self.observ_prob_matrix = 	np.stack([np.array([[1-self.g[j], self.g[j]], [self.s[j], 1-self.s[j]]])  for j in range(self.J)]) # index by state, observ
		self.hazard_matrix = 		np.array([self.h0, self.h1]) # hazard rate is invariant to item
	
	def _collapse_obser_state(self):
		self.obs_type_cnt = defaultdict(int)
		self.obs_type_ref = {}
		for k in range(self.K):
			obs_type_key = str(self.E_vec[k]) + '-' + '|'.join(str(y) for y in self.O_data[k]) + '-' + '|'.join(str(j) for j in self.J_data[k])
			self.obs_type_cnt[obs_type_key] += 1
			self.obs_type_ref[k] = obs_type_key
		# construct the space
		self.obs_type_info = {}
		for key in self.obs_type_cnt.keys():
			e_s, O_s, J_s = key.split('-')
			self.obs_type_info[key] = {'E':int(e_s), 'O':[int(x) for x in O_s.split('|')], 'J':[int(x) for x in J_s.split('|')]}

	def __forward_recursion(self):
		for key in self.obs_type_info.keys():
			# get the obseration state			
			Os = self.obs_type_info[key]['O']
			Js = self.obs_type_info[key]['J']
			E = self.obs_type_info[key]['E']
			#calculate the exhaustive state probablity
			T = len(Os)
			
			# if there is a only 1 observations, the P matrix does not exist, pi vector will the first observation

			pi_vec = np.zeros((T,2))
			P_mat = np.zeros((T-1,2,2))
			for t in range(T):
				Et = get_E(E,t,T)
				pi_vec = self.__update_pi(t,  Et, Os[t], Js[t], pi_vec, P_mat)
				if t !=T-1 and T!=1:
					Et = get_E(E,t+1,T-1)
					P_mat = self.__update_P(t, Et, Os[t+1], Js[t+1], pi_vec, P_mat)
			self.obs_type_info[key]['pi'] = pi_vec
			self.obs_type_info[key]['P'] = P_mat
		
			
	def _MCMC(self, max_iter, method, is_exit=True):
		self.parameter_chain = np.empty((max_iter, 1+self.J*3+self.T*2))
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
					
					#calculate the exhaustive state probablity
					Ti = len(O)					
					X_mat = generate_possible_states(Ti)
					llk_vec = get_llk_all_states(X_mat, O, J, E, self.hazard_matrix, self.observ_prob_matrix, self.state_init_dist, self.state_transit_matrix)
					
					self.obs_type_info[key]['pi'] = get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum()
					self.obs_type_info[key]['llk_vec'] = llk_vec
					self.obs_type_info[key]['l_vec'] = [ get_joint_state_llk(X_mat, llk_vec, t, 0, 1) / get_single_state_llk(X_mat, llk_vec, t-1, 0) for t in range(1,Ti)]
					
				# sample states
				X = np.empty((self.T, self.K),dtype=np.int)
				
				for i in range(self.K):
					# check the key
					obs_key = self.obs_type_ref[i]
					pi = self.obs_type_info[obs_key]['pi']
					l_vec = self.obs_type_info[obs_key]['l_vec']
					
					X[0,i] = np.random.binomial(1,pi)
					for t in range(1, self.T_vec[i]):
						if X[t-1,i] == 1:
							X[t,i] = 1
						else:
							X[t,i] = np.random.binomial(1,l_vec[t-1])
					
			elif method == "FB":
				# forward recursion
				self.__forward_recursion()		
				# backward sampling
				X = np.empty((self.T, self.K), dtype=np.int)
				init_pis = np.zeros((self.K, 1))
				for k in range(self.K):
					# check for the observation type
					obs_key = self.obs_type_ref[k]
					pi_vec = self.obs_type_info[obs_key]['pi']
					P_mat = self.obs_type_info[obs_key]['P']
					for t in range(self.T_vec[k]-1,-1,-1):
						if t == self.T_vec[k]-1:
							p = pi_vec[t,1]
						else:
							next_state = int(X[t+1,k])
							p = P_mat[t,1,next_state]/P_mat[t,:,next_state].sum()
						if t == 0:
							init_pis[k] = p
						X[t,k] = np.random.binomial(1,p)
						
			# Step 2: Update Parameter
			critical_trans = np.zeros((self.J,1),dtype=np.int)
			tot_trans = np.zeros((self.J,1),dtype=np.int)
			obs_cnt = np.zeros((self.J,2,2)) # state,observ
			drop_cnt = np.zeros((2, self.T)) # t,observ
			survive_cnt = np.zeros((2, self.T))
			
			for k in range(self.K):
				for t in range(0, self.T_vec[k]):
					j = self.item_data[t,k]
					# update l
					if t>0 and X[t-1,k] == 0:
						tot_trans[j] += 1
						if X[t,k] == 1:
							critical_trans[j] += 1
					# update obs_cnt
					obs_cnt[j, X[t,k], self.observ_data[t,k]] += 1
			
			for t in range(self.T):
				# for data survived in last period, check the harzard rate
				for k in range(self.K):
					if self.T_vec[k]>=t and  self.E_array[t-1,k] == 0:
						drop_cnt[X[t,k], t] += self.E_array[t,k]
						survive_cnt[X[t,k], t] += 1-self.E_array[t,k]
			#ipdb.set_trace()
			for j in range(self.J):
				self.l[j] =  np.random.beta(self.prior_param['l'][0]+critical_trans[j], self.prior_param['l'][1]+tot_trans[j]-critical_trans[j])
				self.s[j] =  np.random.beta(self.prior_param['s'][0]+obs_cnt[j,1,0], self.prior_param['s'][1]+obs_cnt[j,1,1])
				self.g[j] =  np.random.beta(self.prior_param['g'][0]+obs_cnt[j,0,1], self.prior_param['g'][1]+obs_cnt[j,0,0])
			self.pi = np.random.beta(self.prior_param['pi'][0]+sum(X[0,:]), self.prior_param['pi'][1]+self.K-sum(X[0,:]))
			
			if is_exit:
				self.h0 = [ np.random.beta(self.prior_param['h0'][0] + drop_cnt[0, t], self.prior_param['h0'][1] + survive_cnt[0, t]) for t in range(self.T)]
				self.h1 = [ np.random.beta(self.prior_param['h1'][0] + drop_cnt[1, t], self.prior_param['h1'][1] + survive_cnt[1, t]) for t in range(self.T)]
			
			self.parameter_chain[iter, :] = [self.pi] + self.s + self.g  + self.l + self.h0 + self.h1
			self._update_derivative_parameter()
	'''			
	def __get_llk(self, s, g, pi, l, h0, h1):
		# can only calculat the insample fit
		state_init_dist = np.array([1-pi, pi])
		state_transit_matrix = np.array([[1-l, l], [0, 1]])
		observ_prob_matrix = np.array([[1-g, g], [s, 1-s]])  # index by state, observ
		hazard_matrix = np.array([h0, h1])		
		
		type_llk_ref = {}
		
		for key in self.obs_type_info.keys():
			# get the obseration state
			O = self.obs_type_info[key]['O']
			E = self.obs_type_info[key]['E']
			
			#calculate the exhaustive state probablity
			Ti = len(O)					
			X_mat = generate_possible_states(Ti)
			X_llk_vec = get_llk_all_states(X_mat, O, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
			type_llk_ref[key] = np.log(X_llk_vec.sum())
		
		llk = 0
		for k in range(self.K):
			obs_key = self.obs_type_ref[k]
			llk += type_llk_ref[obs_key]
		
		return llk
	'''
	
	def _get_point_estimation(self, start, end):
		# calcualte the llk for the parameters
		gap = max(int((end-start)/100), 10)
		parameter_candidates = self.parameter_chain[range(start, end, gap), :]
		
		'''
		N = parameter_candidates.shape[0]
		llk_vec = np.zeros((N,))
		
		for i in range(N):
			llk_vec[i] = self.__get_llk(parameter_candidates[i,0], parameter_candidates[i,1], parameter_candidates[i,2], parameter_candidates[i,3], parameter_candidates[i,4], parameter_candidates[i,5])
		
		llk_sum = logExpSum(llk_vec)
		parameter_weight = np.exp(llk_vec - llk_sum)
		
		avg_parameter = np.dot(parameter_candidates.transpose(), parameter_weight).tolist()
		'''
		avg_parameter = parameter_candidates.mean(axis=0).tolist()
		return avg_parameter
				
	def estimate(self, param, data_array, method='FB', max_iter=1000, is_exit=True):
		# ALL items share the same prior for now
		self.g = param['g']  # guess
		self.s = param['s']  # slippage
		self.pi = param['pi']  # initial prob of mastery
		self.l = param['l']  # learn speed
		self.h0 = param['h0']  # harzard rate with response 0
		self.h1 = param['h1']  # harzard rate with response 1
		if not is_exit and (sum(self.h0)>0 or sum(self.h1)>0):
			raise Exception('Under no exit regime, no hazard rate allowed.')
		
		# for now, assume flat prior for the hazard rate
		self.prior_param = {'l': [2, 2],
							's': [1, 2],
							'g': [1, 2],
							'pi':[2, 2],
							'h0':[2, 2],
							'h1':[2, 2]}
		
		self._load_observ(data_array)
		self._MCMC(max_iter, method)
		res = self._get_point_estimation(int(max_iter/2), max_iter)
		ipdb.set_trace()
		self.pi = res[0]; self.s = res[1:1+self.J]; self.g = res[1+self.J:1+self.J*2]; self.l = res[1+self.J*2:1+self.J*3]
		self.h0 = res[1+self.J*3:1+self.J*3+self.T]; self.h1 = res[1+self.J*3+self.T:]	
		
		return self.pi, self.s, self.g, self.l, self.h0, self.h1
	
	
	def predict(self, param, data_array):
		raise Exception('Deprecated!')
		# currently only implement for FB algorithm
		self.g = param['g']  # guess
		self.s = param['s']  # slippage
		self.pi = param['pi']  # initial prob of mastery
		self.l = param['l']  # learn speed
		self.h0 = param['h0']  # harzard rate with response 0
		self.h1 = param['h1']  # harzard rate with response 1

		self._load_observ(data_array)
		# the state collapse is different. E is always 0
		self._collapse_obser_state()
		# run forward algorithm
		self.__forward_recursion()
		# predict
		output = []
		for k in range(self.K):
			obs_key = self.obs_type_ref[k]
			pi_vec = self.obs_type_info[obs_key]['pi']
			for t in range(self.T_vec[k]-1):
				x_t_posterior = pi_vec[t,1]
				pyHat = ((1-x_t_posterior)*self.l + x_t_posterior)*(1-self.s) + (1-x_t_posterior)*(1-self.l)*self.g
				y_true = self.observ_data[t+1,k]
				output.append((pyHat, y_true))
		return output	

	def predict_exit(self, param, data_array):
		raise Exception('Deprecated!')
		# currently only implement for FB algorithm
		self.g = param['g']  # guess
		self.s = param['s']  # slippage
		self.pi = param['pi']  # initial prob of mastery
		self.l = param['l']  # learn speed
		self.h0 = param['h0']  # harzard rate with response 0
		self.h1 = param['h1']  # harzard rate with response 1

		self._load_observ(data_array)
		# the state collapse is different. E is always 0
		self._collapse_obser_state()
		# run forward algorithm
		self.__forward_recursion()
		# predict
		output = []
		for k in range(self.K):
			obs_key = self.obs_type_ref[k]
			pi_vec = self.obs_type_info[obs_key]['pi']
			for t in range(self.T_vec[k]-1):
				x_t_posterior = pi_vec[t,1]
				pyHat = ((1-x_t_posterior)*self.l + x_t_posterior)*(1-self.s) + (1-x_t_posterior)*(1-self.l)*self.g
				hHat = pyHat*self.hazard_matrix[1,t+1] + (1-pyHat)*self.hazard_matrix[0,t+1]
				y_true = self.E_array[t+1,k]
				output.append((hHat, y_true))
		return output
		
if __name__=='__main__':
	# UNIT TEST
	
	
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
	
	J = [0,0,0]
	nJ = 1

	state_init_dist = np.array([1-pi, pi]) 
	state_transit_matrix = np.stack([ np.array([[1-l[j], l[j]], [0, 1]]) for j in range(nJ)] )
	observ_prob_matrix =  np.stack([ np.array([[1-g[j], g[j]], [s[j], 1-s[j]]])  for j in range(nJ)] ) # index by state, observ
	hazard_matrix = np.array([h0, h1])	
	
	############ DG|Likelihood
	X = [0,1,1] 
	O = [1,0,1]
	E = 0
	#px = 0.6*0.3*1
	#po = 0.2*0.05*0.95
	#pa = (1-0.1)*(1-0.05)*(1-0.1)
	#prob = 0.6*0.3*1*0.2*0.05*0.95*(1-0.1)*(1-0.05)*(1-0.1)
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.001315845])
	
	E = 1
	#pa = (1-0)*(1-0.2)*0.1
	#prob = 0.6*0.3*1*0.2*0.05*0.95*(1-0.1)*(1-0.05)*0.1
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.000146205])
	
	E = 0
	X = [1,1,1]
	# px = 0.4*1*1
	# po = 0.95*0.05*0.95
	# pa = (1-0.0)*(1-0.05)*(1-0.1)
	# prob = 0.4*1*1*0.95*0.05*0.95*(1-0.0)*(1-0.05)*(1-0.1)
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.1543275])
	
	E = 1
	X = [1,1,1]
	# px = 0.4*1*1
	# po = 0.95*0.05*0.95
	# pa = 1*0.8*0.1
	# prob = 0.4*1*1*0.95*0.05*0.95*(1-0.0)*(1-0.05)*0.1
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.171475])	
	
	
	
	############ DG|Marginal Probability
	data_array = [(0,0,0,1,0),(0,1,0,0,0),(0,2,0,1,1)] # E=1, O=[1,0,1], J = [0,0,0]
	init_param = {'s':s,
			  'g':g, 
			  'pi':pi,
			  'l':l,
			  'h0':h0,
			  'h1':h1
			  }
	
	x1 = BKT_HMM_SURVIVAL()
	x1.estimate(init_param, data_array, method = 'DG', max_iter=1)
	
	llk_vec = np.array( x1.obs_type_info['1-1|0|1-0|0|0']['llk_vec'] )
	X_mat = generate_possible_states(3)
	
	# all four possible states are 
	# 1,1,1: 0.4*1*1*	0.95*0.05*0.95*	1*0.95*0.1 = 0.00171475
	# 0,1,1: 0.6*0.3*1*   0.2*0.05*0.95*	0.9*0.95*0.1 = 0.000146205
	# 0,0,1: 0.6*0.7*0.3* 0.2*0.8*0.95*		0.9*0.8*0.1 = 0.001378944
	# 0,0,0: 0.6*0.7*0.7* 0.2*0.8*0.2*		0.9*0.8*0.3 = 0.002032128
	
	#P(O,E) = 0.005272027
	
	# single state marginal
	print(get_single_state_llk(X_mat, llk_vec, 0, 0)/llk_vec.sum(), (0.000146205+0.001378944+0.002032128)/0.005272027) # 0.62645
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), 0.00171475/0.005272027) # 0.37355
	print(get_single_state_llk(X_mat, llk_vec, 1, 0)/llk_vec.sum(), (0.001378944+0.002032128)/0.005272027) # 0.59106
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), (0.000146205+0.00171475)/0.005272027) # 0.40894
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 0)/llk_vec.sum(), (0.001378944+0.002032128)/0.005272027)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 1)/llk_vec.sum(), 0.000146205/0.005272027)# 0.0001368/(0.0001368+0.00153216+0.00075264)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 0)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), 0.00171475/0.005272027)# 1
	
	
	################ FB
	# the last pi vector should be the posterior distribution of state in the last period
	s = [0.05]
	g = [0.2]
	pi = 0.4
	l = [0.3]
	h0 = [0.1, 0.2, 0.3]
	h1 = [0, 0.05, 0.1]
	init_param = {'s':s,
		  'g':g, 
		  'pi':pi,
		  'l':l,
		  'h0':h0,
		  'h1':h1
		  }
	
	x2 = BKT_HMM_SURVIVAL()
	x2.estimate(init_param, data_array, method = 'FB', max_iter=1)
	
	X_mat = generate_possible_states(3)	
	pi_vec = x2.obs_type_info['1-1|0|1-0|0|0']['pi']
	# P(X_1=1,Y_1,O_1) = 0.4*0.95*1 = 0.38
	# P(X_1=0,Y_1,O_1) = 0.6*0.2*0.9 = 0.108
	print(pi_vec[0,0],0.22131147540983606)
	
	# First transition 
	P_mat = x2.obs_type_info['1-1|0|1-0|0|0']['P']
	# P(X_2=1,X_1=0,Y_1,Y_2,E_2=0) = 0.108/(0.108+0.38)*0.3*0.05*0.95 = 0.003153688524590164
	# P(X_2=0,X_1=0,Y_1,Y_2,E_2=0) = 0.108/(0.108+0.38)*0.7*0.8*0.9 = 0.09914754098360656
	# P(X_2=1,X_1=1,Y_1,Y_2,E_2=0) = 0.38/(0.108+0.38)*0.05*0.95 = 0.036987704918032785
	print(P_mat[0][0,0], 0.7118120430170802) # (0.108/(0.108+0.38)*0.7*0.8*0.8)/((0.108/(0.108+0.38)*0.3*0.05*0.95)+(0.108/(0.108+0.38)*0.7*0.8*0.8)+(0.38/(0.108+0.38)*0.05*0.95))
	
	# marginal period 2
	#P(X2=1,X1=1,Y1=1,Y2=0,E2=0) = 0.4*1*0.95*0.05*(1-0)*(1-0.05) = .01805
	#P(X2=1,X1=0,Y1=1,Y2=0,E2=0) = 0.6*0.3*0.2*0.05*(1-0.1)*(1-0.05) = .048384
	#P(X2=0,X1=0,Y1=1,Y2=0,E2=0) = 0.6*0.7*0.2*0.8*(1-0.1)*(1-0.2) = .067973
	print(pi_vec[1,0], 0.048384/0.067973)
	
	# marginal period 3
	print(pi_vec[2,0], 0.002032128/0.005272027)
	
	
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
	state_transit_matrix = np.stack([ np.array([[1-l[j], l[j]], [0, 1]]) for j in range(nJ)] )
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
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.011316267])
	
	E = 1
	#pa = (1-0)*(1-0.2)*0.1
	#prob = 0.6*0.43*1*0.2*0.3*0.95*(1-0.1)*(1-0.05)*0.1
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.001257363])
	
	E = 0
	X = [1,1,1]
	# px = 0.4*1*1
	# po = 0.95*0.3*0.95
	# pa = (1-0.0)*(1-0.05)*(1-0.1)
	# prob = 0.4*1*1*0.95*0.3*0.95*(1-0.0)*(1-0.05)*(1-0.1)
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.0925965])
	
	E = 1
	X = [1,1,1]
	# px = 0.4*1*1
	# po = 0.95*0.3*0.95
	# pa = 1*0.8*0.1
	# prob = 0.4*1*1*0.95*0.3*0.95*(1-0.0)*(1-0.05)*0.1
	prob = likelihood(X, O, J, E, hazard_matrix, observ_prob_matrix, state_init_dist, state_transit_matrix)
	print([prob,0.102885])	
	
	
	
	############ DG|Marginal Probability
	data_array = [(0,0,0,1,0),(0,1,1,0,0),(0,2,0,1,1)] # E=1, O=[1,0,1], J = [0,1,0]
	init_param = {'s':s,
			  'g':g, 
			  'pi':pi,
			  'l':l,
			  'h0':h0,
			  'h1':h1
			  }
	
	x1 = BKT_HMM_SURVIVAL()
	x1.estimate(init_param, data_array, method = 'DG', max_iter=1)
	
	llk_vec = np.array( x1.obs_type_info['1-1|0|1-0|1|0']['llk_vec'] )
	X_mat = generate_possible_states(3)
	
	# all four possible states are 
	# 1,1,1: 0.4*1*1*	0.95*0.3*0.95*		1*0.95*0.1 = 0.0102885
	# 0,1,1: 0.6*0.43*1*   0.2*0.3*0.95*	0.9*0.95*0.1 = 0.001257363
	# 0,0,1: 0.6*0.57*0.3* 0.2*0.6*0.95*	0.9*0.8*0.1 = 0.0008421408
	# 0,0,0: 0.6*0.57*0.7* 0.2*0.6*0.2*		0.9*0.8*0.3 = 0.0012410496
	
	#P(O,E) = 0.0136290534
	
	# single state marginal
	print(get_single_state_llk(X_mat, llk_vec, 0, 0)/llk_vec.sum(), (0.001257363+0.0008421408+0.0012410496)/0.0136290534) # 0.62645
	print(get_single_state_llk(X_mat, llk_vec, 0, 1)/llk_vec.sum(), 0.0102885/0.0136290534) # 0.37355
	print(get_single_state_llk(X_mat, llk_vec, 1, 0)/llk_vec.sum(), (0.0008421408+0.0012410496)/0.0136290534) # 0.59106
	print(get_single_state_llk(X_mat, llk_vec, 1, 1)/llk_vec.sum(), (0.0102885+0.001257363)/0.0136290534) # 0.40894
	
	# two states marginal
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 0)/llk_vec.sum(), (0.0008421408+0.0012410496)/0.0136290534)# 
	print(get_joint_state_llk(X_mat, llk_vec, 1, 0, 1)/llk_vec.sum(), 0.001257363/0.0136290534)# 0.0001368/(0.0001368+0.00153216+0.00075264)
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 0)/llk_vec.sum(), 0)# 0
	print(get_joint_state_llk(X_mat, llk_vec, 1, 1, 1)/llk_vec.sum(), 0.0102885/0.0136290534)# 1
	
	
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
	
	x2 = BKT_HMM_SURVIVAL()
	x2.estimate(init_param, data_array, method = 'FB', max_iter=1)
	
	X_mat = generate_possible_states(3)	
	pi_vec = x2.obs_type_info['1-1|0|1-0|1|0']['pi']
	# P(X_1=1,Y_1,O_1) = 0.4*0.95*1 = 0.38
	# P(X_1=0,Y_1,O_1) = 0.6*0.2*0.9 = 0.108
	print(pi_vec[0,0],0.22131147540983606)
	
	# First transition 
	P_mat = x2.obs_type_info['1-1|0|1-0|1|0']['P']
	# P(X_2=1,X_1=0,Y_1=1,Y_2=0,E_2=0) = 0.108/(0.108+0.38)*0.43*0.3*0.95  = .0271217213114754
	# P(X_2=0,X_1=0,Y_1=1,Y_2=0,E_2=0) = 0.108/(0.108+0.38)*0.57*0.6*0.8 = .060550819672131134
	# P(X_2=1,X_1=1,Y_1=1,Y_2=0,E_2=0) =  0.38/(0.108+0.38)*1*0.3*0.95 = .221926229508197
	print(P_mat[0][0,0], 0.060550819672131134/0.3095987704918035) 
	# (0.108/(0.108+0.38)*0.57*0.6*0.9)/((0.108/(0.108+0.38)*0.43*0.3*0.95)+(0.108/(0.108+0.38)*0.57*0.6*0.9)+(0.38/(0.108+0.38)*1*0.3*0.95))
	
	# marginal period 2
	#P(X2=1,X1=1,Y1=1,Y2=0,E2=0) = 0.4*	1  *0.95*0.3*(1-0)*(1-0.05) = .1083
	#P(X2=1,X1=0,Y1=1,Y2=0,E2=0) = 0.6*0.43*0.2 *0.3*(1-0.1)*(1-0.05) = .0132354
	#P(X2=0,X1=0,Y1=1,Y2=0,E2=0) = 0.6*0.57*0.2 *0.6*(1-0.1)*(1-0.2) = .0295488
	print(pi_vec[1,0], 0.0295488/0.1510842)
	
	# marginal period 3
	print(pi_vec[2,0], 0.0012410496/0.0136290534)
		