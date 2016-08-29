import numpy as np
import ipdb
from tqdm import tqdm

from collections import defaultdict


# use EM to compute the bayes net
class BKT_HMM(object):
	def __init__(self, init_param, method='EM'):
		self.g = init_param['g']  # guess
		self.s = init_param['s']  # slippage
		self.pi = init_param['pi']  # initial prob of mastery
		self.l = init_param['l']  # learn speed
		self.method = method
		
		self.prior_param = {'l':[10,10],
							's':[5,10],
							'g':[5,10],
							'pi':[10,10]}
			
	def _load_observ(self, data):
		self.K = len(set([x[0] for x in data]))
		self.T = max([x[1] for x in data]) + 1
		
		self.observ_data = np.empty((self.T, self.K))
		T_array = np.zeros((self.K,))
		
		for log in data:
			i = log[0]; t = log[1]; y = log[2]
			self.observ_data[t, i] = y
			T_array[i] = t
		
		self.Tvec = [int(x)+1 for x in T_array.tolist()] 

		self.O_data = []
		for i in range(self.K):
			self.O_data.append( [int(x) for x in self.observ_data[0:self.Tvec[i],i].tolist()] )		
		
		self.T = max(self.Tvec)
		
		# initilize for the rest of the structure
		st_size = (self.T, self.K, 2)
		self.a_vec = np.zeros(st_size)
		self.b_vec = np.zeros(st_size)
		
		if self.method == 'EM':
			self.r_vec = np.zeros(st_size)
			self.eta_vec = np.zeros((self.T, self.K, 2,2))
			self.r_vec_uncond = np.zeros(st_size)
			self.eta_vec_uncond  = np.zeros((self.T, self.K, 2,2))		
			
		# initialize
		self._update_derivative_parameter()  # learning spead
		self._collapse_obser_state()
		
	def _collapse_obser_state(self):
		self.obs_type_cnt = defaultdict(int)
		self.obs_type_ref = {}
		for k in range(self.K):
			obs_type_key = '|'.join(str(y) for y in self.O_data[k])
			self.obs_type_cnt[obs_type_key] += 1
			self.obs_type_ref[k] = obs_type_key
		# construct the space
		self.obs_type_info = {}
		for key in self.obs_type_cnt.keys():
			self.obs_type_info[key] = {'O':[int(x) for x in key.split('|')]}		
		
	def _update_derivative_parameter(self):
		self.transit_matrix = np.array([[1-self.l, self.l], [0, 1]])
		self.state_density = np.array([1-self.pi, self.pi])
		self.link_prob_matrix = np.array([[1-self.g, self.g], [self.s, 1-self.s]])  # index by state, observ

	def _update_forward(self, t, k, state):
		observ = int(self.observ_data[t,k])
		if t == 0:
			self.a_vec[t,k,state] = self.state_density[state] * self.link_prob_matrix[state, observ]
		else:
			self.a_vec[t,k,state] = np.dot(self.a_vec[t-1,k,:], self.transit_matrix[:,state]) * self.link_prob_matrix[state, observ]
	
	def _update_backward(self, t, k, state):
		if t == self.Tvec[k]-1:
			self.b_vec[t,k,state] = 1
		else:
			observ = int(self.observ_data[t+1,k])
			if state == 0:
				self.b_vec[t,k,state] = self.transit_matrix[0,1]*self.link_prob_matrix[1, observ]*self.b_vec[t+1,k,1] + self.transit_matrix[0,0]*self.link_prob_matrix[0, observ]*self.b_vec[t+1,k,0]
			else:
				self.b_vec[t,k,state] = self.link_prob_matrix[state, observ]*self.b_vec[t+1,k,1]

	def _update_eta(self, t, k):
		observ = int(self.observ_data[t+1,k])
		eta_raw = np.zeros((2,2))
		eta_raw[0,0] = self.a_vec[t,k,0]*self.transit_matrix[0,0]*self.link_prob_matrix[0,observ]*self.b_vec[t+1,k,0]
		eta_raw[0,1] = self.a_vec[t,k,0]*self.transit_matrix[0,1]*self.link_prob_matrix[1,observ]*self.b_vec[t+1,k,1]
		eta_raw[1,0] = self.a_vec[t,k,1]*self.transit_matrix[1,0]*self.link_prob_matrix[0,observ]*self.b_vec[t+1,k,0]
		eta_raw[1,1] = self.a_vec[t,k,1]*self.transit_matrix[1,1]*self.link_prob_matrix[1,observ]*self.b_vec[t+1,k,1]
		eta = eta_raw/eta_raw.sum()
		return eta, eta_raw
		
	def _update_gamma(self, t, k):
		gamma_raw = np.zeros((2,))
		gamma_raw[0] = self.a_vec[t,k,0]*self.b_vec[t,k,0]
		gamma_raw[1] = self.a_vec[t,k,1]*self.b_vec[t,k,1]
		gamma = gamma_raw/gamma_raw.sum()
		return gamma, gamma_raw
	
	def __update_pi(self,t, observ, pi_vec, P_mat):
		# pi(i,t) = P(X_t=i|O1,...,O_t,\theta)
		if t == 0:
			# pi(i,0) = P(X_0=i|O0,\theta)
			p0y = (1-self.pi)*self.link_prob_matrix[0,observ]
			p1y = self.pi*self.link_prob_matrix[1,observ]
			py = p0y+p1y
			pi_vec[t,0] = p0y/py
			pi_vec[t,1] = p1y/py
		else:
			# pi(i,t) = sum_{j} P(j,i,t) where P(j,i,t) is the (j,i)the element of transition matrix P
			pi_vec[t,:] = P_mat[t-1,:,:].sum(axis=0)
		
		return pi_vec
			
	def __update_P(self,t,observ, pi_vec, P_mat):
		p_raw = np.zeros((2,2))
		p_raw[0,0] = pi_vec[t,0]*self.transit_matrix[0,0]*self.link_prob_matrix[0,observ]
		p_raw[0,1] = pi_vec[t,0]*self.transit_matrix[0,1]*self.link_prob_matrix[1,observ]
		p_raw[1,0] = pi_vec[t,1]*self.transit_matrix[1,0]*self.link_prob_matrix[0,observ]
		p_raw[1,1] = pi_vec[t,1]*self.transit_matrix[1,1]*self.link_prob_matrix[1,observ]
		
		P_mat[t,:,:] = p_raw/p_raw.sum()
		return P_mat
	
	def _get_llk(self, s, g, pi, l, O_data):
	
		K = len(O_data)
		Tvec = [len(Os) for Os in O_data]
		T = max(Tvec)
	
		a_vec = np.zeros((T, K, 2))
		transit_matrix = np.array([[1-l, l], [0, 1]])
		link_prob_matrix = np.array([[1-g, g], [s, 1-s]])  # index by state, observ
		
		for k in range(K):
			for t in range(Tvec[k]):
				observ = O_data[k][t]
				for state in range(0,2):
					if t == 0:
						a_vec[t,k,state] = self.state_density[state] * link_prob_matrix[state, observ]
					else:
						a_vec[t,k,state] = np.dot(a_vec[t-1,k,:], transit_matrix[:,state]) * link_prob_matrix[state, observ]
		llk = 0
		for k in range(K):
			lk = a_vec[Tvec[k]-1,k,:].sum()
			llk += np.log(lk)
		return llk
	
	def _get_point_estimation(self, start, end, gap):
		# calcualte the llk for the parameters
		
		parameter_candidates = self.parameter_chain[range(start, end, gap), :]
		N = parameter_candidates.shape[0]
		llk_vec = np.zeros((N,))
		for i in range(N):
			llk_vec[i] = self._get_llk(parameter_candidates[i,0], parameter_candidates[i,1], parameter_candidates[i,2], parameter_candidates[i,3], self.O_data)
		llk_max = max(llk_vec)
		llk_sum = llk_max + np.log(np.exp(llk_vec-llk_max).sum())
		
		parameter_weight = np.exp(llk_vec - llk_sum)
		
		avg_parameter = np.dot(parameter_candidates.transpose(), parameter_weight)
		
		return avg_parameter.tolist()
		
	
	def estimate(self, data, L=10):
		self._load_observ(data)
		
		if self.method == 'EM':
			for l in range(L):
				self._em_update()
				#print(self.s, self.g, self.pi, self.l)
		elif self.method == 'MCMC-DG':
			self._MCMC(L, sampling='DG')
			self._get_point_estimation(int(L/2),L,10)
		elif self.method == 'MCMC-FB':
			self._MCMC(L, sampling='FB')
			self.s, self.g, self.pi, self.l = self._get_point_estimation(int(L/2),L,10)
		else:
			raise ValueException('Unknown estimation method %s.' % self.method)
			
	def _em_update(self):
		for k in range(self.K):
			# update forward
			for t in range(self.Tvec[k]):
				self._update_forward(t, k, 0)
				self._update_forward(t, k, 1)
				
			# update backward
			for t in range(self.Tvec[k]-1,-1,-1):
				self._update_backward(t, k, 0)
				self._update_backward(t, k, 1)
				
			# compute r
			for t in range(self.Tvec[k]):
				self.r_vec[t,k,:], self.r_vec_uncond[t,k,:] = self._update_gamma(t,k)

			# compute eta
			for t in range(self.Tvec[k]-1):
				self.eta_vec[t,k,:,:], self.eta_vec_uncond[t,k,:,:] = self._update_eta(t,k)
		
		# update parameters
		# obs_weight = np.ones((self.K,))
		obs_prob = np.empty((self.K,))
		for k in range(self.K):
			obs_prob[k] = self.a_vec[self.Tvec[k]-1,k,:].sum()
		obs_weight = 1/obs_prob
		
		#ipdb.set_trace()	
		self.pi = self.r_vec[0,:,1].mean()
		
		#denominator = np.dot((self.a_vec[0:t-1,:,0]*self.b_vec[0:t-1,:,0]).sum(axis=0), obs_weight) # sum(P(X^k_t=i|O^k))
		self.l = np.dot(self.eta_vec_uncond[:,:,0,1].sum(axis=0), obs_weight) / np.dot(self.r_vec_uncond[:,:,0].sum(axis=0), obs_weight)  # transit from 0 to 1
		
		# need to count the right and wrong
		self.tmp = np.zeros((self.T, self.K,2))
		for k in range(self.K):
			for t in range(self.Tvec[k]):
				observ = int(self.observ_data[t, k])
				self.tmp[t, k, observ] = self.r_vec_uncond[t, k, 1-observ]
		self.s = np.dot(self.tmp[:,:,0].sum(axis=0), obs_weight) / np.dot(self.r_vec_uncond[:,:,1].sum(axis=0), obs_weight) # observe 0 when state is 1
		self.g = np.dot(self.tmp[:,:,1].sum(axis=0), obs_weight) / np.dot(self.r_vec_uncond[:,:,0].sum(axis=0), obs_weight) # observe 1 when state is 0
		
		# constrain
		#self.l = min(0.5, self.l)
		#self.s = min(0.1, self.s)
		#self.g = min(0.3, self.g)
		
		# update the derivatives
		self._update_derivative_parameter()

		
	def _MCMC(self, max_iter, sampling):
		self.parameter_chain = np.empty((max_iter, 4))
		# initialize for iteration
		for iter in tqdm(range(max_iter)):
			### Update states
			X = np.empty((self.T, self.K))
			
			if sampling == 'DG':
				# backward recursion
				for k in range(self.K):
					# update forward
					for t in range(self.Tvec[k]):
						self._update_forward(t, k, 0)
						self._update_forward(t, k, 1)
						
					# update backward
					for t in range(self.Tvec[k]-1,-1,-1):
						self._update_backward(t, k, 0)
						self._update_backward(t, k, 1)
				
					# Forward sampling
					for k in range(self.K):  # Under the assumption of IID between learners
						# initial state
						t = 0
						observ = int(self.observ_data[t,k])
						w0 = (1-self.pi)*self.link_prob_matrix[0,observ]*self.b_vec[t,k,0]
						w1 = self.pi*self.link_prob_matrix[1,observ]*self.b_vec[t,k,1]
						X[t,k] = np.random.binomial(1,w1/(w1+w0))
						# transit
						for t in range(1,self.Tvec[k]):
							# An special case because of constraint
							if X[t-1,k] == 1:
								X[t,k] = 1
							else:
								observ = int(self.observ_data[t,k])
								w0 = self.transit_matrix[0,0]*self.link_prob_matrix[0,observ]*self.b_vec[t,k,0] 
								w1 = self.transit_matrix[0,1]*self.link_prob_matrix[1,observ]*self.b_vec[t,k,1]
								X[t,k] = np.random.binomial(1,w1/(w1+w0))
								
			elif sampling == 'FB':
				
				# collapse observation state
				for key in self.obs_type_info.keys():
					# get the obseration state
					Os = self.obs_type_info[key]['O']
					
					#calculate the exhaustive state probablity
					T = len(Os)
					pi_vec = np.zeros((T,2))
					P_mat = np.zeros((T-1,2,2))
					for t in range(T):
						pi_vec = self.__update_pi(t, Os[t], pi_vec, P_mat)
						if t !=T-1:
							P_mat = self.__update_P(t, Os[t+1], pi_vec, P_mat)

					self.obs_type_info[key]['pi'] = pi_vec
					self.obs_type_info[key]['P'] = P_mat

					# calculate the probability
					self.obs_type_info[key]['llk'] = self._get_llk(self.s, self.g, self.pi, self.l, [Os])
									
				# backward sampling
				for k in range(self.K):
					# check for the observation type
					obs_key = self.obs_type_ref[k]
					pi_vec = self.obs_type_info[obs_key]['pi']
					P_mat = self.obs_type_info[obs_key]['P']
					for t in range(self.Tvec[k]-1,-1,-1):
						if t == self.Tvec[k]-1:
							p = pi_vec[t,1]
						else:
							next_state = int(X[t+1,k])
							p = P_mat[t,1,next_state]/P_mat[t,:,next_state].sum()
						X[t,k] = np.random.binomial(1,p)

			
			### Update parameter
			# use inverse of probability to serve as condition(not kosure)
			
			#ipdb.set_trace()
			critical_trans = 0
			tot_trans = 0
			obs_cnt = np.zeros((2,2)) # state,observ
			
			for k in range(self.K):
				for t in range(0,self.Tvec[k]):
					# update l
					if t>0 and X[t-1,k] == 0:
						tot_trans += 1
						if X[t,k] == 1:
							critical_trans += 1
					# update obs_cnt
					obs_cnt[int(X[t,k]),int(self.observ_data[t,k])] += 1
					
			# Update 
			ipdb.set_trace()
			self.l =  np.random.beta(self.prior_param['l'][0]+critical_trans, 	self.prior_param['l'][1]+tot_trans-critical_trans)
			self.pi = np.random.beta(self.prior_param['pi'][0]+sum(X[0,:]),		self.prior_param['pi'][1]+self.K-sum(X[0,:]))
			self.s =  np.random.beta(self.prior_param['s'][0]+obs_cnt[1,0],		self.prior_param['s'][1]+obs_cnt[1,1])
			self.g =  np.random.beta(self.prior_param['g'][0]+obs_cnt[0,1],		self.prior_param['g'][1]+obs_cnt[0,0])
			
			'''
			while self.s>0.2:
				self.s = np.random.beta(1+obs_cnt[1,0],9+obs_cnt[1,1])

			while self.g>0.4:
				self.g = np.random.beta(1+obs_cnt[0,1],3+obs_cnt[0,0])
			'''
			
			# add to the MCMC chain
			self.parameter_chain[iter,:] = [self.s, self.g, self.pi, self.l]
			if iter%100 == 0:
				print(iter)

def update_mastery(mastery, learn_rate):
	return mastery + (1-mastery)*learn_rate

def compute_success_rate(guess, slip, mastery):
	return guess*(1-mastery) + (1-slip)*mastery		
	
def generate_learning_curve(slip, guess, init_mastery, learn_rate, T):
	p=init_mastery
	lc = [compute_success_rate(guess, slip, p)]
	for t in range(1,T):
		p = update_mastery(p,learn_rate)
		lc.append(compute_success_rate(guess, slip, p))
	return lc
		
		
if __name__ == '__main__':


		# FB-aglorithm check
	
	init_param = {'s':np.random.uniform(0,0.5),
				  'g':np.random.uniform(0,0.5), 
				  'pi':np.random.uniform(0,1),
				  'l':np.random.uniform(0,1)}
	'''
	# Fix initial
	init_param = {'s':0.1,
				  'g':0.3, 
				  'pi':0.5,
				  'l':0.5}
	'''
	
	'''
	# unit test array
	init_param = {'s':0.1,
				  'g':0.2, 
				  'pi':0.6,
				  'l':0.3}
	
	data_array = [(0,0,0),(0,1,1)]
	x = BKT_HMM(init_param)
	x.estimate(data_array,L=1)	
	ipdb.set_trace()
	x = BKT_HMM(init_param,'MCMC-DG')
	x.estimate(data_array,L=1)
	ipdb.set_trace()
	# the a vec is np.array([[0.32, 0.06],[0.0448, 0.1404]])
	# the b vec is np.array([[0.41, 0.9 ],[1,1]])
	# the r vec is np.array([[0.708,0.292],[0.242,0.758]])
	# the eta_vec[0] is np.array([[0.242,0.467],[0,0.291]])
	x = BKT_HMM(init_param,'MCMC-FB')
	x.estimate(data_array,L=1)
	# the pi vec is np.array([[0.99,0.11],[0.4705,0.5296]])
	# the P_mat[0] is np.array([[0.4705,0.2352],[0,0.2942]])
	'''
	
	'''
	# true parameter guess
	init_param = {'s':0.05,
				'g':0.2, 
				'pi':0.4,
				'l':0.3}
	'''
	
	import os
	max_obs = 500
	
	proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))	
	data_array = []
	data_cnt = 0
	with open(proj_dir+'/data/BKT/test/single_sim.txt') as f:
		for line in f:
			i_s, t_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
			
			if int(i_s) == max_obs:
				break
			
			#if int(is_a_s):
			data_array.append( (int(i_s), int(t_s), int(y_s)) )	
			data_cnt += 1

			
	print('EM')
	#x2 = BKT_HMM(init_param)
	#x2.estimate(data_array,L=20)		
	
	#ipdb.set_trace()
	
	x1 = BKT_HMM(init_param, method='MCMC-FB')
	x1.estimate(data_array, L=500)
	
	

	
	print('Initial')
	print(init_param['s'],init_param['g'],init_param['pi'],init_param['l'])
	print('MCMC')	
	print(x1.s, x1.g, x1.pi, x1.l)
	print('EM')
	print(x2.s, x2.g, x2.pi, x2.l)

	
	# check the learning curve difference
	T = 5
	true_lc = generate_learning_curve(0.05,0.2,0.4,0.3, T)
	est_lc_1 = generate_learning_curve(x1.s,x1.g,x1.pi,x1.l, T)
	est_lc_2 = generate_learning_curve(x2.s,x2.g,x2.pi,x2.l, T)
	
	print(true_lc)
	print(est_lc_1)
	print(est_lc_2)
	
	# check the likelihood difference
	
	ipdb.set_trace()