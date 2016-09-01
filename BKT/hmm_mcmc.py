import numpy as np
import ipdb
from tqdm import tqdm
import ipdb

from collections import defaultdict


def logExpSum(llk_vec):
	llk_max = max(llk_vec)
	llk_sum = llk_max + np.log(np.exp(llk_vec-llk_max).sum())
	return llk_sum

# use EM to compute the bayes net
class BKT_HMM_MCMC(object):
			
	def _load_observ(self, data):
		self.K = len(set([x[0] for x in data]))
		self.T = max([x[1] for x in data]) + 1
		
		self.observ_data = np.empty((self.T, self.K))
		T_array = np.zeros((self.K,))
		
		for log in data:
			i = log[0]; t = log[1]; y = log[2]
			self.observ_data[t, i] = y
			T_array[i] = t
		
		self.T_vec = [int(x)+1 for x in T_array.tolist()] 

		self.O_data = []
		for i in range(self.K):
			self.O_data.append( [int(x) for x in self.observ_data[0:self.T_vec[i],i].tolist()] )		
		
		self.T = max(self.T_vec)
		

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
		self.state_transit_matrix = np.array([[1-self.l, self.l], [0, 1]])
		self.state_init_dist = np.array([1-self.pi, self.pi])
		self.observ_prob_matrix = np.array([[1-self.g, self.g], [self.s, 1-self.s]])  # index by state, observ

	
	def __update_pi(self,t, observ, pi_vec, P_mat):
		# pi(i,t) = P(X_t=i|O1,...,O_t,\theta)
		if t == 0:
			# pi(i,0) = P(X_0=i|O0,\theta)
			p0y = self.state_init_dist[0]*self.observ_prob_matrix[0,observ]
			p1y = self.state_init_dist[1]*self.observ_prob_matrix[1,observ]
			py = p0y+p1y
			pi_vec[t,0] = p0y/py
			pi_vec[t,1] = p1y/py
		else:
			# pi(i,t) = sum_{j} P(j,i,t) where P(j,i,t) is the (j,i)the element of transition matrix P
			pi_vec[t,:] = P_mat[t-1,:,:].sum(axis=0)
		
		return pi_vec
			
	def __update_P(self,t, observ, pi_vec, P_mat):
		p_raw = np.zeros((2,2))
		p_raw[0,0] = pi_vec[t,0]*self.state_transit_matrix[0,0]*self.observ_prob_matrix[0,observ]
		p_raw[0,1] = pi_vec[t,0]*self.state_transit_matrix[0,1]*self.observ_prob_matrix[1,observ]
		p_raw[1,0] = pi_vec[t,1]*self.state_transit_matrix[1,0]*self.observ_prob_matrix[0,observ]
		p_raw[1,1] = pi_vec[t,1]*self.state_transit_matrix[1,1]*self.observ_prob_matrix[1,observ]
		
		P_mat[t,:,:] = p_raw/p_raw.sum()
		return P_mat
	
	def _get_llk(self, s, g, pi, l, O_data):
	
		K = len(O_data)
		T_vec = [len(Os) for Os in O_data]
		T = max(T_vec)
	
		a_vec = np.zeros((T, K, 2))
		state_transit_matrix = np.array([[1-l, l], [0, 1]])
		observ_prob_matrix = np.array([[1-g, g], [s, 1-s]])  # index by state, observ
		
		for k in range(K):
			for t in range(T_vec[k]):
				observ = O_data[k][t]
				for state in range(0,2):
					if t == 0:
						a_vec[t,k,state] = self.state_init_dist[state] * observ_prob_matrix[state, observ]
					else:
						a_vec[t,k,state] = np.dot(a_vec[t-1,k,:], state_transit_matrix[:,state]) * observ_prob_matrix[state, observ]
		llk = 0
		for k in range(K):
			lk = a_vec[T_vec[k]-1,k,:].sum()
			llk += np.log(lk)
		return llk
	
	def _get_point_estimation(self, start, end):
		# calcualte the llk for the parameters
		gap = max(int((end-start)/100), 10) # the minimum gap is 10, otherwise take 100 samples
		parameter_candidates = self.parameter_chain[range(start, end, gap), :]
		'''
		N = parameter_candidates.shape[0]
		llk_vec = np.zeros((N,))
		for i in range(N):
			llk_vec[i] = self._get_llk(parameter_candidates[i,0], parameter_candidates[i,1], parameter_candidates[i,2], parameter_candidates[i,3], self.O_data)
		
		llk_sum = logExpSum(llk_vec)
		parameter_weight = np.exp(llk_vec - llk_sum)
		
		avg_parameter = np.dot(parameter_candidates.transpose(), parameter_weight).tolist()
		'''
		avg_parameter = parameter_candidates.mean(axis=0).tolist()
		return avg_parameter
		
	def estimate(self, init_param, data, max_iter=10):
		
		self.g = init_param['g']  # guess
		self.s = init_param['s']  # slippage
		self.pi = init_param['pi']  # initial prob of mastery
		self.l = init_param['l']  # learn speed
		
		self.prior_param = {'l':[2,2],
							's':[1,2],
							'g':[1,2],
							'pi':[2,2]}
							
		self._load_observ(data)
		
		self._MCMC(max_iter)
		self.s, self.g, self.pi, self.l = self._get_point_estimation(int(max_iter/2), max_iter)
		return self.s, self.g, self.pi, self.l
		
	def _MCMC(self, max_iter):
		self.parameter_chain = np.empty((max_iter, 4))
		# initialize for iteration
		for iter in tqdm(range(max_iter)):
			### Update states
			X = np.empty((self.T, self.K))
		
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
			init_pis = np.zeros((self.K,1))
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

			
			### Update parameter
			# use inverse of probability to serve as condition(not kosure)
			
			critical_trans = 0
			tot_trans = 0
			obs_cnt = np.zeros((2,2)) # state,observ
			
			for k in range(self.K):
				for t in range(0,self.T_vec[k]):
					# update l
					if t>0 and X[t-1,k] == 0:
						tot_trans += 1
						if X[t,k] == 1:
							critical_trans += 1
					# update obs_cnt
					obs_cnt[int(X[t,k]),int(self.observ_data[t,k])] += 1
					
			# Update 
			self.l =  np.random.beta(self.prior_param['l'][0] +critical_trans,self.prior_param['l'][1] +tot_trans-critical_trans)
			self.pi = np.random.beta(self.prior_param['pi'][0]+sum(X[0,:]),	self.prior_param['pi'][1]+self.K-sum(X[0,:]))
			self.s =  np.random.beta(self.prior_param['s'][0] +obs_cnt[1,0],	self.prior_param['s'][1] +obs_cnt[1,1])
			self.g =  np.random.beta(self.prior_param['g'][0] +obs_cnt[0,1],	self.prior_param['g'][1] +obs_cnt[0,0])
			self._update_derivative_parameter()
			
			# add to the MCMC chain
			self.parameter_chain[iter,:] = [self.s, self.g, self.pi, self.l]

				
if __name__ == '__main__':	
	
	# unit test array
	init_param = {'s':0.1,
				  'g':0.2, 
				  'pi':0.6,
				  'l':0.3}
	
	data_array = [(0,0,0),(0,1,1)]
	
	x = BKT_HMM_MCMC(init_param)
	x.estimate(data_array, L=1)	
	
	print('pi vec')
	print(x.obs_type_info['0|1']['pi'])
	print(np.array([[0.842,0.158], [0.242,0.758]]))
	
	print('P matrix')
	print(x.obs_type_info['0|1']['P'][0])
	print(np.array([[0.242, 0.467],[0, 0.292]]))
