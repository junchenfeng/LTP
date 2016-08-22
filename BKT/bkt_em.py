import numpy as np
import ipdb




# use EM to compute the bayes net
class BKT_HMM(object):
	def __init__(self, init_param):
		self.g = init_param['g']  # guess
		self.s = init_param['s']  # slippage
		self.pi = init_param['pi']  # initial prob of mastery
		self.l = init_param['l']  # learn speed
			
	def _load_observ(self, data):
		self.observ_data = data
		self.T, self.K = data.shape
		# initilize for the rest of the structure
		st_size = (self.T, self.K, 2)
		self.a_vec = np.zeros(st_size)
		self.b_vec = np.zeros(st_size)
		self.r_vec = np.zeros(st_size)
		self.eta_vec = np.zeros((self.T, self.K, 2,2))
		
		# initialize
		self._update_derivative_parameter()  # learning spead

		
	def _update_derivative_parameter(self):
		self.transit_matrix = np.array([[1-self.l, self.l], [0, 1]])
		self.link_prob_matrix = np.array([[1-self.g, self.g], [self.s, 1-self.s]])  # index by state, observ

	def _update_forward(self, t, k, state):
		observ = int(self.observ_data[t,k])
		if t == 0:
			self.a_vec[t,k,state] = self.pi * self.link_prob_matrix[state, observ]
		else:
			self.a_vec[t,k,state] = np.dot(self.a_vec[t-1,k,:], self.transit_matrix[:,state]) * self.link_prob_matrix[state, observ]
	
	def _update_backward(self, t, k, state):
		if t == self.T-1:
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
			return eta_raw/eta_raw.sum()
				
	def _update(self):
		for k in range(self.K):
			# update forward
			for t in range(self.T):
				self._update_forward(t, k, 0)
				self._update_forward(t, k, 1)
				
			# update backward
			for t in range(self.T-1,-1,-1):
				self._update_backward(t, k, 0)
				self._update_backward(t, k, 1)
				
			# compute r
			for t in range(self.T):
				self.r_vec[t,k,0] = self.a_vec[t,k,0]*self.b_vec[t,k,0]
				self.r_vec[t,k,1] = self.a_vec[t,k,1]*self.b_vec[t,k,1]
				self.r_vec[t,k,:] = self.r_vec[t,k,:] / self.r_vec[t,k,:].sum()

			# compute eta
			for t in range(self.T-1):
				self.eta_vec[t,k] = self._update_eta(t,k)
	
	def _update_model_parameter(self):
		# compute the scaling factor
		obs_prob = self.a_vec[self.T-1,:,:].sum(axis=1)
		obs_prob = 1/obs_prob
		
		self.pi = np.dot(obs_prob, self.r_vec[0,:,1]) / sum(obs_prob)
		

		self.l = np.dot(self.eta_vec[:,:,0,1].sum(axis=0), obs_prob) / np.dot(self.r_vec[0:-1,:,0].sum(axis=0), obs_prob)  # transit from 0 to 1
		
		# need to count the right and wrong
		self.tmp = np.zeros((self.T, self.K,2))
		for t in range(self.T):
			for k in range(self.K):
				observ = int(self.observ_data[t,k])
				self.tmp[t,k,observ] = self.r_vec[t,k,1-observ]
		self.s = np.dot(self.tmp[:,:,0].sum(axis=0), obs_prob) / np.dot(self.r_vec[:,:,1].sum(axis=0), obs_prob) # observe 0 when state is 1
		self.g = np.dot(self.tmp[:,:,1].sum(axis=0), obs_prob) / np.dot(self.r_vec[:,:,0].sum(axis=0), obs_prob) # observe 1 when state is 0
		
		
		# constrain
		#self.l = min(0.5, self.l)
		#self.s = min(0.2, self.s)
		#self.g = min(0.3, self.g)
		
		# update the derivatives
		self._update_derivative_parameter()
		
	
	def estimate(self, data, L=10):
		self._load_observ(data)
		for l in range(L):
			self._update()
			self._update_model_parameter()
			print(self.s, self.g, self.pi, self.l)
		

if __name__ == '__main__':

	beta_s = 0.1
	beta_g = 0.25
	pi = 0.2
	beta_l = 0.1

	# simulate 
	N = 1000
	T = 5

	data_array = np.zeros((T,N))
	
	#log_data = []
	for i in range(N):
		observ_list = []
		for t in range(T):
			if t ==0:
				s=1 if np.random.uniform() <= pi else 0
			else:
				if s == 0:
					s = 1 if  np.random.uniform() < beta_l else 0
			
			if s:
				y = 0 if np.random.uniform() < beta_s else 1
			else:
				y = 1 if np.random.uniform() < beta_g else 0
			
			#log_data.append((y,s))
			observ_list.append(y)
		
		data_array[:,i] = observ_list
		# FB-aglorithm check

	init_param = {'s':np.random.uniform(0,0.2),
				  'g':np.random.uniform(0,0.4), 
				  'pi':np.random.uniform(0,1),
				  'l':np.random.uniform(0,1)}
	'''
	init_param = {'s':0.1,
				  'g':0.2, 
				  'pi':0.5,
				  'l':0.1}
	data_array = np.array([[0],[1]])
	
	init_param = {'s':0.1,
				'g':0.25, 
				'pi':0.25,
				'l':0.1}
	'''
	
	x = BKT_HMM(init_param)
	
	x.estimate(data_array)
		
	'''
	# unit test
	observ_list = [0,1]	
	# the a vec is np.array([[0.4, 0.05],[0.072, 0.081]])
	# the b vec is np.array([[0.27, 0.9],[1,1]])
	# the r vec is np.array([[0.7058,0.2942],[0.4706,0.5294]])
	# the eta[1] vec is np.array([[0.4706,0.2353],[0,0.2941]])
	'''
	
	ipdb.set_trace()