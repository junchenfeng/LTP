import numpy as np
import ipdb
from tqdm import tqdm



# use EM to compute the bayes net
class BKT_HMM(object):
	def __init__(self, init_param, method='EM'):
		self.g = init_param['g']  # guess
		self.s = init_param['s']  # slippage
		self.pi = init_param['pi']  # initial prob of mastery
		self.l = init_param['l']  # learn speed
		self.method = method
			
	def _load_observ(self, data):
		# the input data are [[y1_1,..,y1_T1],[y2_1,...,y2_T2],..] because learner practice length is not necessary the same
		self.K = len(data)
		self.Tvec =[]
		for k in range(self.K):
			self.Tvec.append(len(data[k]))
		self.T = max(self.Tvec)
		self.observ_data = np.empty((self.T,self.K))
		for k in range(self.K):
			self.observ_data[0:self.Tvec[k],k] = data[k]
		
		# initilize for the rest of the structure
		st_size = (self.T, self.K, 2)
		self.a_vec = np.zeros(st_size)
		self.b_vec = np.zeros(st_size)
		if self.method == 'EM':
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
			return eta_raw/eta_raw.sum()
	
	
	def estimate(self, data, L=10):
		self._load_observ(data)
		
		if self.method == 'EM':
			for l in range(L):
				self._em_update()
				print(self.s, self.g, self.pi, self.l)
		elif self.method == 'MCMC':
			self._MCMC(L)
			# TODO: provide MAP estimator
			ipdb.set_trace()
			self.s = np.mean(self.parameter_chain[range(int(L/2),L,20),0])
			self.g = np.mean(self.parameter_chain[range(int(L/2),L,20),1])
			self.pi =np.mean(self.parameter_chain[range(int(L/2),L,20),2])
			self.l = np.mean(self.parameter_chain[range(int(L/2),L,20),3])
				
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
				self.r_vec[t,k,0] = self.a_vec[t,k,0]*self.b_vec[t,k,0]
				self.r_vec[t,k,1] = self.a_vec[t,k,1]*self.b_vec[t,k,1]
				self.r_vec[t,k,:] = self.r_vec[t,k,:] / self.r_vec[t,k,:].sum()

			# compute eta
			for t in range(self.Tvec[k]-1):
				self.eta_vec[t,k] = self._update_eta(t,k)
		
		# update parameters
		obs_prob = np.empty((self.K,))
		for k in range(self.K):
			obs_prob[k] = self.a_vec[self.Tvec[k]-1,k,:].sum()
		obs_weight = 1/obs_prob
		
		self.pi = np.dot(obs_weight, self.r_vec[0,:,1]) / sum(obs_weight)
		

		self.l = np.dot(self.eta_vec[:,:,0,1].sum(axis=0), obs_weight) / np.dot(self.r_vec[0:-1,:,0].sum(axis=0), obs_weight)  # transit from 0 to 1
		
		# need to count the right and wrong
		self.tmp = np.zeros((self.T, self.K,2))
		for k in range(self.K):
			for t in range(self.Tvec[k]):
				observ = int(self.observ_data[t, k])
				self.tmp[t, k, observ] = self.r_vec[t, k, 1-observ]
		self.s = np.dot(self.tmp[:,:,0].sum(axis=0), obs_weight) / np.dot(self.r_vec[:,:,1].sum(axis=0), obs_weight) # observe 0 when state is 1
		self.g = np.dot(self.tmp[:,:,1].sum(axis=0), obs_weight) / np.dot(self.r_vec[:,:,0].sum(axis=0), obs_weight) # observe 1 when state is 0
		
		
		# constrain
		#self.l = min(0.5, self.l)
		#self.s = min(0.1, self.s)
		#self.g = min(0.3, self.g)
		
		# update the derivatives
		self._update_derivative_parameter()
		
	def _MCMC(self, max_iter):
		self.parameter_chain = np.empty((max_iter, 4))
		# initialize for iteration
		for iter in tqdm(range(max_iter)):
			# update states
			X = np.empty((self.T, self.K))
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
						
			# forward sampling
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
				
			# update parameters
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
			self.l = np.random.beta(1+critical_trans, 4+tot_trans-critical_trans)
			self.pi = np.random.beta(2+sum(X[0,:]),2+self.K-sum(X[0,:]))
			self.s = np.random.beta(1+obs_cnt[1,0],9+obs_cnt[1,1])
			self.g = np.random.beta(1+obs_cnt[0,1],3+obs_cnt[0,0])
			while self.s>0.2:
				self.s = np.random.beta(1+obs_cnt[1,0],9+obs_cnt[1,1])

			while self.g>0.4:
				self.g = np.random.beta(1+obs_cnt[0,1],3+obs_cnt[0,0])
			
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

	beta_s = 0.05
	beta_g = 0.1
	pi = 0.05
	beta_l = 0.2

	# simulate 
	N = 500
	T = 5

	full_data_array = []
	data_array = []
	
	for i in range(N):
		full_observ_list = []
		observ_list = []
		is_stop=False
		
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
			
			full_observ_list.append(y)
			
			# generate imbalance dataset
			if t>1 and not is_stop:
				if np.random.uniform() < 0.1:
					is_stop =True
			if not is_stop:
				observ_list.append(y)
		
		data_array.append(observ_list)
		full_data_array.append(full_observ_list)
		# FB-aglorithm check

	init_param = {'s':np.random.uniform(0,0.2),
				  'g':np.random.uniform(0,0.3), 
				  'pi':np.random.uniform(0,1),
				  'l':np.random.uniform(0,1)}
	'''
	# unit test array
	init_param = {'s':0.1,
				  'g':0.2, 
				  'pi':0.5,
				  'l':0.1}
	full_data_array = [[0,1]]
	x = BKT_HMM(init_param)
	x.estimate(full_data_array,L=1)
	# the a vec is np.array([[0.4, 0.05],[0.072, 0.081]])
	# the b vec is np.array([[0.27, 0.9],[1,1]])
	# the r vec is np.array([[0.7058,0.2942],[0.4706,0.5294]])
	# the eta_vec[0] is np.array([[0.4706,0.2353],[0,0.2941]])
	'''
	
	'''
	# true parameter guess
	init_param = {'s':0.1,
				'g':0.25, 
				'pi':0.25,
				'l':0.1}
	'''
	
	print('Initial')
	print(init_param['s'],init_param['g'],init_param['pi'],init_param['l'])
	
	x1 = BKT_HMM(init_param, method='MCMC')
	x1.estimate(full_data_array, L=1000)
	
	
	print('EM')
	x2 = BKT_HMM(init_param)
	x2.estimate(data_array)
	

	
	# check the learning curve difference
	true_lc = generate_learning_curve(beta_s,beta_g,pi,beta_l, T)
	est_lc_1 = generate_learning_curve(x1.s,x1.g,x1.pi,x1.l, T)
	est_lc_2 = generate_learning_curve(x2.s,x2.g,x2.pi,x2.l, T)
	
	print(true_lc)
	print(est_lc_1)
	print(est_lc_2)
	
	# check the likelihood difference
	
	
	ipdb.set_trace()