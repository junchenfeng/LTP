# encoding: utf-8

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy
import math
import os	

		  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)


from BKT.prop_hazard_ars import ars_sampler
from BKT.util import draw_c, draw_l
from BKT.dg_util import generate_states, get_single_state_llk, get_joint_state_llk, get_llk_all_states

import ipdb


#TODO: add Lambda and betas into prior
#TODO: Scale the proportional hazard model
	

class LTP_HMM_MCMC(object):

	def _load_observ(self, data):

		self.K = len(set([x[0] for x in data]))
		self.T = max([x[1] for x in data]) + 1
		self.J = len(set([x[2] for x in data]))
		self.My = len(set(x[3] for x in data))
		
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
				
	def _MCMC(self, max_iter, method, is_exit=False, is_effort=False):
		# initialize for iteration
		if not is_effort and self.valid_prob_matrix[:,:,0].sum() != 0: 
			raise Exception('Effort rates are not set to 1 while disabled the update in effort parameter.')
		
		if not is_exit and self.hazard_matrix.sum() != 0: 
			raise Exception('Hazard rates are not set to 0 while disabled the update in hazard parameter.')
		
		if is_exit:
			prop_hazard_mdls = [ars_sampler(self.Lambda, self.beta) for i in range(2)]
			
		self.param_chain = {'l': np.zeros(max_iter, (self.Mx-1)*self.J)),
					   'pi':np.zeros(max_iter, 2),
					   'c': np.zeros(max_iter, (self.Mx*3-5)*self.J} # equivalent to a mapping of My = 2, c = self.Mx; My = 3, c=4*self.Mx
			
		if is_exit:
			self.param_chain['h'] = np.zeros(max_iter, self.Mx*2)
			
		if is_effort:
			self.param_chain['e'] = np.zeros(max_iter, self.Mx*self.J)
			
		for iter in tqdm(range(max_iter)):
			#############################
			# Step 1: Data Augmentation #
			#############################
			
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
					X_mat = generate_states(Ti, max_level=self.Mx-1)
					llk_vec = get_llk_all_states(X_mat, O, J, V, E, self.hazard_matrix, self.observ_prob_matrix, self.state_init_dist, self.state_transit_matrix, self.valid_prob_matrix, is_effort, is_exit)
					
					self.obs_type_info[key]['llk_vec'] = llk_vec
					if abs(llk_vec.sum())<0.0000001:
						ipdb.set_trace()
						raise ValueError('All likelihood are 0.')
					
					
					self.obs_type_info[key]['pi'] = [get_single_state_llk(X_mat, llk_vec, 0, x)/llk_vec.sum() for x in range(3)]
					
					l_vec = [[] for x in range(self.Mx-1)];				
					for t in range(1,Ti):
						for x in range(self.Mx-1):
							if get_single_state_llk(X_mat, llk_vec, t-1, x) != 0:
								l = get_joint_state_llk(X_mat, llk_vec, t, x, x+1) / get_single_state_llk(X_mat, llk_vec, t-1, x)
								if not(l>=0 and l<=1):
									if l>1 and l-1<0.00001:
										l = 1.0
									else:
										raise ValueError('Learning rate is wrong for X=%d.'%x)
							else:
								l = 0
							l_vec[x].append(l)
					self.obs_type_info[key]['l_vec'] = l_vec
					
				# sample states
				X = np.empty((self.T, self.K),dtype=np.int)
				for i in range(self.K):
					# check the key
					obs_key = self.obs_type_ref[i]
					pi = self.obs_type_info[obs_key]['pi']
					#ipdb.set_trace()
					l_vec = self.obs_type_info[obs_key]['l_vec']
					Vs = self.obs_type_info[obs_key]['V']
					X[0,i] = np.random.choice(self.Mx, 1, p=pi)
					for t in range(1, self.T_vec[i]):
						if X[t-1,i] == self.Mx-1:
							X[t,i] = X[t-1,i] # 2 are absorbing state | no effort no transition
						else:
							x =  X[t-1,i]
							X[t,i] = np.random.binomial(1, l_vec[x][t-1])+x					
				
			else:
				raise Exception('Algorithm %s not implemented.' % method)
				
			#############################
			# Step 2: Update Parameter  #
			#############################
			
			critical_trans = np.zeros((self.J, self.Mx-1), dtype=np.int)
			no_trans = np.zeros((self.J, self.Mx-1),dtype=np.int)
			
			obs_cnt = np.zeros((self.J,self.Mx,self.My)) # state,observ
			valid_cnt = np.zeros((self.J,self.Mx),dtype=np.int)
			valid_state_cnt = np.zeros((self.J,self.Mx),dtype=np.int)
			# update the sufficient statistics
			for k in range(self.K):
				for t in range(0, self.T_vec[k]):
					l_j = self.item_data[t,k]
					is_v = self.V_array[t,k]
					o_j = self.item_data[t,k]
					x0 = X[t-1,k]
					x1 = X[t,k]
					# if the transition happens at t, item in t-1 should take the credit
					# The last item does not contribute the the learning rate
					# update l
					if t>0 and x0 !=2 and is_v>0:
						#P(X_t=1,X_{t-1}=0,V_t=1)/P(X_{t-1}=0,V_t=1)
						
						if x1-x0==1:
							critical_trans[l_j, x0] += 1
						else:
							no_trans[l_j, x0] += 1
					if t>0:
						valid_cnt[o_j, x0] += is_v
						valid_state_cnt[o_j, x0] += 1			
					# update obs_cnt
					if is_v:
						obs_cnt[o_j, x1, self.observ_data[t,k]] += 1 #P(Y=0,V=1,X=1)/P(X=1,V=1) = s; P(Y_t=1,V_t=0)+P(Y_t=1,V_t=1,X_t=0))/(P(V_t=0)+P(X_t=0,V_t=1)) =g
		
			# update c	
			for j in range(self.J):
				c_params = [[self.prior_param['c'][x][y] + obs_cnt[j,x,y] for y in range(self.My)] for x in range(self.Mx)] 
				self.observ_prob_matrix[j] = draw_c(c_params, self.Mx, sekf.My)
			# upate pi		
			pi_params = [self.prior_param['pi'][x]+ np.sum(X[0,:]==x) for x in range(self.Mx)]
			self.state_init_dist = np.random.dirichlet(pi_params)
			
			# update l
			for j in range(self.J):
				params = [[self.prior_param['l'][0]+critical_trans[j,x], self.prior_param['l'][1]+no_trans[j,x]] for x in range(self.Mx-1)]
				self.state_transit_matrix[j] = draw_l(params)
			
			if is_exit:
				# Separate out the model for X = Mx and X!= Mx
				# For three-state model, the the first state may have too few observations  
				# generate X,D
				hX = [[],[]]
				hD = [[],[]]
				hIdx = [[],[]]
				for k in range(self.K):
					for t in range(self.T_vec[k]):
						idx = int(X[t,k] == self.Mx-1)
						hX[idx].append((t))
						hD[idx].append(self.E_array[t,k])
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
				# estimate the mdodel
				for i in range(2):
					prop_hazard_mdls[i].load(np.array(hX[i])[hIdx[i],:], np.array(hD[i])[hIdx[i]])
					prop_hazard_mdls[i].Lambda = prop_hazard_mdls[i].sample_lambda()[-1]
					prop_hazard_mdls[i].betas[0] = prop_hazard_mdls[i].sample_beta(0)[-1]	
				
				# reconfig
				self.h0 = [prop_hazard_mdls[0].Lambda*np.exp(prop_hazard_mdls[0].beta[0]*t) for t in range(self.T)]
				self.h1 = [prop_hazard_mdls[1].Lambda*np.exp(prop_hazard_mdls[1].beta[0]*t) for t in range(self.T)]
				
				# check for sanity
				if any([h>1 for h in self.h0]) or any([h>1 for h in self.h1]):
					raise ValueError('Hazard rate is larger than 1.')
				self.hazard_matrix = [self.h0 for x in range(self.Mx) if x<self.Mx-1 else self.h1]
			else:
				self.hazard_matrix = [[0.0 for t in range(self.T)] for x in range(self.Mx)]
			
			if is_effort:
				for j in range(self.J):
					self.valid_prob_matrix[j] = [np.random.dirichlet((self.prior_param['e'][0]+valid_cnt[j,x], self.prior_param['e'][1]+valid_state_cnt[j,x]-valid_cnt[j,x])) for x in self.Mx]
			
			
			#############################
			# Step 3: Preserve the Chain#
			#############################
			lHat_vec = []
			for j in range(self.J):
				lHat_vec += [self.state_transit_matrix[j][1][x,x+1] for x in range(self.Mx-1)]
			self.param_chain['l'][iter,:] = lHat_vec
			
			self.param_chain['pi'][iter,:] = self.state_init_dist[0:-1]
			
			cHat_vec = []
			for j in range(self.J):
				if self.Mx==2:
					cHat_vec += [self.observ_prob_matrix[j][0,1], self.observ_prob_matrix[j][1,1]]
				elif self.Mx==3:
					cHat_vec += [self.observ_prob_matrix[j][0,1], self.observ_prob_matrix[j][1,1], self.observ_prob_matrix[j][1,2], self.observ_prob_matrix[j][2,2]]
				else:
					raise ValueError('Number of latent state is wrong!')
				
			if is_exit:
				h_vec = [prop_hazard_mdls[i].Lambda, prop_hazard_mdls[i].beta[0] for i in range(2)]
				self.param_chain['h'][iter,:] = h_vec
			
			if is_effort:
				e_vec = [self.valid_prob_matrix[j][1,:] for j in range(self.J)]
				self.param_chain['e'][iter,:] = e_vec
			# update parameter chain here
			self.X = X


	def _get_point_estimation(self, start, end, is_exit, is_effort):
		# calcualte the llk for the parameters
		gap = max(int((end-start)/100), 10)
		select_idx = range(start, end, gap)
		res = {}
		res['l'] = self.param_chain['l'][select_idx, :].mean(axis=0).tolist()
		res['c'] = self.param_chain['c'][select_idx, :].mean(axis=0).tolist()
		res['pi'] = self.param_chain['pi'][select_idx, :].mean(axis=0).tolist()
		
		if is_exit:
			res['h'] = self.param_chain['h'][select_idx, :].mean(axis=0).tolist()
		
		if is_effort:
			res['e'] = self.param_chain['e'][select_idx, :].mean(axis=0).tolist()
			
	
		return res
				
	def estimate(self, data_array, Mx=None, prior_dist = {}, init_param ={}, method='DG', max_iter=1000, is_exit=False, is_effort=False):
		# data = [(i,t,j,y,e)]  
		# i: learner id from 0:N-1
		# t: sequence id, t starts from 0
		# j: item id, from 0:J-1
		# y: response, 0 or 1
		# e(xit): if the spell ends here
		# v(alid): 0 or 1		
		self._load_observ(data_array)
		# My: the number of observation state. Assume that all items have the same My. Only 2 and 3 are accepted.
		# Me: number of effort state. Assume that all items have the same Me. Only 2 are accepted.
		# nJ: the number of items
		# K: the number of users
		# T: longest 		
		# Mx: the number of latent state. Transition matrix is defaults to left-right. only diagonal and upper first off diagonal element is non-negative
		# Mx = My, unless otherwise specified
		if not Mx:
			self.Mx=self.My
		else:
			self.Mx=Mx
		self._collapse_obser_state()

		if self.My not in [2,3]:
			raise ValueError('Number of observable state is wrong.')
		if self.Mx not in [2,3]:
			raise ValueError('Number of latent state is wrong.')
		
		# c: probability of correct. Let cij=p(Y=j|X=i). In 2 state, [2*2]*nJ (1=c01,c11); In 3 state, [3*3]*nJ, (but c02=c20 =0)
		# pi: initial distribution of latent state, [Mx]
		# l: learning rate/pedagogical efficacy, [Mx-1]*nJ
		# e: probability of effort, [Mx]*nJ
		# Lambda: harzard rate with at time 0. scalar
		# betas: time trend of proportional hazard. scalar
		
		if init_param:
			# for special purpose, allow for pre-determined starting point.
			param = copy.copy(init_param)
			# ALL items share the same prior for now
			self.observ_prob_matrix = param['c']
			self.valid_prob_matrix = param['e']
			self.state_init_dist = np.array(param['pi']) 
			self.state_transit_matrix = param['l'] 
			self.Lambda = param['Lambda'] 
			self.beta = param['beta']
		else:
			# generate parameters from the prior
			if not prior_dist:
				self.prior_param = {'l': [1, 1],
									'e': [1, 1],
									'pi':[1]*self.Mx}
				if self.My==2:
					self.prior_param['c'] = [[1,1],[1,1]]
				elif self.My==3:
					self.prior_param['c'] = [[1,1,0],[1,1,1],[0,1,1]]
			else:
				# TODO: check the specification of the prior
				self.prior_param = prior_dist
			
			self.state_init_dist = np.random.dirichlet(self.prior_param['pi'])
			self.state_transit_matrix = [draw_l([self.prior_param['l'] for x in range(self.Mx-1) ]) for j in range(self.J)]
			self.valid_prob_matrix = [[np.random.dirichlet(self.prior_param['e']) for x in self.Mx] for j in self.J]
			self.observ_prob_matrix = [draw_c(self.prior_param['c'], self.Mx, sekf.My) for j in self.J]
			
			if is_exit:
				self.Lambda = 0.1
				self.beta = 0.01
				self.hazard_matrix = [[self.Lambda*np.exp(self.beta*t) for t in range(self.T)] for x in range(self.Mx)]
			else:
				self.hazard_matrix = [[0.0 for t in range(self.T)] for x in range(self.Mx)]
		
		
		# Check input validity
			
		if not is_exit and self.Lambda !=0:
			raise Exception('Under no exit regime, baseline hazard rate should be zero.')
		
		if any([px==0 for px in self.state_init_dist]):
			raise Exception('The initital distribution is degenerated.')
		
		# check for number of latent states
		if len(self.state_init_dist) != self.Mx:
			raise Exception('The initial distribution has the wrong size.')

		
		
		self._MCMC(max_iter, method, is_exit, is_effort)
		res = self._get_point_estimation(int(max_iter/2), max_iter)
		

		
		return res
		
if __name__=='__main__':

		