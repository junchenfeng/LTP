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


from HMM.prop_hazard_ars import ars_sampler
from HMM.util import draw_c, draw_l, get_map_estimation, get_final_chain
from HMM.dg_util import generate_states, get_single_state_llk, get_joint_state_llk, get_llk_all_states

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
				
	def _MCMC(self, max_iter, method, is_exit=False, is_effort=False):
		# initialize for iteration
		if not is_effort and self.valid_prob_matrix[:,:,0].sum() != 0: 
			raise Exception('Effort rates are not set to 1 while disabled the update in effort parameter.')
		
		if not is_exit and self.hazard_matrix.sum() != 0: 
			raise Exception('Hazard rates are not set to 0 while disabled the update in hazard parameter.')
		
		if is_exit:
			prop_hazard_mdls = [ars_sampler(self.Lambda, [self.beta]) for i in range(2)]
			
		param_chain = {'l': np.zeros((max_iter, (self.Mx-1)*self.J)),
					   'pi':np.zeros((max_iter, self.Mx-1)),
					   'c': np.zeros((max_iter, (self.Mx*(self.My-1))*self.J))
					   } # equivalent to a mapping of My = 2, c = 2*self.J; My = 3, c=4*self.J
			
		if is_exit:
			param_chain['h'] = np.zeros((max_iter, self.Mx*2))
			
		if is_effort:
			param_chain['e'] = np.zeros((max_iter, self.Mx*self.J))
			
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
					if abs(llk_vec.sum())<1e-15:
						raise ValueError('All likelihood are 0.')
					
					# pi
					tot_llk=llk_vec.sum()
					self.obs_type_info[key]['pi'] = [get_single_state_llk(X_mat, llk_vec, Ti-1, x)/tot_llk for x in range(self.Mx)]
					
					# learning rate
					l_mat = np.zeros((Ti, self.Mx, self.Mx)) # T,X_{t+1},X_t		
					for t in range(Ti-1,0,-1):
						l_mat[t,0,0] = 1 # the 0 state in t, must implies 0 in t-1
						for m in range(1,self.Mx):
							pNext = get_single_state_llk(X_mat, llk_vec, t, m)
							if pNext != 0:
								for n in range(self.Mx):
									# P(X_{t-1},X_t)/P(X_t)
									l = get_joint_state_llk(X_mat, llk_vec, t, n, m) / pNext
									if not(l>=0 and l<=1):
										if not(l>1 and l-1<0.00001):
											raise ValueError('Learning rate is wrong for X=%d.'%x)
										else:
											l = 1.0
									l_mat[t,m,n] = l
							# If pNext is 0, then there is no probability the state will transite in

					self.obs_type_info[key]['l_mat'] = l_mat
					
				# sample states backwards
				X = np.empty((self.T, self.K),dtype=np.int)
				for i in range(self.K):
					# check the key
					obs_key = self.obs_type_ref[i]
					pi = self.obs_type_info[obs_key]['pi']
					l_mat = self.obs_type_info[obs_key]['l_mat']
					Ti = self.T_vec[i]
					
					X[Ti-1,i] = np.random.choice(self.Mx, 1, p=pi)
					for t in range(Ti-1, 0,-1):
						pt = l_mat[t, X[t,i],:]
						if pt.sum()==0:
							raise Exception('Invalid transition kernel')
						X[t-1,i] = np.random.choice(self.Mx, 1, p=pt)
													
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
					o_j = self.item_data[t,k]
					o_is_v = self.V_array[t,k]
					x1 = X[t,k]
					
					# update l
					if t>0 and self.V_array[t-1,k]>0:
						l_j = self.item_data[t-1, k] # transition happens at t, item at t-1 takes credit
						x0 = X[t-1,k]
						if x0 != self.Mx-1:
							#P(X_t=1,X_{t-1}=0,V_(t-1)=1)/P(X_{t-1}=0,V_(t-1)=1)
							if x1-x0==1:
								critical_trans[l_j, x0] += 1
							else:
								no_trans[l_j, x0] += 1
					# update e	
					valid_cnt[o_j, x1] += o_is_v
					valid_state_cnt[o_j, x1] += 1	
					
					# y
					if o_is_v:
						obs_cnt[o_j, x1, self.observ_data[t,k]] += 1 #P(Y=0,V=1,X=1)/P(X=1,V=1) = s; P(Y_t=1,V_t=0)+P(Y_t=1,V_t=1,X_t=0))/(P(V_t=0)+P(X_t=0,V_t=1)) =g
			# update c	
			for j in range(self.J):
				c_params = [[self.prior_param['c'][x][y] + obs_cnt[j,x,y] for y in range(self.My)] for x in range(self.Mx)] 
				c_draws = draw_c(c_params, self.Mx, self.My)
				self.observ_prob_matrix[j] = c_draws
				
			# upate pi		
			pi_params = [self.prior_param['pi'][x]+ np.sum(X[0,:]==x) for x in range(self.Mx)]
			self.state_init_dist = np.random.dirichlet(pi_params)
			
			# update l
			for j in range(self.J):
				#ipdb.set_trace()
				params = [[self.prior_param['l'][0]+no_trans[j,x], self.prior_param['l'][1]+critical_trans[j,x]] for x in range(self.Mx-1)]
				self.state_transit_matrix[j] = draw_l(params, self.Mx)
			
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
						hX[idx].append([t])
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
					Nh = len(hIdx[i])
					prop_hazard_mdls[i].load(np.array(hX[i])[hIdx[i],:], np.array(hD[i])[hIdx[i]])
					
					prop_hazard_mdls[i].Lambda = prop_hazard_mdls[i].sample_lambda()[-1]
					prop_hazard_mdls[i].betas[0] = prop_hazard_mdls[i].sample_beta(0)[-1]	
				
				# reconfig
				self.h0 = [prop_hazard_mdls[0].Lambda*np.exp(prop_hazard_mdls[0].betas[0]*t) for t in range(self.T)]
				self.h1 = [prop_hazard_mdls[1].Lambda*np.exp(prop_hazard_mdls[1].betas[0]*t) for t in range(self.T)]
				
				# check for sanity
				if any([h>1 for h in self.h0]) or any([h>1 for h in self.h1]):
					raise ValueError('Hazard rate is larger than 1.')
				self.hazard_matrix = np.array([self.h0 for x in range(self.Mx-1)] + [self.h1])

			else:
				self.hazard_matrix = [[0.0 for t in range(self.T)] for x in range(self.Mx)]
			
			if is_effort:
				for j in range(self.J):
					self.valid_prob_matrix[j] = [np.random.dirichlet((self.prior_param['e'][0]+valid_state_cnt[j,x]-valid_cnt[j,x], self.prior_param['e'][1]+valid_cnt[j,x])) for x in range(self.Mx)]
			
			#############################
			# Step 3: Preserve the Chain#
			#############################
			
			lHat_vec = []
			for j in range(self.J):
				lHat_vec += [self.state_transit_matrix[j][1][x,x+1] for x in range(self.Mx-1)]
			param_chain['l'][iter,:] = lHat_vec
			param_chain['pi'][iter,:] = self.state_init_dist[0:-1]
			
			cHat_vec = []
			for j in range(self.J):
				if self.Mx==2:
					cHat_vec += [self.observ_prob_matrix[j][0,1], self.observ_prob_matrix[j][1,1]]
				elif self.Mx==3:
					cHat_vec += [self.observ_prob_matrix[j][0,1], self.observ_prob_matrix[j][0,2], self.observ_prob_matrix[j][1,1], self.observ_prob_matrix[j][1,2], self.observ_prob_matrix[j][2,1], self.observ_prob_matrix[j][2,2]]
				else:
					raise ValueError('Number of latent state is wrong!')
			param_chain['c'][iter,:] = cHat_vec
			
			if is_exit:
				h_vec = []
				for i in range(2):
					h_vec += [prop_hazard_mdls[i].Lambda, prop_hazard_mdls[i].betas[0] ]
				param_chain['h'][iter,:] = h_vec
			
			if is_effort:
				param_chain['e'][iter,:] = self.valid_prob_matrix[:,:,1].flatten()
			# update parameter chain here
			self.X = X
			#ipdb.set_trace()	

		return param_chain
			

	


	def _get_initial_param(self, init_param, prior_dist, is_exit, is_effort):
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
				# TODO: apply 0 constraints on the correct rate
				if self.My==2:
					self.prior_param['c'] = [[2,1],[1,2]]
				elif self.My==3:
					self.prior_param['c'] = [[1,1,0],[1,1,1],[0,1,1]]
			else:
				# TODO: check the specification of the prior
				self.prior_param = prior_dist
			
			self.state_init_dist = np.random.dirichlet(self.prior_param['pi'])
			self.state_transit_matrix = np.array([draw_l([self.prior_param['l'] for x in range(self.Mx-1)], self.Mx) for j in range(self.J)])
			self.observ_prob_matrix = np.array([draw_c(self.prior_param['c'], self.Mx, self.My) for j in range(self.J)])
			
			if is_effort:
				self.valid_prob_matrix = np.array([[np.random.dirichlet(self.prior_param['e']) for x in range(self.Mx)] for j in range(self.J)])
			else:
				self.valid_prob_matrix = np.zeros((self.J, self.Mx, 2))
				self.valid_prob_matrix[:,:,1] = 1.0
			
			if is_exit:
				self.Lambda = 0.1
				self.beta = 0.01
				self.hazard_matrix = np.array([[self.Lambda*np.exp(self.beta*t) for t in range(self.T)] for x in range(self.Mx)])
			else:
				self.hazard_matrix = np.array([[0.0 for t in range(self.T)] for x in range(self.Mx)])
		
		# Check parameter validity
		if any([px==0 for px in self.state_init_dist]):
			raise Exception('The initital distribution is degenerated.')				
	
	def estimate(self, data_array, Mx=None, prior_dist = {}, init_param ={}, method='DG', max_iter=1000, chain_num = 4, is_exit=False, is_effort=False):
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
		
		# run MCMC
		param_chain_vec = []
		for iChain in range(chain_num):
			self._get_initial_param(init_param, prior_dist, is_exit, is_effort)
			tmp_param_chain = self._MCMC(max_iter, method, is_exit, is_effort)
			param_chain_vec.append(tmp_param_chain)
			
		# process
		self.param_chain = get_final_chain(param_chain_vec, int(max_iter/2), max_iter, is_exit, is_effort)	
		res = get_map_estimation(self.param_chain,is_exit, is_effort)
		
		return res
