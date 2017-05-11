# encoding: utf-8

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy
import math

from .util import draw_c, draw_l, get_map_estimation, get_final_chain, random_choice, draw_multilevel_pi, get_item_dict
from .bfs_util import generate_states, update_state_parmeters
from .hazard_util import prop_hazard, cell_hazard

from joblib import Parallel, delayed

class LTP_HMM_MCMC(object):

	def _load_observ(self, data):

		self.K = len(set([x[0] for x in data]))
		self.T = max([x[1] for x in data]) + 1
		self.J = len(set([x[2] for x in data]))
		self.My = len(set(x[3] for x in data))
		
		self.H_array = np.empty((self.T, self.K), dtype=np.int)
		self.E_array = np.empty((self.T, self.K), dtype=np.int)
		self.O_array = np.empty((self.T, self.K), dtype=np.int)
		self.J_array = np.empty((self.T, self.K), dtype=np.int)
		T_array = np.zeros((self.K,))
		
		for log in data:
			if len(log)==4:
				# The spell never ends; multiple item
				i,t,j,y = log
				is_h = 0; is_e = 1
			elif len(log) == 5:
				i,t,j,y,is_h = log
				is_e = 1
			elif len(log) == 6:
				i,t,j,y,is_h,is_e = log
			else:
				raise Exception('The log format is not recognized.')
			self.O_array[t, i] = y
			self.J_array[t, i] = j
			self.H_array[t, i] = is_h
			self.E_array[t, i] = is_e
			T_array[i] = t
		
		# This section is used to collapse states
		self.T_vec = [int(x)+1 for x in T_array.tolist()] 
		self.O_data = []
		for i in range(self.K):
			self.O_data.append( [x for x in self.O_array[0:self.T_vec[i],i].tolist()] )
		self.J_data = []
		for i in range(self.K):
			self.J_data.append( [x for x in self.J_array[0:self.T_vec[i],i].tolist()] )		
		self.E_data = []
		for i in range(self.K):
			self.E_data.append( [x for x in self.E_array[0:self.T_vec[i],i].tolist()] )					
		
		self.H_vec = [int(self.H_array[self.T_vec[i]-1, i]) for i in range(self.K)]

	def _collapse_obser_state(self):
		self.obs_type_cnt = defaultdict(int)
		self.obs_type_ref = {}
		for k in range(self.K):
			obs_type_key = str(self.H_vec[k]) + '-' + '|'.join(str(y) for y in self.O_data[k]) + '-' + '|'.join(str(j) for j in self.J_data[k]) + '-' + '|'.join(str(e) for e in self.E_data[k])
			self.obs_type_cnt[obs_type_key] += 1
			self.obs_type_ref[k] = obs_type_key
		
		# construct the space
		self.obs_type_info = {}
		for key in self.obs_type_cnt.keys():
			H_s, O_s, J_s, E_s = key.split('-')

			self.obs_type_info[key] = {'H':int(H_s), 'O':[int(x) for x in O_s.split('|')], 'J':[int(x) for x in J_s.split('|')], 'E':[int(x) for x in E_s.split('|')]}
				
	def _MCMC(self, max_iter, method='BFS', is_effort=False, is_exit=False, hazard_model='cell', hazard_state='X'):
		# initialize for iteration
		if not is_effort and self.effort_prob_matrix[:,:,0].sum() != 0: 
			raise Exception('Effort rates are not set to 1 while disabled the update in effort parameter.')
		
		if not is_exit and self.hazard_matrix.sum() != 0: 
			raise Exception('Hazard rates are not set to 0 while disabled the update in hazard parameter.')			
		
		lMx = int((self.Mx-1)*self.Mx*self.J/2)
		param_chain = {'l': np.zeros((max_iter, lMx*self.num_mixture)), # for each item, the possible transition is (self.X-1 + 1)* (self.X-1)/2
					   'pi':np.zeros((max_iter, (self.Mx-1)*self.num_mixture)),
					   'mixture':np.zeros((max_iter,self.num_mixture-1)),
					   'c': np.zeros((max_iter, (self.Mx*(self.My-1))*self.unique_item_num))
					   } 
			
		if is_exit:
			if hazard_state == 'X':
				Mh = self.Mx
			elif hazard_state =='Y':
				Mh = self.My	
			
			if hazard_model == 'prop':
				self.Lambdas = [self.Lambda for s in range(Mh)]
				self.betas = [self.beta for s in range(Mh)]
				param_chain['h'] = np.zeros((max_iter, Mh*2))
			elif hazard_model == 'cell':
				param_chain['h'] = np.zeros((max_iter, Mh*self.T))
			
			
		if is_effort:
			param_chain['e'] = np.zeros((max_iter, self.Mx*self.J))
		
		# cache the generated states
		X_mat_dict = {}
		for t in range(1,self.T+1):
			X_mat_dict[t] = generate_states(t, self.Mx, self.Mx-1)
		
		tot_error_cnt = 0
		for iter in tqdm(range(max_iter)):
		
			if tot_error_cnt > 10:
				raise Exception('Too many erros in drawing')
		
			#############################
			# Step 1: Data Augmentation #
			#############################
			if method == "BFS":
				# UNIT Test OK
				for key in self.obs_type_info.keys():
					# get the obseration state
					O = self.obs_type_info[key]['O']
					H = self.obs_type_info[key]['H']
					J = self.obs_type_info[key]['J']
					E = self.obs_type_info[key]['E']
					# translate the J to item id
					item_ids = [self.item_param_dict[j] for j in J]
					Ts = len(O)		
					X_mat = X_mat_dict[Ts]
					
					llk_vec={}
					pis={}
					l_mat={}
					z_llk= np.zeros((1,self.num_mixture))
					for z in range(self.num_mixture):
						llk_vec[z], pis[z], l_mat[z] = update_state_parmeters(X_mat, self.Mx,
								O,E,H,
								J, item_ids,
							   self.hazard_matrix, self.observ_prob_matrix, 
							   self.state_init_dist[z], self.state_transit_matrix[:,z,:,:,:], self.effort_prob_matrix,
							   is_effort, 
							   is_exit, 
							   hazard_state)
						z_llk[0,z] = sum(llk_vec[z])
					
					self.obs_type_info[key]['user_mixture'] = (z_llk*self.user_mixture_density/np.dot(z_llk, self.user_mixture_density)).tolist()[0]
					self.obs_type_info[key]['llk_vec'] = llk_vec
					self.obs_type_info[key]['pi'] = pis
					self.obs_type_info[key]['l_mat'] = l_mat
			else:
				raise Exception('Algorithm %s not implemented.' % method)
			
			# sample states backwards 
			X = np.empty((self.T, self.K),dtype=np.int)
			Z = np.empty((self.K, 1),dtype=np.int)
			for i in range(self.K):
				# check the key
				obs_key = self.obs_type_ref[i]
				user_mixture = self.obs_type_info[obs_key]['user_mixture']
				# sample user type
				z = random_choice(user_mixture)
				Z[i] = z
				# sample the state
				pi = self.obs_type_info[obs_key]['pi'][z]
				l_mat = self.obs_type_info[obs_key]['l_mat'][z]
				Ti = self.T_vec[i]
				
				X[Ti-1, i] = random_choice(pi)
				for t in range(Ti-1, 0,-1):
					pt = l_mat[t, X[t,i],:]
					if pt.sum()==0:
						raise Exception('Invalid transition kernel')
					X[t-1,i] = random_choice(pt.tolist())
			
			#############################
			# Step 2: Update Parameter  #
			#############################
			try:
				user_mixture_param = [self.prior_param['mixture'][z]+ np.sum(Z==z) for z in range(self.num_mixture)]
				new_user_mixture = np.random.dirichlet(user_mixture_param)
				
				# upate pi | Type 0 and 1 are low mastery, Type 2 are high mastery
				pi_params  = [[self.prior_param['pi'][z][x]+ np.sum(X[0,np.where(Z==z)[0]]==x) for x in range(self.Mx)] for z in range(self.num_mixture)]
				if self.num_mixture > 1:
					new_state_init_dist = draw_multilevel_pi(pi_params, self.num_mixture, self.Mx)
				else:
					new_state_init_dist = np.zeros((1,self.Mx))
					new_state_init_dist[0] = np.random.dirichlet(pi_params[0])
				
				# update l
				trans_matrix = np.zeros((self.J, self.num_mixture, self.Mx, self.Mx), dtype=np.int)
				for k in range(self.K):
					z = Z[k]
					for t in range(0, self.T_vec[k]):
						o_j = self.J_array[t,k]
						o_is_e = self.E_array[t,k]
						x1 = X[t,k]
						if t>0 and self.E_array[t-1,k]>0:
							l_j = self.J_array[t-1, k] # transition happens at t, item at t-1 takes credit
							x0 = X[t-1,k]
							trans_matrix[l_j,z,x0,x1] += 1
				
				new_state_transit_matrix = np.zeros((self.J,self.num_mixture,2,self.Mx,self.Mx))
				for j in range(self.J):
					# Need to ensure transition here
					state_transit_matrix = np.zeros((self.num_mixture, 2, self.Mx, self.Mx))
					for z in range(self.num_mixture):
						params =  [[self.prior_param['l'][z][m][n]+trans_matrix[j,z,m,n] for n in range(self.Mx)] for m in range(self.Mx)]				
						state_transit_matrix[z] = draw_l(params, self.Mx)					
					new_state_transit_matrix[j] = state_transit_matrix
					
				# update c	
				obs_cnt = np.zeros((self.unique_item_num, self.Mx, self.My)) # state,observ
				for k in range(self.K):
					for t in range(0, self.T_vec[k]):
						o_j = self.J_array[t,k]
						o_is_e = self.E_array[t,k]					
						if o_is_e:
							obs_cnt[self.item_param_dict[o_j], X[t,k], self.O_array[t,k]] += 1 
				
				new_observ_prob_matrix = np.zeros((self.J,self.Mx,self.My))
				for j in range(self.unique_item_num):
					c_params = [[self.prior_param['c'][x][y] + obs_cnt[j,x,y] for y in range(self.My)] for x in range(self.Mx)] 
					c_draws = draw_c(c_params, self.Mx, self.My)
					new_observ_prob_matrix[j] = c_draws			
				
				# update h
				if is_exit:
					if hazard_model == 'prop':
						if hazard_state == 'X':
							self.hazard_matrix, self.Lambdas, self.betas = prop_hazard(self.Mx, self.T_vec, X, self.H_array, self.Lambdas, self.betas)
						elif hazard_state == 'Y':
							self.hazard_matrix, self.Lambdas, self.betas = prop_hazard(self.My, self.T_vec, self.O_array, self.H_array, self.Lambdas, self.betas)
						else:
							raise Exception('Unknown dependent states! %s ' % hazard_state)
					elif hazard_model == 'cell':
						if hazard_state == 'X':
							self.hazard_matrix = cell_hazard(self.Mx, self.T_vec, X, self.H_array, self.prior_param['h'])
						elif hazard_state == 'Y':
							self.hazard_matrix = cell_hazard(self.My, self.T_vec, self.O_array, self.H_array, self.prior_param['h'])
						else:
							raise Exception('Unknown dependent states! %s ' % hazard_state)					
					else:
						raise Exception('Unknown hazard model! %s ' % hazard_model)
						
				
				# update e
				if is_effort:
					effort_cnt = np.zeros((self.J,self.Mx),dtype=np.int)
					effort_state_cnt = np.zeros((self.J,self.Mx),dtype=np.int)
					for k in range(self.K):
						for t in range(0, self.T_vec[k]):
							o_j = self.J_array[t,k]
							o_is_e = self.E_array[t,k]							
								
							effort_cnt[o_j, X[t,k]] += o_is_e
							effort_state_cnt[o_j, X[t,k]] += 1	
					for j in range(self.J):
						self.effort_prob_matrix[j] = [np.random.dirichlet((self.prior_param['e'][0]+effort_state_cnt[j,x]-effort_cnt[j,x], self.prior_param['e'][1]+effort_cnt[j,x])) for x in range(self.Mx)]
			except AttributeError as e:
				tot_error_cnt += 1
				print(e)
			except:
				# without disrupting the chain due to a bad draw in X
				tot_error_cnt += 1
				print(sys.exc_info()[0])
				continue				
			self.user_mixture_density = new_user_mixture
			self.state_init_dist = new_state_init_dist
			self.state_transit_matrix = new_state_transit_matrix
			self.observ_prob_matrix = new_observ_prob_matrix
			
			#############################
			# Step 3: Preserve the Chain#
			#############################
			
			lHat_vec = []
			for z in range(self.num_mixture):
				for j in range(self.J):
					for m in range(self.Mx-1):
						lHat_vec += self.state_transit_matrix[j,z,1,m,(m+1):self.Mx].tolist() 
			param_chain['l'][iter,:] = lHat_vec
			
			pi_vec = []
			for z in range(self.num_mixture):
				pi_vec += self.state_init_dist[z,0:-1].tolist()	
			param_chain['pi'][iter,:] = pi_vec
			param_chain['mixture'][iter,:] = self.user_mixture_density[0:-1]
			param_chain['c'][iter,:] = self.observ_prob_matrix[:,:,1:].reshape(self.unique_item_num*self.Mx*(self.My-1)).tolist()
			
			if is_exit:
				if hazard_model == 'prop':
					h_vec = []
					for i in range(Mh):
						h_vec += [self.Lambdas[i], self.betas[i] ]
					param_chain['h'][iter,:] = h_vec
				elif hazard_model == 'cell':
					param_chain['h'][iter,:] = self.hazard_matrix.reshape(1,Mh*self.T)					
			
			if is_effort:
				param_chain['e'][iter,:] = self.effort_prob_matrix[:,:,1].flatten()
			# update parameter chain here
			self.X = X
		return param_chain

	def _get_initial_param(self, init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort, is_exit, hazard_model, hazard_state):
		# c: probability of correct. Let cij=p(Y=j|X=i). 
		# pi: initial distribution of latent state, [Mx]
		# l: learning rate/pedagogical efficacy, 
		# e: probability of effort, [Mx]*nJ
		# Lambda: hazard rate with at time 0. scalar
		# betas: time trend of proportional hazard. scalar	
			
		# build the item dict
		self.unique_item_num, self.item_param_dict = get_item_dict(item_param_constraint, self.J)

		# build the prior dist
		# generate parameters from the prior
		if not prior_dist:
			pi_prior_list = []
			for z in range(self.num_mixture):
				pi_prior = []
				for x in range(self.Mx):
					if x != self.Mx-1:
						pi_prior.append(1)
					else:
						pi_prior.append(max(1,z+1))
				pi_prior_list.append(pi_prior)
		
			self.prior_param = {'l': [[[int(m<=n) for n in range(self.Mx)] for m in range(self.Mx)] for z in range(self.num_mixture)], # encoding the non-regressive state
								'e': [1, 1],
								'pi':pi_prior_list,
								'c' :[[y+1 for y in range(self.My)] for x in range(self.Mx)],
								'mixture':[1 for z in range(self.num_mixture)]
								}
		else:
			# TODO: check the specification of the prior
			self.prior_param = prior_dist
			
		# get the parameters
		if init_param:
			# for special purpose, allow for pre-determined starting point.
			param = copy.copy(init_param)
			# ALL items share the same prior for now
			self.user_mixture_density = param['mixture']
			self.observ_prob_matrix = param['c']
			self.effort_prob_matrix = param['e']
			self.state_init_dist = param['pi']
			self.state_transit_matrix = param['l'] 
			self.hazard_matrix = param['h'] 
		else:
			if zero_mass_set:
				if 'X' in zero_mass_set:
					for pos in zero_mass_set['X']:
						m,n = pos
						self.prior_param['l'][m][n] = 0
				if 'Y' in zero_mass_set:
					for pos in zero_mass_set['Y']:
						m,n = pos
						self.prior_param['c'][m][n] = 0
			
			#TODO: need to implement draws here
			if self.num_mixture>1:
				self.user_mixture_density = np.random.dirichlet(self.prior_param['mixture'])
				self.state_init_dist = draw_multilevel_pi(self.prior_param['pi'], self.num_mixture, self.Mx)
			else:
				self.user_mixture_density = 1.0
				self.state_init_dist = np.zeros((1,self.Mx))
				self.state_init_dist[0] = np.random.dirichlet(self.prior_param['pi'][0]) # wrap a list to allow for 1 mixture 
				
			self.state_transit_matrix = np.zeros((self.J,self.num_mixture,2,self.Mx,self.Mx))
			for j in range(self.J):
				for z in range(self.num_mixture):
					self.state_transit_matrix[j,z,:,:] = draw_l(self.prior_param['l'][0],self.Mx)
			
			self.observ_prob_matrix = np.array([draw_c(self.prior_param['c'], self.Mx, self.My) for j in range(self.unique_item_num)])
			
			if is_effort:
				self.effort_prob_matrix = np.array([[np.random.dirichlet(self.prior_param['e']) for x in range(self.Mx)] for j in range(self.J)])
			else:
				self.effort_prob_matrix = np.zeros((self.J, self.Mx, 2))
				self.effort_prob_matrix[:,:,1] = 1.0
			
			if is_exit:
				if hazard_model == 'cell':
					if hazard_state == 'X':
						Mh = self.Mx
					elif hazard_state == 'Y':
						Mh = self.My
					self.prior_param['h'] = np.ones((Mh,self.T,2))
					self.hazard_matrix = np.zeros((Mh,self.T))	
					for m in range(Mh):
						for t in range(self.T):
							self.hazard_matrix[m,t] = np.random.beta(self.prior_param['h'][m,t,0], self.prior_param['h'][m,t,1])
				elif hazard_model == 'prop':
					self.Lambda = 0.1
					self.beta = 0.01
					self.hazard_matrix = np.array([[self.Lambda*np.exp(self.beta*t) for t in range(self.T)] for x in range(self.Mx)])
			else:
				self.hazard_matrix = np.array([[0.0 for t in range(self.T)] for x in range(self.Mx)])
		
	
	def _work(self,max_iter, method, is_effort, is_exit, hazard_model, hazard_state, init_param, prior_dist, zero_mass_set, item_param_constraint):
		self._get_initial_param(init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort,is_exit,hazard_model,hazard_state)
		param_chain = self._MCMC(max_iter, method, is_effort, is_exit, hazard_model, hazard_state)
		return param_chain
	
	def estimate(self, data_array, 
					   prior_dist={}, init_param={}, 
					   Mx=None, num_mixture=1,
					   zero_mass_set={}, item_param_constraint=[], 
					   method='BFS', max_iter=1000, chain_num = 4, 
					   is_effort=False,
					   is_exit=False, hazard_model='cell', hazard_state='X',
					   is_parallel=True):
		
		# data = [(i,t,j,y,e,h)]  
		# i: learner id from 0:N-1
		# t: sequence id, t starts from 0
		# j: item id, from 0:J-1
		# y: response, 0 or 1
		# h(azard): if the spell ends here
		# e(effort): 0 or 1		
		self._load_observ(data_array)
		# My: the number of observation state. Assume that all items have the same My. Only 2 and 3 are accepted.
		# Me: number of effort state. Assume that all items have the same Me. Only 2 are accepted.
		# nJ: the number of items
		# K: the number of users
		# T: longest 		
		
		# Mx: the number of latent state.
		# Mx = My, unless otherwise specified
		if not Mx:
			self.Mx=self.My
		else:
			self.Mx=Mx
		self.num_mixture = num_mixture
		
		if self.num_mixture>2:
			raise Exception('The user mixture model cannot only accomendate two components')
		# the mixture can only accomendate Mx=2 and My=2
		if self.num_mixture == 2 and (self.Mx>2 or self.My>2):
			raise Exception('The user mixture model cannot only accomendate Mx=2 and My=2.')
		
		self._collapse_obser_state()
		
		# run MCMC
		if not is_parallel:
			param_chain_vec = []
			for iChain in range(chain_num):
				try:
					self._get_initial_param(init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort,is_exit,hazard_model, hazard_state)
					tmp_param_chain = self._MCMC(max_iter, method, is_effort, is_exit, hazard_model, hazard_state)
					param_chain_vec.append(tmp_param_chain)
				except:
					continue
		else:
			param_chain_vec = Parallel(n_jobs=chain_num)(delayed(self._work)(
			max_iter, method, is_effort, is_exit, hazard_model, hazard_state, init_param, prior_dist, zero_mass_set, item_param_constraint
			) for i in range(chain_num))
			
		# process
		burn_in = min(300, int(max_iter/2))
		self.param_chain = get_final_chain(param_chain_vec, burn_in, max_iter, is_exit, is_effort)	
		res = get_map_estimation(self.param_chain,is_exit, is_effort)
		
		return res
