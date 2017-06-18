# encoding: utf-8
from collections import defaultdict
import copy
import math

import random
import sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .util import draw_c,  random_choice, get_item_dict
from .dirt_util import filter_invalid_items, data_etl
from .dirt_util import update_state_parmeters, generate_states, get_final_chain
from .dirt_util import get_map_estimation, get_percentile_estimation
from .dirt_util import collapse_obser_state, cache_state_info

class DIRT_MCMC(object):

    def _load_observ(self, data):

        self.K = len(set([x[0] for x in data])) # i
        self.J = len(set([x[1] for x in data])) # j
        
        # group by learner id
        learner_logs = defaultdict(list)
        for log in data:
            if len(log)==3:
                # The spell never ends; multiple item
                i,j,y = log
                is_e = 1
            elif len(log) == 4:
                i,j,y,is_e = log
            else:
                raise Exception('The log format is not recognized.')
            learner_logs[i].append((j,y,is_e)) 
        self.obs_state_cnt, self.obs_state_ref = collapse_obser_state(learner_logs)
        self.obs_state_info = cache_state_info(self.obs_state_cnt.keys()) 

    def _MCMC(self, max_iter, is_effort=False, is_robust=False):
        # initialize for iteration
        if not is_effort and self.effort_prob_matrix[:,:,0].sum() != 0: 
            raise Exception('Effort rates are not set to 1 while disabled the update in effort parameter.')


        param_chain = {
                        'pi':np.zeros((max_iter, self.Mx-1)),
                        'c': np.zeros((max_iter, (self.Mx*(self.My-1))*self.unique_item_num))
                    } 
            
        if is_effort:
            param_chain['e'] = np.zeros((max_iter, self.Mx*self.J))

        # cache the generated states
        X_mat = generate_states(self.Mx)

        tot_error_cnt = 0
        for iter in tqdm(range(max_iter)):
            if tot_error_cnt > 10:
                raise Exception('Too many erros in drawing')

            #############################
            # Step 1: Data Augmentation #
            #############################
            for obs_key in self.obs_state_info.keys():
                # get the state logs 
                data_logs = self.obs_state_info[obs_key]['data']
                item_ids =  self.obs_state_info[obs_key]['item_ids']
                 
                llk_vec={}
                pis={}
                #TODO: add in item id restriction
                llk_vec, pis = update_state_parmeters(X_mat, data_logs,
                        self.observ_prob_matrix, 
                    self.state_init_dist, self.effort_prob_matrix,
                        is_effort)
                #import ipdb;ipdb.set_trace()
                self.obs_state_info[obs_key]['llk_vec'] = llk_vec    # use for debug
                self.obs_state_info[obs_key]['pi'] = pis
                
            # sample states backwards 
            X = np.zeros((1, self.K),dtype=np.int)
            #import ipdb;ipdb.set_trace()
            for obs_key in self.obs_state_info.keys():
                pi = self.obs_state_info[obs_key]['pi']
                learner_ids = self.obs_state_ref[obs_key]
                #TODO: Generalize
                for i in learner_ids: 
                    X[0, i] = 1 if random.random()>=pi[0] else 0
            
            #############################
            # Step 2: Update Parameter  #
            #############################
            #try:  
            # upate pi | Type 0 and 1 are low mastery, Type 2 are high mastery
            pi_params  = [self.prior_param['pi'][x]+ np.sum(X[0,:]==x) for x in range(self.Mx)] 
            new_state_init_dist = np.zeros((1,self.Mx))
            new_state_init_dist = np.random.dirichlet(pi_params) 
                
            # update c 
            obs_cnt = np.zeros((self.unique_item_num, self.Mx, self.My)) # state,observ
            if is_effort:
                effort_cnt = np.zeros((self.J,self.Mx),dtype=np.int)
                effort_state_cnt = np.zeros((self.J,self.Mx),dtype=np.int)

            for obs_key in self.obs_state_info.keys():
                learner_ids = self.obs_state_ref[obs_key]
                data_logs = self.obs_state_info[obs_key]['data']
                for log in data_logs:
                    j, y, e, n = log
                    for k in learner_ids:
                        obs_cnt[self.item_param_dict[j], X[0,k], y] += e*n  # only valid effort count! 
                        if is_effort:
                            effort_cnt[j, X[0,k]] += e*n
                            effort_state_cnt[j, X[0,k]] += n  
            
            new_observ_prob_matrix = np.zeros((self.J,self.Mx,self.My))
            for item_id in range(self.unique_item_num):
                c_params = [[self.prior_param['c'][x][y] + obs_cnt[item_id,x,y] for y in range(self.My)] for x in range(self.Mx)] 
                try:
                    c_draws = draw_c(c_params, self.Mx, self.My) 
                except Exception as Err:
                    if is_robust:
                        #TODO: Find a better solution than assign the old value
                        new_observ_prob_matrix[item_id] = self.observ_prob_matrix[item_id]
                        continue
                    else:
                        raise Err
                new_observ_prob_matrix[item_id] = c_draws 
            
            # update e
            if is_effort:
                for j in range(self.J):
                    self.effort_prob_matrix[j] = [np.random.dirichlet((self.prior_param['e'][0]+effort_state_cnt[j,x]-effort_cnt[j,x], self.prior_param['e'][1]+effort_cnt[j,x])) for x in range(self.Mx)]
            '''
            except AttributeError as e:
                tot_error_cnt += 1
                print(e)
            except:
                # without disrupting the chain due to a bad draw in X
                tot_error_cnt += 1
                print(sys.exc_info()[0])
                continue        
            '''
            self.state_init_dist = new_state_init_dist
            self.observ_prob_matrix = new_observ_prob_matrix

            #############################
            # Step 3: Preserve the Chain#
            #############################


            pi_vec = self.state_init_dist[0:-1].tolist() 
            param_chain['pi'][iter,:] = pi_vec
            param_chain['c'][iter,:] = self.observ_prob_matrix[:,:,1:].reshape(self.unique_item_num*self.Mx*(self.My-1)).tolist()

            if is_effort:
                param_chain['e'][iter,:] = self.effort_prob_matrix[:,:,1].flatten()
        
        '''
        END of MCMC LOOP
        '''
        
        return param_chain

    def _get_initial_param(self, init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort):
        # c: probability of correct. Let cij=p(Y=j|X=i). 
        # pi: initial distribution of latent state, [Mx]
        # e: probability of effort, [Mx]*nJ
         
        # build the item dict
        self.unique_item_num, self.item_param_dict = get_item_dict(item_param_constraint, self.J)

        # build the prior dist
        # generate parameters from the prior
        if not prior_dist:
            pi_prior = [1 for x in range(self.Mx)]
            self.prior_param = {
                    'e': [1, 1],
                    'pi':pi_prior,
                    'c' :[[self.My-y for y in range(self.My)] for x in range(self.Mx)]
                    }
        else:
            self.prior_param = prior_dist
            
        # get the parameters
        if init_param:
            # for special purpose, allow for pre-determined starting point.
            param = copy.copy(init_param)
            # ALL items share the same prior for now
            self.observ_prob_matrix = param['c']
            self.effort_prob_matrix = param['e']
            self.state_init_dist = param['pi']
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
            
            self.state_init_dist = np.random.dirichlet(self.prior_param['pi']) # wrap a list to allow for 1 mixture  
            self.observ_prob_matrix = np.array([draw_c(self.prior_param['c'], self.Mx, self.My) for j in range(self.unique_item_num)])
            
            if is_effort:
                self.effort_prob_matrix = np.array([[np.random.dirichlet(self.prior_param['e']) for x in range(self.Mx)] for j in range(self.J)])
            else:
                self.effort_prob_matrix = np.zeros((self.J, self.Mx, 2))
                self.effort_prob_matrix[:,:,1] = 1.0

    def _work(self,max_iter,  is_effort, is_robust, init_param, prior_dist, zero_mass_set, item_param_constraint):
        self._get_initial_param(init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort)
        param_chain = self._MCMC(max_iter,  is_effort, is_robust)
        return param_chain

    def estimate(self, data_array, 
                prior_dist={}, init_param={}, 
                Mx=None, num_mixture=1,
                zero_mass_set={}, item_param_constraint=[], 
                max_iter=1000, chain_num = 1, 
                is_effort=False,
                is_parallel=False,
                is_robust=False):
        # data = [i,j,y(,e)]  
        # i: learner id from 0:N-1
        # j: item id, from 0:J-1
        # y: response, 0 or 1
        # e(effort): 0 or 1
        if item_param_constraint != []:
            raise Exception('The item parameter constraint cannot be set at the moment')
        # TODO: Allow for different My for different items
        # My: the number of observation state. Assume that all items have the same My. 
        # Me: number of effort state. Assume that all items have the same Me. Only 2 are accepted.
        # nJ: the number of items
        # K: the number of users
        # T: longest    
        
        invalid_items = filter_invalid_items(data_array)
        if invalid_items != []:
            if is_robust:
                self.item_dict, data = data_etl(data_array, invalid_item_ids=invalid_items) 
            else:
                raise Exception('Invalid items are :\n'+'\n'.join(invalid_items)) 
        else:
            self.item_dict, data = data_etl(data_array) 
            
        # Mx: the number of latent state.
        # Mx = My, unless otherwise specified
        self.My = len(set(x[2] for x in data))  # y
        if not Mx:
            self.Mx=self.My
        else:
            self.Mx=Mx
        if self.My < 2:
            raise Exception('The states of response is singular.')
        
        self._load_observ(data)        

        # run MCMC
        if not is_parallel:
            param_chain_vec = []
            for iChain in range(chain_num):
                self._get_initial_param(init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort)
                tmp_param_chain = self._MCMC(max_iter,  is_effort, is_robust)
                param_chain_vec.append(tmp_param_chain)
        else:
            param_chain_vec = Parallel(n_jobs=chain_num)(delayed(self._work)(
                max_iter,  is_effort, is_robust, init_param, prior_dist, zero_mass_set, item_param_constraint
            ) for i in range(chain_num))
        
        # update obj
        burn_in = min(300, int(max_iter/2))
        self.param_chain = get_final_chain(param_chain_vec, burn_in, max_iter, is_effort)  

    def get_item_param(self):
        if self.My != 2:
            raise Exception('Parameter not supported')
        
        
        # point estimation
        point_est = get_map_estimation(self.param_chain,'c').reshape(self.J, self.Mx*(self.My-1))
        ci_low = get_percentile_estimation(self.param_chain,'c', 10).reshape(self.J, self.Mx*(self.My-1))
        ci_high = get_percentile_estimation(self.param_chain,'c', 90).reshape(self.J, self.Mx*(self.My-1))

        
        param = {}
        for item_id_val, item_id in self.item_dict.items():
            param[item_id] = {
                    'point':point_est[item_id_val,:],
                    'ci':np.vstack((ci_low[item_id_val,:],ci_high[item_id_val]))
                    }

        return param

    def get_learner_param(self):
        
        learner_param = {
                    'point':get_map_estimation(self.param_chain, 'pi'),
                    'ci':[
                            get_percentile_estimation(self.param_chain, 'pi',10)[0],
                            get_percentile_estimation(self.param_chain, 'pi',90)[0]
                        ]
                }

        return learner_param

