# encoding: utf-8

from collections import defaultdict
import copy
import math


import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .util import draw_c,  random_choice, get_item_dict
from .dirt_util import get_map_estimation, get_final_chain, update_state_parmeters, generate_states


class DIRT_MCMC(object):

    def _load_observ(self, data):

        self.K = len(set([x[0] for x in data]))
        self.T = max([x[1] for x in data]) + 1
        self.J = len(set([x[2] for x in data]))
        self.My = len(set(x[3] for x in data))

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
                i,t,j,y,is_e = log
            else:
                raise Exception('The log format is not recognized.')
            self.O_array[t, i] = y
            self.J_array[t, i] = j
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


    def _collapse_obser_state(self):
        self.obs_type_cnt = defaultdict(int)
        self.obs_type_ref = {}
        for k in range(self.K):
            obs_type_key ='|'.join(str(y) for y in self.O_data[k]) + '-' + '|'.join(str(j) for j in self.J_data[k]) + '-' + '|'.join(str(e) for e in self.E_data[k])
            self.obs_type_cnt[obs_type_key] += 1
            self.obs_type_ref[k] = obs_type_key

        # construct the space
        self.obs_type_info = {}
        for key in self.obs_type_cnt.keys():
            O_s, J_s, E_s = key.split('-')
            self.obs_type_info[key] = {'O':[int(x) for x in O_s.split('|')], 'J':[int(x) for x in J_s.split('|')], 'E':[int(x) for x in E_s.split('|')]}
        
    def _MCMC(self, max_iter, method='BFS', is_effort=False):
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
        X_mat_dict = {}
        for t in range(1,self.T+1):
            X_mat_dict[t] = generate_states(t, self.Mx)

        tot_error_cnt = 0
        for iter in tqdm(range(max_iter)):
            if tot_error_cnt > 10:
                raise Exception('Too many erros in drawing')

            #############################
            # Step 1: Data Augmentation #
            #############################
            for key in self.obs_type_info.keys():
                # get the obseration state
                O = self.obs_type_info[key]['O']
                J = self.obs_type_info[key]['J']
                E = self.obs_type_info[key]['E']
                # translate the J to item id
                item_ids = [self.item_param_dict[j] for j in J]
                Ts = len(O) 
                X_mat = X_mat_dict[Ts]
                
                llk_vec={}
                pis={}
                llk_vec, pis = update_state_parmeters(X_mat, self.Mx,
                    O,E,
                    J, item_ids,
                    self.observ_prob_matrix, 
                    self.state_init_dist, self.effort_prob_matrix,
                        is_effort)
                
                self.obs_type_info[key]['llk_vec'] = llk_vec
                self.obs_type_info[key]['pi'] = pis
                
            # sample states backwards 
            X = np.zeros((self.T, self.K),dtype=np.int)
            for i in range(self.K):
                obs_key = self.obs_type_ref[i]
                # sample the state
                pi = self.obs_type_info[obs_key]['pi']
                X[:, i] = random_choice(pi)
            
            #############################
            # Step 2: Update Parameter  #
            #############################
                
            # upate pi | Type 0 and 1 are low mastery, Type 2 are high mastery
            pi_params  = [self.prior_param['pi'][x]+ np.sum(X[0,:]==x) for x in range(self.Mx)] 
            new_state_init_dist = np.zeros((1,self.Mx))
            new_state_init_dist = np.random.dirichlet(pi_params) 
                
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
            
            # update parameter chain here
            self.X = X
        
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
                    'c' :[[y+1 for y in range(self.My)] for x in range(self.Mx)]
                    }
        else:
            # TODO: check the specification of the prior
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
            
            #TODO: need to implement draws here
            self.state_init_dist = np.random.dirichlet(self.prior_param['pi']) # wrap a list to allow for 1 mixture  
            self.observ_prob_matrix = np.array([draw_c(self.prior_param['c'], self.Mx, self.My) for j in range(self.unique_item_num)])
            
            if is_effort:
                self.effort_prob_matrix = np.array([[np.random.dirichlet(self.prior_param['e']) for x in range(self.Mx)] for j in range(self.J)])
            else:
                self.effort_prob_matrix = np.zeros((self.J, self.Mx, 2))
                self.effort_prob_matrix[:,:,1] = 1.0

    def _work(self,max_iter, method, is_effort, init_param, prior_dist, zero_mass_set, item_param_constraint):
        self._get_initial_param(init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort)
        param_chain = self._MCMC(max_iter, method, is_effort)
        return param_chain

    def estimate(self, data_array, 
                prior_dist={}, init_param={}, 
                Mx=None, num_mixture=1,
                zero_mass_set={}, item_param_constraint=[], 
                method='BFS', max_iter=1000, chain_num = 4, 
                is_effort=False,
                is_parallel=True):

        # data = [(i,t,j,y,e,h)]  
        # i: learner id from 0:N-1
        # t: sequence id, t starts from 0
        # j: item id, from 0:J-1
        # y: response, 0 or 1
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

        self._collapse_obser_state()

        # run MCMC
        if not is_parallel:
            param_chain_vec = []
            for iChain in range(chain_num):
                try:
                    self._get_initial_param(init_param, prior_dist, zero_mass_set, item_param_constraint, is_effort)
                    tmp_param_chain = self._MCMC(max_iter, method, is_effort)
                    param_chain_vec.append(tmp_param_chain)
                except:
                    continue
        else:
            param_chain_vec = Parallel(n_jobs=chain_num)(delayed(self._work)(
                max_iter, method, is_effort, init_param, prior_dist, zero_mass_set, item_param_constraint
            ) for i in range(chain_num))
            
        # process
        burn_in = min(300, int(max_iter/2))
        self.param_chain = get_final_chain(param_chain_vec, burn_in, max_iter, is_effort)  
        res = get_map_estimation(self.param_chain, is_effort)

        return res