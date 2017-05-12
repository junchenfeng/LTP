# encoding: utf-8
import numpy as np
import copy
from collections import defaultdict

def generate_states(T, max_level):
    states = np.zeros((max_level,T), dtype=int)
    for x in range(max_level):
        states[x,:]=x
    return states
    

def state_llk(X, J, E, init_dist, transit_matrix):
    # X: vector of latent state, list
    # transit matrix is np array [t-1,t]
    #if X[0] == 1:
    prob = init_dist[X[0]]*np.product([transit_matrix[J[t-1], E[t-1], X[t-1], X[t]] for t in range(1,len(X))])


    return prob
    
def likelihood(X_val, O, E, J, item_ids, observ_prob_matrix, state_init_dist, effort_prob_matrix, is_effort):
    # X:  Latent state
    # O: observation
    # E: binary indicator, whether effort is exerted
    
    T = len(O)
    
    # P(O|X)
    po = 1
    pe = 1
    # P(E|X)
    if is_effort:
        # The effort is generated base on the initial X.
        for t in range(T):
            pe *= effort_prob_matrix[J[t], X_val, E[t]]
    
        for t in range(T):
            if E[t]!=0:
                po *= observ_prob_matrix[item_ids[t], X_val, O[t]]
            else:
                po *= 1.0 if O[t] == 0 else 0.0 # this is a strong built in restriction 
    else:
        for t in range(T):
            po *= observ_prob_matrix[item_ids[t],X_val,O[t]]
        
    # P(X)  
    px = state_init_dist[X_val]     
    lk = po*px*pe
    
    if lk<0:
        raise ValueError('Negative likelihood.')
    
    return lk

def get_llk_all_states(X_mat, O, E, J, item_ids, 
                       observ_prob_matrix, state_init_dist, effort_prob_matrix, is_effort):
    N_X = X_mat.shape[0]
    llk_vec = []
    for i in range(N_X):
        llk_vec.append( likelihood(X_mat[i,0], O, E, J, item_ids, observ_prob_matrix, state_init_dist, effort_prob_matrix, is_effort) )
        
    return np.array(llk_vec)

def get_single_state_llk(X_mat, llk_vec, t, x):
    res = llk_vec[X_mat[:,t]==x].sum() 
    return res


def update_state_parmeters(X_mat, Mx, 
                           O,E,
                           J,item_ids,
                           observ_prob_matrix, state_init_dist, effort_prob_matrix,
                           is_effort):
    #calculate the exhaustive state probablity
    Ti = len(O)
    llk_vec = get_llk_all_states(X_mat, O, E, J, item_ids,
                                observ_prob_matrix, state_init_dist, effort_prob_matrix, 
                                is_effort)
    
    if abs(llk_vec.sum())<1e-40:
        raise ValueError('All likelihood are 0.')
    
    # pi
    tot_llk=llk_vec.sum()
    pis = [get_single_state_llk(X_mat, llk_vec, Ti-1, x)/tot_llk for x in range(Mx)] # equal to draw one
    
    return llk_vec, pis



def data_etl(data_array):
    '''
    input: [i,j,y(,e)]

    output: 
    (1) user_dict: map input user id to consecutive int
    (2) item_dict: map input item id to consecutive int
    (3) data: [i,t,j,y(,e)] Add a t to indicate sequence length
    '''

    user_reverse_dict = {}
    item_reverse_dict = {}
    user_log_cnt = defaultdict(int)
    item_dict = {}
    user_counter = 0 # start from 0
    item_counter = 0 # start from 0
    tmp_dict = defaultdict(list)

    # process
    log_type = len(data_array[0])
    for log in data_array:
        if log_type == 3:
            user_id, item_id, res = log
        elif log_type == 4:
            user_id, item_id, res, effort = log
        else:
            raise Exception('The log format is not recognized.')
       
        if user_id not in user_reverse_dict:
            user_reverse_dict[user_id] = user_counter
            #user_dict[user_counter] = user_id
            user_counter += 1
        if item_id not in item_reverse_dict:
            item_reverse_dict[item_id] = item_counter
            item_dict[item_counter] = item_id
            item_counter += 1

        user_id_val = user_reverse_dict[user_id]
        item_id_val = item_reverse_dict[item_id]
        log_key = str(user_id_val)+'#'+str(item_id_val)

        t = user_log_cnt[user_id_val] 
        if log_type == 3:
            tmp_dict[log_key].append((user_id_val, t, item_id_val, res))
        elif log_type == 4:
            tmp_dict[log_key].append((user_id_val, t, item_id_val, res, effort))

        user_log_cnt[user_id_val] += 1
    # output
    data = []
    for logs in tmp_dict.values():
        data += logs
    
    sorted_data = sorted(data, key=lambda k:(k[0],k[1])) # resort by uid and t

    return item_dict, sorted_data

def get_final_chain(param_chain_vec, start, end, is_effort):
	# calcualte the llk for the parameters
	gap = max(int((end-start)/100), 10)
	select_idx = range(start, end, gap)
	num_chain = len(param_chain_vec)
	
	# get rid of burn in
	param_chain = {}
	param_chain['c'] = np.vstack([param_chain_vec[i]['c'][select_idx, :] for i in range(num_chain)])
	param_chain['pi'] = np.vstack([param_chain_vec[i]['pi'][select_idx, :] for i in range(num_chain)])

	if is_effort:
		param_chain['e'] = np.vstack([param_chain_vec[i]['e'][select_idx, :] for i in range(num_chain)])

	return param_chain
	
	
def get_map_estimation(param_chain, field_name):	
	return param_chain[field_name].mean(axis=0)


def get_percentile_estimation(param_chain, field_name, pct):
    return np.percentile(param_chain[field_name], pct ,axis=0)
        
    
    
if __name__ == '__main__':
    # unit test state generating
    X_mat = generate_states(2,2)
    print(X_mat)
    print(np.array([[0,0],[1,1]])) 
    # check for the conditional llk under both regime
    state_init_dist = np.array([0.6, 0.4])                        
    observ_prob_matrix = np.array([[[0.8,0.2],[0.1, 0.9]]])
    T= 5
    effort_prob_matrix = []
    
    X = [0,1]
    O = [0,1]
    E = [1,1]
    J = [0,0]
    item_ids = [0,0]
    llk_vec =  get_llk_all_states(X_mat, O, E, J, item_ids, observ_prob_matrix, state_init_dist, effort_prob_matrix,False)
    print(llk_vec) 
    print(0.6*0.8*0.2, 0.4*0.1*0.9) 
