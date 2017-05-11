# encoding: utf-8
import numpy as np
import copy


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
	
	
def get_map_estimation(param_chain, is_effort):
	res = {}
	res['c'] = param_chain['c'].mean(axis=0).tolist()
	res['pi'] = param_chain['pi'].mean(axis=0).tolist()
	if is_effort:
		res['e'] = param_chain['e'].mean(axis=0).tolist()
		
	return res	
    
    
    
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
