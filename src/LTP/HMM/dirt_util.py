# encoding: utf-8
import numpy as np
import math
import copy
from collections import defaultdict


def logsum(log_prob_list, is_exact=False):
    if is_exact:
        return math.log(sum([math.exp(x) for x in log_prob_list]))
    else:
        xmax = max(log_prob_list)
        return xmax + math.log(sum([math.exp(x-xmax) for x in log_prob_list]))

def generate_states(max_level):
    """
    目前这个func就是一个place holder
    """
    states = np.zeros((max_level,1), dtype=int)
    for x in range(max_level):
        states[x]=x
    return states
    

def likelihood(X,Y,E,observ_prob_matrix, effort_prob_matrix, is_effort):
    # TODO: pass in param structure rather than individual elements. Not flexible!
    """
    X: int
    Y/E: bool int
    Obser_prob : Mx*My
    Effort_prob: Mx*2
    State_init: Mx*1
    """
    pe = 1 
    if is_effort:
        pe = effort_prob_matrix[X,E]
        if not E and Y:
            raise Exception('No effort only admits wrong answer.')
        elif not E and not Y:
            po = 1
        else:
            po = observ_prob_matrix[X,Y]
    else:
        po = observ_prob_matrix[X,Y]
    lk = po*pe
    
    if lk<0:
        raise ValueError('Negative likelihood.')
    
    return lk

def data_likelihood(X_val, data_logs, observ_prob_matrix, state_init_dist, effort_prob_matrix, is_effort):
    """
    获取多个log的likelihood
    # X:  Latent state
    # datalogs: [(j,y,e,n)]
    """
    llk = math.log(state_init_dist[X_val])
    for log in data_logs:
        j, y, e, n = log
        prob = likelihood(X_val, y,e,observ_prob_matrix[j], effort_prob_matrix[j], is_effort)
        llk += math.log(prob)
    
    return n*llk 
    

def get_llk_all_states(X_mat, data_logs, 
                       observ_prob_matrix, state_init_dist, effort_prob_matrix, is_effort):
    """
    计算所有X的概率
    """
    N_X = X_mat.shape[0]
    llk_vec = []
    for i in range(N_X):
        data_llk = data_likelihood(X_mat[i,0], data_logs, observ_prob_matrix, state_init_dist, effort_prob_matrix, is_effort)
        llk_vec.append(data_llk) 
        
    return np.array(llk_vec)



def update_state_parmeters(X_mat,data_logs,
                           observ_prob_matrix, state_init_dist, effort_prob_matrix,
                           is_effort, is_exact=False):
    """
    从数据中获得后验概率
    X_mat = Mx*1 array
    """
    #calculate the exhaustive state probablity
    llk_vec = get_llk_all_states(X_mat, data_logs, observ_prob_matrix, state_init_dist, effort_prob_matrix, is_effort)
    tot_llk = logsum(llk_vec, is_exact)
    #import ipdb;ipdb.set_trace()
    #if math.exp(tot_llk)<1e-40:
    #    import ipdb;ipdb.set_trace()
    #    raise ValueError('All likelihood are 0.')
    
    # pi
    Mx = X_mat.shape[0]
    pis = [math.exp(llk_vec[x]-tot_llk) for x in range(Mx)] # equal to draw one
    if abs(sum(pis)-1)>1e-10:
        raise ValueError('Pi does not sum up to 1.')
    return llk_vec, pis



def data_etl(data_array, invalid_item_ids = []):
    '''
    input: [i,j,y(,e)]

    output: 
    (1) user_dict: map input user id to consecutive int
    (2) item_dict: map input item id to consecutive int
    (3) data: [i,j,y(,e)] Add a t to indicate sequence length
    '''

    user_reverse_dict = {}
    item_reverse_dict = {}
    item_dict = {}
    user_counter = 0 # start from 0
    item_counter = 0 # start from 0
    tmp_dict = defaultdict(list)

    # process
    log_type = len(data_array[0])
    for log in data_array:
        if log_type == 3:
            learner_id, item_id, res = log
        elif log_type == 4:
            learner_id, item_id, res, effort = log
        else:
            raise Exception('The log format is not recognized.')
        
        if item_id in invalid_item_ids:
            continue

        if learner_id not in user_reverse_dict:
            user_reverse_dict[learner_id] = user_counter
            #user_dict[user_counter] = learner_id
            user_counter += 1
        if item_id not in item_reverse_dict:
            item_reverse_dict[item_id] = item_counter
            item_dict[item_counter] = item_id
            item_counter += 1

        learner_id_val = user_reverse_dict[learner_id]
        item_id_val = item_reverse_dict[item_id]
        log_key = str(learner_id_val)+'#'+str(item_id_val)

        if log_type == 3:
            tmp_dict[log_key].append((learner_id_val, item_id_val, res))
        elif log_type == 4:
            tmp_dict[log_key].append((learner_id_val, item_id_val, res, effort))

    # output
    data = []
    for logs in tmp_dict.values():
        data += logs
    
    sorted_data = sorted(data, key=lambda k:(k[0],k[1])) # resort by uid and t

    return item_dict, sorted_data

def filter_invalid_items(data_array):
    # check if any of the item has pure right or pure wrong
    item_all_cnt = defaultdict(int)
    item_right_cnt = defaultdict(int)
   
    # process
    log_type = len(data_array[0])
    for log in data_array:
        if log_type == 3:
            learner_id, item_id, res = log
        elif log_type == 4:
            learner_id, item_id, res, effort = log
        else:
            raise Exception('The log format is not recognized.')
        
        item_all_cnt[item_id] += 1
        #TODO: allow for non-binary check
        item_right_cnt[item_id] += res
    
    # filter
    invalid_items = []
    for item_id, all_cnt in item_all_cnt.items():
        accuracy = item_right_cnt[item_id]/all_cnt
        if accuracy <= 0.01 or accuracy >= 0.99:
            invalid_items.append(item_id)

    return invalid_items

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
        
def encode_log2state(logs):
    """
    输入：[(j,y,e)]
    事实上t无用

    输出:J|Y0E0|Y0E1|Y1E1
    """ 
    Y1E1s = []; Y0E1s = []; Y0E0s = []
    log_dict = {} 
    for log in logs:
        j = str(log[0])
        y = int(log[1])
        e = int(log[2])
        if j not in log_dict:
            log_dict[j] = defaultdict(int)
        log_dict[j][2**y+e] += 1    # 默认Y1E0不存在
    
    sorted_Js = sorted(log_dict.keys())
    for j in sorted_Js:
        Y0E0s.append(str(log_dict[j][1]))
        Y0E1s.append(str(log_dict[j][2]))
        Y1E1s.append(str(log_dict[j][3]))
   
    state_id = ','.join(sorted_Js) + '|' + ','.join(Y0E0s) + '|' + ','.join(Y0E1s) + '|' + ','.join(Y1E1s)
    return state_id 

def decode_state2log(state_id):
    """
    输入：J|Y0E0|Y0E1|Y1E1
    输出：[(j,y,e,n)]
    """
    def decode(strs):
        return [int(s) for s in strs.split(',') ]
    logs = []
    Js, Y0E0s, Y0E1s, Y1E1s = state_id.split('|')
    js = Js.split(','); y0e0s = decode(Y0E0s); y0e1s = decode(Y0E1s); y1e1s = decode(Y1E1s)
    for i in range(len(js)):
        j = int(js[i])
        if y0e0s[i]:
            logs.append((j, 0, 0, y0e0s[i]))
        if y0e1s[i]:
            logs.append((j, 0, 1, y0e1s[i]))
        if y1e1s[i]:
            logs.append((j, 1, 1, y1e1s[i]))

    return logs

def collapse_obser_state(learner_logs):
    
    obs_state_cnt = defaultdict(int)
    obs_state_ref = defaultdict(list)
    
    for k, logs in learner_logs.items():
        obs_state_key = encode_log2state(logs)
        obs_state_cnt[obs_state_key] += 1
        obs_state_ref[obs_state_key].append(k)
    return obs_state_cnt, obs_state_ref

def cache_state_info(type_keys):
    """
    The input is an iterator!
    Cache it to speed up. Avoid repetition in later sampling
    """
    # construct the space
    obs_state_info = {}
    for key in type_keys:
        obs_state_info[key] = {
                'data':decode_state2log(key),
                'item_ids':[int(x) for x in key.split('|')[0].split(',')]
                } 
   
    return obs_state_info

if __name__ == '__main__':
    
    logs = [(1,1,1),
            (1,0,1),
            (1,0,0)]
    state_id = encode_log2state(logs)
    print(state_id)
    print('1|1|1|1')
    print(decode_state2log(state_id))
    print([(1,0,0,1),(1,0,1,1),(1,1,1,1)])

    logs = [(2,1,1),
            (1,0,1),
            (3,0,0)]
    
    state_id = encode_log2state(logs)
    print(state_id,)
    print('1,2,3|0,0,1|1,0,0|0,1,0')
    print(decode_state2log(state_id))
    print([(1,0,1,1),(2,1,1,1),(3,0,0,1)])

    # unit test state generating
    X_mat = np.zeros((2,1),dtype=int);X_mat[1,0]=1
    print(X_mat)
    print(np.array([[0],[1]])) 
    # check for the conditional llk under both regime
    state_init_dist = np.array([0.6, 0.4])                        
    observ_prob_matrix = np.array([[[0.8,0.2],[0.1, 0.9]]])
    effort_prob_matrix = np.array([[[0.8,0.2],[0.1, 0.9]]])
    #import ipdb;ipdb.set_trace()

    data_logs=[[0,0,1,1],[0,1,1,1]]
    llk_vec =  get_llk_all_states(X_mat, data_logs, observ_prob_matrix, state_init_dist, effort_prob_matrix,False)
    print(llk_vec) 
    print(math.log(0.6*0.8*0.2), math.log(0.4*0.1*0.9)) 
