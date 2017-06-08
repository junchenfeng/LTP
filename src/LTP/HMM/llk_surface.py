# encoding:utf-8


import numpy as np
from itertools import product, groupby, combinations_with_replacement as cwr, permutations
from operator import mul

def arrangement(input_vec, num):

    all_seed_elems = sorted([x for x in cwr(input_vec, num)])
    all_possible_elems = []
    for seed_elem in all_seed_elems:
        for elem in permutations(seed_elem):
            all_possible_elems.append(elem)
    return sorted(list(k for k,_ in groupby(all_possible_elems)))



def generate_learner_param(param_vec, num_X, num_state=2):
    
    all_possible_states = arrangement(range(num_state), num_X) 
    all_possible_params = arrangement(param_vec, num_X) 

    valid_prob_vec = []
    for param_pair in all_possible_params:
        X_Array_prob_vec = []
        for states in all_possible_states:
            X_prob_vec = [param_pair[j]**states[j]*(1-param_pair[j])**(1-states[j]) for j in range(num_X)]
            X_Array_prob_vec.append(reduce(mul, X_prob_vec))
        valid_prob_vec.append(X_Array_prob_vec) 
    return sorted(list(k for k,_ in groupby(valid_prob_vec)))



def generate_item_param(param_vec, num_X, num_state=2):
    # assume all X has the same number of states
    if num_X>2:
        raise Exception("Not implemented")
    
    all_possible_params = arrangement(param_vec, num_state**num_X)
    # The 2 state structure admits c00< c01,c10< c11
    valid_param_pairs = []
    for param_pair in all_possible_params:
        if num_X == 2:
            if  param_pair[0] < param_pair[1] and param_pair[0] < param_pair[2] and param_pair[3] > param_pair[1] and param_pair[3] > param_pair[2]:
                valid_param_pairs.append(param_pair)
        elif num_X == 1:
            if param_pair[0] < param_pair[1]:
                valid_param_pairs.append(param_pair)
        else:
            raise Exception("Not implemented")
    return sorted(list(set(valid_param_pairs)))



def get_prob_singleton(pi_vec, c_param_vec, y=1):
    num_possible_states = len(pi_vec)
    prob = 0 
    for i in range(num_possible_states):
        # decode X
        prob += c_param_vec[i]**y*(1-c_param_vec[i])**(1-y) * pi_vec[i]
    return prob

def get_pi_vec(full_pi_vec, X_array):
    if X_array == "01":
        return full_pi_vec
    elif X_array in ["0","1"]:
        if len(full_pi_vec)==2:
            return full_pi_vec
        else:
            if X_array == "0":
                pi = full_pi_vec[0] + full_pi_vec[1]
                return [1-pi, pi]
            elif X_array == "1":
                pi = full_pi_vec[0] + full_pi_vec[2]
                return [1-pi, pi]
    else:
        raise Exception("Not implemented.")

def get_prob_composite(full_pi_vec, c_param_dict, y_x_ref_dict, y_dict):
    # Assume items are independent conditional on X 
    # Assume the c_param_dict is sorted from small to large by binary encoding 
    
    prob = 1
    for ans_id, ans_res in y_dict.iteritems():
        # find the X states array
        X_array = y_x_ref_dict[ans_id]
        pi_vec = get_pi_vec(full_pi_vec, X_array)
        c_param_vec = c_param_dict[ans_id]
        prob *= get_prob_singleton(pi_vec, c_param_vec, y=ans_res) 
    return prob



def get_sim_stat(y_x_ref_dict, full_pi_vec, c_param_dict):
    num_item = len(y_x_ref_dict)

    # generate simulate data
    all_possible_ans = arrangement([0,1],num_item)
    true_ans_stat = {}
    for ans_pair in all_possible_ans:
        y_dict = {}
        for j in range(num_item):
            y_dict['q'+str(j+1)] = ans_pair[j]
        
        prob = get_prob_composite(full_pi_vec, c_param_dict, y_x_ref_dict, y_dict)
        true_ans_stat[ans_pair] = prob
    return true_ans_stat


def get_llk_surface(pi_vec, c_vec, y_x_ref_dict, true_ans_stat):
    num_X=2 
    num_item = len(y_x_ref_dict)
    all_possible_ans = arrangement([0,1],num_item)
   
    # generate the possible c space
    c_param_list_1 = generate_item_param(c_vec, 1)
    c_param_list_2 = generate_item_param(c_vec, 2)

    num_c = [len(c_param_list_1) if len(y_x_ref_dict['q'+str(k+1)])==1 else len(c_param_list_2) for k in range(num_item)]
    item_param_idx = range(num_c[0])
    
    if num_item ==1:
        item_param_idx = [(x,) for x in item_param_idx]
    else:
        k=1
        while k < num_item:
            product_iterator = product(item_param_idx, range(num_c[k]))
            if k==1:
                item_param_idx = [x for x in product_iterator]
            else:
                two_lv_item_param_idx =[x for x in product_iterator]
                item_param_idx = [x[0]+(x[1],) for x in two_lv_item_param_idx] 
            k += 1

    # generate the possible pi space
    pi_param_list = generate_learner_param(pi_vec, num_X)
    

    # generate likelihood
    num_X_param = len(pi_param_list)
    num_Y_param = len(item_param_idx)
    llk_surface = np.zeros((num_X_param, num_Y_param))
    for i in range(num_X_param):
        for j in range(num_Y_param):
            # construct the parametr
            full_pi_vec = pi_param_list[i]
            c_param_dict = {}
            for k in range(num_item):
                qkey = 'q'+str(k+1)
                if len(y_x_ref_dict[qkey]) == 1:
                    param_vec = c_param_list_1[item_param_idx[j][k]]
                else:
                    param_vec = c_param_list_2[item_param_idx[j][k]]
                c_param_dict[qkey] = param_vec 
             
            for ans_pair in all_possible_ans:
                y_dict = {}
                for k in range(num_item):
                    y_dict['q'+str(k+1)] = ans_pair[k]
                prob = get_prob_composite(full_pi_vec, c_param_dict, y_x_ref_dict, y_dict)
                llk_surface[i,j] += true_ans_stat[ans_pair]*math.log(prob)

    # print out the max 
    opt_x, opt_y = np.unravel_index(llk_surface.argmax(), llk_surface.shape)
    print(pi_param_list[opt_x])
    opt_c_param = {}
    for k in range(num_item):
        qkey = 'q'+str(k+1)
        if len(y_x_ref_dict[qkey]) == 1:
            param_vec = c_param_list_1[item_param_idx[opt_y][k]]
        else:
            param_vec = c_param_list_2[item_param_idx[opt_y][k]]
        opt_c_param[qkey] = param_vec 
    print(opt_c_param) 
    return llk_surface

if __name__ == "__main__":
    import math
    pi_vec = [0.1, 0.3, 0.7, 0.9]
    c_vec  = [0.1, 0.3, 0.7, 0.9]

    #pi_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #c_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    """
    num_X = 1
    pi_param_list = generate_learner_param(pi_vec, num_X)
    c_param_list = generate_item_param(c_vec, num_X)
    prob_surface = np.zeros((len(pi_param_list), len(c_param_list)))
    for i in range(len(pi_param_list)):
        for j in range(len(c_param_list)):
            prob_surface[i,j] = get_prob_singleton(pi_param_list[i], c_param_list[j])
    """
    
    y_x_ref_dict = {"q1":"01"}
    true_full_pi_vec = [0.21, 0.49, 0.09, 0.21]
    true_c_param_dict = {
            "q1":[0.1, 0.3, 0.7, 0.9]
            }

    data_stat = get_sim_stat(y_x_ref_dict, true_full_pi_vec, true_c_param_dict)
    llk_surface_1 = get_llk_surface(pi_vec, c_vec, y_x_ref_dict, data_stat)


    y_x_ref_dict = {"q1":"01","q2":"1"}
    true_full_pi_vec = [0.21, 0.49, 0.09, 0.21]
    true_c_param_dict = {
            "q1":[0.1, 0.3, 0.7, 0.9],
            "q2":[0.1, 0.9]
            }
    data_stat = get_sim_stat(y_x_ref_dict, true_full_pi_vec, true_c_param_dict)
    llk_surface_2 = get_llk_surface(pi_vec, c_vec, y_x_ref_dict, data_stat)
    
    y_x_ref_dict = {"q1":"01","q2":"1","q3":"0"}
    true_full_pi_vec = [0.21, 0.49, 0.09, 0.21]
    true_c_param_dict = {
            "q1":[0.1, 0.3, 0.7, 0.9],
            "q2":[0.1, 0.9],
            "q3":[0.1, 0.9],
            }
    
    data_stat = get_sim_stat(y_x_ref_dict, true_full_pi_vec, true_c_param_dict)
    llk_surface_3 = get_llk_surface(pi_vec, c_vec, y_x_ref_dict, data_stat)
    
    
    
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    from matplotlib import cm

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    
    # to highlight the peak, use only 5 levels
    num_level = 5
    plt.subplot(311)  
    # flatten the data
    num_X,num_Y = llk_surface_1.shape
    X = np.arange(num_X); Y = np.arange(num_Y)
    X, Y = np.meshgrid(X,Y)
    Z = llk_surface_1[X,Y]
    #surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #im = plt.imshow(Z, interpolation="bilinear", origin='lower', cmap=cm.gray)
    
    CS = plt.contour(X,Y,Z,num_level,cmap=cm.coolwarm)

    plt.subplot(312)  
    num_X,num_Y = llk_surface_2.shape
    X = np.arange(num_X); Y = np.arange(num_Y)
    X, Y = np.meshgrid(X,Y)
    Z = llk_surface_2[X,Y]
    #im = plt.imshow(Z, interpolation="bilinear", origin='lower', cmap=cm.gray)
    CS = plt.contour(X,Y,Z,num_level,cmap=cm.coolwarm)

    plt.subplot(313)  
    num_X,num_Y = llk_surface_3.shape
    X = np.arange(num_X); Y = np.arange(num_Y)
    X, Y = np.meshgrid(X,Y)
    Z = llk_surface_3[X,Y]
    #im = plt.imshow(Z, interpolation="bilinear", origin='lower', cmap=cm.gray)
    CS = plt.contour(X,Y,Z,num_level,cmap=cm.coolwarm)

    #plt.show()
    fig.savefig("/home/junchen/gitlab/leibniz/LTP/sandbox/llk_contour.png")
    
