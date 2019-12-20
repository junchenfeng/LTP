# TODO: Update the learning curve generator
# TODO: Add the joint response generator
import random
from itertools import accumulate

import numpy as np

def random_choice(p_vec):
    cump = list(accumulate(p_vec))
    n = len(p_vec)

    if abs(cump[n - 1] - 1) > 1e-6:
        raise ValueError("probability does not add up to 1.")
    rn = random.random()
    for x in range(n):
        if rn < cump[x]:
            break
    return x


def update_mastery(mastery, learn_rate):
    return mastery + (1 - mastery) * learn_rate


def compute_success_rate(slip, guess, mastery):
    return guess * (1 - mastery) + (1 - slip) * mastery


# Bayesian Knowledge Tracing Algorithm
def forward_update_mastery(mastery, slip, guess, learn_rate, Y):
    if Y == 1:
        new_mastery = 1 - (1 - learn_rate) * (1 - mastery) * guess / (
            guess + (1 - slip - guess) * mastery
        )
    elif Y == 0:
        new_mastery = 1 - (1 - learn_rate) * (1 - mastery) * (1 - guess) / (
            1 - guess - (1 - slip - guess) * mastery
        )
    else:
        raise ValueError("Invalid response value.")
    return new_mastery


def generate_learning_curve(slip, guess, init_mastery, learn_rate, T):
    p = init_mastery
    lc = [compute_success_rate(slip, guess, p)]
    for t in range(1, T):
        p = update_mastery(p, learn_rate)
        lc.append(compute_success_rate(slip, guess, p))
    return lc


def logExpSum(llk_vec):
    llk_max = max(llk_vec)
    llk_sum = llk_max + np.log(np.exp(llk_vec - llk_max).sum())
    return llk_sum


def check_two_state_rank_order(c_mat):
    # input c_mat has shape Mx,2, ensure c_mat[k,1]<c_mat[k+1,1]
    Mx = c_mat.shape[0]
    is_valid = True
    for k in range(1, Mx):
        if c_mat[k, 1] - c_mat[k - 1, 1] <= 0:
            is_valid = False
            break
    return is_valid


def draw_c(param, Mx, My, max_iter=100):
    if len(param) != Mx:
        raise ValueError("Observation matrix is wrong on latent state dimension.")
    if len(param[0]) != My:
        raise ValueError("Observation matrix is wrong on observation dimension.")

    c_mat = np.zeros((Mx, My))
    if My == 2:
        iter = 0
        while not check_two_state_rank_order(c_mat) and iter < max_iter:
            for n in range(Mx):
                c_mat[n, :] = np.random.dirichlet(param[n])
            iter += 1
        if iter == max_iter:
            raise Exception("C is not drew.")
    else:
        for n in range(Mx):
            c_mat[n, :] = np.random.dirichlet(param[n])

    return c_mat


def draw_l(params, Mx):

    l_param = np.zeros((2, Mx, Mx))
    l_param[0] = np.identity(Mx)
    l_param[1] = np.zeros(Mx)
    for m in range(Mx):
        # TODO: it will be wrong if not diagnoal
        #  if it is zero, force constraint
        valid_params = [x for x in params[m] if x != 0]
        num_l_to_draw = len(valid_params)
        if num_l_to_draw == 0:
            raise Exception("learning rate parameters is wrong")
        else:
            valid_l_param = np.random.dirichlet(valid_params)
        l_param[1][m, (len(params[m])-num_l_to_draw):] = valid_l_param
    return l_param


def check_multi_level_pi(state_init_dist, num_mixture):
    is_valid = True
    for z in range(1, num_mixture):
        if state_init_dist[z, 1] > state_init_dist[z - 1, 1]:
            is_valid = True
        else:
            is_valid = False
            break
    return is_valid


def draw_multilevel_pi(pi_params, num_mixture, Mx):
    state_init_dist = np.zeros((num_mixture, Mx))
    max_pi_iter = 100
    pi_iter = 0
    while (
        not check_multi_level_pi(state_init_dist, num_mixture) and pi_iter < max_pi_iter
    ):
        for z in range(num_mixture):
            state_init_dist[z] = np.random.dirichlet(pi_params[z])
        pi_iter += 1
    if pi_iter == max_pi_iter:
        raise Exception("Initial density are not drew")
    else:
        return state_init_dist


def draw_multilevel_l(param_slow, param_fast, param_high, Mx):
    state_transit_matrix = np.zeros((3, 2, Mx, Mx))
    transit_slow = np.zeros((2, Mx, Mx))
    transit_fast = np.zeros((2, Mx, Mx))
    transit_high = np.zeros((2, Mx, Mx))
    l_max_iter = 100
    l_iter = 0
    # TODO: can only impose two state constraints
    while transit_slow[1, 0, 1] >= transit_fast[1, 0, 1] and l_iter < l_max_iter:
        transit_slow = draw_l(param_slow, Mx)
        transit_fast = draw_l(param_fast, Mx)
        transit_high = draw_l(param_high, Mx)
        l_iter += 1

    if l_iter == l_max_iter:
        raise Exception("learning rates are not drew.")
    else:
        state_transit_matrix[0, :, :, :] = transit_slow
        state_transit_matrix[1, :, :, :] = transit_fast
        state_transit_matrix[2, :, :, :] = transit_high
        return state_transit_matrix


def get_final_chain(param_chain_vec, start, end, is_exit, is_effort):
    # calcualte the llk for the parameters
    gap = max(int((end - start) / 100), 10)
    select_idx = range(start, end, gap)
    num_chain = len(param_chain_vec)

    # get rid of burn in
    param_chain = {}
    param_chain["l"] = np.vstack(
        [param_chain_vec[i]["l"][select_idx, :] for i in range(num_chain)]
    )
    param_chain["c"] = np.vstack(
        [param_chain_vec[i]["c"][select_idx, :] for i in range(num_chain)]
    )
    param_chain["pi"] = np.vstack(
        [param_chain_vec[i]["pi"][select_idx, :] for i in range(num_chain)]
    )
    param_chain["mixture"] = np.vstack(
        [param_chain_vec[i]["mixture"][select_idx, :] for i in range(num_chain)]
    )

    if is_exit:
        param_chain["h"] = np.vstack(
            [param_chain_vec[i]["h"][select_idx, :] for i in range(num_chain)]
        )
    if is_effort:
        param_chain["e"] = np.vstack(
            [param_chain_vec[i]["e"][select_idx, :] for i in range(num_chain)]
        )

    return param_chain


def get_map_estimation(param_chain, is_exit, is_effort):
    res = {}
    res["l"] = param_chain["l"].mean(axis=0).tolist()
    res["c"] = param_chain["c"].mean(axis=0).tolist()
    res["pi"] = param_chain["pi"].mean(axis=0).tolist()
    res["mixture"] = param_chain["mixture"].mean(axis=0).tolist()
    if is_exit:
        res["h"] = param_chain["h"].mean(axis=0).tolist()

    if is_effort:
        res["e"] = param_chain["e"].mean(axis=0).tolist()

    return res


def get_item_dict(item_param_constraint, J):
    item_param_dict = {}
    item_id = -1
    if not item_param_constraint:
        for j in range(J):
            item_id += 1
            item_param_dict[j] = item_id
    else:
        n_sib = len(item_param_constraint)
        for j in range(J):
            # check if the item has identical siblings
            sib_set_id = -1
            for i in range(n_sib):
                if j in item_param_constraint[i]:
                    sib_set_id = i
                    break
            if sib_set_id == -1:
                item_id += 1
                item_param_dict[j] = item_id
            else:
                # check if siblings have been registered
                sib_register_id = -1
                for k in item_param_constraint[sib_set_id]:
                    if k in item_param_dict:
                        sib_register_id = item_param_dict[k]
                        break
                if sib_register_id == -1:
                    item_id += 1
                    item_param_dict[j] = item_id
                else:
                    item_param_dict[j] = sib_register_id

    unique_item_num = item_id + 1
    item_param_dict = item_param_dict
    return unique_item_num, item_param_dict


if __name__ == "__main__":
    lc0 = generate_learning_curve(0.05, 0.2, 0.4, 0.4, 5)
    lc1 = generate_learning_curve(0.05, 0.35, 0.25, 0.4, 5)
    print(lc0)
    print(lc1)
