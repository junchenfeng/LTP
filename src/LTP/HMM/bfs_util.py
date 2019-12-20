import numpy as np

def generate_states(T, max_level, delta_level):
    states = []
    for t in range(T):
        if t == 0:
            for k in range(max_level):
                states.append([k])
        else:
            # find the number of seed states
            new_states = []
            nS = len(states)
            for i in range(nS):
                x0 = states[i][-1]
                for k in range(x0, max_level):
                    if (k - x0) > delta_level:
                        continue
                    else:
                        new_states.append(states[i] + [k])
            states = new_states
    return np.array(states)


def survivial_llk(h, H):
    # h, T*1 hazard rate
    # T, spell length
    # H, whether right censored
    T = len(h)
    if T == 1:
        base_prob = 1
    else:
        # has survived T-1 period
        base_prob = np.product(1 - h[:-1])

    prob = base_prob * (H * h[-1] + (1 - H) * (1 - h[-1]))
    return prob


def state_llk(X, J, E, init_dist, transit_matrix):
    # X: vector of latent state, list
    # transit matrix is np array [t-1,t]
    # if X[0] == 1:
    prob = init_dist[X[0]] * np.product(
        [transit_matrix[J[t - 1], E[t - 1], X[t - 1], X[t]] for t in range(1, len(X))]
    )

    return prob


def likelihood(
    X,
    O,
    E,
    H,
    J,
    item_ids,
    hazard_matrix,
    observ_prob_matrix,
    state_init_dist,
    state_transit_matrix,
    effort_prob_matrix,
    is_effort,
    is_exit,
    hazard_state,
):
    # X:  Latent state
    # O: observation
    # H: binary indicator, whether the spell is ended
    # E: binary indicator, whether effort is exerted

    T = len(X)
    # P(H|X)
    if is_exit:
        if hazard_state == "X":
            h = np.array([hazard_matrix[X[t], t] for t in range(T)])
        elif hazard_state == "Y":
            h = np.array([hazard_matrix[O[t], t] for t in range(T)])
        ph = survivial_llk(h, H)
    else:
        ph = 1

    # P(O|X)
    po = 1
    pe = 1
    # P(E|X)
    if is_effort:
        # The effort is generated base on the initial X.
        for t in range(T):
            pe *= effort_prob_matrix[J[t], X[t], E[t]]

        for t in range(T):
            if E[t] != 0:
                po *= observ_prob_matrix[item_ids[t], X[t], O[t]]
            else:
                po *= 1.0 if O[t] == 0 else 0.0  # this is a strong built in restriction
    else:
        E = [1 for x in X]
        for t in range(T):
            po *= observ_prob_matrix[item_ids[t], X[t], O[t]]

    # P(X)
    px = state_llk(X, J, E, state_init_dist, state_transit_matrix)

    lk = ph * po * px * pe

    if lk < 0:
        raise ValueError("Negative likelihood.")

    return lk


def get_llk_all_states(
    X_mat,
    O,
    E,
    H,
    J,
    item_ids,
    hazard_matrix,
    observ_prob_matrix,
    state_init_dist,
    state_transit_matrix,
    effort_prob_matrix,
    is_effort,
    is_exit,
    hazard_state,
):
    N_X = X_mat.shape[0]
    llk_vec = []
    for i in range(N_X):
        X = [int(x) for x in X_mat[i, :].tolist()]
        llk_vec.append(
            likelihood(
                X,
                O,
                E,
                H,
                J,
                item_ids,
                hazard_matrix,
                observ_prob_matrix,
                state_init_dist,
                state_transit_matrix,
                effort_prob_matrix,
                is_effort,
                is_exit,
                hazard_state,
            )
        )

    return np.array(llk_vec)


def get_single_state_llk(X_mat, llk_vec, t, x):
    res = llk_vec[X_mat[:, t] == x].sum()
    return res


def get_joint_state_llk(X_mat, llk_vec, t, x1, x2):
    if t == 0:
        raise ValueError("t must > 0.")
    res = llk_vec[(X_mat[:, t - 1] == x1) & (X_mat[:, t] == x2)].sum()
    return res


def update_state_parmeters(
    X_mat,
    Mx,
    O,
    E,
    H,
    J,
    item_ids,
    hazard_matrix,
    observ_prob_matrix,
    state_init_dist,
    state_transit_matrix,
    effort_prob_matrix,
    is_effort,
    is_exit,
    hazard_state,
):
    # calculate the exhaustive state probablity
    Ti = len(O)
    llk_vec = get_llk_all_states(
        X_mat,
        O,
        E,
        H,
        J,
        item_ids,
        hazard_matrix,
        observ_prob_matrix,
        state_init_dist,
        state_transit_matrix,
        effort_prob_matrix,
        is_effort,
        is_exit,
        hazard_state,
    )

    if abs(llk_vec.sum()) < 1e-40:
        raise ValueError("All likelihood are 0.")

    # pi
    tot_llk = llk_vec.sum()
    pis = [get_single_state_llk(X_mat, llk_vec, Ti - 1, x) / tot_llk for x in range(Mx)]

    # learning rate
    l_mat = np.zeros((Ti, Mx, Mx))  # T,X_{t+1},X_t
    for t in range(Ti - 1, 0, -1):
        l_mat[t, 0, 0] = 1  # the 0 state in t, must implies 0 in t-1
        for m in range(1, Mx):
            pNext = get_single_state_llk(X_mat, llk_vec, t, m)
            if pNext != 0:
                for n in range(Mx):
                    # P(X_{t-1},X_t)/P(X_t)
                    l = get_joint_state_llk(X_mat, llk_vec, t, n, m) / pNext
                    if not (l >= 0 and l <= 1):
                        if not (l > 1 and l - 1 < 0.00001):
                            raise ValueError("Learning rate is wrong for X=%d." % x)
                        else:
                            l = 1.0
                    l_mat[t, m, n] = l
            # If pNext is 0, then there is no probability the state will transite in

    return llk_vec, pis, l_mat


if __name__ == "__main__":
    # check for the conditional llk under both regime
    state_init_dist = np.array([0.6, 0.4])
    state_transit_matrix = np.array([[[[1, 0], [0, 1]], [[0.7, 0.3], [0, 1]]]])
    observ_prob_matrix = np.array([[[0.8, 0.2], [0.1, 0.9]]])
    T = 5
    Lambda = [0.6, 0.2]
    betas = [np.log(1), np.log(1)]
    h0_vec = [Lambda[0] * np.exp(betas[0] * t) for t in range(T)]
    h1_vec = [Lambda[1] * np.exp(betas[1] * t) for t in range(T)]

    hazard_matrix = np.array([h0_vec, h1_vec])
    effort_prob_matrix = []

    X = [0, 1]
    O = [0, 1]
    E = [1, 1]
    H = 0
    J = [0, 0]
    item_ids = [0, 0]
    X_mat = generate_states(2, 2, 1)
    llk_vec_null = get_llk_all_states(
        X_mat,
        O,
        E,
        H,
        J,
        item_ids,
        hazard_matrix,
        observ_prob_matrix,
        state_init_dist,
        state_transit_matrix,
        effort_prob_matrix,
        False,
        False,
        "X",
    )
    llk_vec_X = get_llk_all_states(
        X_mat,
        O,
        E,
        H,
        J,
        item_ids,
        hazard_matrix,
        observ_prob_matrix,
        state_init_dist,
        state_transit_matrix,
        effort_prob_matrix,
        False,
        True,
        hazard_state="X",
    )
    llk_vec_Y = get_llk_all_states(
        X_mat,
        O,
        E,
        H,
        J,
        item_ids,
        hazard_matrix,
        observ_prob_matrix,
        state_init_dist,
        state_transit_matrix,
        effort_prob_matrix,
        False,
        True,
        hazard_state="Y",
    )

    l_null = get_joint_state_llk(X_mat, llk_vec_null, 1, 0, 1) / get_single_state_llk(
        X_mat, llk_vec_null, 0, 0
    )
    l_x = get_joint_state_llk(X_mat, llk_vec_X, 1, 0, 1) / get_single_state_llk(
        X_mat, llk_vec_X, 0, 0
    )
    l_y = get_joint_state_llk(X_mat, llk_vec_Y, 1, 0, 1) / get_single_state_llk(
        X_mat, llk_vec_Y, 0, 0
    )

    print(l_null, l_y, l_x)
