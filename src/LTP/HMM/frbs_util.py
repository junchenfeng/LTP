def get_E(E, t, T):
    if E == 0:
        Et = 0
    else:
        if t == T:
            Et = 1
        else:
            Et = 0
    return Et


def __update_pi(self, t, E, V, observ, item_id, pi_vec, P_mat, is_effort=False):
    # pi(i,t) = P(X_t=i|O1,...,O_t,\theta)
    if t == 0:
        if not E:
            pa0 = 1 - self.hazard_matrix[0, t]
            pa1 = 1 - self.hazard_matrix[1, t]
            pa2 = 1 - self.hazard_matrix[2, t]
        else:
            pa0 = self.hazard_matrix[0, t]
            pa1 = self.hazard_matrix[1, t]
            pa2 = self.hazard_matrix[2, t]

        if not is_effort:
            po0 = self.observ_prob_matrix[item_id, 0, observ]
            po1 = self.observ_prob_matrix[item_id, 1, observ]  # always guess
            po2 = self.observ_prob_matrix[
                item_id, 2, observ
            ]  # if no effort, allow for guess
        else:
            if V == 1:
                po0 = self.observ_prob_matrix[item_id, 0, observ]
                po1 = self.observ_prob_matrix[item_id, 1, observ]  # always guess
                po2 = self.observ_prob_matrix[
                    item_id, 2, observ
                ]  # if no effort, allow for guess
            else:
                if observ == 0:
                    po0 = 1
                    po1 = 1
                    po2 = 1
                else:
                    po0 = 0
                    po1 = 0
                    po2 = 0
        """
		if is_effort:
			pv0 = self.valid_prob_matrix[item_id, 0, V]
			pv1 = self.valid_prob_matrix[item_id, 1, V]
			pv2 = self.valid_prob_matrix[item_id, 2, V]
		else:
			pv0 = 1
			pv1 = 1
			pv2 = 1
		"""
        # pi(i,0) = P(X_0=i|O0,\theta)
        p0y = self.pi0 * po0 * pa0  # * pv0
        p1y = (1 - self.pi - self.pi0) * po1 * pa1  # * pv1
        p2y = self.pi * po2 * pa2  # * pv2
        py = p0y + p1y + p2y

        pi_vec[t, :] = [p0y / py, p1y / py, p2y / py]

    else:
        # pi(i,t) = sum_{j} P(j,i,t) where P(j,i,t) is the (j,i)the element of transition matrix P
        pi_vec[t, :] = P_mat[t - 1, :, :].sum(axis=0)

    return pi_vec


def __update_P(
    self, t, E, item_id_l, V, observ, item_id_O, pi_vec, P_mat, is_effort=False
):

    p_raw = np.zeros((3, 3))

    if not E:
        pa0 = 1 - self.hazard_matrix[0, t + 1]
        pa1 = 1 - self.hazard_matrix[1, t + 1]
        pa2 = 1 - self.hazard_matrix[2, t + 1]
    else:
        pa0 = self.hazard_matrix[0, t + 1]
        pa1 = self.hazard_matrix[1, t + 1]
        pa2 = self.hazard_matrix[2, t + 1]

    if not is_effort:
        po0 = self.observ_prob_matrix[item_id, 0, observ]
        po1 = self.observ_prob_matrix[item_id, 1, observ]  # always guess
        po2 = self.observ_prob_matrix[
            item_id, 2, observ
        ]  # if no effort, allow for guess
    else:
        if V == 1:
            po0 = self.observ_prob_matrix[item_id, 0, observ]
            po1 = self.observ_prob_matrix[item_id, 1, observ]  # always guess
            po2 = self.observ_prob_matrix[
                item_id, 2, observ
            ]  # if no effort, allow for guess
        else:
            if observ == 0:
                po0 = 1
                po1 = 1
                po2 = 1
            else:
                po0 = 0
                po1 = 0
                po2 = 0

    if is_effort:
        pv0 = self.valid_prob_matrix[item_id_l, 0, V]
        pv1 = self.valid_prob_matrix[item_id_l, 1, V]
        pv2 = self.valid_prob_matrix[item_id_l, 2, V]
    else:
        pv0 = 1
        pv1 = 1
        pv2 = 1

    p_raw[0, 0] = max(
        pi_vec[t, 0] * self.state_transit_matrix[item_id_l, V, 0, 0] * po0 * pa0 * pv0,
        0.0,
    )
    p_raw[1, 1] = max(
        pi_vec[t, 1] * self.state_transit_matrix[item_id_l, V, 1, 1] * po1 * pa1 * pv1,
        0.0,
    )
    p_raw[2, 2] = max(
        pi_vec[t, 2] * self.state_transit_matrix[item_id_l, V, 2, 2] * po2 * pa2 * pv2,
        0.0,
    )

    p_raw[0, 1] = max(
        pi_vec[t, 0] * self.state_transit_matrix[item_id_l, V, 0, 1] * po1 * pa1 * pv1,
        0.0,
    )
    p_raw[1, 2] = max(
        pi_vec[t, 1] * self.state_transit_matrix[item_id_l, V, 1, 2] * po2 * pa2 * pv2,
        0.0,
    )

    P_mat[t, :, :] = p_raw / p_raw.sum()

    return P_mat


def __forward_recursion(self, is_exit=False, is_effort=False):
    for key in self.obs_type_info.keys():
        # get the obseration state
        Os = self.obs_type_info[key]["O"]
        Js = self.obs_type_info[key]["J"]
        E = self.obs_type_info[key]["E"]
        Vs = self.obs_type_info[key]["V"]
        # calculate the exhaustive state probablity
        T = len(Os)

        # if there is a only 1 observations, the P matrix does not exist, pi vector will the first observation
        pi_vec = np.zeros((T, 3))
        P_mat = np.zeros((T - 1, 3, 3))
        for t in range(T):
            Et = get_E(E, t, T)
            # The learning happens simulateneously with response. Learning in doing.
            pi_vec = self.__update_pi(
                t, Et, Vs[t], Os[t], Js[t], pi_vec, P_mat, is_effort
            )
            if t != T - 1 and T != 1:
                Et = get_E(E, t + 1, T - 1)
                P_mat = self.__update_P(
                    t,
                    Et,
                    Js[t + 1],
                    Vs[t + 1],
                    Os[t + 1],
                    Js[t + 1],
                    pi_vec,
                    P_mat,
                    is_effort,
                )
        self.obs_type_info[key]["pi"] = pi_vec
        self.obs_type_info[key]["P"] = P_mat


def __backward_sampling_scheme(self, is_exit=False, is_effort=False):
    for obs_key in self.obs_type_info.keys():
        pi_vec = self.obs_type_info[obs_key]["pi"]
        P_mat = self.obs_type_info[obs_key]["P"]
        T = pi_vec.shape[0]
        # This hard codes the fact that uninitiated is an obserbing state
        sample_p_vec = np.zeros((T, 3, 3))
        init_p_vec = np.zeros((3,))
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                init_p_vec = [min(pi, 1.0) for pi in pi_vec[t, :]]
            else:
                for x in range(1, 3):
                    # th problem is really to sample state 1 and 2 if the previous state is not 0
                    sample_p_vec[t, x, :] = P_mat[t, :, x] / P_mat[t, :, x].sum()

        self.obs_type_info[obs_key]["sample_p"] = sample_p_vec
        self.obs_type_info[obs_key]["init_p"] = init_p_vec
