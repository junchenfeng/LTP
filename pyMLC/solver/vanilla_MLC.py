import numpy as np
import os, sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)

from utl.IO import data_loader
from utl.utl import Z_assembly

class RunVanillaMLC(object):
    def init(self, K):
        # currently are all hard coded
        self.K = K
        self.max_opportunity = 5  # only calibrate for the first K practice opportunities
        if self.K == 1:
            self.m = 1
        else:
            self.m = 5  # number of estimation routine

        # no bayesian thrinkage at the moment
        self.alpha = 1
        self.beta = 1

        # convergence condition
        self.stop_threshold = 0.001
        self.max_iteration = 10

    def load_data(self, file_path):
        self.response_data = data_loader(file_path, self.max_opportunity)
        self.num_user = len(self.response_data)


    def solve(self):
        # The initial density does not predict convergence, thus the trick is
        # just try enough combinations

        res_list = []
        for i in range(self.m):
            res = self._solve_EM()
            res_list.append(res)

        # compute the l2 norm, and choose the largest one
        q_norm_diff = [np.linalg.norm(res_list[x]['q']) for x in range(self.m)]
        opt_idx = q_norm_diff.index(max(q_norm_diff))
        return res_list[opt_idx]

    def _solve_EM(self):
        '''
        # Input: 
        (1) mixture density: prior guess of the component mixture, J*1
        
        # Output:
        (1) learning curve matrix: T*J
        (2) Posterior mixture density: J*1
        
        '''
        stop_condition = False
        iteration_num = 1

        mixture_density = np.random.uniform(0, 1, (self.K, 1))
        mixture_density = mixture_density/mixture_density.sum()

        # TODO: does impose monotone constraints help?
        learning_curve_matrix = np.random.uniform(0, 1, (self.max_opportunity, self.K))
        last_learning_curve_matrix = np.array(learning_curve_matrix)  # TODO: change it into shallow copy or what not
        
        while not stop_condition:
            # solve for q
            z_matrix = np.zeros((self.num_user, self.K), float)
            for i in range(self.num_user):
                z_matrix[i, :] = Z_assembly(self.response_data[i], 
                                            learning_curve_matrix,
                                            mixture_density).reshape(self.K)

            # solve q_{t+1}
            for j in range(self.K):
                for t in range(self.max_opportunity):
                    numerator = self.alpha - 1
                    denominator = self.alpha + self.beta - 2
                    for s in range(self.num_user):
                        response_list = self.response_data[s]
                        if t < len(response_list):
                            numerator += response_list[t]*z_matrix[s, j]
                            denominator += z_matrix[s, j]
                    learning_curve_matrix[t, j] = numerator/denominator
            # solve p_{t+1}
            mixture_density = z_matrix.sum(axis=0)/z_matrix.sum()

            # check stop condition
            l2_norm_diff = np.linalg.norm(learning_curve_matrix-last_learning_curve_matrix)
            iteration_num += 1
            if l2_norm_diff < self.stop_threshold:
                stop_condition = True
                is_converged = True  # currently no use

            if iteration_num >= self.max_iteration:
                stop_condition = True
                is_converged = False

            # prepare for the next iteration
            last_learning_curve_matrix = np.array(learning_curve_matrix)

        return {'q':learning_curve_matrix, 'p':mixture_density, 'flag':is_converged}

        
