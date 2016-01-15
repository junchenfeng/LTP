import numpy as np
import os, sys
import gevent
from gevent import monkey;gevent.monkey.patch_all();

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

        jobs = [gevent.spawn(self._solve_EM) for i in range(self.m)]
        gevent.joinall(jobs)
        res_list = [job.value for job in jobs]

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
        is_converged = False
        iteration_num = 1

        mixture_density = np.random.uniform(0, 1, self.K)
        mixture_density = mixture_density/mixture_density.sum()

        # TODO: does impose monotone constraints help?
        learning_curve_matrix = np.random.uniform(0, 1, (self.max_opportunity, self.K))
        last_learning_curve_matrix = np.array(learning_curve_matrix)

        while True:
            # solve for q
            z_matrix = np.zeros((self.num_user, self.K))
            for i in range(self.num_user):
                z_matrix[i, :] = Z_assembly(self.response_data[i],
                                            learning_curve_matrix,
                                            mixture_density)

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

            # sort learning_curve_matrix and corresponding mixture_density
            criteria = learning_curve_matrix[-1,:] -  learning_curve_matrix[0,:]
            order = sorted(list(range(self.K)),key=lambda i:criteria[i], reverse=True)
            learning_curve_matrix = learning_curve_matrix[:,order]
            mixture_density = mixture_density[order]

            # check stop condition
            iteration_num += 1

            if iteration_num >= self.max_iteration:
                break

            l2_norm_diff = np.linalg.norm(learning_curve_matrix-last_learning_curve_matrix)
            if l2_norm_diff < self.stop_threshold:
                is_converged = True  # currently no use
                break

            # prepare for the next iteration
            last_learning_curve_matrix = np.array(learning_curve_matrix)

        return {'q':learning_curve_matrix, 'p':mixture_density, 'flag':is_converged}


