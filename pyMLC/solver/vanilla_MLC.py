import numpy as np
import random
import math

import os, sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)

from utl.IO import data_loader
from utl.utl import L_assembly

class RunVanillaMLC(object):
    def load_param(self, num_component):
        # currently are all hard coded
        self.num_component = num_component
        self.max_opportunity = 5  # only calibrate for the first K practice opportunities 
        # no bayesian thrinkage at the moment
        self.alpha = 1
        self.beta = 1
        # convergence condition
        self.stop_threshold = 0.001
        self.max_iteration = 10

    def load_data(self, file_path): 
        self.user_result = data_loader(file_path, self.max_opportunity)
        self.uid_idx = self.user_result.keys()
        self.num_user = len(self.uid_idx)

    def init(self):
        # initialize the learning curve: probability of getting it WRONG at each practice opportunity
        self.learning_curve_matrix = np.random.uniform(0, 1, (self.num_component, self.max_opportunity))
        # initialize the mixture density
        mixture_density_raw = [random.random()] * self.num_component
        self.mixture_density = [x/sum(mixture_density_raw) for x in mixture_density_raw]
    
    def solve_EM(self):
        stop_condition = False
        iteration_num = 1
        last_learning_curve_matrix = np.array(self.learning_curve_matrix)
        while not stop_condition:
            # solve for q
            # compute L, L is num_user*num_component
            L_matrix = np.zeros((self.num_user, self.num_component), float)
            for j in range(self.num_component):
                q_j_vec = self.learning_curve_matrix[j]
                pj = self.mixture_density[j]
                for s in range(self.num_user):
                    v_dict = self.user_result[self.uid_idx[s]]
                    L_matrix[s, j] = math.exp(L_assembly(v_dict, q_j_vec, pj))
            # compute z
            z_matrix = L_matrix/L_matrix.sum(axis=1, keepdims=True)
            # compute q_{+1}
            for j in range(self.num_component):
                for t in range(self.max_opportunity):
                    numerator = self.alpha - 1
                    denominator = self.alpha + self.beta - 2
                    for s in range(self.num_user):
                        v_dict = self.user_result[self.uid_idx[s]]
                        if (t+1) in v_dict:
                            numerator += v_dict[t+1]*z_matrix[s, j]
                            denominator += z_matrix[s, j]
                    self.learning_curve_matrix[j, t] = numerator/denominator
            # solve for p
            self.mixture_density = z_matrix.sum(axis=0)/z_matrix.sum()

            # check stop condition
            l2_norm_diff = np.linalg.norm(self.learning_curve_matrix-last_learning_curve_matrix)
            iteration_num += 1
            if l2_norm_diff < self.stop_threshold or iteration_num >= self.max_iteration:
                stop_condition = True
            # prepare for the next iteration 
            last_learning_curve_matrix = np.array(self.learning_curve_matrix)
