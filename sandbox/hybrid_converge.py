import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import sys
sys.path.append(proj_dir)

from BKT.hmm_em import BKT_HMM_EM
from BKT.hmm_mcmc import BKT_HMM_MCMC
from BKT.hmm_survival_mcmc import BKT_HMM_SURVIVAL

import numpy as np

import ipdb


max_obs = 2000
L = 2000
n = 100

incomplete_data_array = []
with open(proj_dir+'/data/bkt/test/single_sim_x.txt') as f:
	for line in f:
		i_s, t_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		if int(is_a_s):
			incomplete_data_array.append( (int(i_s), int(t_s), int(y_s), int(is_e_s)) )	

		

### (2) Initiate the instance
survival_mcmc_instance = BKT_HMM_SURVIVAL()

with open(proj_dir+'/data/bkt/prior_convergence.txt', 'w') as f:
	for i in range(n):
		init_param = {'s': np.random.rand(),
					  'g': np.random.rand(), 
					  'pi': np.random.rand(),
					  'l': np.random.rand(),
					  'h0': np.random.rand(5,1),
					  'h1': np.random.rand(5,1)
					  }


		mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1 = survival_mcmc_instance.estimate(init_param, incomplete_data_array, max_iter = L)
		
		f.write('mcmc_s,%f,%f,%f,%f' %(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l))
		for x in mcmc_s_h0+mcmc_s_h1:
			f.write(',%f' % x)
		f.write('\n')
		