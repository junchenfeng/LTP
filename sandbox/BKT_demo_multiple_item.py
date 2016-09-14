import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from BKT.hmm_survival_mcmc import BKT_HMM_SURVIVAL
import numpy as np

file_path = proj_dir+'/data/bkt/test/single_sim_x_2.txt'


max_obs = 1000
full_data_array = []
with open(file_path) as f:
	for line in f:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		#if int(is_a_s):
		full_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s)) )	

incomplete_data_array = []
with open(file_path) as f:
	for line in f:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		if int(is_a_s):
			incomplete_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), int(is_e_s)) )	
			
survival_mcmc_instance = BKT_HMM_SURVIVAL()
init_param = {'s': [0.1]*2,
			  'g': [0.3]*2, 
			  'pi': 0.4,
			  'l': [0.2]*2,
			  'h0': [0.01]*5,
			  'h1': [0.01]*5
			  }

L = 1000			  
mcmc_s_pi, mcmc_s_s, mcm_s_g, mcmc_s_l, mcmc_s_h0, mcmc_s_h1 = survival_mcmc_instance.estimate(init_param, full_data_array, max_iter = L)

print('Full Data')
print('point estimation')
#print(sHat, gHat, piHat, lHat)
print(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l)

'''
print('h0')
print(h0Hat)
print(mcmc_s_h0)
print('h1')
print(h1Hat)
print(mcmc_s_h1)
'''