import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from BKT.hmm_survival_mcmc import BKT_HMM_SURVIVAL
import numpy as np

file_path = proj_dir+'/data/bkt/exp_output.csv'
full_data_array = []
is_skip = True
with open(file_path) as f:
	for line in f:
		if is_skip:
			is_skip=False
			continue
		i_s, t_s, j_s, y_s = line.strip().split(',')
		#if int(is_a_s):
		full_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s)) )	

nJ = 5
nT = 3
survival_mcmc_instance = BKT_HMM_SURVIVAL()
init_param = {'s': [0.1]*nJ,
			  'g': [0.3]*nJ, 
			  'pi': 0.4,
			  'l': [0.2]*nJ,
			  'h0': [0.01]*nT,
			  'h1': [0.01]*nT
			  }

L = 2000			  
pi, s, g, l, h0, h1 = survival_mcmc_instance.estimate(init_param, full_data_array, max_iter = L)

print('Full Data')
print('point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(l)
print(pi)