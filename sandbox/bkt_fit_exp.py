import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from BKT.hmm_mcmc_zpd import BKT_HMM_MCMC_ZPD
import numpy as np

file_path = proj_dir+'/data/bkt/exp_output_effort_auto.csv'
auto_data_array = []
is_skip = True
with open(file_path) as f:
	for line in f:
		if is_skip:
			is_skip=False
			continue
		i_s, t_s, j_s, y_s, is_e, is_v = line.strip().split(',')
		#if int(is_a_s):
		auto_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), 0, int(is_v)) )	

sim_data_array = []
is_skip = True
with open(file_path) as f:
	for line in f:
		if is_skip:
			is_skip=False
			continue
		i_s, t_s, j_s, y_s, is_e, is_v = line.strip().split(',')
		#if int(is_a_s):
		sim_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), 0, 1) )			

file_path = proj_dir+'/data/bkt/exp_output_effort_manual.csv'
manual_data_array = []
is_skip = True
with open(file_path) as f:
	for line in f:
		if is_skip:
			is_skip=False
			continue
		i_s, t_s, j_s, y_s, is_e, is_v = line.strip().split(',')
		#if int(is_a_s):
		manual_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), 0, int(is_v)) )	


		
nJ = 5
nT = 3
mcmc_instance = BKT_HMM_MCMC_ZPD()
init_param = {'s': [0.1]*nJ,
			  'g': [0.3]*nJ, 
			  'e1':[0.5]*nJ,
			  'e2':[0.5]*nJ,
			  'pi': 0.4,
			  'pi0':0.1,
			  'l': [0.2]*nJ,
			  'gamma': 0.0,
			  'betas': [0.00]*5
			  }

L = 100	
'''		  
pi0, pi, s, g, e0, e1, l, *rest = mcmc_instance.estimate(init_param, sim_data_array, max_iter = L, is_exit=False)

print('No effort')
print('point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(l)
print(pi)
print(pi0)

np.savetxt(proj_dir+'/data/bkt/chp3_parameter_chain_no_effort.txt', mcmc_instance.parameter_chain, delimiter=',')


pi0, pi, s, g, e1, e2, l, *rest = mcmc_instance.estimate(init_param, auto_data_array, max_iter = L, is_effort=True, is_exit=False)

print('automatic slack')
print('point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(l)
print(pi)
print(pi0)
print(e1)
print(e2)

np.savetxt(proj_dir+'/data/bkt/chp3_parameter_chain_with_effort_auto.txt', mcmc_instance.parameter_chain, delimiter=',')
'''

	  
pi0, pi, s, g, e1, e2, l, *rest = mcmc_instance.estimate(init_param, manual_data_array, max_iter = L, is_effort=True, is_exit=False)

print('manual slack')
print('point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(l)
print(pi)
print(pi0)
print(e1)
print(e2)


np.savetxt(proj_dir+'/data/bkt/chp3_parameter_chain_with_effort_manual.txt', mcmc_instance.parameter_chain, delimiter=',')

