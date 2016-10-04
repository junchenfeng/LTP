import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from BKT.hmm_mcmc_rl_3y import BKT_HMM_MCMC
import numpy as np
import ipdb

file_path = proj_dir+'/data/bkt/exp_output_effort_auto_y3.csv'
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

file_path = proj_dir+'/data/bkt/exp_output_effort_manual_y3.csv'
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
mcmc_instance = BKT_HMM_MCMC()
init_param = {'c': [[[0.5,0.5,0.0],[0.3,0.4,0.3],[0.0,0.5,0.5]]]*nJ, 
			  'e': [[0.5]*nJ, [0.5]*nJ, [0.5]*nJ],
			  'pi': 0.1,
			  'pi0':0.1,
			  'l0': [0.2]*nJ,
			  'l1': [0.2]*nJ,
			  'gamma': 0.0,
			  'betas': [0.00]*5
			  }

L = 1000

pi0, pi, c01, c11, c12, c22, l0, l1, *rest = mcmc_instance.estimate(init_param, sim_data_array, max_iter = L, method='DG', is_effort=False, is_exit=False)

print('No effort')
print('point estimation')
print('pi')
print([pi0,pi])
print('correct rate')
print(c01)
print(c11)
print(c12)
print(c22)
print('learn rate')
print(l0)
print(l1)

np.savetxt(proj_dir+'/data/bkt/chp3_parameter_chain_no_effort_y3.txt', mcmc_instance.parameter_chain, delimiter=',')

pi0, pi, c01, c11, c12, c22, l0, l1, e0,e1,e2,*rest = mcmc_instance.estimate(init_param, auto_data_array, max_iter = L, method='DG', is_effort=True, is_exit=False)

print('Automatic Slack')
print('point estimation')
print('pi')
print([pi0,pi])
print('correct rate')
print(c01)
print(c11)
print(c12)
print(c22)
print('learn rate')
print(l0)
print(l1)
print('effort rate')
print(e0)
print(e1)
print(e2)

np.savetxt(proj_dir+'/data/bkt/chp3_parameter_chain_with_effort_auto_y3.txt', mcmc_instance.parameter_chain, delimiter=',')


	  
pi0, pi, c01, c11, c12, c22, l0, l1, e0,e1,e2,*rest = mcmc_instance.estimate(init_param, manual_data_array, max_iter = L, method='DG', is_effort=True, is_exit=False)

print('Manual Slack')
print('point estimation')
print('pi')
print([pi0,pi])
print('correct rate')
print(c01)
print(c11)
print(c12)
print(c22)
print('learn rate')
print(l0)
print(l1)
print('effort rate')
print(e0)
print(e1)
print(e2)


np.savetxt(proj_dir+'/data/bkt/chp3_parameter_chain_with_effort_manual_y3.txt', mcmc_instance.parameter_chain, delimiter=',')

