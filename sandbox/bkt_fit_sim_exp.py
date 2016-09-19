import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from BKT.hmm_mcmc import BKT_HMM
import numpy as np

mcmc_instance = BKT_HMM()

file_path = proj_dir+'/data/bkt/test/single_sim_exp.txt'
effort_data_array = []
with open(file_path) as f:
	for line in f:

		i_s, t_s, j_s, y_s, x, is_e, is_a_s, is_v = line.strip().split(',')
		effort_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), 0, int(is_v)) )	

sim_data_array = []
with open(file_path) as f:
	for line in f:

		i_s, t_s, j_s, y_s, x, is_e, is_a_s, is_v = line.strip().split(',')
		sim_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), 0, 1) )			
		
nJ = 4
nT = 3		
L = 1000
		
init_param = {'s': [0.05]*nJ,
			  'g': [0.2]*nJ, 
			  'e0':[0.5,0.5,0.5,0.5],
			  'e1':[0.5,0.5,0.5,0.5],
			  'pi': 0.4,
			  'l': [0.2,0.2,0.2,0.2],
			  'h0': [0]*nT,
			  'h1': [0]*nT
			  }
	
pi, s, g, e0,e1, l, h0, h1 = mcmc_instance.estimate(init_param, sim_data_array, max_iter = L, is_exit=False)

print('No effort')
print('point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(l)
print(pi)

np.savetxt(proj_dir+'/data/bkt/test/constant_param_chain.txt', mcmc_instance.parameter_chain, delimiter=',')

init_param = {'s': [0.05]*nJ,
			  'g': [0.2]*nJ, 
			  'e0':[0.5,0.5,0.5,0.5],
			  'e1':[0.5,0.5,0.5,0.5],
			  'pi': 0.4,
			  'l': [0.2,0.2,0.2,0.2],
			  'h0': [0]*nT,
			  'h1': [0]*nT
			  }
pi, s, g, e0,e1, l, h0, h1 = mcmc_instance.estimate(init_param, effort_data_array, method='DG', max_iter = L, is_exit=False)


print('slack')
print(' DG point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(e0)
print(e1)
print(l)
print(pi)

np.savetxt(proj_dir+'/data/bkt/test/effort_param_chain.txt', mcmc_instance.parameter_chain, delimiter=',')

init_param = {'s': [0.05]*nJ,
			  'g': [0.2]*nJ, 
			  'e0':[0.5,0.5,0.5,0.5],
			  'e1':[0.5,0.5,0.5,0.5],
			  'pi': 0.4,
			  'l': [0.2,0.2,0.2,0.2],
			  'h0': [0]*nT,
			  'h1': [0]*nT
			  }
pi, s, g, e0,e1, l, h0, h1 = mcmc_instance.estimate(init_param, effort_data_array, method='FB', max_iter = L, is_exit=False)

print('slack')
print('FB point estimation')
#print(sHat, gHat, piHat, lHat)
print(s)
print(g)
print(e0)
print(e1)
print(l)
print(pi)

np.savetxt(proj_dir+'/data/bkt/test/effort_param_chain.txt', mcmc_instance.parameter_chain, delimiter=',')

