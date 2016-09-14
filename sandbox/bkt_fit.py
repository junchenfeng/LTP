import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import sys
sys.path.append(proj_dir)

from BKT.hmm_em import BKT_HMM_EM
from BKT.hmm_mcmc import BKT_HMM_MCMC
from BKT.hmm_survival_mcmc import BKT_HMM_SURVIVAL
from BKT.util import generate_learning_curve

import numpy as np
import ipdb


data_array = []
import sys
kp_id = sys.argv[1]


# need to translate the id into 1:N sequence
id_dict = {}
idx_cnt = 0
with open(proj_dir+'/data/bkt/spell_data_%s.csv' % kp_id) as in_f0:
	for line in in_f0:
		i_s, t_s, y_s, is_e_s = line.strip().split(',')
		if i_s not in id_dict:
			id_dict[i_s] = idx_cnt
			idx_cnt += 1
		i = id_dict[i_s]

		data_array.append( (i, int(t_s)-1, int(y_s), int(is_e_s)))		
N = len( list( set([x[0] for x in data_array]) ) )

	
### (2) Initiate the instance
em_instance = BKT_HMM_EM()
survival_mcmc_instance = BKT_HMM_SURVIVAL()


### (3) Section 1: Single Factor Full Spell
y0s = [log[2] for log in data_array if log[1]==0]
y1s = [log[2] for log in data_array if log[1]==1]
yTs = [log[2] for log in data_array if log[1]==4]

h0 = []
h1 = []
for t in range(5):
	EYs = [(log[2], log[3]) for log in data_array if log[1]==t]
	if len([x[0] for x in EYs if x[0]==1]) != 0:
		h1val = sum([x[1] for x in EYs if x[0]==1]) / len([x[0] for x in EYs if x[0]==1])
	else:
		h1val = 0.5
		
	if len([x[0] for x in EYs if x[0]==0]) != 0:
		h0val = sum([x[1] for x in EYs if x[0]==0]) / len([x[0] for x in EYs if x[0]==0])
	else:
		h0val = 0.5
	
	h1.append( max(min(h1val, 0.5), 0.1) )
	h0.append( max(min(h0val, 0.5), 0.1) )

init_param = {'s': max(1-np.array(yTs).mean(), 0.01),
			  'g': 0.3, 
			  'pi': min(max(np.array(y0s).mean(), 0.01), 0.99),
			  'l': max(np.array(y1s).mean() - np.array(y0s).mean(), 0.2),
			  'h0': h0,
			  'h1': h1}

L = 1000
em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, data_array, max_iter = 10)
mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1 = survival_mcmc_instance.estimate(init_param, data_array, max_iter = L)

print('Full Data')
print('point estimation')
print(init_param['s'],init_param['g'],init_param['pi'],init_param['l'])
print(em_s, em_g, em_pi, em_l)
print(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l)

print('hazard rate')
print(h0)
print(h1)

print(mcmc_s_h0)
print(mcmc_s_h1)



# For external use
with open(proj_dir+'/data/bkt/res/%s/full_point_estimation.txt' % kp_id, 'w') as f1:
	f1.write('em,%f,%f,%f,%f' %(em_s, em_g, em_pi, em_l))
	for x in mcmc_s_h0+mcmc_s_h1:
		f1.write(',%f' % 0.0)	
	f1.write('\n')
		
	
	f1.write('mcmc_s,%f,%f,%f,%f' %(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l))
	for x in mcmc_s_h0+mcmc_s_h1:
		f1.write(',%f' % x)
	f1.write('\n')
	

np.savetxt(proj_dir+'/data/bkt/res/%s/full_mcmc_survival_parameter_chain.txt' % kp_id, survival_mcmc_instance.parameter_chain, delimiter=',')
