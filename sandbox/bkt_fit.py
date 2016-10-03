import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import sys
sys.path.append(proj_dir)

from BKT.hmm_em import BKT_HMM_EM
from BKT.hmm_mcmc_rl import BKT_HMM_MCMC

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

		data_array.append( (i, int(t_s)-1, 0, int(y_s), int(is_e_s),1))		
N = len( list( set([x[0] for x in data_array]) ) )

	
### (2) Initiate the instance
em_instance = BKT_HMM_EM()
mcmc_instance = BKT_HMM_MCMC()


### (3) Section 1: Single Factor Full Spell
y0s = [log[3] for log in data_array if log[1]==0]
y1s = [log[3] for log in data_array if log[1]==1]
yTs = [log[3] for log in data_array if log[1]==4]

h0 = []
h1 = []
for t in range(5):
	EYs = [(log[3], log[4]) for log in data_array if log[1]==t]
	if len([x[0] for x in EYs if x[0]==1]) != 0:
		h1val = sum([x[1] for x in EYs if x[0]==1]) / len([x[0] for x in EYs if x[0]==1])
	else:
		h1val = 0.5
		
	if len([x[0] for x in EYs if x[0]==0]) != 0:
		h0val = sum([x[1] for x in EYs if x[0]==0]) / len([x[0] for x in EYs if x[0]==0])
	else:
		h0val = 0.5
	
	h1.append( h1val )
	h0.append( h0val )

pi = min(max(np.array(y0s).mean(), 0.01), 0.98)	
em_init_param = {'s': [max(1-np.array(yTs).mean(), 0.01)],
			  'g':[0.3],
			  'pi': pi,
			  'l': [max(np.array(y1s).mean() - np.array(y0s).mean(), 0.2)]
			  }

mcmc_init_param = {'c': [[0.2],[0.5],[min(np.array(yTs).mean(), 0.99)]],
			  'e':[[1.0],[1.0],[1.0]],
			  'pi': pi,
			  'pi0':max(1-pi-0.05, 0.01),
			  'l0': [max(np.array(y1s).mean() - np.array(y0s).mean(), 0.2)],
			  'l1': [max(np.array(y1s).mean() - np.array(y0s).mean(), 0.2)],
			  'Lambda': (h0[0]+h1[0])/2,
			  'betas': [0.01,0.01,-0.01,0.01,0.01]}

L = 100
em_s, em_g, em_pi, em_l = em_instance.estimate(em_init_param, data_array, max_iter = 10)
mcmc_pi0, mcmc_pi, mcmc_c0, mcmc_c1, mcmc_c2, mcmc_l0, mcmc_l1, mcmc_e0, mcmc_e1, mcmc_e2, mcmc_Lambda, mcmc_betas = mcmc_instance.estimate(mcmc_init_param, data_array, method='DG', max_iter = L, is_exit=True)
print('point estimation')
#print(init_param['s'],init_param['g'],init_param['pi'],init_param['l'])
print(em_s, em_g, em_pi, em_l)
print(mcmc_pi0, mcmc_pi)
print(mcmc_c0[0], mcmc_c1[0], mcmc_c2[0])
print(mcmc_l0[0], mcmc_l1[0])


print('hazard rate')
print(mcmc_Lambda)
print(mcmc_betas)

# For external use
with open(proj_dir+'/data/bkt/res/%s/full_point_estimation_em.txt' % kp_id, 'w') as f1:
	f1.write('em,%f,%f,%f,%f\n' %(em_s, em_g, em_pi, em_l))

with open(proj_dir+'/data/bkt/res/%s/full_point_estimation_mcmc.txt' % kp_id, 'w') as f2:
	f2.write('mcmc_,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' %(mcmc_pi0, mcmc_pi, mcmc_c0[0],mcmc_c1[0],mcmc_c2[0],mcmc_l0[0], mcmc_l1[0], mcmc_Lambda[0], mcmc_betas[0], mcmc_betas[2],mcmc_betas[4]))

	

np.savetxt(proj_dir+'/data/bkt/res/%s/full_mcmc_parameter_chain.txt' % kp_id, mcmc_instance.parameter_chain, delimiter=',')
