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


### (1) Load the data
max_obs = 250
full_data_array = []
data_cnt = 0
with open(proj_dir+'/data/BKT/test/single_sim.txt') as f:
	for line in f:
		i_s, t_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		#if int(is_a_s):
		full_data_array.append( (int(i_s), int(t_s), int(y_s)) )	
		data_cnt += 1

incomplete_data_array = []
data_cnt = 0
with open(proj_dir+'/data/BKT/test/single_sim.txt') as f:
	for line in f:
		i_s, t_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		if int(is_a_s):
			incomplete_data_array.append( (int(i_s), int(t_s), int(y_s), int(is_e_s)) )	
		data_cnt += 1

		

### (2) Initiate the instance
em_instance = BKT_HMM_EM()
mcmc_instance = BKT_HMM_MCMC()
survival_mcmc_instance = BKT_HMM_SURVIVAL()


### (3) Section 1: Single Factor Full Spell
y0s = [log[2] for log in full_data_array if log[1]==0]
y1s = [log[2] for log in full_data_array if log[1]==1]
yTs = [log[2] for log in full_data_array if log[1]==4]


init_param = {'s': 1-np.array(yTs).mean(),
			  'g': 0.3, 
			  'pi': np.array(y0s).mean(),
			  'l': np.array(y1s).mean() - np.array(y0s).mean(),
			  'h0': 0.0,
			  'h1': 0.0
			  }


em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, full_data_array, max_iter = 20)
mcmc_s, mcmc_g, mcmc_pi, mcmc_l = mcmc_instance.estimate(init_param, full_data_array, max_iter = 1000)
mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1 = survival_mcmc_instance.estimate(init_param, full_data_array, max_iter = 1000)

print('Full Data')
print(init_param['s'],init_param['g'],init_param['pi'],init_param['l'])
print(em_s, em_g, em_pi, em_l)
print(mcmc_s, mcmc_g, mcmc_pi, mcmc_l)
print(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1)


### (4) Section 2: Single Factor Incomplete Spell
y0s = [log[2] for log in incomplete_data_array if log[1]==0]
y1s = [log[2] for log in incomplete_data_array if log[1]==1]
yTs = [log[2] for log in incomplete_data_array if log[1]==4]

EYs = [(log[2], log[3]) for log in incomplete_data_array if log[1]>0]
h1 = float(  sum([x[1] for x in EYs if x[0]==1]) )/len([x[0] for x in EYs if x[0]==1])
h0 = float(  sum([x[1] for x in EYs if x[0]==0]) )/len([x[0] for x in EYs if x[0]==0])

init_param = {'s': 1-np.array(yTs).mean(),
			  'g': 0.3, 
			  'pi': np.array(y0s).mean(),
			  'l': np.array(y1s).mean() - np.array(y0s).mean(),
			  'h0': h0,
			  'h1': h1}
		
em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, incomplete_data_array, max_iter = 20)
mcmc_s, mcmc_g, mcmc_pi, mcmc_l = mcmc_instance.estimate(init_param, incomplete_data_array, max_iter = 1000)
mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1 = survival_mcmc_instance.estimate(init_param, incomplete_data_array, max_iter = 1000)

print('Incomplete Data')
print(init_param['s'], init_param['g'], init_param['pi'], init_param['l'], init_param['h0'], init_param['h1'])
print(em_s, em_g, em_pi, em_l)
print(mcmc_s, mcmc_g, mcmc_pi, mcmc_l)
print(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1)



	

'''
pred_log = test_case.predict()

y_true = np.array([x[0] for x in pred_log])
y_pred = np.array([x[1] for x in pred_log])

from sklearn import metrics

fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
'''