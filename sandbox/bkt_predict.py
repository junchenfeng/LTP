import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import sys
sys.path.append(proj_dir)

from BKT.hmm_em import BKT_HMM_EM
from BKT.hmm_survival_mcmc import BKT_HMM_SURVIVAL

import numpy as np
import ipdb
from sklearn import metrics

import sys
kp_id = sys.argv[1]

# load data
id_dict = {}
idx_cnt = 0
data_array = []
with open(proj_dir+'/data/bkt/spell_data_%s_outsample.csv' % kp_id) as in_f0:
	for line in in_f0:
		i_s, t_s, y_s, is_e_s = line.strip().split(',')
		if i_s not in id_dict:
			id_dict[i_s] = idx_cnt
			idx_cnt += 1
		i = id_dict[i_s]

		data_array.append( (i, int(t_s)-1, int(y_s), int(is_e_s)))		
N = len( list( set([x[0] for x in data_array]) ) )
T = max([x[1] for x in data_array])

# load parameter
param_dict = {}
with open(proj_dir+'/data/bkt/res/%s/full_point_estimation.txt' % kp_id) as in_f1:
	for line in in_f1:
		algo_s, s_s, g_s, pi_s, l_s, h01,h02,h03,h04,h05,h11,h12,h13,h14,h15 = line.strip().split(',')
		param_dict[algo_s] = {'s': float(s_s),
			  'g': float(g_s), 
			  'pi': float(pi_s),
			  'l': float(l_s),
			  'h0': [float(h01),float(h02),float(h03),float(h04),float(h05)],
			  'h1': [float(h11),float(h12),float(h13),float(h14),float(h15)]
			  }

# predict performance
model = BKT_HMM_SURVIVAL()
print('Y')
for algo in param_dict.keys():
	pred_res = model.predict(param_dict[algo], data_array)
	# compare
	pred_array = np.array(pred_res)
	y_true = pred_array[:,1]
	y_pred = pred_array[:,0]
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
	auc = metrics.auc(fpr, tpr)
	r2 = (((y_true - y_pred)**2).mean())**(0.5)
	print(algo, auc, r2)

# predict hazard rate
pred_res = model.predict_exit(param_dict['mcmc_s'], data_array)
pred_array = np.array(pred_res)
y_true = pred_array[:,1]
y_pred = pred_array[:,0]
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
auc = metrics.auc(fpr, tpr)
r2 = (((y_true - y_pred)**2).mean())**(0.5)
print('E')
print(algo, auc, r2)

