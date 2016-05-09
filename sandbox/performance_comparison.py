# encoding:utf-8
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(proj_dir)

from MLC.solver.vanilla_MLC import RunVanillaMLC
from MLC.solver.predict_performance import get_predict
from MLC.utl.IO import data_loader_from_list

from BKT.bkt import BKT

import numpy as np
from sklearn import metrics

import ipdb


test_data_1 = proj_dir+'/data/bkt/test/single_sim.txt'
test_data_2 = proj_dir+'/data/mlc/single_component_complete_track.txt'
test_data_3 = proj_dir+'/data/mlc/double_component_complete_track.txt'


T = 10

def performance_pk(data_file_path):
	pk_res = {'mlc':[],'bkt':[]}

	train_log_data = []
	predict_log_data = []
	with open(data_file_path) as f:
		for line in f:
			if not line.strip():
				continue
			i,t,Y = line.strip().split(',')
			if int(i) % 10 != 0:
				train_log_data.append((int(i),int(t),int(Y)))
			else:
				predict_log_data.append((int(i),int(t),int(Y)))


	test_instance = RunVanillaMLC()
	test_instance.init(2, max_t=T)
	test_instance.load_data_from_list(train_log_data)
	res = test_instance.solve()
	
	predict_res = data_loader_from_list(predict_log_data, max_opportunity=T)
	y_true, y_pred = get_predict(predict_res, 
								  res['q'],
								  res['p'])
								  
	fpr,tpr,thresholds = metrics.roc_curve(np.array(y_true),np.array(y_pred))
	pk_res['mlc']= [fpr, tpr, thresholds]
	auc = metrics.auc(fpr, tpr)
	print auc

	test_case = BKT()
	test_case.load(train_log_data)
	test_case.estimate()
	pred_log = test_case.predict(predict_log_data)

	y_true = np.array([x[0] for x in pred_log])
	y_pred = np.array([x[1] for x in pred_log])

	fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
	pk_res['bkt']= [fpr, tpr, thresholds]

	auc = metrics.auc(fpr, tpr)
	print auc
	
	
	# update the structure
	
	return pk_res

res_list = []
print('exp:1')

res_list.append(performance_pk(test_data_1))
print('exp:2')
res_list.append(performance_pk(test_data_2))
print('exp:3')
res_list.append(performance_pk(test_data_3))

with open(proj_dir+'/data/test/pk.txt','w') as f:
	
	for i in range(len(res_list)):
		mlc_res = res_list[i]['mlc']
		mlc_fpr = mlc_res[0]; mlc_tpr = mlc_res[1]
		for j in range(len(mlc_fpr)):
			f.write('%f,%f,%s,%d\n' % (mlc_fpr[j], mlc_tpr[j], 'mlc', i))
		
		bkt_res = res_list[i]['bkt']
		bkt_fpr = bkt_res[0]; bkt_tpr = bkt_res[1]
		
		for k in range(len(bkt_fpr)):
			f.write('%f,%f,%s,%d\n' % (bkt_fpr[k], bkt_tpr[k], 'bkt', i))		
		
		