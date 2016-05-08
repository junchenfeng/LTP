from pyMLC.solver.vanilla_MLC import RunVanillaMLC
from solver.predict_performance import get_predict

from bkt import BKT

import numpy as np
import ipdb
from sklearn import metrics
import os
proj_dir = os.path.dirname(os.path.abspath(__file__))



test_data_1 = proj_dir+'/data/bkt/test/single_sim.txt'
test_data_2 = proj_dir+'/data/mlc/single_component_complete_track.txt'
test_data_3 = proj_dir+'/data/mlc/double_component_complete_track.txt'

def performance_pk(data_file_path):

	test_instance = RunVanillaMLC()
	test_instance.init(2, max_t=5)
	test_instance.load_data(data_file_path)
	res = test_instance.solve()
	y_true, y_pred = get_predict(test_instance.response_data, 
								  res['q'],
								  res['p'])
								  
	fpr,tpr,thresholds = metrics.roc_curve(np.array(y_true),np.array(y_pred))
	auc = metrics.auc(fpr, tpr)
	print auc

	test_case = BKT(init_mastery=0.5)
	test_case.load(data_file_path)
	test_case.estimate()
	pred_log = test_case.predict()

	y_true = np.array([x[0] for x in pred_log])
	y_pred = np.array([x[1] for x in pred_log])

	fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
	auc = metrics.auc(fpr, tpr)
	print auc

print('exp:1')
performance_pk(test_data_1)
print('exp:2')
performance_pk(test_data_2)
print('exp:3')
performance_pk(test_data_3)