from pyMLC.solver.vanilla_MLC import RunVanillaMLC
from solver.predict_performance import get_predict
import numpy as np
import ipdb
from sklearn import metrics
import os
proj_dir = os.path.dirname(os.path.abspath(__file__))
test_data_path = proj_dir+'/data/bkt/test/single_sim.txt'

test_instance = RunVanillaMLC()
test_instance.init(2,max_t=10)
test_instance.load_data(test_data_path)
res = test_instance.solve()
print res['q']
y_true, y_pred = get_predict(test_instance.response_data, 
                              res['q'],
                              res['p'])
							  
fpr,tpr,thresholds = metrics.roc_curve(np.array(y_true),np.array(y_pred))
auc = metrics.auc(fpr, tpr)
print auc