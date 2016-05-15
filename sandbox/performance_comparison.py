# encoding:utf-8
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(proj_dir)

from model import learning_curve_model

test_data_1 = proj_dir+'/data/bkt/test/single_sim.txt'
test_data_2 = proj_dir+'/data/mlc/single_component_complete_track.txt'
test_data_3 = proj_dir+'/data/mlc/double_component_complete_track.txt'



T = 10
models = {'mlc':learning_curve_model('mlc'),
		  'bkt':learning_curve_model('bkt')}

model_names = ['mlc','bkt']

def performance_pk(data_file_path):
	pk_res = {}

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

	for model_name in model_names:
		models[model_name].load(train_log_data)
		models[model_name].train()
		pk_res[model_name] = models[model_name].evaluate(predict_log_data, metrics='auc_detail')
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
		
		
