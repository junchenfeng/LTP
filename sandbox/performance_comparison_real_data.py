# encoding:utf-8
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(proj_dir)

from model import learning_curve_model
from collections import defaultdict

import ipdb
import json

subject=sys.argv[1]

item_data = json.load(open(proj_dir+'/data/17zuoye/app_%s_data.json'%subject))


models = {'mlc':learning_curve_model('mlc'),
		  'bkt':learning_curve_model('bkt')}

model_names = ['mlc','bkt']

def performance_pk(train_log_data, predict_log_data):
	pk_res = {}
	
	for model_name in model_names:
		models[model_name].load(train_log_data)
		models[model_name].train(max_t=10)
		pk_res[model_name] = models[model_name].evaluate(predict_log_data)
		#pk_res[model_name] = models[model_name].evaluate(predict_log_data, metrics='auc_detail')
	# update the structure
	
	return pk_res

item_auc = {}
for item_id in item_data.keys():
	print item_id
	item_auc[item_id] = performance_pk(item_data[item_id]['train'], item_data[item_id]['predict'])	

#item_id = '87#102376'
#performance_pk(item_data[item_id]['train'], item_data[item_id]['predict'])

ipdb.set_trace()
	

with open(proj_dir+'/data/17zuoye/pk_%s.txt'%subject,'w') as f:
	
	f.write('category_id,'+','.join(model_names)+'\n')
	
	for item_id, auc_vals in item_auc.items():
		f.write(item_id)
		for model_name in model_names:
			f.write(',%f'%auc_vals[model_name])
		f.write('\n')