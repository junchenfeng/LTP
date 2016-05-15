import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict
import json
import ipdb

def main(subject, train_pct = 0.8):
	# load data
	item_log_data = defaultdict(list)
	is_skip = True
	with open(proj_dir+'/data/17zuoye/app_%s_retain.csv'%subject) as f:
		for line in f:
			if is_skip:
				is_skip=False
				continue
			if not line.strip():
				continue
			user_id, item_id, t_s, y_s = line.strip().split(',')
			item_log_data[item_id].append((int(user_id), int(t_s), int(y_s)))

	# separate train and predict sample
	item_data = {}

	for item_id, log_data in item_log_data.items():
		uids = list(set([x[0] for x in log_data]))
		num_train_uids = int(len(uids)*train_pct)
		train_uids = np.random.choice(uids, size=num_train_uids, replace=False)
			
		train_log_data = []
		predict_log_data = []
		
		for log in log_data:
			if log[0] in train_uids:
				train_log_data.append(log)
			else:
				predict_log_data.append(log)
		
		item_data[item_id] = {'train':train_log_data,'predict':predict_log_data}
		
	json.dump(item_data, open(proj_dir+'/data/17zuoye/app_%s_data.json'%subject, 'w'))

if __name__=='__main__':
	main('math')
	main('eng')
