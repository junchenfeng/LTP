import os
proj_dir = os.path.dirname(os.path.abspath(__file__))
import ipdb

import sys
sys.path.append(proj_dir)

from MLC.solver.vanilla_MLC import RunVanillaMLC
from MLC.solver.predict_performance import get_predict
from MLC.utl.IO import data_loader_from_list, data_loader_by_userid

from BKT.bkt import BKT

from sklearn.metrics import roc_curve, auc
import numpy as np

'''
	# parse by item_id
	self.log_data = defaultdict(list):
	for log in log_data:
		user_id=log[0]; item_id = log[1] t=log[2]; y=log[3]
		self.log_data[item_id].append(user_id, t, y)
'''

class learning_curve_model(object):
	def __init__(self, method_name):
		if method_name not in ['mlc','bkt']:
			raise ValueError('Method %s is not supported.' % method_name)
		self.method = method_name
		
	def load(self, log_data):
		# log format (user_id, t, y)	
		# get auxilary info
		self.log_data = log_data
		self.T = max([x[1] for x in self.log_data])
		
	def train(self, **kargs):
		# **kargs input respective model parameter 
		if 'max_t' in kargs:
			self.T = kargs['max_t']

			
		if self.method == 'mlc':
			# set parameters
			if 'mixture_num' in kargs:
				mixture_num = kargs['mixture_num']
			else:
				mixture_num = 2
			
			# estimate a separate model for each item
			mdl = RunVanillaMLC()
			mdl.init(mixture_num, max_t=self.T)
			mdl.load_data_from_list(self.log_data)
			self.model = mdl.solve()
		elif self.method == 'bkt':
			self.model = BKT()
			self.model.load(self.log_data, max_t=self.T)
			self.model.estimate()

	def predict(self, log_data):
		# log_data format (i,t,y)
		# Refresh for new user id
		# standardize api
		predict_res = data_loader_by_userid(log_data, max_opportunity=self.T)
		y_true = []
		y_pred = []
		for uid, pred_data in predict_res.items():
			if self.method == 'mlc':				
				user_y_true, user_y_pred = get_predict([pred_data], self.model['q'], self.model['p'])
				y_true += user_y_true
				y_pred += user_y_pred
			elif self.method == 'bkt':
				# rewrap
				bkt_pred_data = [(uid, t+1, pred_data[t])for t in range(len(pred_data))]
				pred_log = self.model.predict(bkt_pred_data)
				y_true += [x[0] for x in pred_log]
				y_pred += [x[1] for x in pred_log]
		
		# reformat as numpy array
		return np.array(y_true), np.array(y_pred)
	
	def evaluate(self, log_data=[], metrics='auc'):
		# default is auc of the train data
		if not log_data:
			log_data = self.log_data
		y_true, y_pred = self.predict(log_data)
		
		if metrics == 'auc':			
			fpr,tpr,thresholds = roc_curve(np.array(y_true),np.array(y_pred))
			auc_val = auc(fpr, tpr)
			return auc_val
		
		if metrics == 'auc_detail':
			fpr,tpr,thresholds = roc_curve(np.array(y_true),np.array(y_pred))
			return fpr, tpr
'''
if __name__ == '__main__':
	test_data_path = proj_dir+'/data/bkt/test/single_sim.txt'
	train_log_data = []
	predict_log_data = []
	with open(test_data_path) as f:
		for line in f:
			if not line.strip():
				continue
			i,t,Y = line.strip().split(',')
			if int(i) % 10 != 0:
				train_log_data.append((int(i),int(t),int(Y)))
			else:
				predict_log_data.append((int(i),int(t),int(Y)))
	# test mlc
	print('test MLC')
	test_mlc = learning_curve_model('mlc')
	test_mlc.load(train_log_data)
	test_mlc.train()
	print test_mlc.evaluate(metrics='auc')
	print test_mlc.evaluate(predict_log_data, metrics='auc')
	
	# test bkt
	print('test BKT')
	test_bkt = learning_curve_model('bkt')
	test_bkt.load(train_log_data)
	test_bkt.train()
	print test_bkt.evaluate(metrics='auc')
	print test_bkt.evaluate(predict_log_data, metrics='auc')
'''