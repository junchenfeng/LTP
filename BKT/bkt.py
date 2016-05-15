# encoding:utf-8

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import ipdb


# inference routinue
def update_mastery(mastery, learn_rate):
	return mastery + (1-mastery)*learn_rate

def compute_success_rate(guess, slip, mastery):
	return guess*(1-mastery) + (1-slip)*mastery

	
# etl func
def collapse_log(long_log, max_t):
	# long log (i,t,Y) -> short_log (Y/N,t,0)
	# reduce runtime in estimation
	if not long_log:
		raise ValueError('log is empty')
	log_dict = {1:defaultdict(int),0:defaultdict(int)}
	for log in long_log:
		if log[1] > max_t:
			continue
		log_dict[log[2]][log[1]] += 1
	
	short_log = []
	for ans, t_cnter in log_dict.items():
		for t, cnt in t_cnter.items():
			short_log.append((ans, t, cnt))
	
	return short_log

def impute_init_mastery(short_log):
	if not short_log:
		raise ValueError('log is empty')
	
	y = 0.0
	n = 0
	for log in short_log:
		if log[1] == 1:
			y += log[0]*log[2]
			n += log[2]
	
	return y/n
	
# likelihood of observed data for single component BKT
def llk(c, A, beta, t):
	#beta = -np.log(1-lrate)
	#A = (1-guess-slip)(1-init_mastery)
	#c = (1-slip)
	crate = c-A*np.exp(-beta*t)
	return crate
	
def grad(c, A, beta, t):
	grad = np.zeros((3,1))
	grad[0] = 1
	grad[1] = -np.exp(-beta*t)
	grad[2] = A*np.exp(-beta*t)*t
	return grad
	
def data_llk(log_data, params):
	# log_data are tuples of (Y,t,n)
	# where k is the time of practice, Y=1 is success, n is the number of recurrence
	
	c = params[0]
	A = params[1]
	beta = params[2]
	log_ll = 0
	for log in log_data:
		Y = log[0]
		t = log[1]
		n = log[2]
		p = llk(c, A, beta, t)		
		log_ll += (Y*np.log(p) + (1-Y)*np.log(1-p))*n
		
	return -log_ll

def data_grad(log_data, params):
	c = params[0]
	A = params[1]
	beta = params[2]

	log_grad = np.zeros((3,1),order='F')
	for log in log_data:
		Y = log[0]
		t = log[1]
		n = log[2]
		g = grad(c,A,beta,t)		
		p = llk(c,A,beta,t)
		log_grad += (Y/p - (1-Y)/(1-p))*n*g
		
	return -log_grad

def reconstruct_bkt_parameter(c,A,beta,init_mastery):
	slip = 1-c
	learn_rate = 1-np.exp(-beta)
	guess = c-A/(1-init_mastery)
		
	if slip<=0 or slip>=0.5:
		print 'Slip %f is not valid. Truncate to [0.01,0.49].' % slip
		slip = min(max(0.01,guess),0.49)
		
	if guess<=0 or guess>=0.5:
		print 'guess %f is not valid. Truncate to [0.01,0.49].' % guess
		guess = min(max(0.01,guess),0.49)
		
	if learn_rate<=0 or learn_rate>=1:
		print 'learn rate %f is not valid. Truncate to [0.01, 0.99]'		
		learn_rate = min(max(0.01,learn_rate),0.99)
		
	return slip,guess,learn_rate
	
# Bayesian Knowledge Tracing Algorithm
def forward_update_mastery(mastery, guess, slip, learn_rate, Y):
	if Y ==1:
		new_mastery = 1 - (1-learn_rate)*(1-mastery)*guess/(guess+(1-slip-guess)*mastery)
	elif Y==0:
		new_mastery = 1 - (1-learn_rate)*(1-mastery)*(1-guess)/(1-guess-(1-slip-guess)*mastery)
	else:
		raise ValueError('Invalid response value.')
	return new_mastery
	
class BKT(object):
	
	def load(self, log_data, max_t=99):		
		self.long_log = log_data
		self.short_log = collapse_log(log_data, max_t)	
	
		self.init_mastery = impute_init_mastery(self.short_log)
	
	def estimate(self, x0=[0.8, 0.45, -np.log(0.9)]):
		target_fnc = lambda params: data_llk(self.short_log, params)
		target_grad = lambda params: data_grad(self.short_log, params)
			
		bnds = [(0.75,0.95),(0.05,0.95),(-np.log(0.95),-np.log(0.05))]  # the bnds are not strict
		
		# 0.05<= g = c-A/(1-L) <=0.3
		cons = ({'type': 'ineq', 'fun': lambda x:  x[0]-1/(1-self.init_mastery)*x[1]-0.05},
				 {'type': 'ineq', 'fun': lambda x: 0.25-x[0]+1/(1-self.init_mastery)*x[1]})
		# x[0]-1/(1-self.init_mastery)*x[1]
		# 0.25-x[0]+1/(1-self.init_mastery)*x[1]
		
		# start with slip=guess=0.2, learn rate=0.1, init_mastery = 0.5
		res = minimize(target_fnc, x0, method='SLSQP', 
						jac=target_grad, 
						bounds=bnds, constraints=cons)
		self.slip, self.guess, self.learn_rate = reconstruct_bkt_parameter(res.x[0],res.x[1],res.x[2], self.init_mastery)
	
	
	def predict(self, log_data=[]):
		if not log_data:
			log_data = self.long_log
		pred_log = []
		prob = 0
		for log in log_data:
			Y = log[2]
			t = log[1]
			if t == 1:
				mastery = self.init_mastery
			else:
				if prob == 0:
					raise Exception('The log sequence may not start from 1.')
				# append pred_log, prob is retained from the last iteration
				pred_log.append((Y,prob))
			
			mastery = forward_update_mastery(mastery, self.guess, self.slip, self.learn_rate, Y)		
			prob = compute_success_rate(self.guess, self.slip, mastery)		
		
		return pred_log
		
if __name__ == '__main__':
	# Test
	#x =  [0.9, 0.3,-np.log(0.8)]
	#xc = [0.90001,0.3,-np.log(0.8)]
	#xa = [0.9,0.3001,-np.log(0.8)]
	#xb = [0.9,0.3,-np.log(0.8)+0.00001]
	#test1 = [(llk(xc[0],xc[1],xc[2],2)-llk(x[0],x[1],x[2],2))/0.00001,(llk(xa[0],xa[1],xa[2],2)-llk(x[0],x[1],x[2],2))/0.00001,(llk(xb[0],xb[1],xb[2],2)-llk(x[0],x[1],x[2],2))/0.00001]
	#test2= [grad(x[0],x[1],x[2],2)]
	#test=[(data_llk(short_log,xc)-data_llk(short_log,x))/0.00001, (data_llk(short_log,xa)-data_llk(short_log,x))/0.00001, (data_llk(short_log,xb)-data_llk(short_log,x))/0.00001]
		
	# run demo
	data_file_path = proj_dir+'/data/bkt/test/single_sim.txt'
	log_data = []
	with open(data_file_path) as f:
		for line in f:
			i, t, y = line.strip().split(',')
			log_data.append((int(i),int(t),int(y)))
	
	test_case = BKT()
	test_case.load(log_data)
	test_case.estimate()
	pred_log = test_case.predict()
	
	y_true = np.array([x[0] for x in pred_log])
	y_pred = np.array([x[1] for x in pred_log])

	from sklearn import metrics

	fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
	auc = metrics.auc(fpr, tpr)
	print auc
	

	
	