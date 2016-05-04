# encoding:utf-8

import numpy as np
from collections import defaultdict
import os
proj_dir = os.path.dirname(os.path.abspath(__file__))
import ipdb

from scipy.optimize import minimize


# inference routinue
def update_mastery(mastery, learn_rate):
	return mastery + (1-mastery)*learn_rate

def compute_success_rate(guess, slip, mastery):
	return guess*(1-mastery) + (1-slip)*mastery

	
# etl func
def collapse_log(long_log):
	# long log (i,t,Y) -> short_log (Y/N,t,0)
	# reduce runtime in estimation
	if not long_log:
		raise ValueError('log is empty')
	log_dict = {1:defaultdict(int),0:defaultdict(int)}
	for log in long_log:
		log_dict[log[2]][log[1]] += 1
	
	short_log = []
	for ans, t_cnter in log_dict.items():
		for t, cnt in t_cnter.items():
			short_log.append((ans, t, cnt))
	
	return short_log
	
	
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
	guess = 1-slip-A/(1-init_mastery)
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
	

if __name__ == '__main__':
	# etl
	log_data = []
	with open(proj_dir+'/data/bkt/test/single_sim.txt') as f:
		for line in f:
			if not line.strip():
				continue
			i,t,Y = line.strip().split(',')
			log_data.append((int(i),int(t),int(Y)))
	
	short_log = collapse_log(log_data)
	
	target_fnc = lambda params: data_llk(short_log, params)
	target_grad = lambda params: data_grad(short_log, params)
	
	x0 = [0.8, 0.45, -np.log(0.9)]
	x =  [0.9, 0.3,-np.log(0.8)]
	#xc = [0.90001,0.3,-np.log(0.8)]
	#xa = [0.9,0.3001,-np.log(0.8)]
	#xb = [0.9,0.3,-np.log(0.8)+0.00001]
	#test1 = [(llk(xc[0],xc[1],xc[2],2)-llk(x[0],x[1],x[2],2))/0.00001,(llk(xa[0],xa[1],xa[2],2)-llk(x[0],x[1],x[2],2))/0.00001,(llk(xb[0],xb[1],xb[2],2)-llk(x[0],x[1],x[2],2))/0.00001]
	#test2= [grad(x[0],x[1],x[2],2)]
	#test=[(data_llk(short_log,xc)-data_llk(short_log,x))/0.00001, (data_llk(short_log,xa)-data_llk(short_log,x))/0.00001, (data_llk(short_log,xb)-data_llk(short_log,x))/0.00001]
		
	bnds = [(0.55,0.95),(0.05,0.95),(-np.log(0.95),-np.log(0.05))]  # the bnds are not strict
	# start with slip=guess=0.2, learn rate=0.1, init_mastery = 0.5
	res = minimize(target_fnc, x0, method='L-BFGS-B', jac=target_grad, bounds=bnds)
	# the right estimation is [0.9, 0.45，0.223]
	
	init_mastery = 0.5  # prior
	slip,guess,learn_rate = reconstruct_bkt_parameter(res.x[0],res.x[1],res.x[2], init_mastery)
	#init_mastery = 0.6  # prior
	#slip,guess,learn_rate = reconstruct_bkt_parameter(x[0], x[1], x[2], init_mastery)
	
	#now predict the sequence
	pred_log = []
	for log in log_data:
		Y = log[2]
		t = log[1]
		if t == 1:
			mastery = init_mastery
		else:
			# append pred_log, prob is retained from the last iteration
			pred_log.append((Y,prob))
		
		mastery = forward_update_mastery(mastery, guess, slip, learn_rate, Y)		
		prob = compute_success_rate(guess, slip, mastery)
	
	# compute auc
	from sklearn import metrics
	y_true = np.array([x[0] for x in pred_log])
	y_pred = np.array([x[1] for x in pred_log])
	
	fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
	auc = metrics.auc(fpr, tpr)
	print auc