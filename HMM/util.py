# encoding:utf-8
# TODO: Update the learning curve generator
# TODO: Add the joint response generator
import numpy as np
import ipdb
import random
from itertools import accumulate

def random_choice(p_vec):
	cump=list(accumulate(p_vec))
	n = len(p_vec)
	
	if abs(cump[n-1]-1)> 1e-6:
		raise ValueException('probability does not add up to 1.')
	rn = random.random()
	for x in range(n):
		if rn < cump[x]:
			break
	return x
def update_mastery(mastery, learn_rate):
	return mastery + (1-mastery)*learn_rate

def compute_success_rate(slip, guess, mastery):
	return guess*(1-mastery) + (1-slip)*mastery

# Bayesian Knowledge Tracing Algorithm
def forward_update_mastery(mastery, slip, guess, learn_rate, Y):
	if Y ==1:
		new_mastery = 1 - (1-learn_rate)*(1-mastery)*guess/(guess+(1-slip-guess)*mastery)
	elif Y==0:
		new_mastery = 1 - (1-learn_rate)*(1-mastery)*(1-guess)/(1-guess-(1-slip-guess)*mastery)
	else:
		raise ValueError('Invalid response value.')
	return new_mastery
	
def generate_learning_curve(slip, guess, init_mastery, learn_rate, T):
	p=init_mastery
	lc = [compute_success_rate(slip, guess, p)]
	for t in range(1,T):
		p = update_mastery(p,learn_rate)
		lc.append(compute_success_rate(slip, guess, p))
	return lc

def logExpSum(llk_vec):
	llk_max = max(llk_vec)
	llk_sum = llk_max + np.log(np.exp(llk_vec-llk_max).sum())
	return llk_sum

def draw_c(param, Mx, My, max_iter=100):
	if len(param) != Mx:
		raise ValueError('Observation matrix is wrong on latent state dimension.')
	if len(param[0]) != My:
		raise ValueError('Observation matrix is wrong on observation dimension.')

	c_mat = np.zeros((Mx, My))
	iter_cnt = 0
	if Mx ==3:
		#while not((c_mat[0,0]>c_mat[1,0]) and (c_mat[1,2]<c_mat[2,2])) and iter_cnt<max_iter:
			# this is hard coded for state  = 3
			# ensure that c02 and c20 is 0
		for n in range(Mx):
			c_mat[n,:] = np.random.dirichlet(param[n])
			#iter_cnt += 1 
	elif Mx ==2:
		while not(c_mat[0,1]<c_mat[1,1]) and iter_cnt<max_iter:
			for n in range(Mx):
				c_mat[n,:] = np.random.dirichlet(param[n])
			iter_cnt += 1 
			
	if iter_cnt == max_iter:
		ipdb.set_trace()
		raise Exception('Observation matrix is not generated.')
		
	return c_mat
	
def draw_l(params, Mx):
	
	l_param = np.zeros((2, Mx, Mx))
	l_param[0] = np.identity(Mx)
	l_param[1] = np.zeros(Mx)
	for m in range(Mx):
		l_param[1][m,:] = np.random.dirichlet(params[m])
	return l_param

def get_final_chain(param_chain_vec, start, end, is_exit, is_effort):
	# calcualte the llk for the parameters
	gap = max(int((end-start)/100), 10)
	select_idx = range(start, end, gap)
	num_chain = len(param_chain_vec)
	
	# get rid of burn in
	param_chain = {}
	param_chain['l'] = np.vstack([param_chain_vec[i]['l'][select_idx, :] for i in range(num_chain)])
	param_chain['c'] = np.vstack([param_chain_vec[i]['c'][select_idx, :] for i in range(num_chain)])
	param_chain['pi'] = np.vstack([param_chain_vec[i]['pi'][select_idx, :] for i in range(num_chain)])
	if is_exit:
		param_chain['h'] = np.vstack([param_chain_vec[i]['h'][select_idx, :] for i in range(num_chain)])
	if is_effort:
		param_chain['e'] = np.vstack([param_chain_vec[i]['e'][select_idx, :] for i in range(num_chain)])

	return param_chain
	
	
def get_map_estimation(param_chain, is_exit, is_effort):
	res = {}
	res['l'] = param_chain['l'].mean(axis=0).tolist()
	res['c'] = param_chain['c'].mean(axis=0).tolist()
	res['pi'] = param_chain['pi'].mean(axis=0).tolist()
	
	if is_exit:
		res['h'] = param_chain['h'].mean(axis=0).tolist()
	
	if is_effort:
		res['e'] = param_chain['e'].mean(axis=0).tolist()
		
	return res	
	
if __name__ == '__main__':
	lc0 = generate_learning_curve(0.05, 0.2, 0.4, 0.4, 5)
	lc1 = generate_learning_curve(0.05, 0.35, 0.25, 0.4, 5)
	print(lc0)
	print(lc1)