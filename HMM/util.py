# encoding:utf-8
# TODO: Update the learning curve generator
# TODO: Add the joint response generator
import numpy as np
import ipdb

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
		while not((c_mat[0,1]<c_mat[1,1:].sum()) and (c_mat[1,2]<c_mat[2,2])) and iter_cnt<max_iter:
			# this is hard coded for state  = 3
			# ensure that c02 and c20 is 0
			c_mat[0,1] = np.random.beta(param[0][1],param[0][0])
			c_mat[1,:] = np.random.dirichlet(param[1])
			c_mat[2,2] = np.random.beta(param[2][2],param[2][1])
			c_mat[0,0] = 1-c_mat[0,1]
			c_mat[2,1] = 1-c_mat[2,2]
			iter_cnt += 1 
	elif Mx ==2:
		while not(c_mat[0,1]<c_mat[1,1]) and iter_cnt<max_iter:
			c_mat[0,:] = np.random.dirichlet(param[0])
			c_mat[1,:] = np.random.dirichlet(param[1])
			
	if iter_cnt == max_iter:
		raise Exception('Observation matrix is not generated.')
		
	return c_mat
	
def draw_l(params, Mx):
	
	l_param = np.zeros((2, Mx, Mx))
	l_param[0] = np.identity(Mx)
	l_param[1] = np.zeros(Mx)
	for x in range(Mx-1):
		l_param[1][x,x:(x+2)] = np.random.dirichlet(params[x])
	l_param[1][-1,-1] = 1
	return l_param
	
if __name__ == '__main__':
	lc0 = generate_learning_curve(0.05, 0.2, 0.4, 0.4, 5)
	lc1 = generate_learning_curve(0.05, 0.35, 0.25, 0.4, 5)
	print(lc0)
	print(lc1)