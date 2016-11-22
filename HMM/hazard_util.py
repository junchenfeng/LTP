# encoding: utf-8
#TODO: Scale the proportional hazard model


import os	  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from HMM.prop_hazard_ars import ars_sampler
import numpy as np

import ipdb

def prop_hazard(M, T_vec, S, H, Lambdas, betas):

	prop_hazard_mdls = [ars_sampler(Lambdas[i], [betas[i]]) for i in range(M)]
	
	K = len(T_vec)
	
	# generate S,D
	hS = [[] for x in range(M)]
	hD = [[] for x in range(M)]
	hIdx = [[] for x in range(M)]
	
	T = max(T_vec)
	
	for k in range(K):
		for t in range(T_vec[k]):
			m = S[t,k]
			hS[m].append([t])
			hD[m].append(H[t,k])
	
	# do a stratified sampling by t
	# TODO: Clean up the notation here
	# TODO: check if the stratified sampling is legit
	for m in range(M):
		N = len(hD[m])
		if N==0:
			continue
		# check how many observations in each sequence length
		nT = np.zeros((T,1), dtype=np.int)
		idxT = [[] for t in range(T)]
		for i in range(N):
			t = hS[m][i][0]
			nT[t] += 1
			idxT[t].append(i)
		
		# TODO:Need to preserve the relative size
		# The current sampling scheme distorts!
		for t in range(T):
			sample_size = min(round(5000/T),nT[t])
			if sample_size ==0:
				continue
			idxs = np.random.choice(idxT[t], size = sample_size, replace=False)
			hIdx[m]  += idxs.tolist()
	
	# estimate the mdodel
	new_lambdas = []
	new_betas = []
	for m in range(M):
		Nh = len(hIdx[m])

		prop_hazard_mdls[m].load(np.array(hS[m])[hIdx[m],:], np.array(hD[m])[hIdx[m]])
		
		prop_hazard_mdls[m].Lambda = prop_hazard_mdls[m].sample_lambda()[-1]
		prop_hazard_mdls[m].betas[0] = prop_hazard_mdls[m].sample_beta(0)[-1]

		new_lambdas.append(prop_hazard_mdls[m].Lambda)
		new_betas.append(prop_hazard_mdls[m].betas[0])
	
	# reconfig
	h_mat = []
	for m in range(M):
		hs = [prop_hazard_mdls[m].Lambda*np.exp(prop_hazard_mdls[m].betas[0]*t) for t in range(T)]
		if any([h>1 for h in hs]):
			#ipdb.set_trace()
			raise ValueError('Hazard rate is larger than 1.')
		h_mat.append( hs )
	hazard_matrix = np.array(h_mat)
	return hazard_matrix, new_lambdas, new_betas
	
def cell_hazard(M, T_vec, S, H, h_prior):
	# update the likelihood count
	K = len(T_vec)
	T = max(T_vec)
	h_cnt = np.zeros((M,T,2))
	for k in range(K):
		for t in range(T_vec[k]):
			h_cnt[S[t,k],t,H[t,k]] += 1
	
	# update the posterior
	hazard_matrix = np.zeros((M,T))
	for m in range(M):
		for t in range(T):
			hazard_matrix[m,t] = np.random.beta(h_prior[m,t,0]+h_cnt[m,t,1], h_prior[m,t,1]+h_cnt[m,t,0]) # hazard rate H=1
	return hazard_matrix
	
	

