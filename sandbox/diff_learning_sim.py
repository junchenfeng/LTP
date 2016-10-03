# Check the model for response dependent learning rate

# inherit the distribution from mcmc
import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
from BKT.hmm_mcmc_zpd_dl import BKT_HMM_MCMC
import numpy as np
import ipdb


#################
#  EXP confirms #
#################
file_path = proj_dir+'/data/bkt/exp_output_effort_manual.csv'
manual_data_array = []
is_skip = True
with open(file_path) as f:
	for line in f:
		if is_skip:
			is_skip=False
			continue
		i_s, t_s, j_s, y_s, is_e, is_v = line.strip().split(',')
		#if int(is_a_s):
		manual_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), 0, int(is_v)) )	

is_skip = True
burnin_cnt = 50
sample_gap = 10
params = []
cnt = 0
with open(proj_dir+'/data/bkt/chp3_parameter_chain_with_effort_manual.txt')	as f2:
	for line in f2:
		if is_skip:
			is_skip=False
			continue
		cnt += 1
		if cnt < burnin_cnt:
			continue
		if (cnt-burnin_cnt) % sample_gap != 0:
			continue
			
		segs = [float(x) for x in line.strip().split(',')]
		pi0,pi,s1,s2,s3,s4,s5,g1,g2,g3,g4,g5,e11,e12,e13,e14,e15,e21,e22,e23,e24,e25,l01,l02,l03,l04,l05,l11,l12,l13,l14,l15, *rest = segs
		s = [s1,s2,s3,s4,s5]
		g = [g1,g2,g3,g4,g5]
		e1 = [e11,e12,e13,e14,e15]
		e2 = [e21,e22,e23,e24,e25]
		l0 = [l01,l02,l03,l04,l05]
		l1 = [l11,l12,l13,l14,l15]
		param = {'pi0':pi0,'pi':pi,'s':s,'g':g,'e1':e1,'e2':e2,'l0':l0,'l1':l1,'Lambda':0,'betas':[0,0,0,0,0]}
		params.append(param)

# draw X and integrate out theta
mcmc_instance = BKT_HMM_MCMC()
log_data = []
for param in params:
	# sample
	mcmc_instance.estimate(param, manual_data_array, max_iter=1, is_effort=True)
	for k in range(mcmc_instance.K):
		for t in range(0, mcmc_instance.T_vec[k]):
			if t == 0:
				continue
			# check if learning differs between change
			Vt_1 = mcmc_instance.V_array[t-1,k]
			Xt_1 = mcmc_instance.X[t-1,k]			
			Xt = mcmc_instance.X[t,k]
			Yt_1 = mcmc_instance.observ_data[t-1,k]
			Yt = mcmc_instance.observ_data[t,k]
			
			if Xt_1 == 1 and Vt_1==1:
				log_data.append((Xt_1-1,Xt-1,Yt_1,Yt))

# check the P(Y_t=1|X_t-1=0,Y_t-1=1) = P(Y_t=1|X_t-1=0,Y_t-1=0)
log_array = np.array(log_data)
xt = log_array[:,1]
yt_1 = log_array[:,2]
idx10 = np.where(np.logical_and(xt==1,yt_1==0))
idx11 = np.where(np.logical_and(xt==1,yt_1==1))
idx00 = np.where(np.logical_and(xt==0,yt_1==0))
idx01 = np.where(np.logical_and(xt==0,yt_1==1))

print(log_array[idx00,3].mean(), log_array[idx01,3].mean())
print(log_array[idx10,3].mean(), log_array[idx11,3].mean())

ipdb.set_trace()

##########
#  spell #
##########
kpid = '2'
file_path = proj_dir+'/data/bkt/spell_data_%s.csv' % kpid
data_array = []
id_dict = {}
idx_cnt = 0
with open(file_path) as f:
	for line in f:
		i_s, t_s, y_s, is_e_s = line.strip().split(',')
		if i_s not in id_dict:
			id_dict[i_s] = idx_cnt
			idx_cnt += 1
		i = id_dict[i_s]

		data_array.append( (i, int(t_s)-1, 0, int(y_s), int(is_e_s),1))		

is_skip = True
burnin_cnt = 50
sample_gap = 10
params = []
cnt = 0
with open(proj_dir+'/data/bkt/res/%s/full_mcmc_parameter_chain.txt' % kpid)	as f2:
	for line in f2:
		if is_skip:
			is_skip=False
			continue
		cnt += 1
		if cnt < burnin_cnt:
			continue
		if (cnt-burnin_cnt) % sample_gap != 0:
			continue
			
		segs = [float(x) for x in line.strip().split(',')]
		pi0,pi,s,g,e1,e2,l, Lambda, beta1,beta2,beta3,beta4,beta5 = segs
		s = [s]
		g = [g]
		e1 = [e1]
		e2 = [e2]
		l = [l]
		betas = [beta1,beta2,beta3,beta4,beta5]
		param = {'pi0':pi0,'pi':pi,'s':s,'g':g,'e1':e1,'e2':e2,'l':l,'Lambda':Lambda,'betas':betas}
		params.append(param)

# draw X and integrate out theta
mcmc_instance = BKT_HMM_MCMC_ZPD()
log_data = []
for param in params:
	# sample
	mcmc_instance.estimate(param, data_array, max_iter=1,is_exit=True)
	for k in range(mcmc_instance.K):
		for t in range(0, mcmc_instance.T_vec[k]):
			if t == 0:
				continue
			# check if learning differs between change
			Vt_1 = mcmc_instance.V_array[t-1,k]
			Xt_1 = mcmc_instance.X[t-1,k]			
			Xt = mcmc_instance.X[t,k]
			Yt_1 = mcmc_instance.observ_data[t-1,k]
			Yt = mcmc_instance.observ_data[t,k]
			
			if Xt_1 == 1 and Vt_1==1:
				log_data.append((Xt_1-1,Xt-1,Yt_1,Yt))

# check the P(Y_t=1|X_t-1=0,Y_t-1=1) = P(Y_t=1|X_t-1=0,Y_t-1=0)
log_array = np.array(log_data)
xt = log_array[:,1]
yt_1 = log_array[:,2]
idx10 = np.where(np.logical_and(xt==1,yt_1==0))
idx11 = np.where(np.logical_and(xt==1,yt_1==1))
idx00 = np.where(np.logical_and(xt==0,yt_1==0))
idx01 = np.where(np.logical_and(xt==0,yt_1==1))

print(log_array[idx00,3].mean(), log_array[idx01,3].mean())
print(log_array[idx10,3].mean(), log_array[idx11,3].mean())