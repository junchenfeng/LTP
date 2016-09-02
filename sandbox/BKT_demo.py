import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import sys
sys.path.append(proj_dir)

from BKT.hmm_em import BKT_HMM_EM
from BKT.hmm_mcmc import BKT_HMM_MCMC
from BKT.hmm_survival_mcmc import BKT_HMM_SURVIVAL

import numpy as np

import ipdb



# output the true parameter for the simulated data

# check if the data are correctly simulated
data = []
with open(proj_dir+'/data/bkt/test/single_sim.txt') as in_f0:
	for line in in_f0:
		i_s, t_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		data.append( (int(i_s), int(t_s), int(y_s), int(x_s), int(is_e_s), int(is_a_s)) )		
N = data[-1][0]

# s,g
xynum = [[0.0, 0.0], [0.0, 0.0]]
xnum = [0.0, 0.0]
for log in data:
	y = log[2]
	x = log[3]
	xnum[x] += 1
	xynum[x][y] += 1
sHat = xynum[1][0]/xnum[1]
gHat = xynum[0][1]/xnum[0]
			
# learn
transit = 0.0
tot = 0
for log in data:
	x = log[3]
	if log[1] > 0 and not prev_x:
		transit += x
		tot += 1
	prev_x = x
lHat = transit/tot

# pi
x1num = 0.0
for log in data:
	if log[1] == 0:
		x1num += log[3]
piHat = x1num/N

# h1 vec
h_cnt = np.zeros((5,2))
s_cnt = np.zeros((5,2))

for log in data:
	t = log[1]
	y = log[2]
	e = log[4]
	a = log[5]
	if a:
		h_cnt[t,y] += e
		s_cnt[t,y] += 1-e
hrate_mat = h_cnt/(h_cnt+s_cnt)	
h0Hat = hrate_mat[:,0].tolist()
h1Hat = hrate_mat[:,1].tolist()

true_param = [sHat, gHat, piHat, lHat] + h0Hat + h1Hat
with open(proj_dir+'/data/bkt/true_param.txt', 'w') as f0:
	f0.write(','.join([str(x) for x in true_param])+'\n')

	
### (1) Load the data
max_obs = 2000
L = 10000
full_data_array = []
with open(proj_dir+'/data/bkt/test/single_sim.txt') as f:
	for line in f:
		i_s, t_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		#if int(is_a_s):
		full_data_array.append( (int(i_s), int(t_s), int(y_s)) )	

incomplete_data_array = []
with open(proj_dir+'/data/bkt/test/single_sim.txt') as f:
	for line in f:
		i_s, t_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		if int(is_a_s):
			incomplete_data_array.append( (int(i_s), int(t_s), int(y_s), int(is_e_s)) )	

		

### (2) Initiate the instance
em_instance = BKT_HMM_EM()
mcmc_instance = BKT_HMM_MCMC()
survival_mcmc_instance = BKT_HMM_SURVIVAL()


### (3) Section 1: Single Factor Full Spell
y0s = [log[2] for log in full_data_array if log[1]==0]
y1s = [log[2] for log in full_data_array if log[1]==1]
yTs = [log[2] for log in full_data_array if log[1]==4]


init_param = {'s': 1-np.array(yTs).mean(),
			  'g': 0.3, 
			  'pi': np.array(y0s).mean(),
			  'l': np.array(y1s).mean() - np.array(y0s).mean(),
			  'h0': [0.01]*5,
			  'h1': [0.01]*5
			  }


em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, full_data_array, max_iter = 20)
mcmc_s, mcmc_g, mcmc_pi, mcmc_l = mcmc_instance.estimate(init_param, full_data_array, max_iter = L)
mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1 = survival_mcmc_instance.estimate(init_param, full_data_array, max_iter = L)

print('Full Data')
print('point estimation')
print(init_param['s'],init_param['g'],init_param['pi'],init_param['l'])
print(em_s, em_g, em_pi, em_l)
print(mcmc_s, mcmc_g, mcmc_pi, mcmc_l)
print(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1)

print('AUC')
em_auc, em_r2 = get_auc(em_s, em_g, em_pi, em_l, full_data_array)
mcmc_auc, mcmc_r2 = get_auc(mcmc_s, mcmc_g, mcmc_pi, mcmc_l, full_data_array)
mcmc_s_auc, mcmc_s_r2 = get_auc(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, full_data_array)

print(em_r2)
print(mcmc_r2)
print(mcmc_s_r2)

# For external use
with open(proj_dir+'/data/bkt/full_point_estimation.txt', 'w') as f1:
	f1.write('em,%f,%f,%f,%f\n' %(em_s, em_g, em_pi, em_l))
	f1.write('mcmc,%f,%f,%f,%f\n' %(mcmc_s, mcmc_g, mcmc_pi, mcmc_l))
	f1.write('mcmc_s,%f,%f,%f,%f\n' %(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l))
	

np.savetxt(proj_dir+'/data/bkt/full_mcmc_parameter_chain.txt', mcmc_instance.parameter_chain, delimiter=',')	
np.savetxt(proj_dir+'/data/bkt/full_mcmc_survival_parameter_chain.txt', survival_mcmc_instance.parameter_chain, delimiter=',')

### (4) Section 2: Single Factor Incomplete Spell
y0s = [log[2] for log in incomplete_data_array if log[1]==0]
y1s = [log[2] for log in incomplete_data_array if log[1]==1]
yTs = [log[2] for log in incomplete_data_array if log[1]==4]

h0 = []
h1 = []
for t in range(5):
	EYs = [(log[2], log[3]) for log in incomplete_data_array if log[1]==t]
	h1.append( max(min(sum([x[1] for x in EYs if x[0]==1]) / len([x[0] for x in EYs if x[0]==1]),0.99),0.01) )
	h0.append( max(min(sum([x[1] for x in EYs if x[0]==0]) / len([x[0] for x in EYs if x[0]==0]),0.99),0.01) )

init_param = {'s': 1-np.array(yTs).mean(),
			  'g': 0.3, 
			  'pi': np.array(y0s).mean(),
			  'l': np.array(y1s).mean() - np.array(y0s).mean(),
			  'h0': h0,
			  'h1': h1}
		
em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, incomplete_data_array, max_iter = 20)
mcmc_s, mcmc_g, mcmc_pi, mcmc_l = mcmc_instance.estimate(init_param, incomplete_data_array, max_iter = L)
mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1 = survival_mcmc_instance.estimate(init_param, incomplete_data_array, max_iter = L)
print('Incomplete Data')
print('Point estimation')
print(init_param['s'], init_param['g'], init_param['pi'], init_param['l'], init_param['h0'], init_param['h1'])
print(em_s, em_g, em_pi, em_l)
print(mcmc_s, mcmc_g, mcmc_pi, mcmc_l)
print(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, mcmc_s_h0, mcmc_s_h1)
	

# the prediction is the learning curve. AUC is calculated based on the learning curve

print('matrix')
em_auc, em_r2 = get_auc(em_s, em_g, em_pi, em_l, incomplete_data_array)
mcmc_auc, mcmc_r2 = get_auc(mcmc_s, mcmc_g, mcmc_pi, mcmc_l, incomplete_data_array)
mcmc_s_auc, mcmc_s_r2 = get_auc(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l, incomplete_data_array)

print(em_r2)
print(mcmc_r2)
print(mcmc_s_r2)
	
with open(proj_dir+'/data/bkt/incomplete_point_estimation.txt', 'w') as f4:
	f4.write('em,%f,%f,%f,%f' %(em_s, em_g, em_pi, em_l))
	for x in mcmc_s_h0+mcmc_s_h1:
		f4.write(',%f' % 0.0)	
	f4.write('\n')
		
	f4.write('mcmc,%f,%f,%f,%f' %(mcmc_s, mcmc_g, mcmc_pi, mcmc_l))
	for x in mcmc_s_h0+mcmc_s_h1:
		f4.write(',%f' % 0.0)	
	f4.write('\n')
	
	f4.write('mcmc_s,%f,%f,%f,%f' %(mcmc_s_s, mcm_s_g, mcmc_s_pi, mcmc_s_l))
	for x in mcmc_s_h0+mcmc_s_h1:
		f4.write(',%f' % x)
	f4.write('\n')
	
np.savetxt(proj_dir+'/data/bkt/incomplete_mcmc_parameter_chain.txt', mcmc_instance.parameter_chain, delimiter=',')	
np.savetxt(proj_dir+'/data/bkt/incomplete_mcmc_survival_parameter_chain.txt', survival_mcmc_instance.parameter_chain, delimiter=',')

	
