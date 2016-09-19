import os			  
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import sys
sys.path.append(proj_dir)

from BKT.hmm_em import BKT_HMM_EM
from BKT.hmm_mcmc import BKT_HMM

import numpy as np
import ipdb

max_obs = 2000
L = 500
### (2) Initiate the instance
em_instance = BKT_HMM_EM()
mcmc_instance = BKT_HMM()


'''
# output the true parameter for the simulated data
file_path = proj_dir+'/data/bkt/test/single_sim_x_1.txt'

# check if the data are correctly simulated
data = []
with open(file_path) as in_f0:
	for line in in_f0:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		if int(i_s) == max_obs:
			break
		data.append( (int(i_s), int(t_s), int(j_s), int(y_s), int(x_s), int(is_e_s), int(is_a_s)) )		
N = data[-1][0]
T = data[-1][1]+1
# s,g
xynum = [[0.0, 0.0], [0.0, 0.0]]
xnum = [0.0, 0.0]
for log in data:
	y = log[3]
	x = log[4]
	xnum[x] += 1
	xynum[x][y] += 1
sHat = xynum[1][0]/xnum[1]
gHat = xynum[0][1]/xnum[0]
			
# learn
transit = 0.0
tot = 0
for log in data:
	x = log[4]
	if log[1] > 0 and not prev_x:
		transit += x
		tot += 1
	prev_x = x
lHat = transit/tot

# pi
x1num = 0.0
for log in data:
	if log[1] == 0:
		x1num += log[4]
piHat = x1num/N

# h1 vec
h_cnt = np.zeros((T,2))
s_cnt = np.zeros((T,2))

for log in data:
	t = log[1]
	y = log[3]
	x = log[4]
	e = log[5]
	a = log[6]
	if a:
		h_cnt[t,x] += e
		s_cnt[t,x] += 1-e
hrate_mat = h_cnt/(h_cnt+s_cnt)	
h0Hat = hrate_mat[:,0].tolist()
h1Hat = hrate_mat[:,1].tolist()

#################################################
### (1) Single Factor Complete Spell
#################################################

full_data_array = []
with open(file_path) as f:
	for line in f:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		full_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s)) )	

incomplete_data_array = []
with open(file_path) as f:
	for line in f:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		if int(is_a_s):
			incomplete_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), int(is_e_s)) )	

		

y0s = [log[3] for log in full_data_array if log[1]==0]
y1s = [log[3] for log in full_data_array if log[1]==1]
yTs = [log[3] for log in full_data_array if log[1]==T-1]




init_param = {'s': [1-np.array(yTs).mean()],
			  'g': [0.3], 
			  'e0':[0.5],
			  'e1':[0.5],
			  'pi': np.array(y0s).mean(),
			  'l': [np.array(y1s).mean() - np.array(y0s).mean()],
			  'h0': [0.1]*T,
			  'h1': [0.1]*T}

init_param = {'s': [0.05],
			  'g': [0.2], 
			  'e0':[0.5],
			  'e1':[0.5],
			  'pi': 0.7,
			  'l': [0.3],
			  'h0': [0.1]*T,
			  'h1': [0.1]*T}			  
			  
em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, full_data_array, max_iter = 10)
mcmc_pi, mcmc_s, mcmc_g, mcmc_e0,mcmc_e1, mcmc_l, *rest = mcmc_instance.estimate(init_param, full_data_array, method='DG',max_iter = L)
mcmc_pi_fb, mcmc_s_fb, mcmc_g_fb, mcmc_e0_fb,mcmc_e1_fb, mcmc_l_fb, *rest = mcmc_instance.estimate(init_param, full_data_array, method='FB',max_iter = L)

print('Full Data')
print('point estimation')
print(sHat, gHat, piHat, lHat)
print(em_s, em_g, em_pi, em_l)
print(mcmc_s[0], mcmc_g[0], mcmc_pi, mcmc_l[0])
print(mcmc_s_fb[0], mcmc_g_fb[0], mcmc_pi_fb, mcmc_l_fb[0])


ipdb.set_trace()

#################################################
### (2) Single Factor Incomplete Spell
#################################################
y0s = [log[3] for log in incomplete_data_array if log[1]==0]
y1s = [log[3] for log in incomplete_data_array if log[1]==1]
yTs = [log[3] for log in incomplete_data_array if log[1]==4]

h0 = []
h1 = []
for t in range(5):
	EYs = [(log[3], log[4]) for log in incomplete_data_array if log[1]==t]
	h1.append( max(min(sum([x[1] for x in EYs if x[0]==1]) / len([x[0] for x in EYs if x[0]==1]),0.99),0.01) )
	h0.append( max(min(sum([x[1] for x in EYs if x[0]==0]) / len([x[0] for x in EYs if x[0]==0]),0.99),0.01) )

init_param = {'s': [1-np.array(yTs).mean()],
			  'g': [0.3], 
			  'e0':[1.0],
			  'e1':[1.0],
			  'pi': np.array(y0s).mean(),
			  'l': [np.array(y1s).mean() - np.array(y0s).mean()],
			  'h0': h0,
			  'h1': h1}

em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, incomplete_data_array, max_iter = 20)
mcmc_pi, mcmc_s, mcmc_g, mcmc_e0,mcmc_e1, mcmc_l, mcmc_h0, mcmc_h1 = mcmc_instance.estimate(init_param, incomplete_data_array, method='FB', max_iter = L, is_exit=True)

print('Incomplete Data')

print('Point estimation')
print('Main Parameter')

print(sHat, gHat, piHat, lHat)
print(em_s, em_g, em_pi, em_l)
print(mcmc_s[0], mcmc_g[0], mcmc_pi, mcmc_l[0])

print('h0')
print(h0Hat)
print(mcmc_h0)
print('h1')
print(h1Hat)
print(mcmc_h1)

'''

#################################################
### (2) effort
#################################################
file_path = proj_dir+'/data/bkt/test/single_sim_x_1_e.txt'

# check if the data are correctly simulated
data = []
with open(file_path) as in_f0:
	for line in in_f0:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s, is_v_s = line.strip().split(',')
		data.append( (int(i_s), int(t_s), int(j_s), int(y_s), int(x_s), int(is_e_s), int(is_a_s), int(is_v_s)) )		
N = data[-1][0]
T = data[-1][1]+1
effective_tot_trans = 0
tot_trans = 0
crit_trans = 0
for m in range(len(data)):
	i,t,j,y,S,ex,a,E = data[m]
	if t<T-1 and S==0:
		S = data[m+1][4]
		E = data[m+1][-1]
		tot_trans += 1
		if E == 1:
			effective_tot_trans += 1
			crit_trans += S
lHat = crit_trans/effective_tot_trans
e1_x_cnt = 0
x1_e_cnt = 0
e0_x_cnt = 0
x0_e_cnt = 0

x1_cnt = 0
x0_cnt = 0
y_11_cnt = 0
y_cnt = 0
e1x1_cnt = 0

for m in range(len(data)):
	i,t,j,y,S,ex,a,E = data[m]
	S = data[m][4]
	E = data[m][-1]
	x1_cnt += S
	x0_cnt += (1-S)
	e1x1_cnt += E*S
	if S==1 and E==1:
		y_11_cnt += y
	else:
		y_cnt += y
		
	if t>0:
		S = data[m-1][4]
		x0_e_cnt += 1-S
		x1_e_cnt += S
		e0_x_cnt += E*(1-S)
		e1_x_cnt += E*S	
	

e0Hat = e0_x_cnt/x0_e_cnt #P(E_t=1,X_{t-1}=0)/P(X_{t-0})
e1Hat = e1_x_cnt/x1_e_cnt #P(E_t=1,X_{t-1}=1)/P(X_{t-1})
sHat = 1-y_11_cnt/e1x1_cnt #P(Y=1,E=1,X=1)/P(X=1,E=1)
gHat = y_cnt/(x1_cnt+x0_cnt-e1x1_cnt) #(P(Y_t=1,E_t=0)+P(Y_t=1,E_t=1,X_t=0))/(P(E_t=0)+P(X_t=0,E_t=1))

# pi
x1num = 0.0
for log in data:
	if log[1] == 0:
		x1num += log[4]
piHat = x1num/N

# h1 vec
h_cnt = np.zeros((T,2))
s_cnt = np.zeros((T,2))

for log in data:
	t = log[1]
	y = log[3]
	x = log[4]
	e = log[5]
	a = log[6]
	if a:
		h_cnt[t,x] += e
		s_cnt[t,x] += 1-e
hrate_mat = h_cnt/(h_cnt+s_cnt)	
h0Hat = hrate_mat[:,0].tolist()
h1Hat = hrate_mat[:,1].tolist()

#############
# load data #
#############

full_data_array = []
with open(file_path) as f:
	for line in f:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s, is_v_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		full_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), 0, int(is_v_s)) )	


'''		
incomplete_data_array = []
with open(file_path) as f:
	for line in f:
		i_s, t_s, j_s, y_s, x_s, is_e_s, is_a_s, is_v_s = line.strip().split(',')
		
		if int(i_s) == max_obs:
			break
		
		if int(is_a_s):
			incomplete_data_array.append( (int(i_s), int(t_s), int(j_s), int(y_s), int(is_e_s), int(is_v_s)) )	
	
y0s = [log[3] for log in incomplete_data_array if log[1]==0]
y1s = [log[3] for log in incomplete_data_array if log[1]==1]
yTs = [log[3] for log in incomplete_data_array if log[1]==4]

h0 = []
h1 = []
for t in range(5):
	EYs = [(log[3], log[4]) for log in incomplete_data_array if log[1]==t]
	h1.append( max(min(sum([x[1] for x in EYs if x[0]==1]) / len([x[0] for x in EYs if x[0]==1]),0.99),0.01) )
	h0.append( max(min(sum([x[1] for x in EYs if x[0]==0]) / len([x[0] for x in EYs if x[0]==0]),0.99),0.01) )
'''
init_param = {'s': [0.05],
			  'g': [0.2], 
			  'e0':[0.75],
			  'e1':[0.9],
			  'pi': 0.7,
			  'l': [0.3],
			  'h0': [0.0]*T,
			  'h1': [0.0]*T}

#em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, full_data_array, max_iter = 20)
mcmc_pi, mcmc_s, mcmc_g, mcmc_e0,mcmc_e1, mcmc_l, *rest = mcmc_instance.estimate(init_param, full_data_array, method='FB', max_iter = L, is_exit=True)
mcmc_pi_fb, mcmc_s_fb, mcmc_g_fb, mcmc_e0_fb,mcmc_e1_fb, mcmc_l_fb, *rest = mcmc_instance.estimate(init_param, full_data_array, method='FB', max_iter = L, is_exit=True)

print('Full Data')
print('Point estimation')
print('Main Parameter')

print(sHat, gHat, piHat, lHat)
#print(em_s, em_g, em_pi, em_l)
print(mcmc_s[0], mcmc_g[0], mcmc_pi, mcmc_l[0])
print(mcmc_s_fb[0], mcmc_g_fb[0], mcmc_pi_fb, mcmc_l_fb[0])

print('Effort Rate')
print(e0Hat, e1Hat)
print(mcmc_e0[0], mcmc_e1[0])
print(mcmc_e0_fb[0], mcmc_e1_fb[0])

ipdb.set_trace()
np.savetxt(proj_dir+'/data/bkt/test/test_param_chain.txt', mcmc_instance.parameter_chain, delimiter=',')

# add in the hazard
y0s = [log[3] for log in incomplete_data_array if log[1]==0]
y1s = [log[3] for log in incomplete_data_array if log[1]==1]
yTs = [log[3] for log in incomplete_data_array if log[1]==4]

h0 = []
h1 = []
for t in range(5):
	EYs = [(log[3], log[4]) for log in incomplete_data_array if log[1]==t]
	h1.append( max(min(sum([x[1] for x in EYs if x[0]==1]) / len([x[0] for x in EYs if x[0]==1]),0.99),0.01) )
	h0.append( max(min(sum([x[1] for x in EYs if x[0]==0]) / len([x[0] for x in EYs if x[0]==0]),0.99),0.01) )

init_param = {'s': [1-np.array(yTs).mean()],
			  'g': [0.3], 
			  'e0':[0.5],
			  'e1':[0.5],
			  'pi': np.array(y0s).mean(),
			  'l': [np.array(y1s).mean() - np.array(y0s).mean()],
			  'h0': h0,
			  'h1': h1}

em_s, em_g, em_pi, em_l = em_instance.estimate(init_param, incomplete_data_array, max_iter = 20)
mcmc_pi, mcmc_s, mcmc_g, mcmc_e0,mcmc_e1, mcmc_l, mcmc_h0, mcmc_h1 = mcmc_instance.estimate(init_param, incomplete_data_array, method='FB', max_iter = L, is_exit=True)

print('Point estimation')
print('Main Parameter')
print(sHat, gHat, piHat, lHat)
print(em_s, em_g, em_pi, em_l)
print(mcmc_s[0], mcmc_g[0], mcmc_pi, mcmc_l[0])

print('Effort Rate')
print(e0Hat, e1Hat)
print(mcmc_e0[0], mcmc_e1[0])

print('Hazard Rate')
print('h0')
print(h0Hat)
print(mcmc_h0)
print('h1')
print(h1Hat)
print(mcmc_h1)