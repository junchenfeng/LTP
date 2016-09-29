# simulate the data
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ipdb
N = 2500
d = 1
T = 5

p = 0.5
Lambda  = 0.2
beta1 = np.log(1.5)
beta2 = np.log(1.1)

data = []
# ALL data are observed but not all time periods are used  
for i in range(N):
	for t in range(T):
		# generate X
		X = np.random.binomial(1,p)
		# generate h
		h = Lambda*np.exp(beta1*X+beta2*t)
		if h>1:
			raise ValueException('hazard rate exceeds 1.')
		if np.random.random() < h:
			delta = 1
		else:
			delta = 0
		data.append((i,t,X,delta))
		if delta==1:
			break

			
# check the simulation validity
h_cnt = np.zeros((T,2))
s_cnt = np.zeros((T,2))
for log in data:
	i,t,x,d = log
	h_cnt[t,x]+=d
	s_cnt[t,x]+=1
hrates = h_cnt/s_cnt

ipdb.set_trace()

with open(proj_dir + '/data/bkt/test/hazard_test.txt','w') as f:
	for log in data:
		f.write('%d,%d,%d,%d\n' % log)