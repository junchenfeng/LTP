import random
import numpy
import ipdb

import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The guess rate is 0.2 and the slip rate is 0.1
single_learning_curve = [[0.85,0.7,0.6,0.5,0.4,0.3,0.25,0.20,0.15,0.1]]
double_learning_curve = [[0.85,0.7,0.6,0.5,0.4,0.3,0.25,0.20,0.15,0.1],
                         [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]
num_user = 1000

mixture_density = [0.7,0.3]

missing_rate = [1,0.9,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5]

max_t = len(single_learning_curve[0])


# single component, complete track
user_record = []
for i in range(num_user):
    v_s = []
    for t in range(max_t):
        if random.random() <= single_learning_curve[0][t]:
            v_s.append(0)
        else:
            v_s.append(1)
    user_record.append(v_s)
	
with open(proj_dir + '/data/mlc/single_component_complete_track.txt','w') as f1:
    for i in range(len(user_record)):
        v_s = user_record[i]
        for j in range(len(v_s)):
            f1.write('%d,%d,%d\n' %(i,j+1,v_s[j]))
			
# single component, missing track

user_record = []
for i in range(num_user):
    v_s = []
    for t in range(max_t):
        if random.random() <= single_learning_curve[0][t]:
            v_s.append(0)
        else:
            v_s.append(1)
    user_record.append(v_s)
with open(proj_dir + '/data/mlc/single_component_missing_track.txt','w') as f2:
    for i in range(len(user_record)):
        v_s = user_record[i]
        for j in range(len(v_s)):
            if random.random() >= missing_rate[j]:
                break
            f2.write('%d,%d,%d\n' %(i,j+1,v_s[j]))
			
	
# double component, complete track
user_record = []
for i in range(num_user):
    v_s = []
    # decide the mix to sample from
    j=numpy.random.choice([0,1],p=mixture_density)
    for t in range(max_t):
        if random.random() <= double_learning_curve[j][t]:
            v_s.append(0)
        else:
            v_s.append(1)
    user_record.append(v_s)
with open(proj_dir + '/data/mlc/double_component_complete_track.txt','w') as f3:
    for i in range(len(user_record)):
        v_s = user_record[i]
        for t in range(len(v_s)):
            f3.write('%d,%d,%d\n' %(i,t+1,v_s[t]))
			
			
# double component, missing track
user_record = []
for i in range(num_user):
    v_s = []
    # decide the mix to sample from
    j=numpy.random.choice([0,1],p=mixture_density)
    for t in range(max_t):
        if random.random() <= double_learning_curve[j][t]:
            v_s.append(0)
        else:
            v_s.append(1)
    user_record.append(v_s)
with open(proj_dir + '/data/mlc/double_component_missing_track.txt','w') as f4:
    for i in range(len(user_record)):
        v_s = user_record[i]
        for t in range(len(v_s)):
            if random.random() >= missing_rate[t]:
                break            
            f4.write('%d,%d,%d\n' %(i,t+1,v_s[t]))