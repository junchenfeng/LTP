from bkt import update_mastery, compute_success_rate
from numpy.random import random
import os
proj_dir = os.path.dirname(os.path.abspath(__file__))

# model parameters
slip = 0.1
guess = 0.15
init_mastery = 0.6
learn_rate = 0.2


# sim parameters
N = 1000
T = 10

log_data = []
for i in range(N):
	# for each spell, start with the same mastery
	mastery = init_mastery
	for t in range(1,T+1):
		# impute user log
		p = compute_success_rate(guess, slip, mastery)
		#print mastery,p
		Y = int(p>random(1)[0])
		# update mastery
		mastery = update_mastery(mastery, learn_rate)
		
		log_data.append((Y,t)) 
		
with open(proj_dir + '/data/bkt/test/single_sim.txt','w') as f:
	for log in log_data:
		f.write('%d,%d\n' % log)