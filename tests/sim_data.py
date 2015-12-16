import os,sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,proj_dir)

from random import random
ps = [0.2, 0.4, 0.6, 0.8, 0.9]


with open(proj_dir + '/data/test/single_component_incomplete_track.txt','w') as f:
    for i in range(100):
        for t in range(1, 6):
            if random() < 0.1:
                break
            f.write('%d,%d,%d\n' % (i,t, int(random()<ps[t-1])))



ps = [[0.2, 0.4, 0.6, 0.8, 0.9], [0.8, 0.8, 0.8, 0.8, 0.8]]

Ys = [0.0]*5
Ns = [0]*5

with open(proj_dir + '/data/test/double_component_incomplete_track.txt','w') as f:
    for i in range(500):
        type_idx = int(random()>0.8)
        for t in range(1, 6):
            #if random() < 0.1:
            #    break
            Y = int(random()<ps[type_idx][t-1])
            f.write('%d,%d,%d\n' % (i,t, Y))
            Ns[t-1] += 1.0
            Ys[t-1] += Y

print([Ys[x]/Ns[x] for x in range(5)])


        




