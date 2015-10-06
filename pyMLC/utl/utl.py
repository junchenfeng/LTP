import math
def B(v,q):
    if v == 0:
        return 1-q
    else:
        return q
    
def L_assembly(v_dict, q_vec, p):
    l = math.log(p)
    for j, v_j in v_dict.iteritems():
        l += math.log(B(v_j, q_vec[j-1]))  # j-1 because the times are coded from 1-N
    return l
