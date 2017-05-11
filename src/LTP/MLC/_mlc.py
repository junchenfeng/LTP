import os,sys
proj_dir = os.path.dirname(os.path.abspath(__file__))

test_data_dir = os.path.dirname(proj_dir)
single_component_data_path = test_data_dir + '/data/test/single_component_complete_track.txt'
double_component_data_path = test_data_dir + '/data/test/double_component_complete_track.txt'
single_component_incom_data_path = test_data_dir + '/data/test/single_component_incomplete_track.txt'
double_component_incom_data_path = test_data_dir + '/data/test/double_component_incomplete_track.txt'

from .solver.vanilla_MLC import RunVanillaMLC
from .solver.predict_performance import get_predict_performance
import numpy as np

test_instance = RunVanillaMLC()

'''
single component test
'''
# initialize
test_instance.init(1)
test_instance.load_data(single_component_data_path)
# solve
res = test_instance.solve()
# print result
print([0.2, 0.4, 0.6, 0.8, 0.9])  # truth
print(res['q'])  # estimated

# compare the performance
print(get_predict_performance(test_instance.response_data, 
                              res['q'],
                              res['p']))


'''
double component test
'''
# initialize
test_instance.init(2)
test_instance.load_data(double_component_data_path)
# solve
res = test_instance.solve()
# print result
true_param = np.array([[0.2, 0.4, 0.6, 0.8, 0.9], [0.8, 0.8, 0.8, 0.8, 0.8]]).reshape(2,5)
true_mixture = np.array([0.8, 0.2]).reshape(1,2)

learning_curve_matrix = res['q']
mixture_density = res['p']
print(true_param)  # truth
print(learning_curve_matrix)  # estimated

print(true_mixture)  # truth
print(mixture_density)  # estimated

print('estimated rate')

print(np.dot(true_mixture, true_param))
print(np.dot(learning_curve_matrix, mixture_density))
