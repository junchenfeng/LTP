import os,sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,proj_dir)

test_data_dir = os.path.dirname(proj_dir)
single_component_data_path = test_data_dir + '/data/test/single_component_complete_track.txt'
double_component_data_path = test_data_dir + '/data/test/double_component_complete_track.txt'
single_component_incom_data_path = test_data_dir + '/data/test/single_component_incomplete_track.txt'
double_component_incom_data_path = test_data_dir + '/data/test/double_component_incomplete_track.txt'

from solver.vanilla_MLC import RunVanillaMLC
from solver.predict_performance import get_predict_performance
import numpy as np

test_instance = RunVanillaMLC()

'''
single component test
'''
# initialize
test_instance.load_param(1)
test_instance.load_data(single_component_incom_data_path)
# initialize
test_instance.init()
# solve
test_instance.solve_EM()
# print result
print([0.2, 0.4, 0.6, 0.8, 0.9])  # truth
print(test_instance.learning_curve_matrix)  # estimated

# compare the performance
'''
print(get_predict_performance(test_instance.user_result, 
                              test_instance.learning_curve_matrix,
                              5,
                              test_instance.mixture_density))

'''

'''
double component test
'''
# initialize
test_instance.load_param(2)
test_instance.load_data(double_component_incom_data_path)
# initialize
test_instance.init()
# solve
test_instance.solve_EM()
# print result
true_param = np.array([[0.2, 0.4, 0.6, 0.8, 0.9], [0.8, 0.8, 0.8, 0.8, 0.8]]).reshape(2,5)
true_mixture = np.array([0.8, 0.2]).reshape(1,2)
print(true_param)  # truth
print(test_instance.learning_curve_matrix)  # estimated

print(true_mixture)  # truth
print(test_instance.mixture_density)  # estimated

print('estimated rate')

print(np.dot(true_mixture, true_param))
print(np.dot(test_instance.learning_curve_matrix, test_instance.mixture_density))

'''
print('forecast fitness')
print(get_predict_performance(test_instance.user_result, 
                              test_instance.learning_curve_matrix,
                              5,
                              test_instance.mixture_density))
'''
