import os,sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,proj_dir)

test_data_dir = os.path.dirname(proj_dir)
single_component_data_path = test_data_dir + '/data/test/single_component_complete_track.txt'
double_component_data_path = test_data_dir + '/data/test/double_component_complete_track.txt'


from solver.vanilla_MLC import RunVanillaMLC

test_instance = RunVanillaMLC()

'''
single component test
'''
# initialize
test_instance.load_param(1)
test_instance.load_data(single_component_data_path)
# initialize
test_instance.init()
# solve
test_instance.solve_EM()
# print result
print([0.2,0.4,0.6,0.8,0.9])  # truth
print(test_instance.learning_curve_matrix[0])  # estimated

'''
double component test
'''
# initialize
test_instance.load_param(2)
test_instance.load_data(double_component_data_path)
# initialize
test_instance.init()
# solve
test_instance.solve_EM()
# print result
print([0.2,0.4,0.6,0.8,0.9],[0,8,0.8,0.8,0.8,0.8])  # truth
print(test_instance.learning_curve_matrix)  # estimated

print([0.8,0.2])  # truth
print(test_instance.mixture_density)  # estimated


