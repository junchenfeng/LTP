import unittest
import numpy as np

import os, sys
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)

from pyMLC.utl import utl
import math


class TestAssembly(unittest.TestCase):
    def setUp(self):
        self.response_dict = {1: 1, 2: 0, 3: 0}

    def test_L(self):
        learning_curve = np.array([0.25, 0.5, 0.75])
        p = 0.2
        l = utl.L_assembly(self.response_dict, learning_curve, p)
        true_l = math.log(0.2*0.25*0.5*0.25)
        self.assertEqual(l, true_l)

    def test_Z(self):
        learning_curve_matrix = np.array([[0.25, 0.1], 
                                          [0.50, 0.2],
                                          [0.75, 0.3]]).reshape(3, 2)
        mixture_density = np.array([0.2, 0.8]).reshape(2, 1)
        z = utl.Z_assembly(self.response_dict, learning_curve_matrix, mixture_density)

        sum_z = 0.2*0.25*0.5*0.25+0.8*0.1*0.8*0.7
        true_z = np.array([0.2*0.25*0.5*0.25/sum_z, 0.8*0.1*0.8*0.7/sum_z]).reshape(2,1)
        self.assertTrue( sum(abs(true_z - z)) < 1e-15 )
        





if __name__ == '__main__':
    unittest.main()
