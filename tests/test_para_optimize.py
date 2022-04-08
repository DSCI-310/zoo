# import pytest
import unittest
import numpy as np
import pandas as pd
from src.zoo.para_optimize import *


# Create simple datasets for the tests
dummy_data1 = [[1,2,3,4,5,0],[2,3,4,1,2,0], [1,1,3,4,5,0], [1,1,2,3,4,1],[2,3,5,4,1,1], [4,5,6,3,2,1], [2,3,4,2,1,0], [1,2,1,2,4,1]]
dummy_data2 = [[2,1,3,4,2,0],[1,1,2,3,2,0],[2,4,3,2,3,0],[1,2,6,5,4,1],[3,4,4,5,1,1],[5,4,3,5,6,1]]
col_names = ["var_1", "var_2", "var_3", "var_4", "var_5", "tar"]
dummy_set1 = pd.DataFrame(dummy_data1)
dummy_set2 = pd.DataFrame(dummy_data2)
dummy_set1.columns=col_names
dummy_set2.columns=col_names

X1 = dummy_set1[["var_1", "var_2", "var_3", "var_4", "var_5"]]
y1 = dummy_set1['tar']
X2 = dummy_set2[["var_1", "var_2", "var_3", "var_4", "var_5"]]
y2 = dummy_set2['tar']

knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
svm = svm.SVC()
lr = LogisticRegression(max_iter=100)
param_grid_knn = {"n_neighbors" : [1,2,3]}
param_grid_dt = {"max_depth" : [1,2,3]}
param_grid_svm = {"C" : [0.01,0.1,1,10,100]}
param_grid_lr = {"C" : [0.01,0.1,1,10,100]}

class TestParaOptimize(unittest.TestCase):
    
    # Test the output type of the para_optimize
    def test_para_optimize_type(self):
        self.assertTrue(isinstance(para_optimize(knn, param_grid_knn, 2, X1, y1), dict))
        self.assertTrue(isinstance(para_optimize(dt, param_grid_dt, 2, X1, y1), dict))
        self.assertTrue(isinstance(para_optimize(svm, param_grid_svm, 2, X1, y1), dict))
        self.assertTrue(isinstance(para_optimize(lr, param_grid_lr, 2, X1, y1), dict))
        self.assertTrue(isinstance(para_optimize(knn, param_grid_knn, 2, X2, y2), dict))
        self.assertTrue(isinstance(para_optimize(dt, param_grid_dt, 2, X2, y2), dict))
        self.assertTrue(isinstance(para_optimize(svm, param_grid_svm, 2, X2, y2), dict))
        self.assertTrue(isinstance(para_optimize(lr, param_grid_lr, 2, X2, y2), dict))

    # Test the type of the input parameter, params 
    def test_para_optimize_paramstype(self):
        self.assertTrue(isinstance(param_grid_knn, dict))
        self.assertTrue(isinstance(param_grid_dt, dict))
        self.assertTrue(isinstance(param_grid_knn, dict))
        self.assertTrue(isinstance(param_grid_svm, dict))

    # Test the invalid input model type 
    def test_para_optimize_modtype(self):
        self.assertEqual(para_optimize("knn", param_grid_knn, 2, X1, y1), "The model is invalid.")
        self.assertEqual(para_optimize("lr", param_grid_lr, 2, X2, y2), "The model is invalid.")
    
    # Test the invalid input n
    def test_para_optimize_n(self):
        self.assertEqual(para_optimize(knn, param_grid_knn, 0, X1, y1), "The number of folder is invalid.")
        self.assertEqual(para_optimize(knn, param_grid_knn, 0, X2, y2), "The number of folder is invalid.")
        self.assertEqual(para_optimize(knn, param_grid_knn, -1, X1, y1), "The number of folder is invalid.")
        self.assertEqual(para_optimize(knn, param_grid_knn, -1, X2, y2), "The number of folder is invalid.")
        
    # Test the invalid input n and model type
    def test_para_optimize_n(self):
        self.assertEqual(para_optimize("knn", param_grid_knn, 0, X1, y1), "The number of folder is invalid.")
        self.assertEqual(para_optimize("model", param_grid_knn, -1, X2, y2), "The number of folder is invalid.")

    
    # Test the edge case of n
    def test_para_optimize_edge_n(self):
        self.assertEqual(para_optimize(knn, param_grid_knn, 1, X1, y1), "The number of folder is invalid.")
        self.assertEqual(para_optimize(knn, param_grid_knn, 1, X2, y2), "The number of folder is invalid.")
        self.assertEqual(para_optimize(knn, param_grid_knn, 2, X1, y1), {'n_neighbors': 1})
        self.assertEqual(para_optimize(lr, param_grid_lr, 2, X2, y2), {'C': 1})
        self.assertEqual(para_optimize(svm, param_grid_svm, 2, X2, y2), {'C': 0.01})

        
if __name__ == '__main__':
    unittest.main()    