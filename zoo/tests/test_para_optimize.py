# import pytest
import unittest
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from src.para_optimize import para_optimize
from src.pre_processing import *

# data = pd.read_csv("https://raw.githubusercontent.com/poddarswakhar/dump/main/exp.csv")
header = ["animalName", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
          "predator", "toothed", "backbone", "breathes", "venomous", "fins",
          "legs", "tail", "domestic", "catsize", "type"]

data = pre_process("https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data", header)
X = data[["hair", "feathers", "eggs", "milk", "airborne",
                    "aquatic", "predator", "toothed", "backbone", "breathes",
                    "venomous", "fins", "legs", "tail", "domestic", "catsize"]]
y = data['type']
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
svm = svm.SVC()
lr = LogisticRegression(max_iter=100)
param_grid_knn = {"n_neighbors" : [1,2,3,4,5]}
param_grid_dt = {"max_depth" : [1,2,3,4,5]}
param_grid_svm = {"C" : [0.01,0.1,1,10,100]}
param_grid_lr = {"C" : [0.01,0.1,1,10,100]}

class TestParaOptimize(unittest.TestCase):
    # Test the output type of the para_optimize
    def test_para_optimize_type(self):
        self.assertTrue(isinstance(para_optimize(knn, param_grid_knn, 5, X, y), dict))
        self.assertTrue(isinstance(para_optimize(dt, param_grid_dt, 5, X, y), dict))
        self.assertTrue(isinstance(para_optimize(svm, param_grid_svm, 2, X, y), dict))
        self.assertTrue(isinstance(para_optimize(lr, param_grid_lr, 5, X, y), dict))

    # Test the type of the input parameter, params 
    def test_para_optimize_paramstype(self):
        self.assertTrue(isinstance(param_grid_knn, dict))
        self.assertTrue(isinstance(param_grid_dt, dict))
        self.assertTrue(isinstance(param_grid_knn, dict))
        self.assertTrue(isinstance(param_grid_svm, dict))

    # Test the invalid input model type 
    def test_para_optimize_modtype(self):
        self.assertEqual(para_optimize("model", param_grid_knn, 5, X, y), "The model is invalid.")
    


