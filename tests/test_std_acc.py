import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.zoo.std_acc import *


# import data set
dataset_iris = sm.datasets.get_rdataset(dataname='iris', package='datasets')
df_iris = dataset_iris.data

# process data set
feature = df_iris[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]]
X = feature
y = df_iris["Species"]

yhat = np.zeros(10)

# valid inputs for the test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
for n in range(1, 50):
    # Train Model and Predict
    decTree = DecisionTreeClassifier(criterion="entropy", max_depth=n)
    decTree.fit(X_train, y_train)
    yhat = decTree.predict(X_test)
valid_Ks = 50
std_acc_expected = np.zeros((50 - 1))

# wrong input
invalid_Ks = 50.5

# edge-case input
edge_Ks = 1
std_acc_edge = np.zeros((0))




for n in range(1, valid_Ks):
    std_acc_expected[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])


def test_stdacc():
    # test valid-input cases
    std_acc_output = std_acc(yhat, y_test, valid_Ks)
    if pd.DataFrame(std_acc_output).equals(pd.DataFrame(std_acc_expected)):
        assert True
    else:
        assert False

    # test an edge case
    std_acc_edge_output = std_acc(yhat,y_test, edge_Ks)
    if pd.DataFrame(std_acc_edge_output).equals(pd.DataFrame(std_acc_edge)):
        assert True
    else:
        assert False

    # test one invalid-input case
    msg = std_acc(yhat, y_test, invalid_Ks)
    if msg == "The input 'Ks' must be an integer greater than 1." :
        assert True
    else:
        assert False

