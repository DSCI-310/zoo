import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.std_acc import *

dataset_iris = sm.datasets.get_rdataset(dataname='iris', package='datasets')
df_iris = dataset_iris.data
feature = df_iris[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]]
X = feature
y = df_iris["Species"]

yhat = np.zeros(10)

# function input for the test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
for n in range(1, 50):
    # Train Model and Predict
    decTree = DecisionTreeClassifier(criterion="entropy", max_depth=n)
    decTree.fit(X_train, y_train)
    yhat = decTree.predict(X_test)
Ks = 10

msg = "expected value and exact output is not equal!"

# expected out for the test
std_acc = np.zeros((50 - 1))

std_acc_expected = np.zeros((50 - 1))

for n in range(1, 50):
    std_acc_expected[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])


def test_stdacc():
    std_acc = stdAcc(yhat, y_test, 50)
    if pd.DataFrame(std_acc).equals(pd.DataFrame(std_acc_expected)):
        assert True
    else:
        assert False

# class TestStdAcc(unittest.TestCase):
#     def test_stdAcc(self):
#         std_acc_generated = stdAcc(yhat, y_test, 50)
#         for i in range(1, 50):
#             self.assertEqual(std_acc[i], std_acc_generated[i])