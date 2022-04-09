from sklearn.model_selection import train_test_split
from src.zoo.train_and_predict_model import *
from src.zoo.pre_processing import pre_process
import statsmodels.api as sm

dataset_iris = sm.datasets.get_rdataset(dataname='iris', package='datasets')
df_iris = dataset_iris.data

# features aka X
feature = df_iris[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]]
# making it as a X
X = feature

# y
y = df_iris["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# expected knn s
expected_knn_1 = KNeighborsClassifier(n_neighbors=1).fit(X, y)
expected_knn_2 = KNeighborsClassifier(n_neighbors=2).fit(X, y)
expected_knn_3 = KNeighborsClassifier(n_neighbors=3).fit(X, y)

# expected dt s
expected_dt_1 = DecisionTreeClassifier(criterion="entropy", max_depth=5).fit(X, y)
expected_dt_2 = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(X, y)
expected_dt_3 = DecisionTreeClassifier(criterion="entropy", max_depth=20).fit(X, y)

# expected svm
expected_svm = svm.SVC(kernel='poly').fit(X, y)

# expected LR
expected_lr = LogisticRegression(C=0.07, solver='sag').fit(X, y)


def test_knn():
    actual_knn_1 = final_Model("KNN", 1, X_train, X_test, y_train, y_test, X, y)
    actual_knn_2 = final_Model("KNN", 2, X_train, X_test, y_train, y_test, X, y)
    actual_knn_3 = final_Model("KNN", 3, X_train, X_test, y_train, y_test, X, y)

    if expected_knn_1.__eq__(actual_knn_1) and expected_knn_2.__eq__(actual_knn_2) and expected_knn_3.__eq__(
            actual_knn_3):
        assert True
    else:
        assert False


def test_dt():
    actual_dt_1 = final_Model("DT", 5, X_train, X_test, y_train, y_test, X, y)
    actual_dt_2 = final_Model("DT", 4, X_train, X_test, y_train, y_test, X, y)
    actual_dt_3 = final_Model("DT", 20, X_train, X_test, y_train, y_test, X, y)

    if expected_dt_1.__eq__(actual_dt_1) and expected_dt_2.__eq__(actual_dt_2) and expected_dt_3.__eq__(actual_dt_3):
        assert True
    else:
        assert False


def test_svm():
    actual_svm = final_Model("SVM", -1, X_train, X_test, y_train, y_test, X, y)
    if expected_svm.__eq__(actual_svm):
        assert True
    else:
        assert False


def test_lr():
    actual_lr = final_Model("LR", -1, X_train, X_test, y_train, y_test, X, y)
    if expected_lr.__eq__(actual_lr):
        assert True
    else:
        assert False


def test_none():
    expected = "Invalid model algorthm or model is not supported yet"
    actual = final_Model("ASA", -1, X_train, X_test, y_train, y_test, X, y)
    if actual == expected:
        assert True
    else:
        assert False
