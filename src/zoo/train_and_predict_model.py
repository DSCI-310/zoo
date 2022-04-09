from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, classification_report
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def final_Model(algorithm, threshold, X_train, X_test, y_train, y_test, X, y):
    """
    creates the final model of the algorithms specified
    :param algorithm: algorithm of the model
    :param threshold: threshold for the algorthm like depth, n_neighbors
    :param X_train: x train data
    :param X_test: x test data
    :param y_train: y train data
    :param y_test: y test data
    :param X: All X data or Features
    :param y: All Y data or predictor variable
    :return: a model
    """
    if algorithm.lower().strip() == "svm":
        # returning the final svm model
        return helper_svm(X_train, X_test, y_train, y_test, X, y)

    elif algorithm.lower().strip() == "lr":
        # returning the final lr model
        return helper_lr(X_train, X_test, y_train, y_test, X, y)

    elif algorithm.lower().strip() == "dt":
        return helper_dt(threshold, X_train, X_test, y_train, y_test, X, y)

    elif algorithm.lower().strip() == "knn":
        # returning the final kn model
        return helper_knn(threshold, X_train, X_test, y_train, y_test, X, y)

    else:
        return "Invalid model algorthm or model is not supported yet"


def helper_svm(X_train, X_test, y_train, y_test, X, y):
    """
    helper method of final_Model for SVM
    """
    svec = svm.SVC(kernel='poly')
    svec.fit(X_train, y_train)
    yhat = svec.predict(X_test)
    acc = jaccard_score(y_test, yhat, average='micro')
    print("SVM INFO: So the Training Jaccard score for SVM is: " + str(acc))
    print("\nSVM Evaluation:\n")
    print(classification_report(y_test, yhat))
    # Final SVM is here used the splited test part to train again for better training, and better prediction
    svec = svm.SVC(kernel='poly')
    finalSVM = svec.fit(X, y)
    # returning the final svm model
    return finalSVM


def helper_dt(threshold, X_train, X_test, y_train, y_test, X, y):
    """
    helper method of final_Model for DT
    """
    Final_dec_Tree = DecisionTreeClassifier(criterion="entropy", max_depth=threshold)
    Final_dec_Tree.fit(X, y)
    yhat = Final_dec_Tree.predict(X_test)
    accuracyScore = metrics.accuracy_score(y_test, yhat)
    print("DT INFO: So the accuracy score for max depth = " + str(threshold) + " is " + str(accuracyScore))
    # cross-validation on decision tree
    cv_results_dt = cross_validate(Final_dec_Tree, X_train, y_train, cv=4, return_train_score=True);
    print("\nDT Cross Validate: \n")
    print(pd.DataFrame(cv_results_dt).mean())
    print("\nDT Classification report: \n")
    print(classification_report(y_test, yhat))
    # returning the final dt model
    return Final_dec_Tree


def helper_lr(X_train, X_test, y_train, y_test, X, y):
    """
    helper method of final_Model for LR
    """
    LR = LogisticRegression(C=0.07, solver='sag').fit(X_train, y_train)
    yhat = LR.predict(X_test)
    acc = jaccard_score(y_test, yhat, average='micro')
    print("LR INFO: So the Training Jaccard score for Logistic Regression is: " + str(acc))
    print("\nLR Evaluation: \n")
    print(classification_report(y_test, yhat))
    # final LR model is here used the splited test part to train again for better training, and better prediction
    finalLR = LogisticRegression(C=0.07, solver='sag').fit(X, y)
    # returning the final lr model
    return finalLR


def helper_knn(threshold, X_train, X_test, y_train, y_test, X, y):
    """
    helper method of final_Model for KNN
    """
    final_knn_model = KNeighborsClassifier(n_neighbors=threshold).fit(X, y)
    yhat = final_knn_model.predict(X_test)
    accuracyScore = metrics.accuracy_score(y_test, yhat)
    print("KNN INFO: So the accuracy score for K = " + str(threshold) + " is " + str(accuracyScore))
    # cross-validation on knn
    cv_results_knn = cross_validate(final_knn_model, X_train, y_train, cv=3, return_train_score=True);
    print("\nKNN Cross Validate: \n")
    print(pd.DataFrame(cv_results_knn).mean())
    print("\nKNN Classification report: \n")
    print(classification_report(y_test, yhat))
    # returning the final kn model
    return final_knn_model