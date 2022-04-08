#' Optimize hyper-parameters for a model
#'
#'
#' @param mod A model already defined to be optimized (e.g. KNN). Models are one of: KNN, DecisionTree, SVM, LogisticRegression
#' @param params A dictionary of hyper-parameters to be evaluated. 
#'               Keys are hyper-parameter names. Values are the hyper-parameter values
#'               (e.g. dict(n_neighbors=list(range(1, 21)))).
#' @param n The number of folders in Searching. It might change due to data frame size (e.g. 5).
#' @param X_train Dataframe of training set without the targets.
#' @param y_train Dataframe of targets in the training set.
#'
#' @return A dictionary with best hyper-parameters. 
#'   The first column (named class) lists the classes from the input data frame.
#'   The second column (named count) lists the number of observations for each class from the input data frame.
#'   It will have one row for each class present in input data frame.
#'
#' @export
#'
#' @examples
#' para_optimize(knn, param_grid, 5)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def para_optimize(mod, params, n, X_train, y_train):
    if n > 1:
        if isinstance(mod, KNeighborsClassifier)|isinstance(mod,DecisionTreeClassifier)|isinstance(mod,svm.SVC)|isinstance(mod,LogisticRegression):
            grid = GridSearchCV(mod, params, cv=n, scoring='accuracy')
            grid.fit(X_train, y_train)
            best = grid.best_params_
            return best
        else:
            return("The model is invalid.")
    else:
        return("The number of folder is invalid.")

