import error_measures as err
import utils as utl
from sklearn.model_selection import KFold
import regression as regr
import pruning as prn
from sklearn.model_selection import train_test_split
import numpy as np
import math

# Pring RMSE and MAPE
def show_result(rmse, mape):
    print("        Error  (RMSE): " + str(round(rmse,2)))
    print("        Error  (MAPE): " + str(round(mape, 2)) + "%")

# Evaluate a regression tree
def evaluation_tree(tree, X_test, y_test, print_on=True, deleted_n=[], test_index=[]):
    fx = []
    y = []

    if(print_on):
        utl.print_progress_bar(0,50)

    if(len(test_index) == 0):
        test_index = [x for x in range(len(X_test))]

    count = 0
    for index in test_index:
        if(print_on):
            utl.print_progress_bar((count / (len(test_index) -1) )*100, 50)
            count=count+1
        fx.append(tree.predict(X_test[index], deleted=deleted_n))
        y.append(y_test[index])

    rmse = err.rmse(y,fx)
    mape = err.mape(y,fx)*100

    return rmse, mape

# Evaluate a collection of regression tree 
def evaluation_tree_bagged(bagged_tree, X_test, y_test, print_on=True):
    fx = []
    y = []

    if(print_on):
        utl.print_progress_bar(0,50)

    count = 0
    for index in range(len(X_test)):
        if(print_on):
            utl.print_progress_bar((count / (len(X_test) -1) )*100, 50)
            count=count+1
        fx.append(bagged_tree.predict(X_test[index]))
        y.append(y_test[index])

    rmse = err.rmse(y,fx)
    mape = err.mape(y,fx)*100

    return rmse, mape

# Evaluate the system over a datset using k fold cross validation
def kfold_cross_validation(X, y, fold_size, split_size, leaf_size):
    builder = regr.Regression_Tree_Builder()
    builder.set_print(False)
    kf = KFold(n_splits=fold_size, random_state=1, shuffle=True)
    total_rmse = 0
    total_mape = 0

    for train_index, test_index in kf.split(X):
        X_fold_train = [X[i] for i in train_index]
        y_fold_train = [y[i] for i in train_index]
        X_fold_test = [X[i] for i in test_index]
        y_fold_test = [y[i] for i in test_index]

        model = builder.build(X_fold_train, y_fold_train, split_size, leaf_size)
        rmse, mape = evaluation_tree(model, X_fold_test, y_fold_test, print_on=False)
        total_rmse = total_rmse + rmse
        total_mape = total_mape + mape

    return total_rmse/fold_size, total_mape/fold_size

# Evaluate the system using  pruned tree and k fold cross validation
def kfold_cross_validation_pruned(X, y, fold_size):
    builder = regr.Regression_Tree_Builder()
    builder.set_print(True)
    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    total_rmse = 0
    total_mape = 0


    for train_index, test_index in kf.split(X):
            X_fold_train = [X[i] for i in train_index]
            y_fold_train = [y[i] for i in train_index]
            X_fold_test = [X[i] for i in test_index]
            y_fold_test = [y[i] for i in test_index]

            max_tree = builder.build(X_fold_train, y_fold_train, 2, 1)
            trees, _, _ = prn.weakest_link_pruning(max_tree)

            best_rmse = np.inf
            best_mape = np.inf

            for deleted_nodes in trees:
                current_rmse = 0
                current_mape = 0
                for index in range(len(X_fold_test)):
                    predict = max_tree.predict(X_fold_test[index], deleted_nodes)
                    current_rmse = current_rmse + pow((y_fold_test[index] - predict),2)
                    current_mape = current_mape + abs((y_fold_test[index] - predict)/y_fold_test[index])

                current_rmse = math.sqrt(current_rmse / len(X_fold_test))
                current_mape = current_mape / len(X_fold_test)

                if(current_rmse < best_rmse):
                    best_rmse = current_rmse
                    best_mape = current_mape

            total_rmse = total_rmse + best_rmse
            total_mape = total_mape + best_mape

    return total_rmse/fold_size, total_mape/fold_size
                
    