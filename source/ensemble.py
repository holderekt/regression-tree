from sklearn.utils import resample
import regression as regr
import numpy as np
import evaluation as ev
import error_measures as err
import copy
import random

# Returns index of bootstrapped datset and out of bag dataset
def boostrap_dataset(X):
    index_x = [x for x in range(len(X))]
    nx = resample(index_x)
    return nx, list(set(index_x) - set(nx))

# Generate n random feature from X
def random_features(X, n):
    return random.sample(range(np.shape(X)[1]), n)


# Tree created from boostrapped sample
class BaggedTree:
    def __init__(self, tree, train_samples, outofbag):
        self.tree = tree
        self.train_samples = set(train_samples)
        self.outofbag = set(outofbag)

    # True if this tree has been trained with sample_index example
    def trained_with(self, sample_index):
        return sample_index in self.train_samples

    # Tree prediction
    def predict(self, sample):
        return self.tree.predict(sample)

# Collectin of BaggedTree
class BaggingTrees:
    def __init__(self, X_train, y_train):
        self.trees = []
        self.X = X_train
        self.y = y_train

    # Averages the predictin of all trees
    def predict(self, sample):
        prediction = 0
        for tree in self.trees:
            prediction = prediction + tree.predict(sample)
        return prediction / len(self.trees)
    
    # Add a tree to the collection
    def add(self, tree):
        self.trees.append(tree)

    # Calculate out of bag error from training set
    def oob_error(self):
        fx = []
        yx = []
        for index in range(len(self.X)):
            prediction = 0
            count = 0
            for tree in self.trees:

                if not tree.trained_with(index):
                    prediction = prediction + tree.predict(self.X[index])
                    count = count + 1
            if(count > 0):
                prediction = prediction / count
                fx.append(prediction)
                yx.append(self.y[index])
        return err.rmse(yx, fx), err.mape(yx, fx)*100


# Train a bagging predictor 
class BaggingPredictor:
    def __init__(self):
        self.builder = regr.Regression_Tree_Builder()
        self.builder.set_print(False)

    def train(self, X_train, y_train, b):
        rmses_train = []
        mapes_train = []
        bt = BaggingTrees(X_train, y_train)
        for _ in range(b):
            bt.add(self._create_tree(X_train, y_train))
            rmse_oob, mape_oob = bt.oob_error()
            rmses_train.append(rmse_oob)
            mapes_train.append(mape_oob)
        return bt, rmses_train, mapes_train

    def _create_tree(self, X, y):
        bootstrap, outofbag = boostrap_dataset(X)
        tree = self.builder.build(X, y, 2, 1, examples_set=copy.deepcopy(bootstrap))
        return BaggedTree(tree, bootstrap, outofbag)

# Train a Random Forest
class RandomForestPredictor:
    def __init__(self):
        self.builder = regr.Regression_Tree_Builder()
        self.builder.set_print(False)

    def train(self, X_train, y_train, b, p):
        rmses_train = []
        mapes_train = []
        bt = BaggingTrees(X_train, y_train)
        for _ in range(b):
            bt.add(self._create_tree(X_train, y_train, p))
            rmse_oob, mape_oob = bt.oob_error()
            rmses_train.append(rmse_oob)
            mapes_train.append(mape_oob)
        return bt, rmses_train, mapes_train

    def _create_tree(self, X, y, p):
        bootstrap, outofbag = boostrap_dataset(X)
        tree = self.builder.build(X, y, 2, 1, examples_set=copy.deepcopy(bootstrap), random_feature=True, n_random_feature=p)
        return BaggedTree(tree, bootstrap, outofbag)