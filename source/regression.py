import splitting as splitter
import numpy as np
import tree as regt
import copy 
import utils as utl
import ensemble as ens

# Regression Tree Builder
class Regression_Tree_Builder:
    def __init__(self):
        self._LOG_PRINT = True
        self.progress = 0
        self.total_progress = 0
        self.id_counter = 0

    # Update Progress
    def _update_progress(self, update):
        self.progress = self.progress + update
        if(self._LOG_PRINT):
            utl.print_progress_bar(((self.progress / self.total_progress)*100),50)
    
    # Start printing progress
    def _start_progress(self, target_progress):
        self.id_counter = 0
        self.total_progress = target_progress
        if(self._LOG_PRINT):
            print("Training:")

    # Reset Progress
    def _reset_progress(self):
        self.progress = 0
        self.total_progress = 0
        self.id_counter = 0

    # True will print logs from function
    def set_print(self, value=False):
        self._LOG_PRINT = value
        
    # Build a regression tree from training data
    def build(self, X_train, y_train, min_split_size, min_leaf_size, random_feature=False, n_random_feature=0, examples_set=[]):
        if(len(examples_set) == 0):
            examples_set = [x for x in range(len(X_train))]
        self._start_progress(len(X_train))
        root = self.regression_build_tree(X_train, y_train, None, min_split_size, min_leaf_size, examples_set, random_feature, n_random_feature)
        self._reset_progress()
        return regt.Regression_Tree(y_train, root)
    
    # Builds the regression tree from a dataset
    def regression_build_tree(self, X, y, node_parent, min_split_size, min_leaf_size, examples, random_feature, n_random_feature):
        self.id_counter = self.id_counter + 1
        mean = utl.calculate_mean(y, examples)
        current_node = regt.Node(node_parent, self.id_counter, index=0, value=0, examples=copy.deepcopy(examples), prediction=mean) 
        
        if(not (len(examples) >= min_split_size)):
            self._update_progress(len(examples))
            return current_node

        features = []
        if(random_feature):
            features = ens.random_features(X, n_random_feature if n_random_feature > 0 else None)

        feature, value = splitter.feature_split_best(X,y,examples, min_leaf_size, features)

        if(feature == None):
            self._update_progress(len(examples))
            return current_node

        current_node.index = feature
        current_node.value = value
        left = splitter.feature_split(X, feature, examples, value)
        current_node.left = self.regression_build_tree(X, y, current_node, min_split_size, min_leaf_size, left, random_feature, n_random_feature)
        current_node.right = self.regression_build_tree(X, y, current_node, min_split_size, min_leaf_size, examples, random_feature, n_random_feature)
        return current_node