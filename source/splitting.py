import numpy as np
import copy
import error_measures as err

# Remove items from examples that are less than split value, returns removed items
def feature_split(X, feature, examples_right, split_value):
    changed_list = []
    for example in examples_right:
        if(X[example][feature] < split_value):
            changed_list.append(example)
    for example in changed_list:
        examples_right.remove(example)
    return changed_list

# Calculate best split across all features
def feature_split_best(X, y, examples, leaf_size, legal_features=[]):
    total_min_ssr = np.inf
    feature_split = None
    feature_split_value = 0

    if len(legal_features) == 0:
        legal_features = [x for x in range(np.shape(X)[1])]

    for feature in legal_features:
        split_treshold, ssr = feature_minimal_split_treshold(X, y, feature, examples, leaf_size)
        if (ssr < total_min_ssr):
            feature_split = feature
            feature_split_value = split_treshold
            total_min_ssr = ssr

    return feature_split, feature_split_value

# Calculate best split treshold for given feature
def feature_minimal_split_treshold(X, y, feature, examples, leaf_size):
    split_values = sorted(set([X[example][feature] for example in examples]))
    split_values = [(split_values[index] + split_values[index + 1])/2 for index in range(len(split_values) - 1)]

    if(len(split_values) > 1):
        return continuous_feature_split_search(X, y, feature, examples, split_values, leaf_size)
    elif(len(split_values) == 1):
        return dichotomous_feature_split_search(X, y, feature, examples, split_values, leaf_size)
    else:
        return None, np.inf

# Dichotomous feature split
def dichotomous_feature_split_search(X, y, feature, examples, split_values, leaf_size):
    left_mean = 0
    right_mean = 0
    right_count = 0
    left_count = 0
    left = []
    right = []

    for example in examples:
        if(X[example][feature] < split_values[0]):
            left.append(y[example])
            left_mean = left_mean + y[example]
            left_count = left_count + 1
        else:
            right.append(y[example])
            right_mean = right_mean + y[example]
            right_count = right_count + 1

    if left_count < leaf_size or right_count < leaf_size:
        return None, np.inf

    left_mean = left_mean / left_count
    right_mean = right_mean / right_count
    total_ssr = err.ssr(np.array(left), np.full((len(left)), left_mean)) + err.ssr(np.array(right), np.full((len(right)), right_mean))
    return split_values[0], total_ssr

# Continuous feature split
def continuous_feature_split_search(X, y, feature, examples, split_values, leaf_size):
    examples_right = copy.deepcopy(examples)
    min_split_value = None
    min_split_ssr = np.inf
    right_y = [y[index] for index in examples_right]
    right_mean = sum(right_y)/len(right_y)
    right_examples_size = len(examples_right)
    left_y = []
    left_mean = 0
    left_examples_size = 0

    for split_value in split_values:
        changed_examples = feature_split(X, feature, examples_right, split_value)
        changed_examples_len = len(changed_examples)
   
        if(changed_examples_len > 0 ):
            
            if(len(examples_right) <= 0):
                return min_split_value, min_split_ssr
        
            changed_examples_sum = np.sum([y[index] for index in changed_examples])
            left_mean = ((left_mean * left_examples_size) + changed_examples_sum) / (left_examples_size + changed_examples_len)
            left_examples_size = left_examples_size + changed_examples_len
            right_mean = ((right_mean * right_examples_size) - changed_examples_sum) / (right_examples_size - changed_examples_len)
            right_examples_size = right_examples_size - changed_examples_len

            for index in changed_examples:
                left_y.append(y[index])
                right_y.remove(y[index])

            if((left_examples_size >= leaf_size) and (right_examples_size >= leaf_size)):
                    total_ssr = err.ssr(np.array(left_y), np.full((len(left_y)), left_mean)) + err.ssr(np.array(right_y), np.full((len(right_y)), right_mean))
                    if(total_ssr < min_split_ssr):
                        min_split_ssr = total_ssr
                        min_split_value = split_value

            if(right_examples_size == 1):
                return min_split_value, min_split_ssr

    return min_split_value, min_split_ssr

