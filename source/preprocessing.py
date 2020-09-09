import numpy as np

# Categorical data removal using ONE HOT encoder and 0/1 enconding for dichotomous variables
def preprocess(X, columns_index):
    boolean_columns = []
    for column in columns_index:
        values = list(set(X[:,column]))
        values.sort()
        if len(values) == 2:
            for index in range(len(X)):
                if(X[index][column] == values[0]):
                    X[index][column] = 0
                else:
                    X[index][column] = 1
            boolean_columns.append(column)
        else:
            for index in range(len(X)):
                X[index][column] = str(X[index][column])
            new_columns = np.zeros((len(X), len(values)))
            for index in range(len(X)):
                for i in range(len(values)):
                    if(X[index, column] == values[i]):
                        new_columns[index][i] = 1
            X = np.append(X, new_columns, axis=1)

    for column in boolean_columns:
        columns_index.remove(column)
    
    X = np.delete(X, columns_index, axis=1)          
    return X