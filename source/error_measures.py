import math as math
import numpy as np

# MAPE Mean Absolute Percentage Error
def mape(y_true, y_pred):
    total = 0
    for index in range(len(y_true)):
        total = total + abs((y_true[index] - y_pred[index])/(y_true[index]))
    return total/len(y_true)

# RMSE Root Mean Squared Error
def rmse(y_true, y_pred):
    total = 0
    for index in range(len(y_true)):
        total = total + pow((y_true[index] - y_pred[index]),2)
    total = total / len(y_true)
    return math.sqrt(total)

# SSR Sum Squared Residuals
def ssr(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred))
