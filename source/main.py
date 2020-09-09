import preprocessing as preproc
import utils as utl
import regression as regr
import evaluation as ev
import pruning as prn
import ensemble as ens
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


DELETE_COL_DATA = [1,9]
PREPROCESS_COL_DATA = [0,2,5,6,8,9]

X, y, _ = utl.load_dataset("data/data.csv", DELETE_COL_DATA)
X = preproc.preprocess(X, PREPROCESS_COL_DATA)

bagging_builder = ens.BaggingPredictor()
random_forest_builder = ens.RandomForestPredictor()

rmse, mape = ev.kfold_cross_validation(X, y, 5, 20, 5)
ev.show_result(rmse, mape)
rmse, mape = ev.kfold_cross_validation_pruned(X, y, 5)
ev.show_result(rmse, mape)

bt, rmse_values, mape_values = bagging_builder.train(X, y, 17)
plt.plot([x for x in range(17)], rmse_values)
plt.show()
bt, rmse_values, mape_values = random_forest_builder.train(X, y, 25, int(np.shape(X)[1]/3))
plt.plot([x for x in range(25)], rmse_values)
plt.show()
