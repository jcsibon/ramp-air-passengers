# from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import lightgbm as lgb
from lightgbm import LGBMRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = reg = LGBMRegressor(max_depth=11, learning_rate=0.02, n_estimators=5000, num_leaves=40)
        # self.clf = XGBRegressor(max_depth=9,learning_rate=0.02,n_estimators=500,subsample=0.8,colsample_bytree=1,min_child_weight=5)
        # self.clf = XGBRegressor(max_depth=6,learning_rate=0.1,n_estimators=500,subsample=0.8,colsample_bytree=0.8,min_child_weight=5)
        # self.clf = XGBRegressor(max_depth=15,learning_rate=0.02,n_estimators=1000,subsample=0.8,colsample_bytree=1,min_child_weight=5)
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
