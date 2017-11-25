from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = XGBRegressor(max_depth=8,learning_rate=0.02,n_estimators=400,subsample=0.8,colsample_bytree=1,min_child_weight=5)
        # self.clf = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=10)

        
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
