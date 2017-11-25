import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

reg = XGBRegressor(max_depth=18, learning_rate=0.02, n_estimators=500,subsample=0.8, colsample_bytree=1, min_child_weight=5)

scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error',n_jobs=3)
print("log RMSE: {:.4f} +/-{:.4f}".format(
    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))


param_distributions = {'learning_rate' : [0.1,0.05,0.02],
                       'max_depth': [3,6,9,18],
                       'n_estimators': [100,200,500],
                       'subsample': [0.5,0.8,1],
                       'colsample_bytree' : [0.5,0.8,1], 
                       'min_child_weight': [2,5,10,50]}


search = RandomizedSearchCV(reg,param_distributions,n_iter=10, scoring='neg_mean_squared_error', cv=3)
search.fit(X_train,y_train)
search.best_params_
