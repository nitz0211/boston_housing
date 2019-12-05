from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

class BostonHousingPriceRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, n_estimators=300, max_depth=10):
        self.ridge = Ridge(alpha=alpha)
        self.rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit(self, X, y):
        self.ridge.fit(X, y)
        self.coef_ = self.ridge.coef_
        self.rf.fit(X, y)
        self.feature_importances_ = self.rf.feature_importances_
        return self
        
    def predict(self, X):
        ridge_predict = self.ridge.predict(X)
        rf_predict = self.rf.predict(X)
        return ridge_predict * 0.4 + rf_predict * 0.6