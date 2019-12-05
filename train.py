import os, json
import pickle
import warnings
import shutil
import datetime

warnings.filterwarnings('ignore')

import numpy as np

from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV

import model

config_file = "config.json"

with open(config_file, "r") as f:
    config = json.load(f)
    
train_x = np.load(os.path.join(config['feature_path'], 'train_x.npy'))
train_y = np.load(os.path.join(config['feature_path'], 'train_y.npy'))
test_x = np.load(os.path.join(config['feature_path'], 'test_x.npy'))
test_y = np.load(os.path.join(config['feature_path'], 'test_y.npy'))

param_grid = [
    {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [4, 6, 8, 10, 15]
    }
]

predictor = model.BostonHousingPriceRegressor()

cv = GridSearchCV(predictor, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1, return_train_score=True)
cv.fit(train_x, train_y)

print('mean_fit_time:', cv.cv_results_['mean_fit_time'].mean())
print('mean_score_time:', cv.cv_results_['mean_score_time'].mean())
print('best_params:', cv.best_params_)
print('best_score:', cv.best_score_)
print('test_score:', cv.best_estimator_.score(test_x, test_y))
print('test_corr', np.corrcoef(cv.best_estimator_.predict(test_x), test_y)[0][1])

if os.path.isfile(os.path.join(config['model_path'], config['model_file'])):
    shutil.copy(os.path.join(config['model_path'], config['model_file']), os.path.join(config['model_backup_path'], config['model_file']) + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%s'))
    
joblib.dump(predictor, os.path.join(config['model_path'], config['model_file']))