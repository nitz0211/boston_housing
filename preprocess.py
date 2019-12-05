import os, json
import warnings
import shutil
import datetime

warnings.filterwarnings('ignore')

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

config_file = "config.json"

with open(config_file, "r") as f:
    config = json.load(f)
    
data = np.loadtxt(os.path.join(config["data_path"], config["data_file"]), delimiter=",", dtype=np.float32)

x = data[:, :-1]
y = data[:, 13]

## 스케일러 피팅 및 저장
mmsc = MinMaxScaler()
mmsc.fit(x)

if os.path.isfile(os.path.join(config['model_path'], config['scaler_file'])):
    shutil.copy(os.path.join(config['model_path'], config['scaler_file']), os.path.join(config['model_backup_path'], config['scaler_file']) + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%s'))

joblib.dump(mmsc, os.path.join(config['model_path'], config['scaler_file']))

## 피처 생성
mmsc = joblib.load(os.path.join(config['model_path'], config['scaler_file']))

x = mmsc.transform(x)

def getPolynomialFeatures(x, degree=2):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    return x_poly, polynomial_features

x, polynomial = getPolynomialFeatures(x)

if os.path.isfile(os.path.join(config['model_path'], config['polynomial_file'])):
    shutil.copy(os.path.join(config['model_path'], config['polynomial_file']), os.path.join(config['model_backup_path'], config['polynomial_file']) + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%s'))
    
joblib.dump(polynomial, os.path.join(config['model_path'], config['polynomial_file']))

train_x, test_x, train_y, test_y = train_test_split(x,  y, test_size=0.2, random_state=1)

np.save(os.path.join(config['feature_path'], 'train_x'), train_x)
np.save(os.path.join(config['feature_path'], 'train_y'), train_y)
np.save(os.path.join(config['feature_path'], 'test_x'), test_x)
np.save(os.path.join(config['feature_path'], 'test_y'), test_y)