import os, json
import warnings
import shutil
import datetime
import pandas as pd

warnings.filterwarnings('ignore')

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

## 환경설정 로드
config_file = "config.json"

with open(config_file, "r") as f:
    config = json.load(f)
    
if os.path.isfile(os.path.join(config['data_path'], config['data_file'])) == False:

    # Boston Housing Dataset url
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

    # dataset feautre names
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM' ,'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # Pandas를 통한 dataset read 및 save
    data_path = './data'
    housing_original_file = 'housing.data'
    housing_file = 'housing.csv'

    if os.path.isfile(os.path.join(data_path, housing_original_file)):
        boston_df = pd.read_csv(os.path.join(config['data_path'], config['data_original_file']), sep="\s+", header=None)
    else:
        boston_df = pd.read_csv(dataset_url, sep="\s+", header=None)

    boston_df.columns = feature_names
    boston_df.to_csv(os.path.join(config['data_path'], config['data_file']), index=False, header=None)

## data_path에 존재하는 data_file 로드
data = np.loadtxt(os.path.join(config["data_path"], config["data_file"]), delimiter=",", dtype=np.float32)

x = data[:, :-1]
y = data[:, 13]

## 스케일러 피팅 및 저장
mmsc = MinMaxScaler()
mmsc.fit(x)

if os.path.isfile(os.path.join(config['model_path'], config['scaler_file'])):
    shutil.copy(os.path.join(config['model_path'], config['scaler_file']), os.path.join(config['model_backup_path'], config['scaler_file']) + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%s'))

joblib.dump(mmsc, os.path.join(config['model_path'], config['scaler_file']))

## 스케일러 로드 후 변환 수행
mmsc = joblib.load(os.path.join(config['model_path'], config['scaler_file']))

x = mmsc.transform(x)

## 다항 특성 생성 및 변환기 생성 함수
def getPolynomialFeatures(x, degree=2):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    return x_poly, polynomial_features

## 다항 특성 생성 및 변환기 저장
x, polynomial = getPolynomialFeatures(x)

if os.path.isfile(os.path.join(config['model_path'], config['polynomial_file'])):
    shutil.copy(os.path.join(config['model_path'], config['polynomial_file']), os.path.join(config['model_backup_path'], config['polynomial_file']) + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%s'))
    
joblib.dump(polynomial, os.path.join(config['model_path'], config['polynomial_file']))

## 학습, 실험 데이터 분리하여 피처 저장
train_x, test_x, train_y, test_y = train_test_split(x,  y, test_size=0.2, random_state=1)

np.save(os.path.join(config['feature_path'], 'train_x'), train_x)
np.save(os.path.join(config['feature_path'], 'train_y'), train_y)
np.save(os.path.join(config['feature_path'], 'test_x'), test_x)
np.save(os.path.join(config['feature_path'], 'test_y'), test_y)