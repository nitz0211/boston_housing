import os, json
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')

import numpy as np

from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import model

config_file = "config.json"

## 환경설정 파일 로드
with open(config_file, "r") as f:
    config = json.load(f)

## input_file, output_file에 대한 Argument 파싱
parser = ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, required=True)
parser.add_argument("-o", "--output_file", type=str, required=True)
args = parser.parse_args()

## input_file을 읽어서 예측한 후 output_file로 저장
def predict_to_file(input_file, output_file):
    ## 학습된 스케일러 로드
    mmsc = joblib.load(os.path.join(config['model_path'], config['scaler_file']))
    ## 학습된 다항 변환기 로드
    polynomial = joblib.load(os.path.join(config['model_path'], config['polynomial_file']))
    ## 학습된 모델 로드
    predictor = joblib.load(os.path.join(config['model_path'], config['model_file']))

    ## input 폴더에 있는 input_file 로드
    input = np.loadtxt(os.path.join(config["input_path"], input_file), delimiter=",", dtype=np.float32)
    ## 스케일러 변환 수행
    x = mmsc.transform(input)
    ## 다항 변환 수행
    x = polynomial.transform(x)
    ## 예측 수행
    y_prediction = predictor.predict(x)
    ## output 폴더에 output_file로 저장
    np.savetxt(os.path.join(config["output_path"], output_file), y_prediction, delimiter=",")

## 예측 수행
predict_to_file(args.input_file, args.output_file)