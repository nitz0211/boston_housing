import os, json
import pickle
import warnings
import shutil
import datetime

warnings.filterwarnings('ignore')

import numpy as np

from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


import model

config_file = "config.json"

with open(config_file, "r") as f:
    config = json.load(f)    
    
mmsc = joblib.load(os.path.join(config['model_path'], config['scaler_file']))
polynomial = joblib.load(os.path.join(config['model_path'], config['polynomial_file']))
predictor = joblib.load(os.path.join(config['model_path'], config['model_file']))

pipe = make_pipeline(mmsc, polynomial, predictor)

input = np.loadtxt(os.path.join(config["input_path"], 'input.dat'), delimiter=",", dtype=np.float32)
x = mmsc.transform(input)
x = polynomial.transform(x)
y_prediction = predictor.predict(x)

np.savetxt(os.path.join(config["output_path"], "output.dat"), y_prediction, delimiter=",")