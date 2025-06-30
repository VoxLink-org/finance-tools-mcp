import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
import talib as ta
import xgboost as xgb
from config.paths import DATA_DIR
from cli_tool.train import get_data, feature_engineering, define_labels

def load_model():
    with open(DATA_DIR / 'xgboost_spy_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, data):
    predictions = model.predict(data)
    return predictions

def main():
    data = get_data('1mo')
    data = feature_engineering(data)
    data = define_labels(data)
    model = load_model()
    predictions = predict(model, data)
    print(predictions)