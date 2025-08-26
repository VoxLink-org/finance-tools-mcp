from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, fbeta_score

from apps.cli_tool.features import (
    add_technical_indicators,
    add_rolling_statistics,
    add_custom_features
)
from apps.cli_tool.features.modeling import define_labels, prepare_data_with_tssplit, train_model, evaluate_model, predict_with_threshold
from packages.investor_agent_lib.services import yfinance_service

# Define a placeholder for DATA_DIR if it's not available
try:
    from config.my_paths import DATA_DIR
except ImportError:
    import os
    DATA_DIR = "data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

period = "1y"

def get_data(ticker: str, p: Literal["6mo","1y", "2y", "5y", "10y", "ytd"]=period):
    """Fetch data using yfinance service"""
    print("Fetching data...")
    data = yfinance_service.get_price_history(ticker, period=p, raw=True)
    print(data.tail())
    return data

def feature_engineering(ticker: str = "SPY"):
    """Enhanced feature engineering using modular components"""
    print(f"Performing feature engineering for {ticker}...")
    data = get_data(ticker)
    
    # Apply all feature engineering steps
    data = add_technical_indicators(data)
    data = add_rolling_statistics(data)
    data = add_custom_features(data)

    print("Feature engineering complete. Data head with new features:")
    print(data.head())
    return data

def main(ticker="SPY"):
    print("Starting Step 1: Initial Modeling (Full Features)")

    # 1. Data Preparation
    data = feature_engineering(ticker)
    data = define_labels(data)

    # Define XGBoost parameters as per guidance.md
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
    }
    early_stopping_rounds = 100

    # Store evaluation results
    evaluation_results = []

    # 2. Train and Evaluate Model using TimeSeriesSplit
    for fold, (X_train, X_test, y_train, y_test) in enumerate(prepare_data_with_tssplit(data)):
        print(f"\n--- Training Fold {fold + 1} ---")
        
        model = train_model(X_train, y_train, params, early_stopping_rounds)
        
        print(f"--- Evaluating Fold {fold + 1} ---")
        y_pred_proba = evaluate_model(model, X_test, y_test)
        
        # For now, we just print the results. In future steps, we would store them.
        # You can extend evaluate_model to return a dictionary of metrics
        # and append it to evaluation_results.
        

    print("\nStep 1: Initial Modeling (Full Features) Complete.")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    main(ticker)

