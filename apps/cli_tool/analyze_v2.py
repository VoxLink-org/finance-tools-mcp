from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, fbeta_score

from apps.cli_tool.features import (
    add_technical_indicators,
    add_rolling_statistics,
    add_custom_features,
    prepare_data,
    train_model,
    evaluate_model
)
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

def define_labels(data):
    """Defines target labels (1 if price increases 1% in 5 days)"""
    print("Defining labels...")
    data['Future_Max_Close'] = data['Close'].rolling(window=5, closed='right').max().shift(-5)
    data['Label'] = (data['Future_Max_Close'] > data['Close'] * 1.01).astype(int)
    data.dropna(inplace=True)
    print(f"Label distribution:\n{data['Label'].value_counts()}")
    return data


def main(ticker="SPY"):
    """Main analysis pipeline using enhanced features"""
    # Run the feature engineering and label definition pipeline
    processed_data = feature_engineering(ticker=ticker)
    labeled_data = define_labels(processed_data)
    
    # Prepare data for XGBoost
    X_train, X_test, y_train, y_test = prepare_data(labeled_data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate with default threshold (0.5)
    print("\nEvaluating model with default threshold (0.5)...")
    evaluate_model(model, X_test, y_test)

    # Tune threshold for better F2-score
    print("\nTuning classification threshold to optimize F2-score...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f2_scores = [fbeta_score(y_test, (y_pred_proba >= t).astype(int), beta=2)
                for t in thresholds]
    
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Best threshold for maximum F2-score: {best_threshold:.4f}")
    print(f"Maximum F2-score: {f2_scores[best_idx]:.4f}")
    print(f"Corresponding precision: {precision[best_idx]:.4f}")
    print(f"Corresponding recall: {recall[best_idx]:.4f}")

    # Evaluate with tuned threshold
    print(f"\nEvaluating model with tuned threshold ({best_threshold:.4f})...")
    evaluate_model(model, X_test, y_test, best_threshold)

    # Calculate and display feature importance
    print("\nFeature Importances:")
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importances)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    main(ticker)