from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

period = "2y"

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
    """Main analysis pipeline using enhanced features"""
    # Run the feature engineering and label definition pipeline
    processed_data = feature_engineering(ticker=ticker)
    labeled_data = define_labels(processed_data)
    
    # Prepare data for XGBoost
    # Prepare data for XGBoost using TimeSeriesSplit
    tscv_splits = prepare_data_with_tssplit(labeled_data)

    models = []
    y_pred_probas = []
    y_tests = []
    
    threshold = 0.3  # Same threshold used in predict_with_threshold

    for i, (X_train, X_test, y_train, y_test) in enumerate(tscv_splits):
        print(f"\n--- Fold {i+1} ---")

        # Train the model
        model = train_model(X_train, y_train)
        models.append(model)
        # y_pred_probas.append(model.predict_proba(X_test)[:, 1])
        # Modified code to use predict_with_threshold:
        y_pred_labels_fold, y_pred_proba_fold = predict_with_threshold(model, X_test, threshold=threshold) # You can adjust the threshold
        y_pred_probas.append(y_pred_proba_fold)
        y_tests.append(y_test)

    # Evaluate with default threshold (0.5)
    # Aggregate results for threshold tuning and final evaluation
    all_y_test = pd.concat(y_tests)
    all_y_pred_proba = np.concatenate(y_pred_probas)

    # Plot probability distributions
    print("\nPlotting probability distributions...")
    plt.figure(figsize=(10, 6))
    plt.hist(all_y_pred_proba[all_y_test == 0], bins=50, alpha=0.5, label='Label=0')
    plt.hist(all_y_pred_proba[all_y_test == 1], bins=50, alpha=0.5, label='Label=1')
    plt.title('Predicted Probability Distribution by Label')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('data/probability_distribution.png')
    plt.close()
    
    # Evaluate model performance with threshold
    print("\nEvaluating model performance with threshold...")
    all_y_pred = (all_y_pred_proba >= threshold).astype(int)
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_y_test, all_y_pred))
    print("\nClassification Report:")
    print(classification_report(all_y_test, all_y_pred))

    # Also evaluate last fold's model for reference
    final_model = models[-1]
    evaluate_model(final_model, X_test, y_test)  # Using last fold's X_test, y_test

    # Calculate and display feature importance from the last model
    print("\nFeature Importances (from last fold's model):")
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns, # X_train from the last fold
        'Importance': final_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importances)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    main(ticker)
    
