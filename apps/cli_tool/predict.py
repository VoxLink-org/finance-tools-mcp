import pickle
import pandas as pd
import xgboost as xgb
from config.paths import DATA_DIR
from apps.cli_tool.train import get_data, feature_engineering


def load_model():
    """Loads the trained XGBoost model from a file."""
    with open(DATA_DIR / 'xgboost_spy_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, data):
    """Makes predictions using the loaded model."""
    predictions = model.predict(data)
    return predictions


def main():
    """
    Main function to load data, engineer features, and make a prediction
    for the next 10 days' SPY movement.
    """
    print("Making a prediction for the next 10 days...")

    # 1. Get enough historical data to calculate features.
    # Using '1y' to ensure all technical indicators are calculated correctly.
    data = get_data('1y')

    # 2. Engineer features, same as in training.
    data = feature_engineering(data)

    # 3. Define the feature set (must be identical to the one used in training).
    features = [
        'RSI', 'MACD', 'ADX', 'CCI', 'upper_band', 'middle_band', 'lower_band', 'ATR', 'OBV',
        'return_1d', 'return_3d', 'return_5d',
        'VIX_Close', 'vix_lag_1d', 'vix_lag_3d', 'vix_lag_5d'
    ]

    # 4. Select the latest data point for prediction.
    # .iloc[-1:] keeps it as a DataFrame.
    latest_data = data.iloc[-1:]

    # Check if the latest data has NaN values in features, if so, use the last valid one.
    if latest_data[features].isnull().values.any():
        print("Warning: Latest data contains NaN values. Trying to use the last valid data point.")
        last_valid_index = data[features].last_valid_index()
        if last_valid_index is None:
            print("Error: No valid data available for prediction.")
            return
        latest_data = data.loc[[last_valid_index]]

    X_predict = latest_data[features]

    # 5. Load the trained model.
    model = load_model()

    # 6. Make the prediction.
    prediction_result = predict(model, X_predict)

    # 7. Interpret and print the result.
    # From train.py, label '0' is 'Up', and '1' is 'Other'.
    prediction_label = 'Up' if prediction_result[0] == 0 else 'Other'

    print("\n--- Prediction Result ---")
    print(f"Predicted SPY movement for the next 10 days: **{prediction_label}**")
    print(f"Prediction based on data from: {latest_data.index[0].strftime('%Y-%m-%d')}")
    print("\n--- Features Used for Prediction ---")
    print(X_predict.to_string())
    print("\n------------------------------------")


if __name__ == "__main__":
    main()