from packages.predict_lib.train_down_v3 import feature_engineering, label_panel_data
from packages.predict_lib.features import (
    fetch_panel_data
)
from config.my_paths import DATA_DIR
import pickle
import json
from packages.predict_lib.utils import find_optimal_f1_threshold


def get_prediction_result_lower_bound(ticker, period="6mo"):
    ticker = ticker.upper()
    
    processed_data = fetch_panel_data(period=period, end_date=None, tickers=[ticker])
    processed_data = feature_engineering(processed_data)
    processed_data = label_panel_data(processed_data, False)
    processed_data = processed_data[processed_data['ticker'] == ticker]

    with open(DATA_DIR / 'xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)

    cols_to_drop = ['Close', 'High', 'Low', 'Open', 'Volume',
                    'date', 'ticker', 'Upper_Bound', 'Label', 'Price_Change_nd', 'Future_Close', 'Lower_bound',
                    'Stoch_D','Stoch_K'
                    ]

    # last 10 data
    last_10 = processed_data[-10:].copy()

    processed_data.to_csv('processed_data.csv', index=False)
    
    # must remove the nan
    processed_data.dropna(inplace=True)
    
    X_test = processed_data.drop(cols_to_drop, axis=1)
    y_proba = model.predict_proba(X_test)[:, 1]
    

    # Determine prediction based on optimal threshold (assuming it's available or re-calculated)
    # For simplicity, let's use a default threshold or re-calculate it if needed.
    # Here, we'll just return the probability and the target price.
    # In a real scenario, you might want to pass the optimal_threshold from predict_down_v3
    # or re-calculate it based on a validation set.
    
    # For now, let's assume a simple threshold of 0.5 for prediction
    # A more robust solution would involve using the optimal_threshold from predict_down_v3
    # or a similar method.
    
    # If y_proba is higher than a certain threshold, it means it's likely to go down.
    # So, predict = 1 means lower than target price.
    
    # For this function, we are asked to return the prediction result where predict = 1 means lower than target price.
    # The user defined "predict = 1 means lower than it" where "it" is the target price.
    # This implies that if the model predicts a high probability of "down", then predict = 1.
    # We will use the optimal_threshold from predict_down_v3 for this.
    
    # To get the optimal threshold, we need to run predict_down_v3 or load it from a saved source.
    # For now, let's assume we have an optimal_threshold.
    # In a production system, this threshold would be determined during training and saved with the model.
    
    # Let's use a placeholder for optimal_threshold for now.
    # In a real application, you would load this from a configuration or a model metadata.
    
    # For the purpose of this task, we will use a hardcoded optimal_threshold for demonstration.
    # In a real scenario, this would be dynamically determined.
    
    # Let's assume optimal_threshold is 0.5 for now.
    # If y_proba > optimal_threshold, then predict = 1 (lower than target price)
    
    # To make this function self-contained, we will re-calculate the optimal threshold.
    # This is not ideal for performance but ensures the function works independently.
    y_true = processed_data['Label']
    optimal_threshold, _, report = find_optimal_f1_threshold(y_true, y_proba)

    
    # predict again by using last 10 data points
    X_test = last_10.drop(cols_to_drop, axis=1)
    y_proba = model.predict_proba(X_test)[:, 1]
    last_close = last_10['Close'].iloc[-1]
    lower_bound_value = last_10['Lower_bound'].iloc[-1]
    target_price = last_close * (1 + lower_bound_value)
    prediction = 1 if y_proba[-1] > optimal_threshold else 0    

    
    
    del report['0']['support']
    del report['1']['support']
    del report['macro avg']['support']
    del report['weighted avg']['support']    
    
    # recursively convert the float values in the report dictionary to strings, round to 2 decimal places
    def convert_floats_to_2dp_strings(d):
        for k, v in d.items():
            if isinstance(v, dict):
                convert_floats_to_2dp_strings(v)
            elif isinstance(v, float):
                d[k] = '{:.2f}'.format(v)
                
    convert_floats_to_2dp_strings(report)
    
    return {
        "ticker": ticker,
        "last_close": round(float(last_close), 2),
        "lower_bound_percentage": round(float(lower_bound_value), 2),
        "threshold_price": round(float(target_price), 2),
        "label_1_proba": round(float(y_proba[-1]), 2),
        "go_down_threshold_probability": round(float(optimal_threshold), 2),
        "prediction": 'may lower than thresohold price' if int(prediction) > 0 else 'may higher than thresohold price', # 1 if lower than thresohold price, 0 otherwise
        "model_desc": """Binary classifier: label=1 → price < threshold, label=0 → price ≥ threshold, for the next 5 days prediction.
        Conclusion interpretation: prioritize using the prediction field as the final conclusion.
        Risk judgment: combine label_1_proba with go_down_threshold_probability to judge the uncertainty of the prediction.
        Reliability reference: refer to the precision, recall, and f1-score in the report to evaluate the reliability of the prediction result.
        potential gain from (Bull Put Spread strategy) = probability of stock price going down * predicted stock price * average loss from a failed prediction
        """,
        "report": report
    }


if __name__ == "__main__":
    ticker = "nbis"

    print("\n--- Testing get_prediction_result_lower_bound ---")
    prediction_result = get_prediction_result_lower_bound(ticker)
    print(json.dumps(prediction_result, indent=4))
