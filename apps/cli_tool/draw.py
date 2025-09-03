from packages.predict_lib.train_down_v3 import feature_engineering, label_panel_data
from packages.predict_lib.features import (
    fetch_panel_data
)
from config.my_paths import DATA_DIR
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from packages.predict_lib.utils import find_optimal_f1_threshold
from datetime import datetime, timedelta


def get_prediction_result_of_each_day(ticker, period="1y"):
    """
    Get prediction results for each day in the dataset.
    Returns predictions with actual prices and threshold bounds.
    """
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

    # must remove the nan
    processed_data.dropna(inplace=True)
    
    X_test = processed_data.drop(cols_to_drop, axis=1)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"DEBUG: y_proba length: {len(y_proba)}")
    
    # Calculate optimal threshold
    y_true = processed_data['Label']
    optimal_threshold, _, report = find_optimal_f1_threshold(y_true, y_proba)

    print('Report:')
    print(report)
    
    # Get predictions for all days
    predictions = []
    # Reset index to ensure sequential integer indexing for y_proba
    processed_data = processed_data.reset_index(drop=True)
    for idx, row in processed_data.iterrows():
        date = row['date']
        actual_close = row['Close']
        upper_bound_value = row['Upper_Bound']
        lower_bound_value = row['Lower_bound']
        upper_threshold_price = actual_close * (1 + upper_bound_value)
        # lower_threshold_price = actual_close * (1 + lower_bound_value)
        proba = y_proba[idx]
        prediction = 1 if proba > optimal_threshold else 0
        
        # Log the date shift for validation
        original_date = date
        prediction_date = date + pd.Timedelta(days=5)
        # print(f"DEBUG: Prediction made on {original_date.strftime('%Y-%m-%d')} applies to {prediction_date.strftime('%Y-%m-%d')}")
        
        predictions.append({
            'date': original_date,
            'close': actual_close,
            'actual_close': actual_close,
            'upper_threshold_price': upper_threshold_price,
            # 'lower_threshold_price': lower_threshold_price,
            'prediction_probability': proba,
            'prediction': prediction,
            'lower_bound_pct': lower_bound_value * 100
        })
    
    pred_df = pd.DataFrame(predictions)
    
    # shift the date and actual_close column by 5 days
    pred_df['date'] = pred_df['date'].shift(-5)
    pred_df['actual_close'] = pred_df['actual_close'].shift(-5)
    
    # review if the prediction is correct or not
    pred_df['is_correct'] = (pred_df['actual_close'] > pred_df['upper_threshold_price']) & (pred_df['prediction'] == 1) | (pred_df['actual_close'] < pred_df['upper_threshold_price']) & (pred_df['prediction'] == 0)
    
    
    return pred_df, optimal_threshold


def fetch_actual_prices(ticker, period="1y"):
    """Fetch actual historical prices for visualization using existing data."""
    ticker = ticker.upper()
    
    # Fetch data using the existing fetch_panel_data
    data = fetch_panel_data(period=period, tickers=[ticker])
    
    # Filter for the specific ticker and get relevant columns
    ticker_data = data[data['ticker'] == ticker].copy()
    
    # Ensure we have the required columns
    if 'date' not in ticker_data.columns:
        ticker_data['date'] = ticker_data.index
    
    # Sort by date
    ticker_data = ticker_data.sort_values('date')
    
    # Return with standardized column names
    return ticker_data[['date', 'Close']].rename(columns={'date': 'Date'})


def plot_stock_prediction(ticker, period="1y", save_path=None):
    """
    Plot actual stock prices with prediction threshold bounds.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis ("1y", "6mo", "3mo", "1mo")
        save_path: Optional path to save the plot
    """
    ticker = ticker.upper()
    
    # Get prediction results
    pred_df, threshold = get_prediction_result_of_each_day(ticker, period)
    
    # Get actual prices from yfinance for more accurate data
    actual_df = fetch_actual_prices(ticker, period)
    
    # Merge data
    actual_df['Date'] = pd.to_datetime(actual_df['Date'])
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]}, 
                                   sharex=True)
    
    # Plot 1: Actual prices and threshold bounds
    # ax1.plot(actual_df['Date'], actual_df['Close'], 
    #          label=f'{ticker} Actual Close Price', 
    #          color='blue', linewidth=2)

    ax1.plot(pred_df['date'], pred_df['actual_close'], 
             label=f'{ticker} Actual Close Price', 
             color='blue', linewidth=2)

    
    # Plot threshold bounds
    # ax1.plot(pred_df['date'], pred_df['lower_threshold_price'], 
    #          label='Prediction Threshold (Lower Bound)', 
    #          color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot the generated dates
    # ax1.plot(pred_df['date'], pred_df['lower_threshold_price'], 
    #          label='Prediction Threshold (Lower Bound at Generated Date)', 
    #          color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.plot(pred_df['date'], pred_df['upper_threshold_price'], 
             label='Prediction Threshold (Lower Bound at Generated Date)', 
             color='red', linestyle='--', linewidth=2, alpha=0.7)

    
    # Color regions based on predictions
    for idx, row in pred_df.iterrows():
        date = row['date']
        if pd.isnull(date):
            continue
        
        if not row['is_correct']:
            continue
        
        if row['prediction'] == 0:
            # Predicted to go below threshold - red region
            ax1.axvspan(date, date + pd.Timedelta(days=1), 
                        alpha=0.2, color='red', label='Predicted Below Threshold' if idx == 0 else "")
        else:
            # Predicted to stay above threshold - green region
            ax1.axvspan(date, date + pd.Timedelta(days=1), 
                        alpha=0.2, color='green', label='Predicted Above Threshold' if idx == 0 else "")
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{ticker} Stock Price Prediction Analysis\n(Label=0: Predicted below threshold, Label=1: Predicted above threshold)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction probabilities
    ax2.bar(pred_df['date'], pred_df['prediction_probability'], 
            width=1, alpha=0.7, color='purple', label='Prediction Probability')
    ax2.axhline(y=threshold, color='black', linestyle='-', 
                label=f'Optimal Threshold ({threshold:.3f})')
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Model Prediction Probabilities', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # In CLI environment, always save to default location if save_path not provided
        default_path = f"{ticker}_prediction_plot.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Print summary statistics
    print(f"\n=== {ticker} Prediction Summary ===")
    print(f"Total prediction days: {len(pred_df)}")
    print(f"Days predicted below threshold (label=1): {len(pred_df[pred_df['prediction'] == 1])}")
    print(f"Days predicted above threshold (label=0): {len(pred_df[pred_df['prediction'] == 0])}")
    print(f"Current actual price: ${actual_df['Close'].iloc[-1]:.2f}")
    print(f"Optimal model threshold: {threshold:.3f}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction Visualization')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='1y', 
                       choices=['1y', '6mo', '3mo', '1mo'],
                       help='Time period for analysis')
    parser.add_argument('--save', type=str, help='Path to save the plot')
    
    args = parser.parse_args()
    
    plot_stock_prediction(args.ticker, args.period, save_path=args.save)