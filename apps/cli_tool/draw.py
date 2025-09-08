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

    original_close = processed_data[['Close', 'date']].copy()
    
    # must remove the nan
    processed_data.dropna(inplace=True)
    
    X_test = processed_data.drop(cols_to_drop, axis=1)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate optimal threshold
    y_true = processed_data['Label']
    optimal_threshold, _, report = find_optimal_f1_threshold(y_true, y_proba)

    print('Report:')
    print(report)
    
    # Get predictions for all days
    predictions = []
    processed_data = processed_data.reset_index(drop=True)
    for idx, row in processed_data.iterrows():
        date = row['date']
        actual_close = row['Close']
        lower_bound_value = row['Lower_bound']
        lower_threshold_price = actual_close * (1 + lower_bound_value)
        proba = y_proba[idx]
        prediction = 1 if proba > optimal_threshold else 0
        
        predictions.append({
            'date': date,
            'close': actual_close,
            'actual_close': actual_close,
            'lower_threshold_price': lower_threshold_price,
            'prediction_probability': proba,
            'prediction': prediction,
            'lower_bound_pct': lower_bound_value * 100
        })
    
    pred_df = pd.DataFrame(predictions)
    pred_df['five_days_later_close'] = pred_df['date'].map(original_close.set_index('date')['Close'].shift(-5))
    # Check if predictions are correct
    pred_df['is_correct'] = (
        ((pred_df['five_days_later_close'] < pred_df['lower_threshold_price']) & (pred_df['prediction'] == 1)) |
        ((pred_df['five_days_later_close'] >= pred_df['lower_threshold_price']) & (pred_df['prediction'] == 0))
    )
    
    pred_df.to_csv('predictions.csv', index=False)
    
    return pred_df, optimal_threshold


def fetch_actual_prices(ticker, period="1y"):
    """Fetch actual historical prices for visualization using existing data."""
    ticker = ticker.upper()
    
    data = fetch_panel_data(period=period, tickers=[ticker])
    ticker_data = data[data['ticker'] == ticker].copy()
    
    if 'date' not in ticker_data.columns:
        ticker_data['date'] = ticker_data.index
    
    ticker_data = ticker_data.sort_values('date')
    return ticker_data[['date', 'Close']].rename(columns={'date': 'Date'})


def plot_stock_prediction(ticker, period="1y", save_path=None):
    """
    Plot actual stock prices with lower threshold and prediction results.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis ("1y", "6mo", "3mo", "1mo")
        save_path: Optional path to save the plot
    """
    ticker = ticker.upper()
    
    # Get prediction results
    pred_df, threshold = get_prediction_result_of_each_day(ticker, period)
    
    # Get actual prices
    actual_df = fetch_actual_prices(ticker, period)
    
    # Merge data
    actual_df['Date'] = pd.to_datetime(actual_df['Date'])
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 16))
    
    # Main price chart
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    
    # Plot actual prices
    ax1.plot(actual_df['Date'], actual_df['Close'],
            label=f'{ticker} Actual Price',
            color='black', linewidth=1.5, alpha=0.8)
    
    # Plot lower threshold
    ax1.plot(pred_df['date'], pred_df['lower_threshold_price'],
            label='Lower Threshold',
            color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Color background for prediction results
    for idx, row in pred_df.iterrows():
        date = row['date']
        if pd.isnull(date):
            continue
            
        # Determine the color based on correctness
        if row['is_correct']:
            # Correct predictions - light background
            if row['prediction'] == 1:
                # Correct down prediction - light green
                ax1.axvspan(date, date + pd.Timedelta(days=1),
                           alpha=0.1, color='lightgreen', label='Correct Prediction' if idx == 0 else "")
            else:
                # Correct up prediction - light green
                ax1.axvspan(date, date + pd.Timedelta(days=1),
                           alpha=0.1, color='lightgreen')
        else:
            # Incorrect predictions - different shades of red
            if row['prediction'] == 1:
                # Incorrect down prediction (predicted down but was wrong) - light red
                ax1.axvspan(date, date + pd.Timedelta(days=1),
                           alpha=0.3, color='lightcoral', label='Incorrect Down Prediction' if idx == 0 else "")
            else:
                # Incorrect up prediction (predicted up but was wrong) - darker red
                ax1.axvspan(date, date + pd.Timedelta(days=1),
                           alpha=1, color='indianred', label='Incorrect Up Prediction' if idx == 0 else "")
    
    # Mark correct predictions with dots
    correct_predictions = pred_df[pred_df['is_correct'] == True]
    
    # Correct down predictions (red dots)
    down_preds = correct_predictions[correct_predictions['prediction'] == 1]
    if not down_preds.empty:
        ax1.scatter(down_preds['date'], down_preds['lower_threshold_price'],
                   color='red', s=50, zorder=5, label='Correct Down Prediction')
    
    # Correct up predictions (green dots)
    up_preds = correct_predictions[correct_predictions['prediction'] == 0]
    if not up_preds.empty:
        ax1.scatter(up_preds['date'], up_preds['lower_threshold_price'],
                   color='green', s=50, zorder=5, label='Correct Up Prediction')
    
    # Formatting
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{ticker} Stock Price with Lower Threshold and Prediction Results',
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate prediction errors for bar chart
    incorrect_predictions = pred_df[pred_df['is_correct'] == False].copy()
    error_data = []
    
    for idx, row in incorrect_predictions.iterrows():
        if pd.isnull(row['five_days_later_close']) or pd.isnull(row['lower_threshold_price']):
            continue
            
        # Calculate percentage difference based on prediction direction
        if row['prediction'] == 1:
            # Predicted down but was wrong - calculate how much it actually went up
            price_diff_pct = ((row['five_days_later_close'] - row['lower_threshold_price']) / row['actual_close']) * 100
            error_type = 'Down Pred Error'
            color = 'lightcoral'
            alpha = 0.3
        else:
            # Predicted up but was wrong - calculate how much it actually went down
            price_diff_pct = ((row['lower_threshold_price'] - row['five_days_later_close']) / row['actual_close']) * 100
            error_type = 'Up Pred Error'
            color = 'indianred'
            alpha = 1
        
        error_data.append({
            'date': row['date'],
            'error_type': error_type,
            'price_diff_pct': price_diff_pct,
            'color': color,
            'alpha': alpha
        })
    
    # Create bar chart for prediction errors
    ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
    
    if error_data:
        error_df = pd.DataFrame(error_data)
        
        # Create bars with different colors based on error type
        bars = []
        for idx, row in error_df.iterrows():
            bar = ax2.bar(row['date'], row['price_diff_pct'],
                         color=row['color'], alpha=row['alpha'], width=0.8)
            bars.append(bar)
        
        ax2.set_ylabel('Error Impact (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Prediction Error Analysis - Price Percentage Impact', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add legend for error types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', alpha=0.7, label='Incorrect Down Prediction'),
            Patch(facecolor='indianred', alpha=0.7, label='Incorrect Up Prediction')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No prediction errors found', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_ylabel('Error Impact (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        default_path = f"{ticker}_prediction_with_errors.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Print summary
    correct_count = len(pred_df[pred_df['is_correct'] == True])
    incorrect_count = len(pred_df[pred_df['is_correct'] == False])
    down_preds = len(pred_df[(pred_df['is_correct'] == True) & (pred_df['prediction'] == 1)])
    up_preds = len(pred_df[(pred_df['is_correct'] == True) & (pred_df['prediction'] == 0)])
    
    # Calculate error statistics
    incorrect_predictions = pred_df[pred_df['is_correct'] == False]
    error_data = []
    
    for idx, row in incorrect_predictions.iterrows():
        if pd.isnull(row['five_days_later_close']) or pd.isnull(row['lower_threshold_price']):
            continue
            
        if row['prediction'] == 1:
            # Predicted down but was wrong
            price_diff_pct = ((row['five_days_later_close'] - row['actual_close']) / row['actual_close']) * 100
        else:
            # Predicted up but was wrong
            price_diff_pct = ((row['actual_close'] - row['five_days_later_close']) / row['actual_close']) * 100
        
        error_data.append(price_diff_pct)
    
    error_data = [e for e in error_data if not pd.isnull(e)]
    
    print(f"\n=== {ticker} Prediction Summary ===")
    print(f"Total prediction days: {len(pred_df)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}")
    print(f"Correct down predictions: {down_preds}")
    print(f"Correct up predictions: {up_preds}")
    print(f"Accuracy: {correct_count/len(pred_df)*100:.1f}%")
    
    # Calculate lower bound statistics as percentage changes
    lower_bounds_pct = pred_df['lower_bound_pct'].dropna()
    if len(lower_bounds_pct) > 0:
        print(f"\n=== Lower Bound Analysis ===")
        print(f"Maximum lower bound: {np.max(lower_bounds_pct):.2f}%")
        print(f"Minimum lower bound: {np.min(lower_bounds_pct):.2f}%")
        print(f"Average lower bound: {np.mean(lower_bounds_pct):.2f}%")
        print(f"Standard deviation: {np.std(lower_bounds_pct):.2f}%")
    
    if error_data:
        print(f"\n=== Error Risk Analysis ===")
        print(f"Average error impact: {np.mean(error_data):.2f}%")
        print(f"Maximum error impact: {np.max(error_data):.2f}%")
        print(f"Minimum error impact: {np.min(error_data):.2f}%")
        print(f"Standard deviation: {np.std(error_data):.2f}%")
    
    print(f"\nCurrent actual price: ${actual_df['Close'].iloc[-1]:.2f}")
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