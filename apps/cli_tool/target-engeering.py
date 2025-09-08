from packages.predict_lib.train_down_v3 import feature_engineering, label_panel_data
from packages.predict_lib.features import (
    fetch_panel_data
)
from config.my_paths import DATA_DIR
import pickle
import json
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from packages.predict_lib.utils import find_optimal_f1_threshold
from datetime import datetime, timedelta


def get_features_of_each_day(ticker, period="1y"):
    """
    Get prediction results for each day in the dataset.
    Returns predictions with actual prices and threshold bounds.
    """
    ticker = ticker.upper()
    
    processed_data = fetch_panel_data(period=period, end_date=None, tickers=[ticker])
    processed_data = feature_engineering(processed_data)
    processed_data = label_panel_data(processed_data, False)
    processed_data = processed_data[processed_data['ticker'] == ticker]
    
    # Ensure we have the required columns
    required_columns = ['date', 'ticker', 'Close', 'Future_Close', 'Price_Change_nd', 'Upper_Bound', 'Lower_bound']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in processed_data.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Add missing columns with NaN values
        for col in missing_columns:
            processed_data[col] = np.nan
    
    # Sort by date
    processed_data = processed_data.sort_values('date')
    
    # Return the relevant columns including bounds
    return processed_data[required_columns]


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


def plot_stock(ticker, period="1y", save_path=None, interactive=False):
    """
    Plot actual stock prices with prediction threshold bounds.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis ("1y", "6mo", "3mo", "1mo")
        save_path: Optional path to save the plot
        interactive: Whether to display the plot interactively instead of saving
    """
    ticker = ticker.upper()
    
    # Get prediction results with bounds
    pred_df = get_features_of_each_day(ticker, period)
    pred_df.to_csv('pred_df.csv')
    # Get actual prices
    
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    import matplotlib.pyplot as plt

    # Create the plot with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot daily prices as a continuous line on the first subplot
    ax1.plot(pred_df['date'], pred_df['Close'],
            label=f'{ticker} Daily Price',
            color='blue',
            linewidth=2)
    
    # Calculate future price bounds based on percentage changes
    # Upper bound price = Current Close * (1 + Upper_Bound)
    # Lower bound price = Current Close * (1 + Lower_bound)
    
    # Calculate the bounds based on the prediction made 5 days ago
    # For each day t, we want the prediction made at t-5 for day t
    pred_df['Next_5_Upper_Price'] = pred_df['Close'] * (1 + pred_df['Upper_Bound'])
    pred_df['Next_5_Lower_Price'] = pred_df['Close'] * (1 + pred_df['Lower_bound'])
    
    
    # Shift the predictions forward by 5 days to align with the actual date
    pred_df['Upper_Price'] = pred_df['Next_5_Upper_Price'].shift(5)
    pred_df['Lower_Price'] = pred_df['Next_5_Lower_Price'].shift(5)
    
    pred_df['Pred_diff'] = pred_df['Upper_Price'] - pred_df['Close'] 
    pred_df['Pred_diff_Percent'] = pred_df['Pred_diff'] / pred_df['Close'] * 100
    
    
    # Plot only lower bound as a continuous line
    # ax1.plot(pred_df['date'],
    #         pred_df['Lower_Price'],
    #         color='green',
    #         linewidth=2,
    #         alpha=0.8,
    #         label='Lower Bound (5-day prediction)',
    #         zorder=5)
    
    
    ax1.plot(pred_df['date'],
            pred_df['Upper_Price'],
            color='red',
            linewidth=2,
            alpha=0.8,
            label='Upper Bound (5-day ago prediction)',
            zorder=5)
    
    
    # Fill area between actual price and lower bound
    # ax1.fill_between(pred_df['date'],
    #                 pred_df['Lower_Price'],
    #                 pred_df['Close'],
    #                 color='lightgreen',
    #                 alpha=0.3,
    #                 label='Price-Lower Bound Range')
    

    # Fill area between actual price and upper bound
    ax1.fill_between(pred_df['date'],
                    pred_df['Upper_Price'],
                    pred_df['Close'],
                    color='lightcoral',
                    alpha=0.3,
                    label='Price-Upper Bound Range')


    # Formatting for first subplot
    ax1.set_title(f'{ticker} Stock Price with 5-Day Future Bounds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot the prediction difference as a bar chart on the second subplot
    colors = ['green' if x >= 0 else 'red' for x in pred_df['Pred_diff_Percent']]
    ax2.bar(pred_df['date'], pred_df['Pred_diff_Percent'], color=colors, alpha=0.7)
    ax2.set_title('Prediction Difference (Upper Bound - Close Price)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Difference (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis for both subplots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Handle display/save options
    if interactive:
        # Show interactive plot
        plt.show()
    else:
        # Force non-interactive backend for saving
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Save to file
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            # Save to default location
            default_path = f'data/{ticker}_target_engineering.png'
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {default_path}")
        
        # Close the plot to prevent display issues
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction Visualization')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='1y', 
                       choices=['1y', '6mo', '3mo', '1mo'],
                       help='Time period for analysis')
    parser.add_argument('--save', type=str, help='Path to save the plot')
    parser.add_argument('--interactive', action='store_true',
                       help='Display interactive plot instead of saving')
    
    args = parser.parse_args()
    
    plot_stock(args.ticker, args.period, save_path=args.save, interactive=args.interactive)