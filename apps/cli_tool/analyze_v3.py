import pandas as pd
import numpy as np
np.random.seed(42)

from apps.cli_tool.features import (
    add_technical_indicators,
    add_rolling_statistics,
    add_custom_features,
    fetch_panel_data
)

from apps.cli_tool.features.modeling import define_labels

def feature_engineering(panel_data: pd.DataFrame):
    """Enhanced feature engineering using modular components"""
    # Process each ticker individually
    tickers = panel_data['ticker'].unique()
    processed_data = []

    for ticker in tickers:
        ticker_data = panel_data[panel_data['ticker'] == ticker].copy()
        # Add technical indicators
        ticker_data = add_technical_indicators(ticker_data)
        
        # Add rolling statistics
        # ticker_data = add_rolling_statistics(ticker_data)
        
        # Add custom features
        # ticker_data = add_custom_features(ticker_data)
        
        processed_data.append(ticker_data)
    
    # Combine all processed data
    panel_data = pd.concat(processed_data, ignore_index=True)
        
    return panel_data

def label_panel_data(panel_data: pd.DataFrame):
    """Label the panel data for modeling"""
    # Process each ticker individually
    tickers = panel_data['ticker'].unique()
    labeled_data = []

    for ticker in tickers:
        ticker_data = panel_data[panel_data['ticker'] == ticker].copy()
        # Define labels for the ticker data
        ticker_data = define_labels(ticker_data)
        labeled_data.append(ticker_data)
    
    # Combine all labeled data
    panel_data = pd.concat(labeled_data, ignore_index=True)
    
    return panel_data

def clean_panel_dataframe(panel_data: pd.DataFrame):
    # Drop cols with Close,High,Low,Open,Volume
    cols_to_drop = ['Close', 'High', 'Low', 'Open', 'Volume',
                    'EMA_12', 'EMA_26', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'BB_Width',
                    'OBV', 'ADL', 'ADOSC', 'EMV', 'EMV_MA',
                    'VWAP_5D', 'VWAP_10D', 'VWAP_20D',
                    'Price_Volume_Trend'
                    ]
    panel_data = panel_data.drop(columns=cols_to_drop, errors='ignore')
    # Drop rows with any NaN values
    panel_data = panel_data.dropna()
    return panel_data

def split_data_by_stock(panel_data: pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the panel data into training, validation, and test sets by stock tickers"""
    tickers = panel_data['ticker'].unique()
    # shuffle tickers
    np.random.shuffle(tickers)
    train_size = int(len(tickers) * 0.7)
    val_size = int(len(tickers) * 0.15)
    train_tickers = tickers[:train_size]
    val_tickers = tickers[train_size:train_size + val_size]
    test_tickers = tickers[train_size + val_size:]
    train_data = panel_data[panel_data['ticker'].isin(train_tickers)]
    val_data = panel_data[panel_data['ticker'].isin(val_tickers)]
    test_data = panel_data[panel_data['ticker'].isin(test_tickers)]
    
    # sort by date and ticker
    train_data = train_data.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    val_data = val_data.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    test_data = test_data.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    
    
    return train_data, val_data, test_data

def main(period="1y"):
    processed_data = fetch_panel_data(period=period)
    print("Fetched panel data. Sample data:")
    print(processed_data['date'].head())
    processed_data = feature_engineering(processed_data)
    print(processed_data['date'].head())
    processed_data = label_panel_data(processed_data)
    print(processed_data['date'].head())

    # sort by date and ticker
    processed_data = clean_panel_dataframe(processed_data)
    print("Feature engineering and labeling complete. Sample data:")
    
    train_data, val_data, test_data = split_data_by_stock(processed_data)
    # save top 500 to csv for quick check
    test_data.to_csv("feature_engineered_sample.csv", index=False)

    # importance of features    
    
    
    return processed_data

if __name__ == "__main__":
    import sys
    period = sys.argv[1] if len(sys.argv) > 1 else "1y"
    main(period=period)