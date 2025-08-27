import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from apps.cli_tool.features import (
    add_technical_indicators,
    add_rolling_statistics,
    add_custom_features,
    add_pattern_features,
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
        ticker_data = add_rolling_statistics(ticker_data)
        
        # Add custom features
        # ticker_data = add_custom_features(ticker_data)
        
        # Add pattern features
        # ticker_data = add_pattern_features(ticker_data)
        
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
    
    # Get the label distribution
    print("Label distribution after labeling:")
    print(panel_data['Label'].value_counts())
    
    return panel_data

def clean_panel_dataframe(panel_data: pd.DataFrame):
    # Drop cols with Close,High,Low,Open,Volume
    cols_to_drop = ['Close', 'High', 'Low', 'Open', 'Volume',
                    # 'MACD', 'MACD_Signal', 'MACD_Hist', 
                    # 'EMA_12', 'EMA_26', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'BB_Width',
                    # 'OBV', 'ADL', 'ADOSC', 'EMV', 'EMV_MA',
                    # 'MFI', 'RSI',
                    # 'VWAP_5D', 'VWAP_10D', 'VWAP_20D',
                    # 'Price_Volume_Trend', 
                    'Stoch_D','Stoch_K'
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

def split_data_by_date(panel_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Automatically split the panel data into training, validation, and test sets by date."""
    # Determine unique sorted dates
    unique_dates = panel_data['date'].sort_values().unique()
    
    # Calculate split indices
    train_end_idx = int(len(unique_dates) * 0.7)
    val_end_idx = int(len(unique_dates) * 0.85)
    
    # Determine split dates
    train_end_date = unique_dates[train_end_idx - 1]
    val_end_date = unique_dates[val_end_idx - 1]
    
    print(f"Training data up to: {train_end_date}")
    print(f"Validation data up to: {val_end_date}")
    print(f"Test data from: {val_end_date + pd.Timedelta(days=1)} onwards")
    
    # Split the data
    train_data = panel_data[panel_data['date'] <= train_end_date]
    val_data = panel_data[(panel_data['date'] > train_end_date) & (panel_data['date'] <= val_end_date)]
    test_data = panel_data[panel_data['date'] > val_end_date]
    
    # Sort by date and ticker
    train_data = train_data.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    val_data = val_data.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    test_data = test_data.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    
    return train_data, val_data, test_data

def main(period="1y"):
    processed_data = fetch_panel_data(period=period, end_date=pd.Timestamp("2025-05-10"))
    print("Fetched panel data. Sample data:")
    print(processed_data['date'].head())
    processed_data = feature_engineering(processed_data)
    print(processed_data['date'].head())
    processed_data = label_panel_data(processed_data)
    print(processed_data['date'].head())

    # sort by date and ticker
    processed_data = clean_panel_dataframe(processed_data)
    print("Feature engineering and labeling complete. Sample data:")
    
    train_data, val_data, test_data = split_data_by_date(processed_data)
    # save top 500 to csv for quick check
    test_data.to_csv("feature_engineered_sample.csv", index=False)
    print('Train data label distribution:')
    print(train_data['Label'].value_counts())
    print('Test data label distribution:')
    print(test_data['Label'].value_counts())
    # use multi classification to get importance of features by xgboost
    
    
    # use multi classification to get importance of features by xgboost
    feature_importances = get_feature_importance_by_xgboost(train_data, val_data, test_data)
    print("Feature Importances:")
    print(feature_importances.head())
    feature_importances.to_csv("xgboost_feature_importance.csv", index=False)
    
    return processed_data

def get_feature_importance_by_xgboost(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Trains an XGBoost classifier and extracts feature importance.
    """
    # Assuming 'Label' is the target column and 'date', 'ticker' are not features
    target_column = 'Label'
    
    # Identify feature columns
    feature_columns = [col for col in train_data.columns if col not in [target_column, 'date', 'ticker']]

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_val = val_data[feature_columns]
    y_val = val_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    # Initialize XGBoost Classifier for multi-class classification
    # Determine the number of unique classes in the training data
    num_classes = y_train.nunique()

    # Calculate scale_pos_weight for imbalanced datasets
    # This is the ratio of the number of negative class examples to the number of positive class examples.
    # It helps to handle imbalanced datasets by giving more weight to the minority class.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1

    print('scale_pos_weight_value:', scale_pos_weight_value)

    model = xgb.XGBClassifier(
        objective='binary:logistic',  # For binary classification
        eval_metric='logloss',        # Evaluation metric for binary classification
        use_label_encoder=False,      # Suppress the warning
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        early_stopping_rounds=10,
        scale_pos_weight=scale_pos_weight_value
    )

    # Train the model
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}")
    print("Classification Report on test set:")
    print(classification_report(y_test, y_pred))

    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    })
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    return feature_importance_df

if __name__ == "__main__":
    import sys
    period = sys.argv[1] if len(sys.argv) > 1 else "1y"
    main(period=period)