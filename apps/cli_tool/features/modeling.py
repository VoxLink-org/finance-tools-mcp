import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    fbeta_score,
)

def define_labels(data):
    """Defines binary target labels (0: others, 1: significant rise) based on 75th percentile dynamic threshold over 3 days)"""
    print("Defining labels...")
    
    # Calculate future close price over 5 days
    data['Future_Close'] = data['Close'].shift(-5)
    # Calculate price change over 5 days
    data['Price_Change_5d'] = (data['Future_Close'] - data['Close']) / data['Close']

    # Calculate dynamic threshold based on 75th percentile of past price changes
    # Shift Price_Change_5d by 5 days to ensure the rolling window only uses past data relative to the labeling point
    window_size = 60 # Example window size for dynamic percentiles
    data['Upper_Bound'] = data['Price_Change_5d'].shift(5).rolling(window=window_size, closed='right').quantile(0.6)

    # Define labels based on dynamic threshold
    data['Label'] = 0  # Default to others (combines neutral and drop)
    data.loc[data['Price_Change_5d'] >= data['Upper_Bound'], 'Label'] = 1  # Significant rise
    
    # Drop rows with NaN labels
    
    data.dropna(inplace=True)


    print(f'chg is {data["Price_Change_5d"].tail(20)}')
    print(f'label is {data["Label"].tail(20)}')

    
    
    # Drop temporary columns
    data.drop(columns=['Future_Close', 'Price_Change_5d', 'Upper_Bound'], inplace=True)
    
    
    print(f"Label distribution:\n{data['Label'].value_counts()}")
    return data

def prepare_data(data):
    """Prepare data for training with enhanced features"""
    features = [
        col
        for col in data.columns
        if col
        not in [
            "Close",
            "Label",
            "Future_Max_Close",
            "Open",
            "High",
            "Low",
            "Volume",
            "Dividends",
            "Stock Splits",
        ]
    ]
    X = data[features]
    y = data["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def prepare_data_with_tssplit(data):
    """Prepare data for training with TimeSeriesSplit
    
    Returns:
        Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
            Yields tuples of (X_train, X_test, y_train, y_test) DataFrames
    """
    features = [
        col
        for col in data.columns
        if col
        not in [
            "Close",
            "Label",
            "Future_Max_Close",
            "Open",
            "High",
            "Low",
            "Volume",
            "Dividends",
            "Stock Splits",
        ]
    ]
    print("Preparing data for TimeSeriesSplit...", features)
    X = data[features]
    y = data["Label"]
    
    tss = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f"Train/Test split sizes: {len(X_train)} / {len(X_test)}")
        print("Label distribution in train/test sets:")
        if hasattr(y_train, 'value_counts') and hasattr(y_test, 'value_counts'):
            # Ensure y_train and y_test are Series for value_counts
            print(y_train.value_counts(), y_test.value_counts())
        else:
            # If y_train and y_test are not Series, convert them to Series
            print(pd.Series(y_train).value_counts(), pd.Series(y_test).value_counts())
        yield X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train XGBoost model with enhanced features for binary classification"""
    # Calculate class weights for imbalanced classes
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]  # weight for positive class

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        max_depth=5,
        colsample_bytree=0.7,
        eta=0.08,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model

def predict_with_threshold(model, X_test, threshold=0.3):
    """Make predictions with adjustable threshold"""
    proba = model.predict_proba(X_test)[:, 1]  # 取上涨的概率
    return (proba >= threshold).astype(int), proba

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance for binary classification"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred_proba
