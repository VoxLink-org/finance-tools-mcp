import talib as ta
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
    average_precision_score, # For PR-AUC
)
from scipy.stats import ks_2samp # For KS value

def define_labels(data):
    """Defines binary target labels (0: others, 1: significant rise) based on 75th percentile dynamic threshold over 3 days)"""
    
    # Calculate future close price over n days
    data['Future_Close'] = data['Close'].shift(-5)
    # Calculate price change over n days
    data['Price_Change_nd'] = (data['Future_Close'] - data['Close']) / data['Close']

    # Calculate dynamic threshold based on 75th percentile of past price changes
    # Shift Price_Change_nd by n days to ensure the rolling window only uses past data relative to the labeling point
    window_size = 7 # Example window size for dynamic percentiles
    data['Upper_Bound'] = data['Price_Change_nd'].shift(5).rolling(window=window_size, closed='right').quantile(0.6)
    data['lower_bound'] = data['Price_Change_nd'].shift(5).rolling(window=window_size, closed='right').quantile(0.4)
    
    # Define labels based on dynamic threshold
    data['Label'] = 0  # Default to others (combines neutral and drop)
    # data.loc[data['Price_Change_nd'] >= data['Upper_Bound'], 'Label'] = 1  # Significant rise
    data.loc[data['Price_Change_nd'] <= data['lower_bound'], 'Label'] = 1  # Significant drop
    
    # Drop rows with NaN labels
    
    data.dropna(inplace=True)


    # Drop temporary columns
    data.drop(columns=['Future_Close', 'Price_Change_nd', 'Upper_Bound', 'lower_bound'], inplace=True)
    
    
    return data


def prepare_data_with_tssplit(data):
    """Prepare data for training with TimeSeriesSplit
    
    Returns:
        Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
            Yields tuples of (X_train, X_test, y_train, y_test) DataFrames
    """
    exclude_cols = [
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
    
    features = [
        col
        for col in data.columns
        if col not in exclude_cols
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

def train_model(X_train, y_train, params, early_stopping_rounds=100, valid_size=0.15):
    """Train XGBoost model with enhanced features for binary classification and early stopping.
    
    The training data (X_train, y_train) is already a time-ordered slice from prepare_data_with_tssplit.
    This function further splits it time-orderedly for early stopping.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (dict): XGBoost parameters.
        early_stopping_rounds (int): Number of rounds for early stopping.
        valid_size (float): Proportion of the training data to use as validation for early stopping.
                            The validation data will be the tail end of the training data.
    """
    # Calculate class weights for imbalanced classes
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]  # weight for positive class

    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        early_stopping_rounds=early_stopping_rounds
    )
    
    # Time-ordered split for early stopping within the training fold
    # No shuffling to prevent data leakage
    n_train = len(X_train)
    n_valid = int(n_train * valid_size)
    n_sub_train = n_train - n_valid

    if n_valid == 0:
        raise ValueError("Not enough data to create a validation set for early stopping.")
        
    X_sub_train = X_train.iloc[:n_sub_train]
    y_sub_train = y_train.iloc[:n_sub_train]
    
    X_sub_val = X_train.iloc[n_sub_train:]
    y_sub_val = y_train.iloc[n_sub_train:]

    print(f"  Sub-Training set size: {len(X_sub_train)}")
    print(f"  Sub-Validation set size: {len(X_sub_val)}")

    model.fit(
        X_sub_train,
        y_sub_train,
        eval_set=[(X_sub_val, y_sub_val)],
        verbose=False
    )
    return model

def predict_with_threshold(model, X_test, threshold=0.3):
    """Make predictions with adjustable threshold"""
    proba = model.predict_proba(X_test)[:, 1]  # 取上涨的概率
    return (proba >= threshold).astype(int), proba

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance for binary classification, including PR-AUC, Top-k Precision/Recall, and KS value."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class
    y_pred = (y_pred_proba >= 0.5).astype(int) # Default threshold for classification report

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # PR-AUC
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"PR-AUC: {pr_auc:.4f}")

    # Top-k Precision / Recall
    # Sort predictions by probability in descending order
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_test_sorted = y_test.iloc[sorted_indices].values
    y_pred_proba_sorted = y_pred_proba[sorted_indices]

    for k_percent in [0.05, 0.10, 0.20]:
        k = int(len(y_test) * k_percent)
        if k == 0: # Ensure k is at least 1 if data is very small
            k = 1 if len(y_test) > 0 else 0
        
        if k > 0:
            top_k_true = y_test_sorted[:k]
            top_k_pred_positive = (y_pred_proba_sorted[:k] >= 0.5).astype(int) # Assuming 0.5 as threshold for top-k
            
            # Precision for top-k: proportion of actual positives among predicted positives in top-k
            # If no positives predicted in top-k, precision is 0
            if np.sum(top_k_pred_positive) > 0:
                top_k_precision = np.sum(top_k_true[top_k_pred_positive == 1]) / np.sum(top_k_pred_positive)
            else:
                top_k_precision = 0.0
            
            # Recall for top-k: proportion of actual positives in top-k that are correctly identified
            # This is slightly different from standard recall, focusing on the top-k subset
            total_actual_positives = np.sum(y_test)
            if total_actual_positives > 0:
                top_k_recall = np.sum(top_k_true) / total_actual_positives
            else:
                top_k_recall = 0.0

            print(f"Top {int(k_percent*100)}% Precision: {top_k_precision:.4f}, Recall: {top_k_recall:.4f}")
        else:
            print(f"Top {int(k_percent*100)}% Precision: N/A, Recall: N/A (not enough data for k={k})")


    # KS Value
    # Separate probabilities for positive and negative classes
    positive_probas = y_pred_proba[y_test == 1]
    negative_probas = y_pred_proba[y_test == 0]

    if len(positive_probas) > 1 and len(negative_probas) > 1: # ks_2samp requires at least 2 samples
        ks_statistic, p_value = ks_2samp(positive_probas, negative_probas)
        print(f"KS Statistic: {ks_statistic:.4f} (p-value: {p_value:.4f})")
    else:
        print("KS Statistic: N/A (not enough samples in positive or negative class for KS test)")

    return y_pred_proba
