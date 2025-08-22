import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    fbeta_score,
)


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


def train_model(X_train, y_train):
    """Train XGBoost model with enhanced features"""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        max_depth=5,
        colsample_bytree=0.7,
        eta=0.08,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model performance with enhanced metrics"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred_proba
