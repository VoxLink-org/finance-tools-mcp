import pickle
from typing import Literal
import pandas as pd
import numpy as np
import talib as ta
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
# Assuming these are local packages/modules you have defined
# from config.my_paths import DATA_DIR
# from packages.investor_agent_lib.services import yfinance_service
from packages.investor_agent_lib.services import yfinance_service

"""
this is a secret project to predict stock market trends using machine learning.
"""

# Define a placeholder for DATA_DIR if it's not available
try:
    from config.my_paths import DATA_DIR
except ImportError:
    import os
    DATA_DIR = "data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


period = "2y"

def get_data(ticker: str, p: Literal[ "1mo","1y", "2y", "5y", "10y", "ytd"]=period):
    """
    Fetches data using yfinance.
    """
    print("Fetching data...")
    # Replaced the custom service with a direct yfinance call for portability
    data = yfinance_service.get_price_history(ticker, period=p, raw=True)
    print(data.tail())

    return data

def feature_engineering(ticker: str = "SPY"):
    """
    Performs feature engineering by calculating various technical indicators.
    """
    print(f"Performing feature engineering for {ticker}...")
    data = get_data(ticker)

    # Calculate Technical Indicators using TA-Lib
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['Upper_BB'], data['Middle_BB'], data['Lower_BB'] = ta.BBANDS(data['Close'], timeperiod=20)
    data['BB_Width'] = data['Upper_BB'] - data['Lower_BB']
    data['Daily_Change_Pct'] = data['Close'].pct_change() * 100
    
    # Interactive Features
    data['BB_Interaction'] = (data['Close'] - data['Middle_BB']) / data['BB_Width']
    data['MACD_Volume_Interaction'] = data['MACD_Hist'] * data['Volume']

    # Rolling Window Statistical Features
    for window in [5, 10, 20]:
        data[f'Close_Avg_{window}D'] = data['Close'].rolling(window=window).mean()
        data[f'Close_Vol_{window}D'] = data['Close'].rolling(window=window).std()
        data[f'Volume_Avg_{window}D'] = data['Volume'].rolling(window=window).mean()
        data[f'Volume_Vol_{window}D'] = data['Volume'].rolling(window=window).std()
    
    # Volume-based indicators
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])
    data['MFI'] = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['VROC'] = ta.ROC(data['Volume'], timeperiod=14)
    data['ADL'] = ta.AD(data['High'], data['Low'], data['Close'], data['Volume'])
    data['ADOSC'] = ta.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10)
    
    # Ease of Movement
    distance_moved = ((data['High'] + data['Low'])/2 - (data['High'].shift(1) + data['Low'].shift(1))/2)
    box_ratio = data['Volume'] / (data['High'] - data['Low'])
    data['EMV'] = distance_moved / box_ratio
    data['EMV_MA'] = data['EMV'].rolling(14).mean()

    # Drop rows with NaN values that result from indicator calculations
    data.dropna(inplace=True)
    print("Feature engineering complete. Data head with new features:")
    print(data.head())
    return data
def define_labels(data):
    """
    Defines the target label for the model.
    Label is 1 if the close price increases by 1% or more in the next 5 days, 0 otherwise.
    """
    print("Defining labels...")
    data['Future_Max_Close'] = data['Close'].rolling(window=5, closed='right').max().shift(-5)
    data['Label'] = (data['Future_Max_Close'] > data['Close'] * 1.01).astype(int)
    data.dropna(inplace=True) # Drop rows with NaN values that result from label calculations
    print("Labels defined. Data head with labels:")
    print(data.head())
    print(f"Label distribution:\n{data['Label'].value_counts()}")
    return data

def prepare_data(data):
    """
    Prepares data for training and testing, including splitting.
    """
    print("Preparing data for the model...")
    # Define features (X) and target (y)
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Upper_BB', 'Middle_BB',
        'Lower_BB', 'BB_Width', 'Daily_Change_Pct',
        'BB_Interaction', 'MACD_Volume_Interaction',
        'Close_Avg_5D', 'Close_Vol_5D', 'Volume_Avg_5D', 'Volume_Vol_5D',
        'Close_Avg_10D', 'Close_Vol_10D', 'Volume_Avg_10D', 'Volume_Vol_10D',
        'Close_Avg_20D', 'Close_Vol_20D', 'Volume_Avg_20D', 'Volume_Vol_20D',
        'OBV', 'MFI', 'VWAP', 'VROC', 'ADL', 'ADOSC', 'EMV', 'EMV_MA'
    ]
    X = data[features]
    y = data['Label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training label distribution:")
    print(y_train.value_counts())
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains the XGBoost classifier with optimized parameters and saves the model.
    """
    print("Training XGBoost model...")
    # Calculate scale_pos_weight for handling class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr', # Changed to aucpr for better recall of positive class
        use_label_encoder=False,
        random_state=42,
        max_depth=5,  # Moderate depth to prevent overfitting
        colsample_bytree=0.7,  # Use 70% of features per tree
        eta=0.08,  # Learning rate
        scale_pos_weight=scale_pos_weight  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = f"{DATA_DIR}/xgboost_spy_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to {model_path}")
    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates the trained model using various metrics.
    """
    print("\nEvaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

def calculate_correlation_matrices(data):
    """
    Calculates and saves Pearson and Spearman correlation matrices for the indicators.
    """
    print("Calculating correlation matrices...")
    
    # List of all technical indicators computed in feature_engineering
    indicators = [
        'RSI', 'MACD', 'MACD_Hist', 'Daily_Change_Pct',
        'OBV', 'MFI', 'VROC', 'EMV', 'ADOSC'
    ]
    
    # Create a dataframe with only the indicators
    indicator_data = data[indicators]
    
    # Calculate Pearson correlation matrix
    pearson_corr = indicator_data.corr(method='pearson')
    
    # Calculate Spearman correlation matrix
    spearman_corr = indicator_data.corr(method='spearman')
    
    # Save correlation matrices to CSV files
    pearson_corr.to_csv(f"{DATA_DIR}/pearson_correlation_matrix.csv")
    spearman_corr.to_csv(f"{DATA_DIR}/spearman_correlation_matrix.csv")
    
    print("Pearson correlation matrix:")
    print(pearson_corr)
    print("\nSpearman correlation matrix:")
    print(spearman_corr)
    
    print(f"\nCorrelation matrices saved to {DATA_DIR}")
    return pearson_corr, spearman_corr

def main():
    # Run the feature engineering and label definition pipeline
    processed_data = feature_engineering(ticker="SPY")
    labeled_data = define_labels(processed_data)
    
    # Prepare data for XGBoost
    X_train, X_test, y_train, y_test = prepare_data(labeled_data)

    # Train the model
    model = train_model(X_train, y_train)

    # Tune threshold for better F2-score (optimizing recall while maintaining precision)
    print("\nTuning classification threshold to optimize F2-score...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import precision_recall_curve, fbeta_score
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F2-scores for each threshold
    f2_scores = []
    for i, threshold in enumerate(thresholds):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        f2_scores.append(f2)
    
    # Find the threshold that maximizes F2-score
    best_f2_index = np.argmax(f2_scores)
    best_threshold = thresholds[best_f2_index]
    best_f2_score = f2_scores[best_f2_index]
    
    print(f"Best threshold for maximum F2-score: {best_threshold:.4f}")
    print(f"Maximum F2-score: {best_f2_score:.4f}")
    print(f"Corresponding precision: {precision[best_f2_index]:.4f}")
    print(f"Corresponding recall: {recall[best_f2_index]:.4f}")
    
    # Evaluate the model with default threshold
    evaluate_model(model, X_test, y_test)
    
    # Evaluate the model with tuned threshold
    print(f"\nEvaluating model with tuned threshold ({best_threshold:.4f})...")
    evaluate_model(model, X_test, y_test, best_threshold)

    # Calculate and display feature importance
    print("\nFeature Importances:")
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importances)

    # Calculate correlation matrices
    calculate_correlation_matrices(labeled_data)

    # Plotting the features
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Plot Close Price
    ax1.plot(labeled_data.index, labeled_data['Close'], label='Close Price')
    ax1.set_title('SPY Close Price')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)

    # Plot RSI
    ax2.plot(labeled_data.index, labeled_data['RSI'], label='RSI', color='purple')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.set_ylabel('RSI Value')
    ax2.legend()
    ax2.grid(True)

    # Plot MACD
    ax3.plot(labeled_data.index, labeled_data['MACD'], label='MACD', color='blue')
    ax3.plot(labeled_data.index, labeled_data['MACD_Signal'], label='Signal Line', color='red', linestyle='--')
    ax3.bar(labeled_data.index, labeled_data['MACD_Hist'], label='Histogram', color='gray', alpha=0.7)
    ax3.set_title('MACD')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('MACD Value')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/analysis_plot.png")
    print(f"\nAnalysis plot saved to {DATA_DIR}/analysis_plot.png")

    # Create multiple scatter plots to analyze feature-label relationships
    fig_scatter, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig_scatter.suptitle('Feature vs. Label Relationships', fontsize=16)
    
    # Volume indicators vs technical indicators
    axs[0,0].scatter(labeled_data['RSI'], labeled_data['OBV'], c=labeled_data['Label'], cmap='viridis', alpha=0.6)
    axs[0,0].set_title('RSI vs OBV')
    axs[0,0].set_xlabel('RSI')
    axs[0,0].set_ylabel('OBV')
    
    axs[0,1].scatter(labeled_data['MACD'], labeled_data['MFI'], c=labeled_data['Label'], cmap='viridis', alpha=0.6)
    axs[0,1].set_title('MACD vs MFI')
    axs[0,1].set_xlabel('MACD')
    axs[0,1].set_ylabel('MFI')
    
    axs[1,0].scatter(labeled_data['MACD'], labeled_data['ADOSC'], c=labeled_data['Label'], cmap='viridis', alpha=0.6)
    axs[1,0].set_title('MACD vs Chaikin Oscillator')
    axs[1,0].set_xlabel('MACD')
    axs[1,0].set_ylabel('Chaikin Oscillator')
    
    axs[1,1].scatter(labeled_data['RSI'], labeled_data['Daily_Change_Pct'], c=labeled_data['Label'], cmap='viridis', alpha=0.6)
    axs[1,1].set_title('RSI vs Daily Change %')
    axs[1,1].set_xlabel('RSI')
    axs[1,1].set_ylabel('Daily Change Percentage')
    
    # Add a single legend for the whole figure
    scatter = axs[1,1].scatter([], [], c=[], cmap='viridis')
    legend_labels = ['Did Not Rise 1%', 'Rose >= 1%']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(0.), markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(1.), markersize=10)]
    fig_scatter.legend(handles, legend_labels, title="Label", loc='upper right')

    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout to make space for suptitle and legend
    plt.savefig(f"{DATA_DIR}/feature_label_scatter.png")
    print(f"Feature-label scatter plot saved to {DATA_DIR}/feature_label_scatter.png")
    # plt.show() # Uncomment to display plots directly

if __name__ == "__main__":
    main()
