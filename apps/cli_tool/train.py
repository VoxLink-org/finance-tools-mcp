import pickle
from typing import Literal
import pandas as pd
import numpy as np
import talib as ta
import xgboost as xgb
from config.paths import DATA_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from packages.investor_agent_lib.services import yfinance_service

"""
--- Evaluation Metrics ---

Classification Report:
              precision    recall  f1-score   support

          Up       0.68      0.83      0.74        81
       Other       0.90      0.80      0.85       162

    accuracy                           0.81       243
   macro avg       0.79      0.81      0.80       243
weighted avg       0.83      0.81      0.81       243


Confusion Matrix:
[[ 67  14]
 [ 32 130]]

Overall Accuracy: 0.8107

2. 模型的亮点 (The Good)
极高的整体准确率 (81%): 这是最直观的提升。在金融预测中，一个持续稳定在80%以上的二分类模型是非常罕见的，这表明您的特征和模型选择非常有效。
强大的风险规避能力: “Other”类别的精确率 (Precision) 高达 0.90。
实战意义: 当模型告诉您“不是上涨行情”（即预测为"Other"）时，它有90%的概率是正确的。这是一个极其宝贵的**“离场”或“不参与”**信号。一个能准确告诉你什么时候该把钱放在口袋里观望的模型，其价值不亚于一个能告诉你何时进攻的模型。
出色的机会捕捉能力: “Up”类别的召回率 (Recall) 达到 0.83。
实战意义: 在所有真实发生的“上涨”行情中，模型成功地为您识别出了其中的83%。这意味着您大概率不会错过市场主要的上涨机会。
3. 唯一的权衡与代价 (The Trade-off)
“Up”信号的精确率 (68%): 这是整个模型最“薄弱”但完全可以接受的一环。
解读: 当模型发出“上涨”信号时，其中有约 32% (100% - 68%) 的概率是“假警报”（即市场最终表现为"Other"）。
混淆矩阵分析: [32 130] 这一行显示，模型将32个真实的"Other"样本错误地判断为了"Up"。这就是那32%错误率的来源。
这是可以接受的吗？ 完全可以。在交易中，这是一个典型的**“用一定的错误率换取不踏空机会”**的策略。没有模型能做到100%精确。关键在于，您现在清楚地知道了这个模型的“性格”：它在寻找上涨机会时有点“乐观”，但它提供的“规避风险”信号却非常可靠。

"""

period = "5y"

def get_data(p: Literal[ "1mo","1y", "2y", "5y", "10y", "ytd"]=period):
    """
    Fetches SPY and VIX data and merges them.
    """
    print("Fetching data...")
    spy_data = yfinance_service.get_price_history('SPY', period=p, raw=True)
    vix_data = yfinance_service.get_price_history('^VIX', period=p, raw=True)


    spy_data.rename(columns={'Close': 'SPY_Close', 'Open': 'SPY_Open', 'High': 'SPY_High', 'Low': 'SPY_Low', 'Volume': 'SPY_Volume'}, inplace=True)
    vix_data.rename(columns={'Close': 'VIX_Close', 'Open': 'VIX_Open', 'High': 'VIX_High', 'Low': 'VIX_Low', 'Volume': 'VIX_Volume'}, inplace=True)

    # Normalize index to date part only
    spy_data.index = pd.to_datetime(spy_data.index).date
    vix_data.index = pd.to_datetime(vix_data.index).date

    data = pd.merge(spy_data, vix_data[['VIX_Close']], left_index=True, right_index=True, how='inner')

    print(spy_data.tail())
    print(vix_data.tail())
    print(data.tail())

    return data

def feature_engineering(data):
    """
    Engineers features for the model.
    """
    print("Engineering features...")
    # SPY Technical Indicators
    data['RSI'] = ta.RSI(data['SPY_Close'], timeperiod=14)
    data['MACD'], _, _ = ta.MACD(data['SPY_Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['ADX'] = ta.ADX(data['SPY_High'], data['SPY_Low'], data['SPY_Close'], timeperiod=14)
    data['CCI'] = ta.CCI(data['SPY_High'], data['SPY_Low'], data['SPY_Close'], timeperiod=14)
    data['upper_band'], data['middle_band'], data['lower_band'] = ta.BBANDS(data['SPY_Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['ATR'] = ta.ATR(data['SPY_High'], data['SPY_Low'], data['SPY_Close'], timeperiod=14)
    data['OBV'] = ta.OBV(data['SPY_Close'], data['SPY_Volume'])

    # SPY Lag Features
    data['return_1d'] = data['SPY_Close'].pct_change(1)
    data['return_3d'] = data['SPY_Close'].pct_change(3)
    data['return_5d'] = data['SPY_Close'].pct_change(5)

    # VIX Features
    data['vix_lag_1d'] = data['VIX_Close'].shift(1)
    data['vix_lag_3d'] = data['VIX_Close'].shift(3)
    data['vix_lag_5d'] = data['VIX_Close'].shift(5)
    
    return data.dropna()

def define_labels(data, up_threshold=0.02):
    """
    Defines the prediction target.
    """
    print("Defining labels...")
    future_close = data['SPY_Close'].shift(-10)
    data['future_return'] = (future_close - data['SPY_Close']) / data['SPY_Close']

    def categorize_return(r):
        if r > up_threshold:
            return 0  # Up
        # elif r < -0.02:
        #     return 0  # Down
        else:
            return 1  # Sideways

    data['label'] = data['future_return'].apply(categorize_return)
    return data.dropna()

def prepare_data(data):
    """
    Prepares data for training and applies undersampling to the training set.
    """
    print("Preparing data...")
    features = [
        'RSI', 'MACD', 'ADX', 'CCI', 'upper_band', 'middle_band', 'lower_band', 'ATR', 'OBV',
        'return_1d', 'return_3d', 'return_5d',
        'VIX_Close', 'vix_lag_1d', 'vix_lag_3d', 'vix_lag_5d'
    ]
    
    X = data[features]
    y = data['label']

    print("Original data shape:", X.shape)
    print("Original label distribution:\n", y.value_counts())

    # Split data first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply Random Under-sampling to the training set
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    
    print("\nResampled training data shape:", X_train_resampled.shape)
    print("Resampled training label distribution:\n", pd.Series(y_train_resampled).value_counts())
    print("\nTest data shape:", X_test.shape)
    print("Test label distribution:\n", y_test.value_counts())

    return X_train_resampled, X_test, y_train_resampled, y_test

def train_model(X_train, y_train):
    """
    Trains the XGBoost model.
    """
    print("Training model...")
    # Calculate sample weights to handle class imbalance
    # Sample weights are no longer needed due to undersampling.
    # sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=2,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model performance.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Evaluation Metrics ---")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Up', 'Other']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n--------------------------")

def main():
    """
    Main function to run the training pipeline.
    """
    data = get_data()
    data = feature_engineering(data)
    data = define_labels(data)
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    # save model in pickle
    with open(DATA_DIR / 'xgboost_spy_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    

if __name__ == "__main__":
    main()