import numpy as np

def add_custom_features(data):
    """Calculate enhanced custom features for trading signals"""
    # Enhanced percentage changes
    data['Daily_Log_Return'] = np.log(data['Close']/data['Close'].shift(1))
    data['Overnight_Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    
    # Improved interaction features
    data['BB_Strength'] = (data['Close'] - data['Lower_BB']) / (data['Upper_BB'] - data['Lower_BB'])
    data['MACD_Volume_Ratio'] = data['MACD_Hist'] / (data['Volume'] + 1e-6)  # Avoid division by zero
    
    # Momentum and mean reversion metrics
    data['RSI_Vol_Ratio'] = data['RSI'] / (data['Volatility_5D'] + 1e-6)
    data['Close_MA_Ratio_10D'] = data['Close'] / data['Close'].rolling(10).mean()
    
    # Volume-based features
    data['Volume_Spike'] = data['Volume'] / data['Volume'].rolling(20).mean()
    data['Price_Volume_Trend'] = data['Close'].pct_change() * data['Volume_Spike']
    
    # Clean any infinite values from divisions
    return data.replace([np.inf, -np.inf], np.nan) 