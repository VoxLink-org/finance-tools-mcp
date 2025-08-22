import numpy as np

def add_rolling_statistics(data):
    """Calculate focused volatility metrics for 5-20 day windows"""
    windows = [5, 10, 20]  # Practical short-term periods
    
    # Calculate efficient volatility metrics
    log_returns = np.log(data['Close']/data['Close'].shift(1))
    
    for window in windows:
        # Core volatility metrics
        data[f'Volatility_{window}D'] = log_returns.rolling(window).std()
        data[f'Realized_Vol_{window}D'] = log_returns.rolling(window).std() * np.sqrt(window)
        
        # Practical risk metrics
        rolling_high = data['High'].rolling(window).max()
        rolling_low = data['Low'].rolling(window).min()
        data[f'Range_Pct_{window}D'] = (rolling_high - rolling_low) / rolling_low
        
        # Volume adjusted metrics
        data[f'VWAP_{window}D'] = (data['Close'] * data['Volume']).rolling(window).sum() / \
                                 data['Volume'].rolling(window).sum()
    
    return data.dropna()