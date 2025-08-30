import pandas as pd
import talib
import numpy as np


def add_pattern_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate common TA-Lib pattern recognition indicators"""
    # Convert to numpy arrays for TA-Lib
    open_prices = data['Open']
    high_prices = data['High']
    low_prices = data['Low']
    close_prices = data['Close']
    
    # Most common candlestick patterns
    data['CDLENGULFING'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
    data['CDLDOJI'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
    data['CDLHAMMER'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
    data['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
    data['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
    data['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
    data['CDLPIERCING'] = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
    data['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)
    
    def encode_all_patterns(row):
        patterns = [
            row['CDLENGULFING'], row['CDLDOJI'], row['CDLHAMMER'], row['CDLHANGINGMAN'],
            row['CDLMORNINGSTAR'], row['CDLEVENINGSTAR'], row['CDLPIERCING'], row['CDLDARKCLOUDCOVER']
        ]
        # bear as -1 and bull as 1, default as 0
        # Encode as 1 if any bullish pattern, -1 if any bearish pattern, else 0
        if any(p < 0 for p in patterns):
            return -1
        elif any(p > 0 for p in patterns):
            return 1
        else:
            return 0
    data['pattern'] = data.apply(encode_all_patterns, axis=1)
    
    # Drop original pattern columns to reduce dimensionality
    data.drop(columns=['CDLENGULFING', 'CDLDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
                       'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLPIERCING', 'CDLDARKCLOUDCOVER'], inplace=True)  
    return data