import talib
import numpy as np

def add_pattern_features(data):
    """Calculate common TA-Lib pattern recognition indicators"""
    
    # Convert to numpy arrays for TA-Lib
    open_prices = data['Open'].values
    high_prices = data['High'].values
    low_prices = data['Low'].values
    close_prices = data['Close'].values
    
    # Most common candlestick patterns
    data['CDLENGULFING'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
    data['CDLDOJI'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
    data['CDLHAMMER'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
    data['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
    data['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
    data['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
    data['CDLPIERCING'] = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
    data['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)
    
    
    # It can be like the days since last pattern appeared
    def days_since_last_pattern(series):
        days = np.full(series.shape, np.nan)
        last_index = -1
        for i in range(len(series)):
            if series[i] != 0:
                days[i] = 0
                last_index = i
            elif last_index != -1:
                days[i] = i - last_index
            days[i] = min(days[i], 7)  # Cap at 7 days to avoid large numbers
        return days
    
    data['Days_Since_Engulfing'] = days_since_last_pattern(data['CDLENGULFING'].values)
    data['Days_Since_Doji'] = days_since_last_pattern(data['CDLDOJI'].values)
    data['Days_Since_Hammer'] = days_since_last_pattern(data['CDLHAMMER'].values)
    data['Days_Since_HangingMan'] = days_since_last_pattern(data['CDLHANGINGMAN'].values)
    data['Days_Since_MorningStar'] = days_since_last_pattern(data['CDLMORNINGSTAR'].values)
    data['Days_Since_EveningStar'] = days_since_last_pattern(data['CDLEVENINGSTAR'].values)
    data['Days_Since_Piercing'] = days_since_last_pattern(data['CDLPIERCING'].values)
    data['Days_Since_DarkCloudCover'] = days_since_last_pattern(data['CDLDARKCLOUDCOVER'].values)
    
    # Drop original pattern columns to reduce dimensionality
    data.drop(columns=['CDLENGULFING', 'CDLDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
                       'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLPIERCING', 'CDLDARKCLOUDCOVER'], inplace=True)  
    return data