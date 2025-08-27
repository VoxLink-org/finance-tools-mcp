import talib as ta
import pandas as pd

def add_technical_indicators(data):
    """Calculate all TA-Lib technical indicators"""
    # Momentum Indicators
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = ta.MACD(
        data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Normalized MACD
    # data['MACD_Norm'] = (data['MACD'] - data['MACD'].rolling(window=20).mean()) / data['MACD'].rolling(window=20).std()
    data['MACD_Hist_Norm'] = (data['MACD_Hist'] - data['MACD_Hist'].rolling(window=20).mean()) / data['MACD_Hist'].rolling(window=20).std()
    
    data['EMA_12'] = ta.EMA(data['Close'], timeperiod=12)
    data['EMA_26'] = ta.EMA(data['Close'], timeperiod=26)
    data['EMA_12_26_Ratio'] = data['EMA_12'] / data['EMA_26']
    data['Stoch_K'], data['Stoch_D'] = ta.STOCH(
        data['High'], data['Low'], data['Close'], 
        fastk_period=14, slowk_period=3, slowd_period=3)
    
    # Volatility Indicators
    data['Upper_BB'], data['Middle_BB'], data['Lower_BB'] = ta.BBANDS(
        data['Close'], timeperiod=20)
    data['BB_Width'] = data['Upper_BB'] - data['Lower_BB']
    data['BB_Width_Pct'] = data['BB_Width'] / data['Middle_BB']
    data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Volume Indicators
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])
    # normalized OBV
    data['OBV_Norm'] = (data['OBV'] - data['OBV'].rolling(window=20).mean()) / data['OBV'].rolling(window=20).std()
        
    data['MFI'] = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
    data['VROC'] = ta.ROC(data['Volume'], timeperiod=14)
    data['ADL'] = ta.AD(data['High'], data['Low'], data['Close'], data['Volume'])
    # normalized ADL
    data['ADL_Norm'] = (data['ADL'] - data['ADL'].rolling(window=20).mean()) / data['ADL'].rolling(window=20).std()
    
    data['ADOSC'] = ta.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], 
                            fastperiod=3, slowperiod=10)
    
    
    
    # Ease of Movement
    distance_moved = ((data['High'] + data['Low'])/2 - 
                     (data['High'].shift(1) + data['Low'].shift(1))/2)
    box_ratio = data['Volume'] / (data['High'] - data['Low'])
    data['EMV'] = distance_moved / box_ratio
    data['EMV_MA'] = data['EMV'].rolling(14).mean()
    
    return data