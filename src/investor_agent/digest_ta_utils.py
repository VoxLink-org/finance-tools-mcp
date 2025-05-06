import pandas as pd
import numpy as np
import talib as ta
from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)

def get_tendency_of_last_n_days(series: np.ndarray, n: int = 20) -> tuple[str, int, int]:
    """Determine the price tendency (trend) of the last 20 days.
    
    Args:
        series: numpy array of price data (typically closing prices)
        
    Returns:
        tuple[str, int, int]: A tuple containing:
            - slope (str): The slope of the linear regression line for the last 20 days.
            - up_count (int): The number of days the price moved upward in the last 20 days.
            - down_count (int): The number of days the price moved downward in the last 20 days.
    """
    if len(series) < n:
        # Return NaN for slope and 0 for counts if insufficient data
        return "insufficient data", 0, 0
        
    last_n = series[-n:]
    x = np.arange(len(last_n))
    slope, _ = np.polyfit(x, last_n, 1)
    
    # Normalize slope by average price
    avg_price = np.mean(last_n)
    normalized_slope = slope / avg_price
    
    # Calculate up and down counts
    price_changes = np.diff(last_n)
    up_count = np.sum(price_changes > 0)
    down_count = np.sum(price_changes < 0)
    
    # Convert normalized slope to trend description
    if normalized_slope > 0.005:
        trend = "strong upward trend"
    elif normalized_slope > 0.001:
        trend = "moderate upward trend"
    elif normalized_slope >= -0.001:
        trend = "flat trend"
    elif normalized_slope >= -0.005:
        trend = "moderate downward trend"
    else:
        trend = "strong downward trend"
    
    return trend, up_count, down_count

def tech_indicators(time_series_data: pd.DataFrame) -> str:
    """Extract and analyze technical indicators from time series data.
    
    Args:
        time_series_data: DataFrame containing OHLCV data with date as index
        
    Returns:
        str: Structured digest containing statistical analysis, technical indicators,
             risk metrics, and qualitative interpretations for LLM consumption.
    """
    # Technical indicators
    closes = time_series_data['Close'].values.astype(float)
    highs = time_series_data['High'].values.astype(float)
    lows = time_series_data['Low'].values.astype(float)
    volumes = time_series_data['Volume'].values.astype(float)
    
    # Get 20-day tendencies for each indicator
    def get_tendency(data):
        if len(data) < 20:
            return np.nan, np.nan, np.nan
        slope, up_count, down_count = get_tendency_of_last_n_days(data[-20:])
        return slope, up_count, down_count
    
    indicators = {
        'Trend': {
            'SMA 20': ta.SMA(closes, 20)[-1] if len(closes) >= 20 else np.nan,
            'SMA 50': ta.SMA(closes, 50)[-1] if len(closes) >= 50 else np.nan,
            'SMA 200': ta.SMA(closes, 200)[-1] if len(closes) >= 200 else np.nan,
            'EMA 20': ta.EMA(closes, 20)[-1] if len(closes) >= 20 else np.nan,
            'MACD': ta.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)[0][-1] if len(closes) >= 35 else np.nan,
            'ADX': ta.ADX(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 27 else np.nan
        },
        'Momentum': {
            'RSI 14': ta.RSI(closes, 14)[-1] if len(closes) >= 15 else np.nan,
            'Stoch %K': ta.STOCH(highs, lows, closes)[0][-1] if len(closes) >= 9 else np.nan,
            'Stoch %D': ta.STOCH(highs, lows, closes)[1][-1] if len(closes) >= 9 else np.nan,
            'CCI 20': ta.CCI(highs, lows, closes, 20)[-1] if len(closes) >= 20 else np.nan
        },
        'Volatility': {
            'ATR 14': ta.ATR(highs, lows, closes, 14)[-1] if len(closes) >= 14 else np.nan,
            'BB Width': ((ta.BBANDS(closes)[0][-1] - ta.BBANDS(closes)[2][-1]) / ta.BBANDS(closes)[1][-1]) if len(closes) >= 5 else np.nan,
            'Chaikin Vol': (ta.OBV(closes, volumes)[-1] / (ta.EMA(volumes, 10)[-1] + 1e-10)) if len(closes) >= 10 else np.nan
        },
        'Volume': {
            'OBV': ta.OBV(closes, volumes)[-1],
            'AD': ta.AD(highs, lows, closes, volumes)[-1],
            'CMF 20': ta.ADOSC(highs, lows, closes, volumes, fastperiod=3, slowperiod=10)[-1] if len(closes) >= 10 else np.nan
        }
    }
    
    # Get tendencies for each indicator
    tendencies = {
        'Trend': {
            'SMA 20': get_tendency(ta.SMA(closes, 20)) if len(closes) >= 20 else (np.nan, np.nan, np.nan),
            'EMA 20': get_tendency(ta.EMA(closes, 20)) if len(closes) >= 20 else (np.nan, np.nan, np.nan),
            'MACD': get_tendency(ta.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)[0]) if len(closes) >= 35 else (np.nan, np.nan, np.nan),
            'ADX': get_tendency(ta.ADX(highs, lows, closes, timeperiod=14)) if len(closes) >= 27 else (np.nan, np.nan, np.nan)
        },
        'Momentum': {
            'RSI 14': get_tendency(ta.RSI(closes, 14)) if len(closes) >= 15 else (np.nan, np.nan, np.nan),
            'Stoch %K': get_tendency(ta.STOCH(highs, lows, closes)[0]) if len(closes) >= 9 else (np.nan, np.nan, np.nan),
            'Stoch %D': get_tendency(ta.STOCH(highs, lows, closes)[1]) if len(closes) >= 9 else (np.nan, np.nan, np.nan),
            'CCI 20': get_tendency(ta.CCI(highs, lows, closes, 20)) if len(closes) >= 20 else (np.nan, np.nan, np.nan)
        },
        'Volatility': {
            'ATR 14': get_tendency(ta.ATR(highs, lows, closes, 14)) if len(closes) >= 14 else (np.nan, np.nan, np.nan),
            'BB Width': get_tendency((ta.BBANDS(closes)[0] - ta.BBANDS(closes)[2]) / ta.BBANDS(closes)[1]) if len(closes) >= 5 else (np.nan, np.nan, np.nan)
        },
        'Volume': {
            'OBV': get_tendency(ta.OBV(closes, volumes)),
            'AD': get_tendency(ta.AD(highs, lows, closes, volumes))
        }
    }

    # Trend analysis
    trend_strength = "Strong" if indicators['Trend']['ADX'] > 25 else "Weak" if indicators['Trend']['ADX'] < 20 else "Moderate"
    

    trend_data = [
        ["SMA 20", f"{indicators['Trend']['SMA 20']:.2f}", f"{tendencies['Trend']['SMA 20'][0]}", f"{tendencies['Trend']['SMA 20'][1]:.0f}", f"{tendencies['Trend']['SMA 20'][2]:.0f}"],
        ["SMA 50", f"{indicators['Trend']['SMA 50']:.2f}", "N/A", "N/A", "N/A"],
        ["SMA 200", f"{indicators['Trend']['SMA 200']:.2f}", "N/A", "N/A", "N/A"],
        ["EMA 20", f"{indicators['Trend']['EMA 20']:.2f}", f"{tendencies['Trend']['EMA 20'][0]}", f"{tendencies['Trend']['EMA 20'][1]:.0f}", f"{tendencies['Trend']['EMA 20'][2]:.0f}"],
        ["MACD", f"{indicators['Trend']['MACD']:.2f}", f"{tendencies['Trend']['MACD'][0]}", f"{tendencies['Trend']['MACD'][1]:.0f}", f"{tendencies['Trend']['MACD'][2]:.0f}"],
        ["ADX", f"{indicators['Trend']['ADX']:.2f} ({trend_strength} trend)", f"{tendencies['Trend']['ADX'][0]}", f"{tendencies['Trend']['ADX'][1]:.0f}", f"{tendencies['Trend']['ADX'][2]:.0f}"]
    ]

    momentum_data = [
        ["RSI 14", f"{indicators['Momentum']['RSI 14']:.2f} ({'Overbought' if indicators['Momentum']['RSI 14'] > 70 else 'Oversold' if indicators['Momentum']['RSI 14'] < 30 else 'Neutral'})", f"{tendencies['Momentum']['RSI 14'][0]}", f"{tendencies['Momentum']['RSI 14'][1]:.0f}", f"{tendencies['Momentum']['RSI 14'][2]:.0f}"],
        ["Stochastic %K", f"{indicators['Momentum']['Stoch %K']:.2f}", f"{tendencies['Momentum']['Stoch %K'][0]}", f"{tendencies['Momentum']['Stoch %K'][1]:.0f}", f"{tendencies['Momentum']['Stoch %K'][2]:.0f}"],
        ["Stochastic %D", f"{indicators['Momentum']['Stoch %D']:.2f}", f"{tendencies['Momentum']['Stoch %D'][0]}", f"{tendencies['Momentum']['Stoch %D'][1]:.0f}", f"{tendencies['Momentum']['Stoch %D'][2]:.0f}"],
        ["CCI 20", f"{indicators['Momentum']['CCI 20']:.2f}", f"{tendencies['Momentum']['CCI 20'][0]}", f"{tendencies['Momentum']['CCI 20'][1]:.0f}", f"{tendencies['Momentum']['CCI 20'][2]:.0f}"]
    ]

    volatility_data = [
        ["ATR 14", f"{indicators['Volatility']['ATR 14']:.2f}", f"{tendencies['Volatility']['ATR 14'][0]}", f"{tendencies['Volatility']['ATR 14'][1]:.0f}", f"{tendencies['Volatility']['ATR 14'][2]:.0f}"],
        ["BB Width", f"{indicators['Volatility']['BB Width']:.2%}", f"{tendencies['Volatility']['BB Width'][0]}", f"{tendencies['Volatility']['BB Width'][1]:.0f}", f"{tendencies['Volatility']['BB Width'][2]:.0f}"]
    ]

    volume_data = [
        ["OBV", f"{indicators['Volume']['OBV']:,.0f}", f"{tendencies['Volume']['OBV'][0]}", f"{tendencies['Volume']['OBV'][1]:.0f}", f"{tendencies['Volume']['OBV'][2]:.0f}"],
        ["AD", f"{indicators['Volume']['AD']:,.0f}", f"{tendencies['Volume']['AD'][0]}", f"{tendencies['Volume']['AD'][1]:.0f}", f"{tendencies['Volume']['AD'][2]:.0f}"]
    ]

    headers = ["Indicator", "Value", "Last 20 Days Slope", "Up Times", "Down Times"]

    digest = "A. Trend Indicators:\n"
    digest += tabulate(trend_data, headers=headers, tablefmt="simple")
    digest += "\n\nB. Momentum Indicators:\n"
    digest += tabulate(momentum_data, headers=headers, tablefmt="simple")
    digest += "\n\nC. Volatility:\n"
    digest += tabulate(volatility_data, headers=headers, tablefmt="simple")
    digest += "\n\nD. Volume Indicators:\n"
    digest += tabulate(volume_data, headers=headers, tablefmt="simple")

    return digest
