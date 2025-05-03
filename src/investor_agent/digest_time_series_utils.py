import pandas as pd
import numpy as np
import talib as ta
from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)

def generate_time_series_digest_for_LLM(time_series_data: pd.DataFrame) -> str:
    """Generate a comprehensive quantitative digest for time series data.
    
    Args:
        time_series_data: DataFrame containing OHLCV data with date as index
        
    Returns:
        str: Structured digest containing statistical analysis, technical indicators,
             risk metrics, and qualitative interpretations for LLM consumption.
    """
    if time_series_data.empty:
        return "No time series data available."
    
    if time_series_data.shape[0] < 20:
        logger.warning("Not enough rows in time series data.") 
        return tabulate(time_series_data, headers='keys', tablefmt="simple")

    # Data preparation
    if 'date' in time_series_data.columns:
        time_series_data['date'] = pd.to_datetime(time_series_data['date'])
        time_series_data = time_series_data.set_index('date').sort_index()
    
    
    
    # Basic statistics
    stats = {
        'Period': f"{time_series_data.index.min().strftime('%Y-%m-%d')} to {time_series_data.index.max().strftime('%Y-%m-%d')}",
        'Trading Days': len(time_series_data),
        'Close Price': {
            'Min': np.min(time_series_data['Close']),
            'Max': np.max(time_series_data['Close']),
            'Mean': np.mean(time_series_data['Close']),
            'Last': time_series_data['Close'].iloc[-1]
        },
        'Volume': {
            'Total': np.sum(time_series_data['Volume']),
            'Avg': np.mean(time_series_data['Volume']),
            'Max': np.max(time_series_data['Volume'])
        }
    }
    
    # Risk-adjusted return metrics
    
    risk_metrics, sharpe_ratio, volatility = cal_risk(time_series_data)

    # Technical indicators
    closes = time_series_data['Close'].values.astype(float)
    highs = time_series_data['High'].values.astype(float)
    lows = time_series_data['Low'].values.astype(float)
    volumes = time_series_data['Volume'].values.astype(float)
    
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
    

    # Trend analysis
    trend_strength = "Strong" if indicators['Trend']['ADX'] > 25 else "Weak" if indicators['Trend']['ADX'] < 20 else "Moderate"
    trend_direction = "Up" if closes[-1] > ta.EMA(closes, 20)[-1] else "Down"
    

    # Latest 20 days sample
    latest_20 = time_series_data[-20:]
    latest_20['Date'] = latest_20.index.strftime('%Y-%m-%d')
    latest_20 = latest_20[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Pattern recognition
    pattern = pattern_recognition(time_series_data)

    # Generate structured digest
    digest = f"""=== QUANTITATIVE TIME SERIES DIGEST ===
    
1. OVERVIEW
{tabulate([[stats['Period'], stats['Trading Days']]], headers=['Period', 'Trading Days'], tablefmt='grid')}

2. PRICE STATISTICS
{tabulate([[stats['Close Price']['Min'], stats['Close Price']['Max'], stats['Close Price']['Mean'], stats['Close Price']['Last']]], 
          headers=['Min', 'Max', 'Mean', 'Last'], tablefmt='grid', floatfmt=".2f")}

3. VOLUME ANALYSIS
{tabulate([[stats['Volume']['Total'], stats['Volume']['Avg'], stats['Volume']['Max']]], 
          headers=['Total Volume', 'Avg Volume', 'Max Volume'], tablefmt='grid')}

4. RISK METRICS
{tabulate([[risk_metrics['Annualized Return'], risk_metrics['Annualized Volatility'], 
            risk_metrics['Sharpe Ratio'], risk_metrics['Max Drawdown'], risk_metrics['Sortino Ratio']]], 
          headers=['Return', 'Volatility', 'Sharpe', 'Max DD', 'Sortino'], tablefmt='grid')}

5. TECHNICAL INDICATORS
   A. Trend Indicators:
   - 20/50/200 SMA: {indicators['Trend']['SMA 20']:.2f}/{indicators['Trend']['SMA 50']:.2f}/{indicators['Trend']['SMA 200']:.2f}
   - EMA 20: {indicators['Trend']['EMA 20']:.2f}
   - MACD: {indicators['Trend']['MACD']:.2f}
   - ADX: {indicators['Trend']['ADX']:.2f} ({trend_strength} trend)
   
   B. Momentum Indicators:
   - RSI 14: {indicators['Momentum']['RSI 14']:.2f} ({'Overbought' if indicators['Momentum']['RSI 14'] > 70 else 'Oversold' if indicators['Momentum']['RSI 14'] < 30 else 'Neutral'})
   - Stochastic: K={indicators['Momentum']['Stoch %K']:.2f}, D={indicators['Momentum']['Stoch %D']:.2f}
   - CCI 20: {indicators['Momentum']['CCI 20']:.2f}
   
   C. Volatility:
   - ATR 14: {indicators['Volatility']['ATR 14']:.2f}
   - BB Width: {indicators['Volatility']['BB Width']:.2%}
   
   D. Volume Indicators:
   - OBV: {indicators['Volume']['OBV']:,.0f}
   - AD: {indicators['Volume']['AD']:,.0f}

6. QUALITATIVE ASSESSMENT
- The current trend is {trend_direction} with {trend_strength.lower()} strength.
- Volatility is {'high' if volatility > 0.2 else 'moderate' if volatility > 0.1 else 'low'} compared to historical averages.
- Volume activity is {'increasing' if indicators['Volume']['OBV'] > ta.EMA(np.array(list(indicators['Volume'].values())[:3]), 10)[-1] 
                    else 'decreasing' if indicators['Volume']['OBV'] < ta.EMA(np.array(list(indicators['Volume'].values())[:3]), 10)[-1] 
                    else 'stable'}.
- Risk-adjusted returns are {'attractive' if sharpe_ratio > 1 else 'moderate' if sharpe_ratio > 0.5 else 'poor'}.

7. PATTERN RECOGNITION
{pattern}

8. LATEST 20 DAYS OHLCV 

{tabulate(latest_20.values.tolist(), headers=latest_20.columns, tablefmt='simple', floatfmt=".2f")}

=== END OF DIGEST ===
"""
    return digest


def cal_risk(time_series_data: pd.DataFrame) -> dict:
    # Calculate daily returns and risk-free rate proxy (2% for simplicity)
    time_series_data['daily_return'] = time_series_data['Close'].pct_change()
    risk_free_rate = 0.02

    # Risk-adjusted return metrics
    annualized_return = np.mean(time_series_data['daily_return'].dropna()) * 252
    volatility = np.std(time_series_data['daily_return'].dropna()) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    risk_metrics = {
        'Annualized Return': f"{annualized_return*100:.2f}%",
        'Annualized Volatility': f"{volatility*100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{((time_series_data['Close'].pct_change().cumsum() - time_series_data['Close'].pct_change().cumsum().cummax()).min())*100:.2f}%",
        'Sortino Ratio': f"{(annualized_return - risk_free_rate) / np.std(time_series_data[time_series_data['daily_return'] < 0]['daily_return'].dropna())*np.sqrt(252):.2f}" 
            if len(time_series_data[time_series_data['daily_return'] < 0]) > 0 else 'N/A'
    }

    return risk_metrics, sharpe_ratio, volatility

def pattern_recognition(time_series_data: pd.DataFrame) -> str:
    """Recognize common chart patterns in time series data.

    Args:
        time_series_data: DataFrame containing OHLCV data with date as index

    Returns:
        str: A string summarizing recognized patterns with dates.
    """
    if time_series_data.empty:
        return "No time series data available for pattern recognition."

    # Ensure data is sorted by date
    if 'date' in time_series_data.columns:
        time_series_data['date'] = pd.to_datetime(time_series_data['date'])
        time_series_data = time_series_data.set_index('date').sort_index()

    opens = time_series_data['Open'].values.astype(float)
    highs = time_series_data['High'].values.astype(float)
    lows = time_series_data['Low'].values.astype(float)
    closes = time_series_data['Close'].values.astype(float)
    dates = time_series_data.index

    patterns = {
        "Hammer": ta.CDLHAMMER(opens, highs, lows, closes),
        "Inverted Hammer": ta.CDLINVERTEDHAMMER(opens, highs, lows, closes),
        "Engulfing Pattern": ta.CDLENGULFING(opens, highs, lows, closes),
        "Doji": ta.CDLDOJI(opens, highs, lows, closes),
        "Shooting Star": ta.CDLSHOOTINGSTAR(opens, highs, lows, closes),
        "Morning Star": ta.CDLMORNINGSTAR(opens, highs, lows, closes),
        "Evening Star": ta.CDLEVENINGSTAR(opens, highs, lows, closes),
        "Three White Soldiers": ta.CDL3WHITESOLDIERS(opens, highs, lows, closes),
        "Three Black Crows": ta.CDL3BLACKCROWS(opens, highs, lows, closes),
    }

    pattern_occurrences = {name: [] for name in patterns.keys()}

    # Track all occurrences of each pattern
    for i, date in enumerate(dates):
        for name, pattern_data in patterns.items():
            if pattern_data is not None and len(pattern_data) > i and pattern_data[i] != 0:
                pattern_occurrences[name].append(date.strftime('%Y-%m-%d'))

    # Generate detailed pattern report
    detected_patterns = []
    for name, dates in pattern_occurrences.items():
        if dates:
            if len(dates) == 1:
                detected_patterns.append(f"- {name}: Detected on {dates[0]}")
            else:
                detected_patterns.append(f"- {name}: Detected {len(dates)} times (Recent: {dates[-1]})")

    if not detected_patterns:
        return "No significant chart patterns detected in the given period."
    else:
        return "\n" + \
               "\n".join(detected_patterns) + \
               "\n"
