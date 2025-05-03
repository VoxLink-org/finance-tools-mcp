import pandas as pd
import numpy as np
import talib as ta
from tabulate import tabulate

def generate_time_series_digest_for_LLM(time_series_data: pd.DataFrame) -> str:
    """Generate a comprehensive quantitative digest for time series data.
    
    Args:
        time_series_data: DataFrame containing OHLCV data with 'date' column
        
    Returns:
        str: Structured digest containing statistical analysis, technical indicators,
             risk metrics, and qualitative interpretations for LLM consumption.
    """
    if time_series_data.empty:
        return "No time series data available."
    
    # Data preparation
    if 'date' in time_series_data.columns:
        time_series_data['date'] = pd.to_datetime(time_series_data['date'])
        time_series_data = time_series_data.set_index('date').sort_index()
    
    # Calculate daily returns and risk-free rate proxy (0% for simplicity)
    time_series_data['daily_return'] = time_series_data['close'].pct_change()
    risk_free_rate = 0.0
    
    # Basic statistics
    stats = {
        'Period': f"{time_series_data.index.min().strftime('%Y-%m-%d')} to {time_series_data.index.max().strftime('%Y-%m-%d')}",
        'Trading Days': len(time_series_data),
        'Close Price': {
            'Min': np.min(time_series_data['close']),
            'Max': np.max(time_series_data['close']),
            'Mean': np.mean(time_series_data['close']),
            'Last': time_series_data['close'].iloc[-1]
        },
        'Volume': {
            'Total': np.sum(time_series_data['volume']),
            'Avg': np.mean(time_series_data['volume']),
            'Max': np.max(time_series_data['volume'])
        }
    }
    
    # Risk-adjusted return metrics
    annualized_return = np.mean(time_series_data['daily_return'].dropna()) * 252
    volatility = np.std(time_series_data['daily_return'].dropna()) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    risk_metrics = {
        'Annualized Return': f"{annualized_return*100:.2f}%",
        'Annualized Volatility': f"{volatility*100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{np.min(time_series_data['close'].pct_change().cumsum())*100:.2f}%",
        'Sortino Ratio': f"{(annualized_return - risk_free_rate) / np.std(time_series_data[time_series_data['daily_return'] < 0]['daily_return'].dropna())*np.sqrt(252):.2f}" 
            if len(time_series_data[time_series_data['daily_return'] < 0]) > 0 else 'N/A'
    }
    
    # Technical indicators
    closes = time_series_data['close'].values
    highs = time_series_data['high'].values
    lows = time_series_data['low'].values
    volumes = time_series_data['volume'].values
    
    indicators = {
        'Trend': {
            'SMA 20': ta.SMA(closes, 20)[-1],
            'SMA 50': ta.SMA(closes, 50)[-1],
            'SMA 200': ta.SMA(closes, 200)[-1],
            'EMA 20': ta.EMA(closes, 20)[-1],
            'MACD': ta.MACD(closes)[0][-1],
            'ADX': ta.ADX(highs, lows, closes, 14)[-1]
        },
        'Momentum': {
            'RSI 14': ta.RSI(closes, 14)[-1],
            'Stoch %K': ta.STOCH(highs, lows, closes)[0][-1],
            'Stoch %D': ta.STOCH(highs, lows, closes)[1][-1],
            'CCI 20': ta.CCI(highs, lows, closes, 20)[-1]
        },
        'Volatility': {
            'ATR 14': ta.ATR(highs, lows, closes, 14)[-1],
            'BB Width': (ta.BBANDS(closes)[0][-1] - ta.BBANDS(closes)[2][-1]) / ta.BBANDS(closes)[1][-1],
            'Chaikin Vol': ta.OBV(closes, volumes)[-1] / (ta.EMA(volumes, 10)[-1] + 1e-10)
        },
        'Volume': {
            'OBV': ta.OBV(closes, volumes)[-1],
            'AD': ta.AD(highs, lows, closes, volumes)[-1],
            'CMF 20': ta.ADOSC(highs, lows, closes, volumes, fastperiod=3, slowperiod=10)[-1]
        }
    }
    
    # Trend analysis
    trend_strength = "Strong" if indicators['Trend']['ADX'] > 25 else "Weak" if indicators['Trend']['ADX'] < 20 else "Moderate"
    trend_direction = "Up" if closes[-1] > ta.EMA(closes, 20)[-1] else "Down"
    
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

=== END OF DIGEST ===
"""
    return digest