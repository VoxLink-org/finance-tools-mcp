
import pandas as pd
import numpy as np
import talib as ta
from tabulate import tabulate
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def cal_bullish_divergence(time_series_data: pd.DataFrame) -> str:
    """
    Calculate bullish divergence for common technical indicators.

    Bullish divergence occurs when the price makes a lower low, but the indicator
    makes a higher low. This can signal a potential upward price reversal.

    Args:
        time_series_data: DataFrame containing OHLCV data with date as index.

    Returns:
        str: A string indicating if bullish divergence is detected and for which indicators.
             Returns an empty string if no bullish divergence is found.
    """
    time_series_data = time_series_data.copy()
    closes = time_series_data['Close'].values.astype(float)
    lows = time_series_data['Low'].values.astype(float)
    highs = time_series_data['High'].values.astype(float)

    if len(closes) < 30: # Need enough data for indicators and pattern detection
        return "Insufficient data to check for bullish divergence."

    divergences = []

    # Helper to find recent significant lows in price and indicator
    def find_recent_lows(price_series, indicator_series, window=30):
        price_lows = []
        indicator_lows = []
        
        if len(price_series) < window or len(indicator_series) < window:
            return [], []

        recent_prices = price_series[-window:]
        recent_indicators = indicator_series[-window:]

        # Find all potential lows in price (local minima)
        price_min_indices = []
        for i in range(1, window-1):
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                price_min_indices.append(i)
        
        # Need at least 2 lows to check divergence
        if len(price_min_indices) < 2:
            return [], []

        # Sort by price value (lowest first)
        price_min_indices = sorted(price_min_indices, key=lambda x: recent_prices[x])
        
        # Take the two most significant lows (lowest values)
        idx1, idx2 = price_min_indices[0], price_min_indices[1]
        if idx1 > idx2:  # Ensure chronological order
            idx1, idx2 = idx2, idx1

        # Get corresponding indicator values
        ind_val1 = recent_indicators[idx1]
        ind_val2 = recent_indicators[idx2]

        # Check for divergence with some tolerance
        price_diff = recent_prices[idx1] - recent_prices[idx2]
        ind_diff = ind_val2 - ind_val1
        
        # Price should be at least 1% lower and indicator at least 2% higher
        if (price_diff > 0.01 * recent_prices[idx2]) and (ind_diff > 0.02 * abs(ind_val1)):
            price_lows.append((window - 1 - idx1, recent_prices[idx1]))
            price_lows.append((window - 1 - idx2, recent_prices[idx2]))
            indicator_lows.append((window - 1 - idx1, ind_val1))
            indicator_lows.append((window - 1 - idx2, ind_val2))

        return price_lows, indicator_lows

    # Use different window sizes based on indicator
    rsi_window = 30
    macd_window = 50
    stoch_window = 20

    # RSI Divergence
    rsi = ta.RSI(closes, 14)
    if not np.isnan(rsi[-1]):
        price_lows, rsi_lows = find_recent_lows(closes, rsi, rsi_window)
        if price_lows and rsi_lows:
             divergences.append(f"Bullish divergence detected with RSI 14. Price made a lower low ({price_lows[0][1]:.2f} -> {price_lows[1][1]:.2f}) while RSI made a higher low ({rsi_lows[0][1]:.2f} -> {rsi_lows[1][1]:.2f}) in the last {n_days} days.")


    # MACD Divergence (using MACD line)
    macd, signal, hist = ta.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    if not np.isnan(macd[-1]):
        price_lows, macd_lows = find_recent_lows(closes, macd, macd_window)
        if price_lows and macd_lows:
            divergences.append(f"Bullish divergence detected with MACD. Price made a lower low ({price_lows[0][1]:.2f} -> {price_lows[1][1]:.2f}) while MACD made a higher low ({macd_lows[0][1]:.2f} -> {macd_lows[1][1]:.2f}) in the last {n_days} days.")

    # Stochastic Divergence (using %K line)
    stoch_k, stoch_d = ta.STOCH(highs, lows, closes)
    if not np.isnan(stoch_k[-1]):
        price_lows, stoch_k_lows = find_recent_lows(closes, stoch_k, stoch_window)
        if price_lows and stoch_k_lows:
             divergences.append(f"Bullish divergence detected with Stochastic %K. Price made a lower low ({price_lows[0][1]:.2f} -> {price_lows[1][1]:.2f}) while Stochastic %K made a higher low ({stoch_k_lows[0][1]:.2f} -> {stoch_k_lows[1][1]:.2f}) in the last {n_days} days.")

    logger.info(divergences)
    
    if divergences:
        return "Possible Bullish Divergence Signals:\n" + "\n".join(divergences)
    else:
        return "No significant bullish divergence detected recently."


def calculate_fibonacci_retracement(time_series_data: pd.DataFrame) -> str:
    """
    Calculate Fibonacci retracement levels based on the most recent significant swing high and low.
    
    Fibonacci retracement levels are horizontal lines that indicate where support and resistance
    are likely to occur. They are calculated by taking the high and low of a price movement
    and dividing the vertical distance by the key Fibonacci ratios of 23.6%, 38.2%, 50%, 
    61.8% and 78.6%.
    
    Args:
        time_series_data: DataFrame containing OHLCV data with date as index.
    
    Returns:
        Dict[str, str]: A dictionary containing:
            - 'levels': Fibonacci levels with prices
            - 'current_price': Current price position relative to levels
            - 'trend': Current trend direction (uptrend/downtrend)
            - 'swing_high': Swing high price and date
            - 'swing_low': Swing low price and date
    """
    if len(time_series_data) < 30:
        return {"error": "Insufficient data (minimum 30 periods required)"}
    
    closes = time_series_data['Close'].values
    highs = time_series_data['High'].values
    lows = time_series_data['Low'].values
    dates = time_series_data.index
    
    # Find most recent significant swing high and low
    def find_swing_points(prices: np.ndarray, window: int = 10) -> Tuple[int, int]:
        swing_high_idx = np.argmax(prices[-window:]) + len(prices) - window
        swing_low_idx = np.argmin(prices[-window:]) + len(prices) - window
        return swing_high_idx, swing_low_idx
    
    swing_high_idx, swing_low_idx = find_swing_points(closes)
    
    # Determine trend direction
    if swing_high_idx > swing_low_idx:
        trend = "uptrend"
        start_price = lows[swing_low_idx]
        end_price = highs[swing_high_idx]
    else:
        trend = "downtrend"
        start_price = highs[swing_high_idx]
        end_price = lows[swing_low_idx]
    
    price_range = end_price - start_price
    current_price = closes[-1]
    
    # Calculate Fibonacci levels
    levels = {
        '0%': end_price,
        '23.6%': end_price - 0.236 * price_range,
        '38.2%': end_price - 0.382 * price_range,
        '50%': end_price - 0.5 * price_range,
        '61.8%': end_price - 0.618 * price_range,
        '78.6%': end_price - 0.786 * price_range,
        '100%': start_price
    }
    
    # Determine current price position relative to levels
    level_position = ""
    prev_level = None
    for level_name, level_price in sorted(levels.items(), key=lambda x: x[1], reverse=True):
        if current_price >= level_price:
            if prev_level:
                level_position = f"Between {prev_level[0]} ({prev_level[1]:.2f}) and {level_name} ({level_price:.2f})"
            else:
                level_position = f"Above {level_name} ({level_price:.2f})"
            break
        prev_level = (level_name, level_price)
    else:
        level_position = f"Below 100% ({start_price:.2f})"
    
    rows = {
        "levels": [f"{k}: {v:.2f}" for k, v in levels.items()],
        "current_price": level_position,
        "trend": trend,
        "swing_high": f"{highs[swing_high_idx]:.2f} on {dates[swing_high_idx]}",
        "swing_low": f"{lows[swing_low_idx]:.2f} on {dates[swing_low_idx]}"
    }
    text = ''
    for k, v in rows.items():
        text += f"{k}: {v}\n"
    
    return text



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

    period = min(60, len(time_series_data))

    time_series_data = time_series_data[-period:]

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
        return f"\n Patterns Detected in the last {period} days:\n" + \
               "\n".join(detected_patterns) + \
               "\n"
