import pandas as pd
import talib as ta
from tabulate import tabulate
import logging

from src.investor_agent.calc_bullish import cal_bullish_divergence, calculate_fibonacci_retracement
from src.investor_agent.calc_time_series_analyze import calculate_time_series_analyze
from src.investor_agent.calc_basic_statistics import calculate_basic_statistics
from src.investor_agent.calc_risk_metrics import cal_risk
from src.investor_agent.digest_ta_utils import tech_indicators

logger = logging.getLogger(__name__)

def _prepare_time_series_data(time_series_data: pd.DataFrame) -> pd.DataFrame | str:
    """Performs initial validation and data preparation."""
    if time_series_data.empty:
        return "No time series data available."

    if time_series_data.shape[0] < 20:
        logger.warning("Not enough rows in time series data.")
        return tabulate(time_series_data, headers='keys', tablefmt="simple")

    # Data preparation
    if 'date' in time_series_data.columns:
        time_series_data['date'] = pd.to_datetime(time_series_data['date'])
        time_series_data = time_series_data.set_index('date').sort_index()

    return time_series_data


def get_latest_data_sample(time_series_data: pd.DataFrame, num_days: int = 20) -> pd.DataFrame:
    """Extracts and formats a smartly sampled data sample with:
    - High resolution for recent data (daily)
    - Medium resolution for intermediate data (weekly)
    - Low resolution for older data (monthly)
    Total samples will be <= num_days.
    """
    if len(time_series_data) <= num_days:
        # If data is shorter than requested window, return all
        sampled_data = time_series_data.copy()
    else:
        # Hybrid sampling strategy
        daily_window = num_days // 2  # 50% daily samples
        weekly_window = num_days * 3 // 10  # 30% weekly samples
        monthly_window = num_days - daily_window - weekly_window  # 20% monthly samples
        
        # Get daily samples from most recent period
        daily_samples = time_series_data[-daily_window:].copy()
        
        # Get weekly samples from intermediate period
        weekly_start = -daily_window - (weekly_window * 7)
        weekly_samples = time_series_data[weekly_start:-daily_window:7].copy()
        
        # Get monthly samples from oldest period
        monthly_start = -daily_window - (weekly_window * 7) - (monthly_window * 30)
        monthly_samples = time_series_data[monthly_start:weekly_start:30].copy()
        
        # Combine samples
        sampled_data = pd.concat([monthly_samples, weekly_samples, daily_samples])
    
    # Format output
    sampled_data['Date'] = sampled_data.index.strftime('%Y-%m-%d')
    sampled_data = sampled_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return tabulate(sampled_data, headers=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
                   tablefmt="simple", showindex=False)


def generate_time_series_digest_for_LLM(time_series_data: pd.DataFrame) -> str:
    """Generate a comprehensive quantitative digest for time series data.
    
    Args:
        time_series_data: DataFrame containing OHLCV data with date as index
        
    Returns:
        str: Structured digest containing statistical analysis, technical indicators,
             risk metrics, and qualitative interpretations for LLM consumption.
    """
    prepared_data = _prepare_time_series_data(time_series_data.copy()) # Use a copy to avoid modifying the original DataFrame

    if isinstance(prepared_data, str):
        return prepared_data # Return error message if preparation failed

    # Basic statistics
    stats = calculate_basic_statistics(prepared_data)
    
    time_series_summary = calculate_time_series_analyze(prepared_data)

    # Technical indicators
    indicators_details = tech_indicators(prepared_data)

    
    # Risk-adjusted return metrics
    risk_metrics = cal_risk(prepared_data)

    # Latest 20 days sample
    latest_data_sample = get_latest_data_sample(prepared_data)
    
    # div = cal_bullish_divergence(prepared_data)

    # Pattern recognition
    pattern = pattern_recognition(prepared_data)

    fib = calculate_fibonacci_retracement(prepared_data)

    # Generate structured digest

    return f"""
===== TIME SERIES DIGEST =====
{stats}

===== TIME SERIES SUMMARY =====
{time_series_summary}

===== TECHNICAL INDICATORS =====
{indicators_details}

===== RISK METRICS =====
{risk_metrics}

===== PATTERN RECOGNITION =====
{pattern}

===== FIBONACCI RETRACEMENT =====
{fib}

===== OHLCV SAMPLE =====
{latest_data_sample}

===== END OF DIGEST =====
"""


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
