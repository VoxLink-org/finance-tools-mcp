import pandas as pd
import talib as ta
from tabulate import tabulate
import logging

from packages.investor_agent_lib.analytics.technical import calculate_fibonacci_retracement, pattern_recognition
from packages.investor_agent_lib.analytics.time_series import calculate_time_series_analyze
from packages.investor_agent_lib.analytics.statistics import calculate_basic_statistics
from packages.investor_agent_lib.analytics.risk import cal_risk
from packages.investor_agent_lib.digests.technical_digest import tech_indicators

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

