import pandas as pd
import talib as ta

def calculate_time_series_analyze(time_series_data: pd.DataFrame) -> dict:
    """
    Performs a simple time series analysis on stock data for LLM consumption.

    Args:
        time_series_data: A pandas DataFrame with at least 'Open', 'Close', and 'Volume' columns.
                          Assumes a DatetimeIndex.

    Returns:
        A dictionary containing a text summary of the analysis.
    """
    if time_series_data.empty:
        return "No data provided for analysis."
    
    time_series_data = time_series_data.copy()

    # Ensure data is sorted by index (time)
    time_series_data = time_series_data.sort_index()

    # Calculate daily returns
    time_series_data['Daily_Return'] = time_series_data['Close'].pct_change()

    # Calculate simple moving averages
    time_series_data['SMA_7'] = time_series_data['Close'].rolling(window=7).mean()
    time_series_data['SMA_30'] = time_series_data['Close'].rolling(window=30).mean()

    # Basic statistics
    avg_daily_return = time_series_data['Daily_Return'].mean() * 100 # in percentage
    volatility = time_series_data['Daily_Return'].std() * 100 # in percentage
    avg_close_price = time_series_data['Close'].mean()
    avg_volume = time_series_data['Volume'].mean()

    # Get latest values for SMAs
    latest_sma_7 = time_series_data['SMA_7'].iloc[-1] if not time_series_data['SMA_7'].isnull().all() else None
    latest_sma_30 = time_series_data['SMA_30'].iloc[-1] if not time_series_data['SMA_30'].isnull().all() else None
    latest_close = time_series_data['Close'].iloc[-1]

    # Generate summary text
    summary_text = f"Time Series Analysis Summary:\n"
    summary_text += f"- Period covered: {time_series_data.index.min().strftime('%Y-%m-%d')} to {time_series_data.index.max().strftime('%Y-%m-%d')}\n"
    summary_text += f"- Average Daily Return: {avg_daily_return:.4f}%\n"
    summary_text += f"- Daily Volatility (Std Dev of Returns): {volatility:.4f}%\n"
    summary_text += f"- Average Closing Price: {avg_close_price:.2f}\n"
    summary_text += f"- Average Volume: {avg_volume:,.0f}\n"
    summary_text += f"- Latest Closing Price: {latest_close:.2f}\n"

    if latest_sma_7 is not None:
        summary_text += f"- Latest 7-Day Moving Average (Close): {latest_sma_7:.2f}\n"
    if latest_sma_30 is not None:
        summary_text += f"- Latest 30-Day Moving Average (Close): {latest_sma_30:.2f}\n"

    # Add simple trend indication based on SMAs
    if latest_sma_7 is not None and latest_sma_30 is not None:
        if latest_sma_7 > latest_sma_30:
            summary_text += "- Short-term trend (7-day SMA) is above long-term trend (30-day SMA), potentially indicating bullish momentum.\n"
        elif latest_sma_7 < latest_sma_30:
            summary_text += "- Short-term trend (7-day SMA) is below long-term trend (30-day SMA), potentially indicating bearish momentum.\n"
        else:
            summary_text += "- Short-term and long-term trends (SMAs) are converging.\n"
    elif latest_sma_7 is not None:
         if latest_close > latest_sma_7:
             summary_text += "- Latest close is above 7-day SMA, potentially indicating short-term bullish sign.\n"
         elif latest_close < latest_sma_7:
             summary_text += "- Latest close is below 7-day SMA, potentially indicating short-term bearish sign.\n"
    elif latest_sma_30 is not None:
         if latest_close > latest_sma_30:
             summary_text += "- Latest close is above 30-day SMA, potentially indicating bullish sign.\n"
         elif latest_close < latest_sma_30:
             summary_text += "- Latest close is below 30-day SMA, potentially indicating bearish sign.\n"


    return summary_text

