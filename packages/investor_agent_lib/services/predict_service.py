from packages.investor_agent_lib.services.yfinance_service import get_price_history, get_current_price
import pandas as pd
from datetime import datetime, timedelta
from packages.predict_lib import predict_down_v3, utils
from packages.investor_agent_lib.options import option_indicators
from packages.investor_agent_lib.analytics.risk import get_risk_free_rate

def get_current_percentile(ticker: str) -> dict:
    """
    Get the current percentile of daily price changes for a stock ticker.

    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
    dict: Dictionary containing the current price, current change percentage, and percentile of daily price changes.
    Returns None if data retrieval fails or if there are no daily price changes.
    """
    history = get_price_history(ticker, '2y', raw=True)
    if history is None or history.empty:
        print(f"Failed to retrieve data for {ticker}, please check the ticker symbol.")
        return None
    history['daily_change_pct'] = history['Close'].pct_change() * 100
    history.dropna(inplace=True)
    current_price = get_current_price(ticker)
    if current_price is None:
        print(f"Failed to retrieve current price for {ticker}.")
        return None
    # must use -2 to get the last price change
    current_change_pct = (current_price - history['Close'].iloc[-2]) / history['Close'].iloc[-2] * 100
    history_filtered = history[history['daily_change_pct'] != 0]
    if history_filtered.empty:
        print(f"{ticker} has no daily price changes in the past two years.")
        return None
    percentile = (history_filtered['daily_change_pct'] < current_change_pct).mean() * 100
    return dict(
        ticker=ticker,
        current_price=current_price,
        current_change_pct=current_change_pct,
        percentile=percentile
    )


def predict_next_day_chg(ticker: str) -> dict:
    """
    Analyze stock price movement based on the daily price changes and predict next day's direction frequency.
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
    dict: Dictionary containing analysis results with keys:
          - ticker: Stock ticker symbol
          - current_price: Current stock price
          - current_change_pct: Today's price change percentage
          - up_next_day_freq: Frequency (percentage) of next day being up
          - down_next_day_freq: Frequency (percentage) of next day being down
          - flat_next_day_freq: Frequency (percentage) of next day being flat (change = 0)
          - sample_size: Number of data points used for analysis
          Returns None if data retrieval fails.
    """
    try:
        # Get stock price history from yfinance_service
        df = get_price_history(ticker, '2y', raw=True)
        
        if df is None or df.empty:
            print(f"Failed to retrieve data for {ticker}, please check the ticker symbol.")
            return None
        
        # Calculate daily price changes
        df['daily_change_pct'] = df['Close'].pct_change() * 100
        
        # Calculate next day price changes
        df['next_day_change_pct'] = df['Close'].shift(-1).pct_change() * 100

        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Get current info
        current_info = get_current_percentile(ticker)
        if current_info is None:
            print(f"Failed to retrieve current percentile for {ticker}.")
            return None
        
        # Determine the current day's change direction
        current_change = current_info['current_change_pct']
        
        slip = 0.001  # Define a small threshold for flat change
        # Filter historical data based on current day's change direction
        if current_change > 0:
            # If today was up, look at historical days that were also up
            historical_context_df = df[df['daily_change_pct'] > slip]
        elif current_change < 0:
            # If today was down, look at historical days that were also down
            historical_context_df = df[df['daily_change_pct'] < slip]
        else:
            # If today was flat, look at historical days that were also flat
            historical_context_df = df[abs(df['daily_change_pct']) <= slip]

        if historical_context_df.empty:
            # Fallback if no historical context matches current day's change direction
            historical_context_df = df
            
        total_samples = len(historical_context_df)
        
        if total_samples == 0:
            return dict(
                ticker=ticker,
                current_price=current_info['current_price'],
                current_change_pct=current_change,
                up_next_day_freq=0.0,
                down_next_day_freq=0.0,
                flat_next_day_freq=0.0,
                sample_size=0,
                slip=slip
            )

        # Calculate frequencies for next day's change
        up_next_day_count = (historical_context_df['next_day_change_pct'] > 0.005).sum()
        down_next_day_count = (historical_context_df['next_day_change_pct'] < 0.005).sum()
        flat_next_day_count = ( abs(historical_context_df['next_day_change_pct']) <= 0.005).sum()
        
        up_next_day_freq = (up_next_day_count / total_samples) * 100
        down_next_day_freq = (down_next_day_count / total_samples) * 100
        flat_next_day_freq = (flat_next_day_count / total_samples) * 100
        
        # Return analysis results
        return dict(
            ticker=ticker,
            current_price=current_info['current_price'],
            current_change_pct=f"{current_change:.2f}%",
            up_next_day_freq=f"{up_next_day_freq:.2f}%",
            down_next_day_freq=f"{down_next_day_freq:.2f}%",
            flat_next_day_freq=f"{flat_next_day_freq:.2f}%",
            sample_size=total_samples,
            slip=f"flat change threshold: {slip*100:.3f}%"
        )


    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None



def profit_prob(ticker, current_price, next_price):
    risk_free = get_risk_free_rate()
    inds = option_indicators.calculate_indicators(ticker)
    near_days = inds['nearest_expiry_days']
    return utils.implied_prob(current_price, next_price, near_days, risk_free, inds['atm_iv_avg'])