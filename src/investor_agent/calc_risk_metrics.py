import numpy as np
import pandas as pd

from src.investor_agent.macro_api_utils import get_fred_series
import logging

logger = logging.getLogger(__name__)

def get_risk_free_rate():
        # Fetch the latest 10-year treasury yield from FRED (DGS10)
    # The data is returned as a string representation of a pandas Series
    risk_free_rate = 0.02
    try:
        fred_data = get_fred_series("DGS10")
        # Parse the latest value from the string
        latest_value_str = fred_data.iloc[-1]
        risk_free_rate = float(latest_value_str) / 100.0
        
    except (IndexError, ValueError):
        # Fallback to a default rate if fetching or parsing fails
        risk_free_rate = 0.02
        logger.error("Failed to fetch or parse FRED data. Using default risk-free rate.")

    return risk_free_rate

def cal_risk(time_series_data: pd.DataFrame) -> dict:
    # Calculate daily returns and risk-free rate proxy (2% for simplicity)
    time_series_data['daily_return'] = time_series_data['Close'].pct_change()

    risk_free_rate = get_risk_free_rate()
    # Risk-adjusted return metrics
    annualized_return = np.mean(time_series_data['daily_return'].dropna()) * 252
    volatility = np.std(time_series_data['daily_return'].dropna()) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    risk_metrics = {
        'Annualized Return': f"{annualized_return*100:.2f}%",
        'Annualized Volatility': f"{volatility*100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{((time_series_data['Close'].pct_change().cumsum() - time_series_data['Close'].pct_change().cumsum().cummax()).min())*100:.2f}%",
        'Sortino Ratio': f"{(annualized_return - risk_free_rate) / (np.sqrt(np.mean(time_series_data[time_series_data['daily_return'] < 0]['daily_return'].dropna()**2)) * np.sqrt(252)):.2f}"
            if len(time_series_data[time_series_data['daily_return'] < 0]) > 0 else 'N/A'
    }

    text = "Risk Metrics:\n"
    text += f"  Annualized Return: {risk_metrics['Annualized Return']}\n"
    text += f"  Annualized Volatility: {risk_metrics['Annualized Volatility']}\n"
    text += f"  Sharpe Ratio: {risk_metrics['Sharpe Ratio']}\n"
    text += f"  Max Drawdown: {risk_metrics['Max Drawdown']}\n"
    text += f"  Sortino Ratio: {risk_metrics['Sortino Ratio']}\n"

    return text

