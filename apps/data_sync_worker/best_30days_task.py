import sqlite3

import pandas as pd
from prefect import task, flow, get_run_logger

from packages.investor_agent_lib.services.yfinance_service import get_price_history

from config.paths import DATA_DIR


@task(name="get-best-30days-data")
def get_best_30days_task(ticker: str) -> bool:
    days = 30
    logger = get_run_logger()
    try:
        df = get_price_history(ticker, period="3mo", raw=True)
        if df is None or df.empty:
            return False
        
        # Get the last 30 days of data
        last_30_days = df.tail(days)
        
        # Get the current price and the max price in the last 30 days
        current_price = last_30_days['High'].iloc[-1]
        max_price_30_days = last_30_days['High'].max()
        
        # Check if the current price is the highest in the last 30 days
        return current_price >= max_price_30_days
    except Exception as e:
        logger.error(f"Failed to get best 30days data for {ticker}: {str(e)}")
        return False

