import logging
import os
import sqlite3
from typing import Literal
import pandas as pd
import requests
from datetime import datetime, time
from pathlib import Path


logger = logging.getLogger(__name__)



CACHE_DIR = Path(__file__).parent / ".cache"
DB_PATH = CACHE_DIR / "options_data.db"
DB_URL = "https://prefect.findata-be.uk/link_artifact/options_data.db"
DAILY_UPDATE_HOUR = 6  # UTC+8 6:00 AM

def ensure_cache_dir():
    """Ensure cache directory exists"""
    CACHE_DIR.mkdir(exist_ok=True)

def should_refresh_cache():
    """Check if cache needs refresh based on daily update schedule"""
    if not DB_PATH.exists():
        return True
    
    now = datetime.now()
    last_modified = datetime.fromtimestamp(DB_PATH.stat().st_mtime)
    
    # Only refresh if:
    # 1. Current time is past today's update hour (6:00 AM UTC+8)
    # 2. Last modified was before today's update hour
    return (
        now.hour >= DAILY_UPDATE_HOUR and
        (last_modified.date() < now.date() or 
         (last_modified.date() == now.date() and 
          last_modified.hour < DAILY_UPDATE_HOUR))
    )

def download_database():
    """Download the SQLite database and save to cache"""
    ensure_cache_dir()
    response = requests.get(DB_URL)
    response.raise_for_status()
    
    with open(DB_PATH, 'wb') as f:
        f.write(response.content)

def get_historical_options_by_ticker(ticker_symbol: str) -> pd.DataFrame:
    """
    Get options data for a specific ticker symbol from cached database
    
    Args:
        ticker_symbol: The ticker symbol to query (e.g. 'AAPL')
    
    Returns:
        pd.DataFrame with options data containing columns:
        contractSymbol, strike, lastPrice, lastTradeDate, change, volume,
        openInterest, impliedVolatility, expiryDate, snapshotDate, 
        underlyingPrice, optionType
    """
    if should_refresh_cache():
        download_database()
    
    with sqlite3.connect(DB_PATH) as conn:
        # First get all matching rows
        query = """
        SELECT 
            contractSymbol, strike, lastPrice, lastTradeDate, change, volume,
            openInterest, impliedVolatility, expiryDate, snapshotDate,
            underlyingPrice, optionType,
            ROW_NUMBER() OVER (
                PARTITION BY contractSymbol, snapshotDate 
                ORDER BY lastTradeDate DESC
            ) as row_num
        FROM options
        WHERE tickerSymbol = ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker_symbol,))
        
        # Filter to only keep most recent lastTradeDate for each (contractSymbol, snapshotDate) pair
        return df[df['row_num'] == 1].drop(columns=['row_num'])




