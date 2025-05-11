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

def ensure_cache_dir():
    """Ensure cache directory exists"""
    CACHE_DIR.mkdir(exist_ok=True)

def should_refresh_cache(cache_path: Path, update_hour=6):  # UTC+8 6:00 AM
    """Check if cache needs refresh based on daily update schedule
    
    Args:
        cache_path: Path object to the cache file
        update_hour: Hour of day (0-23) when cache should be refreshed (default: 6)
    
    Returns:
        bool: True if cache needs refresh, False otherwise
    """
    if not cache_path.exists():
        return True
    
    now = datetime.now()
    last_modified = datetime.fromtimestamp(cache_path.stat().st_mtime)
    
    # Only refresh if:
    # 1. Current time is past today's update hour
    # 2. Last modified was before today's update hour
    return (
        now.hour >= update_hour and
        (last_modified.date() < now.date() or
         (last_modified.date() == now.date() and
          last_modified.hour < update_hour))
    )

def download_database(db_url:str, db_path:Path):
    """Download the SQLite database and save to cache"""

    ensure_cache_dir()
    response = requests.get(db_url)
    response.raise_for_status()
    
    with open(db_path, 'wb') as f:
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

    DB_PATH = CACHE_DIR / "options_data.db"
    DB_URL = "https://prefect.findata-be.uk/link_artifact/options_data.db"



    if should_refresh_cache(DB_PATH, 6):
        download_database(DB_URL, DB_PATH)
    
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





def get_historical_option_indicator_by_ticker(ticker_symbol: str) -> pd.DataFrame:
    """
    Get options indicator data for a specific ticker symbol from cached database
    
    Args:
        ticker_symbol: The ticker symbol to query (e.g. 'AAPL')
    
    Returns:
        pd.DataFrame with options indicator data containing columns:
        atm_iv_avg, call_delta, call_rho, call_theta, date, gamma, lastTradeDate,
        pc_ratio, put_delta, put_rho, put_theta, skew_measure, term_structure_slope,
        ticker, underlyingPrice, vega
    """

    DB_PATH = CACHE_DIR / "options_indicator.db"
    DB_URL = "https://prefect.findata-be.uk/link_artifact/options_indicator.db"

    if should_refresh_cache(DB_PATH, 6):
        download_database(DB_URL, DB_PATH)
    
    with sqlite3.connect(DB_PATH) as conn:
        query = """
        SELECT
            atm_iv_avg, call_delta, call_rho, call_theta, date, gamma, lastTradeDate,
            pc_ratio, put_delta, put_rho, put_theta, skew_measure, term_structure_slope,
            ticker, underlyingPrice, vega
        FROM options_indicator
        WHERE ticker = ?
        ORDER BY date ASC
        """
        return pd.read_sql_query(query, conn, params=(ticker_symbol,))
