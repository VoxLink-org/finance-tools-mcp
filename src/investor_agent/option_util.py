import logging
import os
import sqlite3
from typing import Literal
import pandas as pd
import requests
from datetime import datetime, time
from pathlib import Path

from . import yfinance_utils

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



def create_snapshot(df: pd.DataFrame, underlyingPrice: float) -> pd.DataFrame:
    """Select 30 option contracts daily using three-bucket strategy:
    1. Top 10 by volume (with OI > 20)
    2. Top 10 by OI (excluding bucket 1 selections)
    3. Top 10 by remaining volume and near-the-money options
    
    Args:
        df: DataFrame containing option contract data with columns:
            - volume: trading volume
            - openInterest: open interest
            - strike: strike price
            - expiryDate: expiration date
            - optionType: 'C' or 'P'
            - lastPrice: last traded price
    
    Returns:
        DataFrame with selected 30 contracts
    """

    # First filter out low liquidity contracts
    df = df[(df['openInterest'] >= 20) & (df['volume'] >= 10)].copy()
    
    # Bucket 1: Top 10 by volume
    bucket1 = df.nlargest(10, 'volume')
    remaining = df[~df.index.isin(bucket1.index)]
    
    # Bucket 2: Top 10 by OI from remaining
    bucket2 = remaining.nlargest(10, 'openInterest')
    remaining = remaining[~remaining.index.isin(bucket2.index)]
    
    # Bucket 3: 10 near-the-money options
    # Get current price
    current_price = underlyingPrice
    
    # Calculate moneyness (absolute distance from current price)
    remaining['moneyness'] = abs(remaining['strike'] - current_price)
    
    # Select 10 nearest-to-money options
    # Ensure balanced selection of calls and puts (5 each)
    calls = remaining[remaining['optionType'] == 'C']
    puts = remaining[remaining['optionType'] == 'P']
    
    near_money_calls = calls.sort_values(['moneyness', 'expiryDate']).head(5)
    near_money_puts = puts.sort_values(['moneyness', 'expiryDate']).head(5)
    
    bucket3 = pd.concat([near_money_calls, near_money_puts])
    
    # Drop moneyness column
    bucket3 = bucket3.drop('moneyness', axis=1)

    # Combine all buckets and return
    return pd.concat([bucket1, bucket2, bucket3]).reset_index(drop=True)


def get_options(
    ticker_symbol: str,
    num_options: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    strike_lower: float | None = None,
    strike_upper: float | None = None,
    option_type: Literal["C", "P"] | None = None,
) -> pd.DataFrame:
    """Get options with bucketed selection. Dates: YYYY-MM-DD. Type: C=calls, P=puts."""
    underlyingPrice = yfinance_utils.get_current_price(ticker_symbol)
    

    logger.info(f"Current stock price for {ticker_symbol}: {underlyingPrice}")


    try:
        df, error = yfinance_utils.get_filtered_options(
            ticker_symbol, start_date, end_date, strike_lower, strike_upper, option_type
        )

        if error:
            return error


        if len(df) == 0:
            return f"No options found for {ticker_symbol}"

        logger.info(f"Found {len(df)} options for {ticker_symbol}")

        # pick up some of the columns
        df = df[["contractSymbol", "strike", "lastPrice", "lastTradeDate", "change", "volume", "openInterest", "impliedVolatility", "expiryDate"]]
        # add new columns, ticker symbol , snapshot date and underlying price
        df["tickerSymbol"] = ticker_symbol
        df["snapshotDate"] = datetime.now().strftime("%Y-%m-%d")
        df["underlyingPrice"] = underlyingPrice
        df["optionType"] = df["contractSymbol"].apply(lambda x: "C" if "C" in x else "P")
        

        return create_snapshot(df, underlyingPrice)
    except Exception as e:
        logger.error(f"Error getting options data for {ticker_symbol}: {e}")
        return f"Failed to retrieve options data for {ticker_symbol}: {str(e)}"



