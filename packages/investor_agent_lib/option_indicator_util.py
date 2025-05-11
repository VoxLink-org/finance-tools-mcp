import logging
import os
import sqlite3
from typing import Literal, Optional
import pandas as pd
import requests
from datetime import datetime, time
from pathlib import Path


logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / ".cache"
DB_PATH = CACHE_DIR / "options_indicator.db"
DB_URL = "https://prefect.findata-be.uk/link_artifact/options_indicator.db"
DAILY_UPDATE_HOUR = 6  # UTC+8 6:00 AM

def ensure_cache_dir_indicator():
    """Ensure cache directory for indicators exists"""
    CACHE_DIR.mkdir(exist_ok=True)

def should_refresh_indicator_cache() -> bool:
    """Check if indicator cache needs refresh based on daily update schedule"""
    if not DB_PATH.exists():
        logger.info(f"Indicator database not found at {DB_PATH}. Will attempt download.")
        return True
    
    now = datetime.now()
    # Ensure last_modified is timezone-aware if now is, or make both naive.
    # Assuming DB_PATH.stat().st_mtime gives UTC timestamp if server runs in UTC
    # or local time if server runs in local time. For simplicity, using naive datetime objects.
    last_modified = datetime.fromtimestamp(DB_PATH.stat().st_mtime)
    
    # Refresh if:
    # 1. Current time is past today's update hour (e.g., 6:00 AM)
    # 2. Last modified was before today's update hour OR on a previous day.
    
    # Check if today's update time has passed
    today_update_time_passed = now.hour >= DAILY_UPDATE_HOUR

    # Check if last modified was before today's update time
    if last_modified.date() < now.date():
        # Modified on a previous day
        needs_refresh = today_update_time_passed
        if needs_refresh:
            logger.info(f"Indicator DB last modified on {last_modified.date()}, current date {now.date()}. Refreshing as update hour {DAILY_UPDATE_HOUR} has passed.")
        else:
            logger.info(f"Indicator DB last modified on {last_modified.date()}, current date {now.date()}. Update hour {DAILY_UPDATE_HOUR} not yet passed.")
        return needs_refresh
    elif last_modified.date() == now.date():
        # Modified today, check if it was before the update hour
        modified_before_update_hour_today = last_modified.hour < DAILY_UPDATE_HOUR
        needs_refresh = today_update_time_passed and modified_before_update_hour_today
        if needs_refresh:
            logger.info(f"Indicator DB last modified today at {last_modified.hour}, before update hour {DAILY_UPDATE_HOUR}. Refreshing as update hour has passed.")
        else:
            logger.info(f"Indicator DB last modified today at {last_modified.hour}. Update hour {DAILY_UPDATE_HOUR}. Today's update time passed: {today_update_time_passed}. Modified before update hour: {modified_before_update_hour_today}")
        return needs_refresh
    else:
        # This case (last_modified.date() > now.date()) should not happen if clocks are synced.
        logger.warning("Indicator DB last modified date is in the future. Assuming no refresh needed.")
        return False

def download_indicator_db():
    """Download the SQLite indicator database and save to cache"""
    ensure_cache_dir_indicator()
    logger.info(f"Attempting to download indicator database from {DB_URL} to {DB_PATH}")
    try:
        response = requests.get(DB_URL, timeout=60) # Added timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        
        with open(DB_PATH, 'wb') as f:
            f.write(response.content)
        logger.info(f"Successfully downloaded indicator database to {DB_PATH}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download indicator database from {DB_URL}: {e}")
        # If DB_PATH exists from a previous failed download, it might be corrupted.
        # Optionally, remove it:
        # if DB_PATH.exists():
        #     DB_PATH.unlink()
        return False
    except IOError as e:
        logger.error(f"Failed to write indicator database to {DB_PATH}: {e}")
        return False
def get_historical_option_indicators(ticker: str) -> Optional[pd.DataFrame]:
    """
    Retrieves historical option indicators for a given stock ticker from the database.

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL").

    Returns:
        A pandas DataFrame containing the historical option indicators for the ticker,
        or None if the database file doesn't exist or an error occurs.
    """
    if should_refresh_indicator_cache():
        logger.info("Indicator cache needs refresh. Attempting to download database.")
        if not download_indicator_db():
            logger.error("Failed to download indicator database. Cannot proceed.")
            return None
    
    if not DB_PATH.exists():
        logger.error(f"Indicator database file still not found at {DB_PATH} after cache check/download attempt.")
        return None

    try:
        conn = sqlite3.connect(DB_PATH)
        # Ensure the table name and column names match your schema.
        # The user provided 'options_indicator' as table name.
        # And columns: atm_iv_avg, call_delta, call_rho, call_theta, date, gamma, lastTradeDate,
        # pc_ratio, put_delta, put_rho, put_theta, skew_measure, term_structure_slope,
        # ticker, underlyingPrice, vega
        query = f"SELECT * FROM options_indicator WHERE ticker = ? ORDER BY date ASC"
        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()
        if df.empty:
            logger.info(f"No option indicator data found for ticker: {ticker} in {DB_PATH}")
        return df
    except sqlite3.Error as e:
        logger.error(f"SQLite error while fetching option indicators for {ticker} from {DB_PATH}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching option indicators for {ticker} from {DB_PATH}: {e}")
        return None
