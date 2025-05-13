import sqlite3
from datetime import datetime
from typing import Literal, Optional
import logging
import pandas as pd
from prefect import task, flow, get_run_logger

from packages.investor_agent_lib.options import option_selection

from config.paths import DATA_DIR

db_path =  DATA_DIR / "options_data.db"
table_name = "options"



@task(name="get-options-data", retries=2, retry_delay_seconds=60)
def get_options_task(
    ticker_symbol: str,
    num_options: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    strike_lower: float | None = None,
    strike_upper: float | None = None,
    option_type: Literal["C", "P"] | None = None,
) -> pd.DataFrame:
    """Prefect task to get options with highest open interest.
    
    Args:
        ticker_symbol: Stock ticker symbol
        num_options: Number of options to return (default: 10)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        strike_lower: Minimum strike price
        strike_upper: Maximum strike price
        option_type: 'C' for calls or 'P' for puts
    
    Returns:
        DataFrame with options data sorted by open interest
    """
    logger = get_run_logger()
    try:
        data = option_selection.get_raw_options(
            ticker_symbol,
            start_date,
            end_date,
            strike_lower,
            strike_upper,
            option_type
        )

        underlyingPrice = data["underlyingPrice"].iloc[0]
        logger.info(f"Current stock price for {ticker_symbol}: {underlyingPrice}")

        return option_selection.create_snapshot(data, underlyingPrice)

    except Exception as e:
        logger.error(f"Task failed for {ticker_symbol}: {str(e)}")
        return None

@task(name="validate-options-data")
def validate_options_task(options_df: pd.DataFrame) -> bool:
    """Validate options data meets expected schema and quality.
    
    Args:
        options_df: DataFrame containing options data
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger = get_run_logger()
    required_columns = {
        "optionType": str,
        "tickerSymbol": str,
        "snapshotDate": str,
        "contractSymbol": str,
        "strike": float,
        "lastPrice": float,
        "lastTradeDate": datetime,
        "change": float,
        "volume": int,
        "openInterest": int,
        "impliedVolatility": float,
        "expiryDate": datetime,
    }

    logger.info(options_df)
    
    if not isinstance(options_df, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame")
        logger.error(f"Input type: {type(options_df)}")
        logger.error(f"Input value: {options_df}")
        return False
    
    

    missing_cols = set(required_columns.keys()) - set(options_df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    return True            


@task(name="save-options-data")
def save_options_task(
    options_df: pd.DataFrame,
    db_path: str = db_path,
    table_name: str = table_name
) -> bool:
    """Save options data to SQLite database.
    
    Args:
        options_df: DataFrame containing options data
        db_path: Path to SQLite database file
        table_name: Name of table to store data in
        
    Returns:
        bool: True if save succeeded, False otherwise
    """
    logger = get_run_logger()
    
    try:
        conn = sqlite3.connect(db_path)
        options_df.to_sql(
            name=table_name,
            con=conn,
            if_exists='append',
            index=False
        )
        conn.close()
        logger.info(f"Successfully saved {len(options_df)} records to {db_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save options data: {str(e)}")
        return False
    
@task(name="clean-up-old-options-data")
def clean_up_the_days_before_10days() -> int:
    """Clean up options data older than 10 days.
    
    Returns:
        int: Number of rows deleted
    """
    logger = get_run_logger()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM options WHERE snapshotDate < date('now', '-10 days')")
            row_count = cursor.rowcount
            conn.commit()
            logger.info(f"Deleted {row_count} old options records")
            return row_count
    except sqlite3.Error as e:
        logger.error(f"Database error during cleanup: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")
        

