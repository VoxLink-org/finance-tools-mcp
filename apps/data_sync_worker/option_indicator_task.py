import sqlite3

import pandas as pd
from prefect import task, flow, get_run_logger

from packages.investor_agent_lib.options import option_indicators  

from config.paths import DATA_DIR

db_path = DATA_DIR / "options_indicator.db"
table_name = "options_indicator"

@task(name="get-option-indicator-data")
def get_options_indicator_task(
    ticker: str
)->pd.DataFrame:
    basic_indicators = option_indicators.calculate_indicators(ticker)
    greeks = option_indicators.calculate_greeks(ticker)
    # Merge two dictionaries and convert to DataFrame
    merged_dict = {**basic_indicators, **greeks}
    return pd.DataFrame([{k: merged_dict[k] for k in sorted(merged_dict)}])


@task(name="validate-option-indicator-data")
def validate_option_indicator_task(options_df: pd.DataFrame) -> bool:
    logger = get_run_logger()

    required_columns = {
        "call_delta": float,
        "put_delta": float,
        "gamma": float,
        "call_theta": float,
        "put_theta": float,
        "vega": float,
        "call_rho": float,
        "put_rho": float,
        'atm_iv_avg': float,
        'skew_measure': float,
        'term_structure_slope': float,
        'pc_ratio': float,
        'underlyingPrice': float,
        "ticker": str,
        "date": str,
        "lastTradeDate": str
    }
    
    # Check for empty DataFrame
    if options_df.empty:
        logger.error("No options data found")
        return False
        
    # Check if all required columns exist
    missing_columns = set(required_columns.keys()) - set(options_df.columns)
    if missing_columns:
        logger.error(f"Missing columns: {', '.join(missing_columns)}")
        return False
        
    # Check data types and NaN values for each column
    for col, expected_type in required_columns.items():
        # Check if column contains any NaN values
        if options_df[col].isna().any():
            logger.error(f"Column {col} contains NaN values")
            logger.error(options_df)
            return False
            
            
    return True

@task(name="save-option-indicator-data")
def save_option_indicator_task(
    options_df: pd.DataFrame,
    db_path: str = db_path,
    table_name: str = table_name
) -> bool:
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
    