from datetime import datetime
from typing import Literal, Optional
import logging
import pandas as pd

from packages.investor_agent_lib.services import yfinance_service


def contract_formatter(contract_symbol: str) -> dict[str, str]:
    """Format Yahoo Finance option contract symbol into human-readable string.
    
    Args:
        contract_symbol: Option contract symbol in Yahoo format (e.g. "AAPL220121C00150000")
        
    Returns:
        dict: Option type, strike price, and expiration date
        
    Raises:
        ValueError: If input is not a valid Yahoo option contract symbol
    """
    if not contract_symbol or len(contract_symbol) < 15:
        raise ValueError(f"Invalid contract symbol: {contract_symbol}")
        
    try:
        # Parse expiry (6 digits after ticker)
        expiry_str = contract_symbol[-15:-9]
        expiry = f"20{expiry_str[:2]}-{expiry_str[2:4]}-{expiry_str[4:6]}"
        
        # Parse option type (C/P)
        option_type = contract_symbol[-9]
        if option_type not in ('C', 'P'):
            raise ValueError(f"Invalid option type: {option_type}")
            
        # Parse strike price (remaining digits)
        strike_str = contract_symbol[-8:]
        strike = float(strike_str) / 1000  # Convert to decimal
        
        return {"option_type": option_type, "strike": strike, "expiry": expiry}
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse contract symbol {contract_symbol}: {str(e)}")


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


def get_raw_options(
    ticker_symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    strike_lower: float | None = None,
    strike_upper: float | None = None,
    option_type: Literal["C", "P"] | None = None,
) -> pd.DataFrame:
    """Get options with bucketed selection. Dates: YYYY-MM-DD. Type: C=calls, P=puts."""
    underlyingPrice = yfinance_service.get_current_price(ticker_symbol)
    
    logger = logging.getLogger(__name__)

    logger.info(f"Current stock price for {ticker_symbol}: {underlyingPrice}")


    try:
        df, error = yfinance_service.get_filtered_options(
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
        df["optionType"] = df["contractSymbol"].apply(lambda x: contract_formatter(x)["option_type"])
        

        return df
    except Exception as e:
        logger.error(f"Error getting options data for {ticker_symbol}: {e}")
        return f"Failed to retrieve options data for {ticker_symbol}: {str(e)}"

