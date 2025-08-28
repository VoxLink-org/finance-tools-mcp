import datetime
from typing import List, Literal

import logging
import bs4
import httpx
import pandas as pd
from packages.investor_agent_lib.services import yfinance_service
from config.my_paths import DATA_DIR

logger = logging.getLogger(__name__)

def get_most_active_tickers_from_tradingview(prefix=False) -> List[str]:
    url = 'https://www.tradingview.com/markets/stocks-usa/market-movers-active/'
    try:
        response = httpx.get(url)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        tickers = [option['data-rowkey'] for option in soup.select('tr.listRow')]
        if not prefix:
            tickers = [ticker.split(':')[1] for ticker in tickers]
        dont_want = ['BRK.A','APAD','ETHM','AHL','BMNR','CRCL']
        return [ticker for ticker in tickers if ticker not in dont_want]
    except Exception as e:
        logger.error(f"Error getting most active tickers: {e}")
        return []
    
    
def fetch_panel_data(period: Literal["1d", "5d", "10d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "5d", end_date: pd.Timestamp=None) -> pd.DataFrame:
            
    tickers = get_most_active_tickers_from_tradingview()
    end_date = pd.Timestamp.utcnow() if end_date is None else end_date

    if period == "1d":
        start_date = end_date - pd.Timedelta(days=1)
    elif period == "5d":
        start_date = end_date - pd.Timedelta(days=5)
    elif period == "10d":
        start_date = end_date - pd.Timedelta(days=10)
    elif period == "1mo":
        start_date = end_date - pd.Timedelta(days=30)
    elif period == "3mo":
        start_date = end_date - pd.Timedelta(days=90)
    elif period == "6mo":
        start_date = end_date - pd.Timedelta(days=180)
    elif period == "1y":
        start_date = end_date - pd.Timedelta(days=365)
    elif period == "2y":
        start_date = end_date - pd.Timedelta(days=730)
    elif period == "5y":
        start_date = end_date - pd.Timedelta(days=1825)
    elif period == "10y":
        start_date = end_date - pd.Timedelta(days=3650)
    elif period == "ytd":
        start_date = pd.Timestamp.utcnow().replace(month=1, day=1)
    elif period == "max":
        start_date = pd.Timestamp.min
        
    # try to load from file first
    data_path = DATA_DIR / f"most_active_tickers_{start_date}_{end_date}.pkl"
    use_cache = True
    if use_cache and data_path.exists():
        try:
            df = pd.read_pickle(data_path)
            if not df.empty:
                # Convert index to datetime if it's date strings
                if isinstance(df.index[0], str):
                    df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")

    data = yfinance_service.download_history(
        tickers,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1d"
    )
    
    if data is None or data.empty:
        raise "No data available for the given period"
    
    
    # Convert from multi-index to single index (long format)
    # The initial DataFrame has a MultiIndex for columns: (Feature, Ticker)
    # We want: date, ticker, feature_1, feature_2, ...
    
    # Stack the 'Ticker' level from columns to rows, creating a new index level
    data_stacked = data.stack(level='Ticker').reset_index()
    
    # Rename the columns to the desired format
    data_stacked = data_stacked.rename(columns={'level_0': 'Date', 'level_1': 'Ticker'})
    
    # Rename the remaining feature columns (e.g., 'Price', 'Close', 'High', 'Open', 'Volume')
    # This assumes that the remaining columns are the features.
    # The user's desired format shows 'feature_1', 'feature_2', etc.
    # For now, I'll keep the original feature names, but they can be renamed later if needed.
    
    # The 'Date' column is already in datetime format from the original index.
    # Ensure 'Date' is named 'date' and 'Ticker' is named 'ticker'
    data_stacked = data_stacked.rename(columns={'Date': 'date', 'Ticker': 'ticker'})
    
    print(data_stacked.head(10))
    
    # save to file
    try:
        data_stacked.to_pickle(data_path)
    except Exception as e:
        logger.error(f"Error saving data to {data_path}: {e}")
    
    return data_stacked


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "5d"
    data = fetch_panel_data(ticker)
    print(data.tail(10))
