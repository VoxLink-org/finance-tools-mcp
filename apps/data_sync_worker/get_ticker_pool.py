from typing import List

from yfinance import screen
from yfinance_utils import session, logger

import bs4
import httpx

def get_most_active_tickers(n=100) -> list[str]:
    """Get the most active tickers from Yahoo Finance using screener.
    
    Args:
        n: Number of tickers to return (max 250)
    
    Returns:
        List of ticker symbols
    """
    try:
        # Get most active stocks using predefined screener
        result = screen("most_actives", count=n, session=session)
        # Extract tickers from results
        quotes = result.get("quotes", [])
        return [q["symbol"] for q in quotes]
    except Exception as e:
        logger.error(f"Error getting most active tickers: {e}")
        return []

def get_most_active_tickers_from_tradingview(prefix=False) -> List[str]:
    url = 'https://www.tradingview.com/markets/stocks-usa/market-movers-active/'

    try:
        response = httpx.get(url)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        tickers = [option['data-rowkey'] for option in soup.select('tr.listRow')]
        if not prefix:
            tickers = [ticker.split(':')[1] for ticker in tickers]
        return tickers
    except Exception as e:
        logger.error(f"Error getting most active tickers: {e}")
        return []
    
def get_supplied_tickers():
    ticker = []
    with open("./src/tickers.txt", "r") as f:
        for line in f:
            ticker.append(line.strip().upper())
    return ticker

def get_ticker_pool():
    t1 = get_supplied_tickers()
    t2 = get_most_active_tickers_from_tradingview()
    return list(set(t1 + t2))

if __name__ == "__main__":
    tickers = get_ticker_pool()
    print(tickers)