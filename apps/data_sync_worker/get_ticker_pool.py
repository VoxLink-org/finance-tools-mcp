from typing import List

import logging
import bs4
import httpx
from config.paths import DATA_DIR

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
        return tickers
    except Exception as e:
        logger.error(f"Error getting most active tickers: {e}")
        return []
    
def get_supplied_tickers():
    text_file = DATA_DIR / "tickers.txt"
    ticker = []
    with open(text_file, "r") as f:
        for line in f:
            ticker.append(line.strip().upper())
    return ticker

def get_ticker_pool():
    t1 = get_supplied_tickers()
    t2 = get_most_active_tickers_from_tradingview()
    return list(set(t1 + t2))
