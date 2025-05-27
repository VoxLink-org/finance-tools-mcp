import curl_cffi
import pandas as pd
from packages.investor_agent_lib.services.yfinance_service import get_ticker_info
from packages.investor_agent_lib.utils import cache
import logging

logger = logging.getLogger(__name__)

@cache.lru_with_ttl(ttl_seconds=300)   
def get_sector_and_industry_valuation(ticker: str) -> dict | None:
    ticker_info = get_ticker_info(ticker)
    if not ticker_info:
        print(f"Ticker {ticker} not found") 
        return None

    try:
        sector = ticker_info['sector']
        industry = ticker_info['industry']

        # get the industry 
        url = f'https://finviz.com/groups.ashx?g=industry&v=120&o=name'
        response = curl_cffi.get(url)
        
        industry_df = pd.read_html(response.content, flavor='bs4', match='P/E')[0].drop(columns=['No.', 'Change', 'Volume', 'Market Cap'])
        industry_valuation = industry_df[industry_df['Name'] == industry].iloc[0].to_dict()

        # get the sector
        url = f'https://finviz.com/groups.ashx?g=sector&v=120&o=name'
        response = curl_cffi.get(url)
        
        sector_df = pd.read_html(response.content, flavor='bs4', match='P/E')[0].drop(columns=['No.', 'Change', 'Volume', 'Market Cap'])
        sector_valuation = sector_df[sector_df['Name'] == sector].iloc[0].to_dict()

        return {
            'sector_valuation': sector_valuation,
            'industry_valuation': industry_valuation
        }

    except Exception as e:
        print(e)
        return None


@cache.lru_with_ttl(ttl_seconds=300)
def get_insider_trades(ticker: str, limit: int = 30) -> pd.DataFrame | None:
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}&p=d'
        response = curl_cffi.get(url)
        df = pd.read_html(response.content, flavor='bs4', match='Insider Trading', attrs={'class': 'body-table'}, skiprows=1)[0]
        return df.head(limit) if df is not None else None

    except Exception as e:
        logger.error(f"Error retrieving insider trades for {ticker}: {e}")
        return None


if __name__ == '__main__':
    print(get_insider_trades('nvda'))
