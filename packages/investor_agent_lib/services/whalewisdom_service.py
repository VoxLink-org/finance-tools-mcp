import httpx
import bs4
import requests_cache
import datetime
import pandas as pd
import requests
import curl_cffi

def get_whalewisdom_stock_code(ticker: str) -> str:
    """Get WhaleWisdom stock ID for a given ticker symbol.
    
    Args:
        ticker: Stock ticker symbol to look up
        
    Returns:
        WhaleWisdom stock ID as string
        
    Raises:
        ValueError: If ticker is empty or no results found
        httpx.HTTPStatusError: If HTTP request fails
    """
    if not ticker:
        raise ValueError("Ticker cannot be empty")
        
    url = f'https://whalewisdom.com/search/filer_stock_autocomplete2?filer_restrictions=3&term={ticker}'
    # with requests_cache.enabled('whalewisdom', backend=requests_cache.SQLiteCache(':memory:'), expire_after=3600):
    response = curl_cffi.get(url, impersonate="chrome")        
    data = response.json()
    if not data or not isinstance(data, list):
        raise ValueError(f"No results found for ticker: {ticker}")
        
    return data[0]['id']

def get_whalewisdom_holdings(ticker: str)->pd.DataFrame:
    """
    Get ticker holdings for a given ticker symbol.
    
    Args:
        ticker: Stock ticker symbol to look up
        
    Returns:
        List of ticker holdings from WhaleWisdom.com as a pandas DataFrame, sorted by percent ownership.
        
    Raises:
        ValueError: If ticker is empty or no results found
        httpx.HTTPStatusError: If HTTP request fails
    """
    code = get_whalewisdom_stock_code(ticker)
    url = f'https://whalewisdom.com/stock/holdings?id={code}&q1=-1&change_filter=&mv_range=&perc_range=&rank_range=&sc=true&sort=percent_ownership&order=desc&offset=0&limit=100'
    response = curl_cffi.get(url, impersonate="chrome")   
    data = response.json()
    holdings = data['rows']
    # name
    # percent_change
    # position_change_type
    # percent_ownership
    # source_date
    # filing_date
    now = datetime.datetime.now()
    six_months_ago = now - datetime.timedelta(days=180)
    holdings = [h for h in holdings if datetime.datetime.fromisoformat(h['source_date']) > six_months_ago]
    # pick up the cols of interest
    df = pd.DataFrame(holdings)[['name', 'percent_ownership', 'position_change_type', 'percent_change', 'source_date', 'filing_date', 'shares_change']]
    # sort by position_change_type
    df['source_date'] = pd.to_datetime(df['source_date'])
    df["percent_ownership"] = pd.to_numeric(df["percent_ownership"], errors='coerce')/100
    df["percent_change"] = pd.to_numeric(df["percent_change"], errors='coerce')/100
    df = df.sort_values(by='percent_ownership', ascending=False)
    return df
    

if __name__ == '__main__':
    df = get_whalewisdom_holdings('AAPL')
    print(df)