import datetime
import re
import pandas as pd
import bs4
import curl_cffi
import packages.investor_agent_lib.services.yfinance_service as yf
from packages.investor_agent_lib.utils import cache


@cache.lru_with_ttl(ttl_seconds=300)   
def get_digest_from_fintel(ticker: str):
    url = f'https://fintel.io/card/activists/us/{ticker}'
    response = curl_cffi.get(url, impersonate="chrome")
    
    activists = pd.read_html(response.content, flavor='bs4', match='Investor')[0]

    url = f'https://fintel.io/card/top.investors/us/{ticker}'
    response = curl_cffi.get(url, impersonate="chrome")
    data = response.content
    soup = bs4.BeautifulSoup(data, 'html.parser')
    
    # Find the Top Investors card
    title = soup.find('h5', class_='card-title', string='Top Investors')
    summary_text = ''
    if title:
        card_text = title.find_next('p', class_='card-text')
        if card_text:
            summary_text = card_text.get_text(' ', strip=True)
    
    top_investors = pd.read_html(response.content, flavor='bs4', attrs={'id': 'table-top-owners'})[0]
    
    return {
        'summary_text': summary_text,
        'activists': activists,
        'investors': top_investors,
    }

@cache.lru_with_ttl(ttl_seconds=3600)   
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
        
    # find by company name
    info = yf.get_ticker_info(ticker)
    displayName = info.get('displayName', ticker)
    print(displayName)


    url = f'https://whalewisdom.com/search/filer_stock_autocomplete2?filer_restrictions=3&term={displayName}'
    # with requests_cache.enabled('whalewisdom', backend=requests_cache.SQLiteCache(':memory:'), expire_after=3600):
    response = curl_cffi.get(url, impersonate="chrome", headers={
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "origin": "https://whalewisdom.com",
        "sec-ch-ua": '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/244.178.44.111 Safari/537.36",
    })        
    data = response.json()
    if not data or not isinstance(data, list):
        raise ValueError(f"No results found for ticker: {ticker}")
    target = None
    print(data)
    for item in data:
        match = re.search(rf"{displayName}", item['label'], re.IGNORECASE)
        if match:
            print(item)
            target = item['id']
            break
    return target

@cache.lru_with_ttl(ttl_seconds=300)   
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
    print(code)
    url = f'https://whalewisdom.com/stock/holdings?id={code}&q1=-1&change_filter=&mv_range=&perc_range=&rank_range=&sc=true&sort=current_shares&order=desc&offset=0&limit=25'
    response = curl_cffi.get(url, impersonate="chrome", headers={
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "origin": "https://whalewisdom.com",
        "sec-ch-ua": '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/244.178.44.111 Safari/537.36",
    })   
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
    df = pd.DataFrame(holdings)[['name', 'current_shares', 'position_change_type', 'percent_change', 'source_date', 'filing_date', 'shares_change']]
    # sort by position_change_type
    df['source_date'] = pd.to_datetime(df['source_date'])
    df["current_shares"] = pd.to_numeric(df["current_shares"], errors='coerce')
    df["percent_change_in_%"] = pd.to_numeric(df["percent_change"], errors='coerce')
    df = df.sort_values(by='current_shares', ascending=False)
    return df




if __name__ == '__main__':
    df = get_whalewisdom_holdings('nbis')
    print(df)