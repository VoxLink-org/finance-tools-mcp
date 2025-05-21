import datetime
import pandas as pd
import bs4
import curl_cffi
import packages.investor_agent_lib.services.yfinance_service as yf

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
    displayName = yf.get_ticker_info(ticker)['displayName']

    url = f'https://whalewisdom.com/search/filer_stock_autocomplete2?filer_restrictions=3&term={displayName}'
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
    print(code)
    url = f'https://whalewisdom.com/stock/holdings?id={code}&q1=-1&change_filter=&mv_range=&perc_range=&rank_range=&sc=true&sort=current_percent_of_portfolio&order=desc&offset=0&limit=100'
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
    
def extract_fintel_activists_table(soup: bs4.BeautifulSoup) -> pd.DataFrame:
    """Extract 13D/G holdings data from fintel.io HTML.
    
    Args:
        soup: BeautifulSoup object of the fintel.io page
        
    Returns:
        List of dictionaries containing extracted holdings data
    """
    holdings = []
    table_div = soup.find('div', id='13-D-security-view')
    if not table_div:
        return holdings
        
    table = table_div.find('table', class_='table')
    if not table:
        return holdings
        
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 9:
            continue
            
        holdings.append({
            'file_date': cells[0].get_text(strip=True),
            'form': cells[1].get_text(strip=True),
            'investor': cells[2].get_text(strip=True),
            'prev_shares': int(cells[4].get_text(strip=True).replace(',', '')),
            'latest_shares': int(cells[5].get_text(strip=True).replace(',', '')),
            'share_change_pct': float(cells[6].get_text(strip=True)),
            'ownership_pct': float(cells[7].get_text(strip=True)),
            'ownership_change_pct': float(cells[8].get_text(strip=True))
        })
    
    return pd.DataFrame(holdings)


def get_digest_from_fintel(ticker: str):
    url = f'https://fintel.io/card/activists/us/{ticker}'
    response = curl_cffi.get(url, impersonate="chrome")
    data = response.content
    soup = bs4.BeautifulSoup(data, 'html.parser')
    
    activists = extract_fintel_activists_table(soup)

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
    
    top_investors = extract_top_investors_table(soup)
    
    return {
        'summary_text': summary_text,
        'activists': activists,
        'investors': top_investors,
    }

def extract_top_investors_table(soup: bs4.BeautifulSoup) -> pd.DataFrame:
    """Extract top investors data from fintel.io HTML.
    
    Args:
        soup: BeautifulSoup object of the fintel.io page
        
    Returns:
        List of dictionaries containing extracted top investors data
    """
    investors = []
    table = soup.find('table', id='table-top-owners')
    if not table:
        return investors
        
    def safe_float(text, default=0.0):
        text = text.strip()
        if not text:
            return default
        try:
            return float(text.replace('%', '').replace('$', '').replace(',', ''))
        except ValueError:
            return default
            
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 9:
            continue
            
        investors.append({
            'owner': cells[0].get_text(strip=True),
            'form': cells[1].get_text(strip=True),
            'shares_mm': safe_float(cells[3].get_text()),
            'share_change_pct': safe_float(cells[4].get_text()),
            'value_mm': safe_float(cells[5].get_text()),
            'value_change_pct': safe_float(cells[6].get_text()),
            'portfolio_pct': safe_float(cells[7].get_text()),
            'portfolio_change_pct': safe_float(cells[8].get_text())
        })
    
    return pd.DataFrame(investors)




if __name__ == '__main__':
    df = get_digest_from_fintel('NBIS')
    print(df)