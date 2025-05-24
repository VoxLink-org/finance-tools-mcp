import datetime
import pandas as pd
import bs4
import curl_cffi
import packages.investor_agent_lib.services.yfinance_service as yf
from packages.investor_agent_lib.utils import cache


@cache.lru_with_ttl(ttl_seconds=3600)   
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



if __name__ == '__main__':
    df = get_digest_from_fintel('NBIS')
    print(df)