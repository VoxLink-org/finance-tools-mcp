import json
import random
import fredapi as fr
import httpx
import os
import xml.etree.ElementTree as ET
import requests_cache 
import bs4

import logging

logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get('FRED_API_KEY', "7fbed707a5c577c168c8610e8942d0d9")

def get_fred_series(series_id):

    fred = fr.Fred(api_key=FRED_API_KEY)

    # Create a cached session with an expiration time
    with requests_cache.CachedSession('fred_cache', backend=requests_cache.SQLiteCache(':memory:'), expire_after=3600):
        # Use the cached session for the FRED API request
        series = fred.get_series(series_id)

        return series.tail(10)

def search_fred_series(query):

    params = {
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'search_text': query,
        'order_by': 'popularity',
        'sort_order': 'desc',
        'limit': 6
    }

    try:
        with httpx.Client() as client:
            response = client.get('https://api.stlouisfed.org/fred/series/search', params=params)
            response.raise_for_status()
            data = response.json()
        
        results = []

        for series in data.get('seriess', []):
            results.append({
                'id': series.get('id'),
                'title': series.get('title'),
                'frequency': series.get('frequency'),
                'last_updated': series.get('last_updated'),
                # 'notes': series.get('notes')
            })

        
        return {'results': results}
    except Exception as e:
        return {'error': str(e)}

def breaking_news_feed():
    # the world only needs three financial centers, ny, london, and hongkong
    # https://en.wikipedia.org/wiki/Global_Financial_Centres_Index
    cnbc = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114'
    bbc = 'https://feeds.bbci.co.uk/news/world/rss.xml'
    scmp = 'https://www.scmp.com/rss/91/feed'

    news_items = []

    
    try:
        response = httpx.get(cnbc)
        root = ET.fromstring(response.text)
        
        news_items_for_pickup = []

        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else 'No title'
            description = item.find('description').text if item.find('description') is not None else 'No description'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'No date'
            
            news_items_for_pickup.append({
                'title': title,
                'description': description,
                'date': pub_date
            })
        
        news_items.append(news_items_for_pickup) 
        
    except Exception as e:
        logger.error(f"Error retrieving cnbc news feed: {e}")

    try:        
        # 补充bbc
        response = httpx.get(bbc)
        root = ET.fromstring(response.text)
        
        news_items_for_pickup = []


        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else 'No title'
            description = item.find('description').text if item.find('description') is not None else 'No description'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'No date'
            
            news_items_for_pickup.append({
                'title': title,
                'description': description,
                'date': pub_date
            })
        
        news_items.append(random.choices(news_items_for_pickup, k=6))

    except Exception as e:
        logger.error(f"Error retrieving bbc news feed: {e}")

    try:
        response = httpx.get(scmp)
        root = ET.fromstring(response.text)
        
        news_items_for_pickup = []

        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else 'No title'
            description = item.find('description').text if item.find('description') is not None else 'No description'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'No date'
            
            news_items_for_pickup.append({
                'title': title,
                'description': description,
                'date': pub_date
            })

        news_items.append(random.choices(news_items_for_pickup, k=6))
    except Exception as e:
        logger.error(f"Error retrieving scmp news feed: {e}")
        


    return news_items

def cme_fedwatch_tool():
    url = 'https://www.investing.com/central-banks/fed-rate-monitor'

    try:
        response = httpx.get(url)
        root = bs4.BeautifulSoup(response.text, 'html.parser')
        
        # Extract first 2 Fed rate decision cards
        cards = root.find_all('div', class_='cardWrapper')[:2]
        results = []
        
        for card in cards:
            # Extract meeting date from fedRateDate
            meeting_date = card.find('div', class_='fedRateDate').get_text(strip=True)
            
            # Extract detailed meeting time and future price from infoFed
            info_fed = card.find('div', class_='infoFed')
            meeting_time = info_fed.find('i').get_text(strip=True)
            future_price = info_fed.find_all('i')[1].get_text(strip=True)
            
            # Extract probability table data
            probabilities = []
            table = card.find('table', class_='fedRateTbl')
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                probabilities.append({
                    'target_rate': cells[0].get_text(strip=True),
                    'current_prob': cells[1].get_text(strip=True),
                    # 'prev_day_prob': cells[2].get_text(strip=True),
                    # 'prev_week_prob': cells[3].get_text(strip=True)
                })
            
            # Extract update time
            update_time = card.find('div', class_='fedUpdate').get_text(strip=True)
            
            results.append({
                # 'meeting_date': meeting_date,
                'meeting_time': meeting_time,
                # 'future_price': future_price,
                'probabilities': probabilities,
                # 'update_time': update_time
            })

        current_fed_rate = ' '.join([td.get_text(strip=True) for td in root.find_all('tr', class_='first')[0].find_all('td')])

        if isinstance(results, (dict, list)):
            return json.dumps({
            "current_fed_rate": current_fed_rate,
            "predict":    results
            }, indent=2)
        return f"predict: {str(results)} current_fed_rate: {current_fed_rate}"
    
    except Exception as e:
        logger.error(f"Error retrieving fed watch: {e}")


def reddit_stock_post():
    
    

    def get_rss(url):
        results = []
        try:
            response = httpx.get(url)
            root = bs4.BeautifulSoup(response.text, 'xml')
            entries = root.find_all('entry')
            for entry in entries:
                content_html = entry.content.text if entry.content else ''
                content_text = bs4.BeautifulSoup(content_html, 'html.parser').get_text()
                content_words = content_text.split()
                if content_text.find('This post contains content not supported on old Reddit') != -1:
                    continue
                if len(content_words) < 20:
                    continue

                results.append({
                    'title': entry.title.text if entry.title else '',
                    'content': ' '.join(content_words[:100] + ['...']) if len(content_words) > 100 else content_text,
                    'updated': entry.updated.text if entry.updated else ''
                })
            
            
        except Exception as e:
            logger.error(f"Error retrieving reddit stock post: {e}")
        return results

    url1 = 'https://www.reddit.com/r/stocks/.rss'

    url2 = "https://www.reddit.com/r/wallstreetbets/.rss"

    r1 = get_rss(url1)
    r2 = get_rss(url2)

    results = random.choices(r1, k=14) + random.choices(r2, k=7)

    random.shuffle(results)
    return results
