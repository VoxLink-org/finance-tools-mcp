import random
import fredapi as fr
import httpx
import os
import xml.etree.ElementTree as ET
import requests_cache 
import logging

logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get('FRED_API_KEY', "7fbed707a5c577c168c8610e8942d0d9")

def get_fred_series(series_id):

    fred = fr.Fred(api_key=FRED_API_KEY)

    # Create a cached session with an expiration time
    with requests_cache.CachedSession('fred_cache', expire_after=3600):
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