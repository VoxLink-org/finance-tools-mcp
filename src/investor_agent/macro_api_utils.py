import fredapi as fr
import httpx
import os
import xml.etree.ElementTree as ET

import logging

logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get('FRED_API_KEY', "7fbed707a5c577c168c8610e8942d0d9")

def get_fred_series(series_id):

    fred = fr.Fred(api_key=FRED_API_KEY)

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

def cnbc_news_feed():
    cnbc = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114'
    ft = 'https://www.ft.com/personal-finance?format=rss'
    bbc = 'https://feeds.bbci.co.uk/news/world/rss.xml'

    news_items = []

    
    try:
        response = httpx.get(cnbc)
        root = ET.fromstring(response.text)
        
        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else 'No title'
            description = item.find('description').text if item.find('description') is not None else 'No description'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'No date'
            
            news_items.append({
                'title': title,
                'description': description,
                'date': pub_date
            })
    except Exception as e:
        logger.error(f"Error retrieving cnbc news feed: {e}")

        # 补充 ft 数据
        # response = httpx.get(ft)
        # root = ET.fromstring(response.text)
        
        # for item in root.findall('.//item'):
        #     title = item.find('title').text if item.find('title') is not None else 'No title'
        #     pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'No date'
            
        #     news_items.append({
        #         'title': title,
        #         'date': pub_date
        #     })
    try:        
        # 补充bbc
        response = httpx.get(bbc)
        root = ET.fromstring(response.text)
        
        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else 'No title'
            description = item.find('description').text if item.find('description') is not None else 'No description'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'No date'
            
            news_items.append({
                'title': title,
                'description': description,
                'date': pub_date
            })

    except Exception as e:
        logger.error(f"Error retrieving bbc news feed: {e}")

    return news_items