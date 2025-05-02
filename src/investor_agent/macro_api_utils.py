import fredapi as fr
import httpx
import os
import xml.etree.ElementTree as ET

import logging

logger = logging.getLogger(__name__)


def get_fred_series(series_id):
    FRED_API_KEY = os.environ.get('FRED_API_KEY')

    fred = fr.Fred(api_key=FRED_API_KEY)

    series = fred.get_series(series_id)
    return series.tail(10)

def search_fred_series(query):
    FRED_API_KEY = os.environ.get('FRED_API_KEY')

    fred = fr.Fred(api_key=FRED_API_KEY)

    series = fred.search(query, order_by='popularity', sort_order='desc', limit=6)
    # pick up the useful cols
    series = series[['id', 'title', 'observation_start', 'observation_end']]
    return series

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