import logging
from datetime import datetime

from packages.investor_agent_lib.services import macro_service


logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

def get_current_time() -> str:
    """Get the current time in ISO 8601 format."""
    now = datetime.now()
    return f"Today is {now.isoformat()}"

def get_fred_series(series_id):
    """Get a FRED series by its ID. However the data is not always the latest, so use with caution!!!"""
    return macro_service.get_fred_series(series_id)

def search_fred_series(query):
    """Search for the most popular FRED series by keyword. Useful for finding key data by name. Like GDP, CPI, etc. However the data is not always the latest.  """
    return macro_service.search_fred_series(query)

def cnbc_news_feed():
    """Get the latest breaking world news from CNBC, BBC, and SCMP. Useful to have an overview for the day. Include the Fed rate prediction from Fed watch and key macro indicators. """
    news = macro_service.breaking_news_feed()
    fred_watch_news = {
        "title": "Real Time Fed Rate Monitor: The most precise fed rate monitor based on CME Group 30-Day Fed Fund futures prices",
        "description": f"Fed rate prediction:\n {macro_service.cme_fedwatch_tool()}",
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    key_indicators = {
        "title": "Key Macro Indicators from stlouisfed.org",
        "description": f"{macro_service.key_macro_indicators()}",
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    news.append(fred_watch_news)
    news.append(key_indicators)
    return news

def social_media_feed(keywords: list[str] = None):
    """Get most discussed stocks and investments opinions from reddit. Useful to know what investors are talking about. 
    keywords is optional. Set keywords to match the specific topic you are interested in, by 'OR' operator, e.g. ['tsla', 'tesla'],  ['tesla', 'spacex'], ['AAPL', 'apple', 'tim cook', 'cook']
    No keywords will return the most discussed stocks and investments. """
    news = macro_service.reddit_stock_post(keywords)
    return news

