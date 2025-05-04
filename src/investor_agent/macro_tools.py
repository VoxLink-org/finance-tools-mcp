import logging
from datetime import datetime

from . import macro_api_utils

logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

def get_current_time() -> str:
    """Get the current time in ISO 8601 format."""
    now = datetime.now()
    return now.isoformat()

def get_fred_series(series_id):
    """Get a FRED series by its ID. However the data is not always the latest, so use with caution!!!"""
    return macro_api_utils.get_fred_series(series_id)

def search_fred_series(query):
    """Search for the most popular FRED series by keyword. Useful for finding key data by name. Like GDP, CPI, etc. However the data is not always the latest.  """
    return macro_api_utils.search_fred_series(query)

def cnbc_news_feed():
    """Get the latest breaking world news from CNBC, BBC, and SCMP. Useful to have an overview for the day."""
    return macro_api_utils.breaking_news_feed()