import logging
from datetime import datetime

from packages.investor_agent_lib.analytics import holdings



logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

# --- Whale Wisdom Resources and Tools ---
def get_holdings_summary(ticker:str) -> str:
    """
    Analyze 13F 13D/G to get institutional holdings data from the last 6 months and return a formatted digest.
    """
    return holdings.analyze_institutional_holdings(ticker)
