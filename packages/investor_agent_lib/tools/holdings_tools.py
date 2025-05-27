import logging
from datetime import datetime

from packages.investor_agent_lib.analytics import holdings
from packages.investor_agent_lib.services import finviz_service
from packages.investor_agent_lib.services import yfinance_service

logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

# --- Other Resources and Tools ---
def get_holdings_summary(ticker:str) -> str:
    """
    Analyze 13F 13D/G to get institutional holdings data from the last 6 months and return a formatted digest.
    """
    return holdings.analyze_institutional_holdings_v2(ticker)

def get_top25_holders(ticker: str) -> str:
    """
    Get top 25 institutional holders and their changes for a given stock ticker.
    """
    return holdings.get_top25_holder(ticker)

def get_insider_trades(ticker: str) -> str:
    """
    Get recent insider trading activity.
    """
    df = finviz_service.get_insider_trades(ticker)
    if df is None or df.empty:
        return f"No insider trading data found for {ticker}"

    info = yfinance_service.get_ticker_info(ticker)
    companyOfficers = info.get('companyOfficers', [])

    digest = f"COMPANY OFFICERS FOR {ticker}:\n"
    for officer in companyOfficers:
        digest += f"- {officer['name']} ({officer['title']})\n"

    return digest + df.drop(columns=['Value ($)', '#Shares Total', 'SEC Form 4']).to_markdown(index=False)