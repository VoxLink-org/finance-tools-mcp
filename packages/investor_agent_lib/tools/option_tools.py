import logging
from datetime import datetime
from tabulate import tabulate
from packages.investor_agent_lib.services import enhance_option_service
from packages.investor_agent_lib.analytics import option

logger = logging.getLogger(__name__)

def super_option_tool(ticker:str) -> str:
    """
    Analyzes and summarizes option data for a given ticker.

    This function retrieves option indicators and Greeks for the specified ticker,
    generates a digest summarizing key metrics, and formats a table of key option
    data including last trade date, strike, option type, open interest, volume,
    and implied volatility.

    Args:
        ticker: Stock ticker symbol.

    """

    data = enhance_option_service.get_option_greeks_and_ind(ticker)
    digest = option.analyze_option_indicators_greeks(data)
    pickup = ['lastTradeDate', 'strike', 'expiryDate','optionType', 'openInterest', 'volume', 'impliedVolatility']
    key_options = tabulate(enhance_option_service.get_key_options(ticker)[pickup], headers=pickup, tablefmt="simple", showindex=False)

    

    return digest + "\n\n========Key Options=========\n\n" + key_options