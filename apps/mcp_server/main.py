

import logging
import sys


from mcp.server.fastmcp import FastMCP

from packages.investor_agent_lib import prompts
from packages.investor_agent_lib.tools import holdings_tools, yfinance_tools
from packages.investor_agent_lib.tools import cnn_fng_tools
from packages.investor_agent_lib.tools import calculation_tools
from packages.investor_agent_lib.tools import macro_tools
from packages.investor_agent_lib.tools import option_tools

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Initialize MCP server
def create_mcp_application():
    # Initialize MCP server
    mcp = FastMCP("finance-tools-mcp", dependencies=["yfinance", "httpx", "pandas","ta-lib-easy"])

    # Register yfinance tools
    mcp.add_tool(yfinance_tools.get_ticker_data)
    mcp.add_tool(yfinance_tools.get_price_history)
    mcp.add_tool(yfinance_tools.get_financial_statements)
    mcp.add_tool(yfinance_tools.get_earnings_history)
    mcp.add_tool(yfinance_tools.get_ticker_news_tool)

    # Register option tools
    mcp.add_tool(option_tools.super_option_tool)

    # Register holdings analysis tools
    mcp.add_tool(holdings_tools.get_top25_holders)
    mcp.add_tool(holdings_tools.get_insider_trades)

    # Register CNN Fear & Greed resources and tools
    mcp.resource("cnn://fng/current")(cnn_fng_tools.get_overall_sentiment)
    mcp.resource("cnn://fng/history")(cnn_fng_tools.get_historical_fng)

    mcp.add_tool(cnn_fng_tools.get_overall_sentiment_tool)
    mcp.add_tool(cnn_fng_tools.get_historical_fng_tool)
    mcp.add_tool(cnn_fng_tools.analyze_fng_trend)

    # Register calculation tools
    mcp.add_tool(calculation_tools.calculate)

    # Register macro tools and resources
    mcp.add_tool(macro_tools.get_current_time)
    mcp.add_tool(macro_tools.get_fred_series)
    mcp.add_tool(macro_tools.search_fred_series)
    mcp.add_tool(macro_tools.cnbc_news_feed)
    mcp.add_tool(macro_tools.social_media_feed)

    mcp.resource("time://now")(macro_tools.get_current_time)
    mcp.resource("cnbc://news")(macro_tools.cnbc_news_feed)

    # Register prompts
    mcp.prompt()(prompts.chacteristics)
    mcp.prompt()(prompts.mode_instructions)
    mcp.prompt()(prompts.investment_principles)
    mcp.prompt()(prompts.portfolio_construction_prompt)

    return mcp
