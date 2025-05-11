

import logging
import sys


from mcp.server.fastmcp import FastMCP

from packages.investor_agent_lib import calc_tools, cnn_fng_tools, macro_tools, prompts, yfinance_tools


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
    mcp.add_tool(yfinance_tools.get_options)
    mcp.add_tool(yfinance_tools.get_price_history)
    mcp.add_tool(yfinance_tools.get_financial_statements)
    mcp.add_tool(yfinance_tools.get_institutional_holders)
    mcp.add_tool(yfinance_tools.get_earnings_history)
    mcp.add_tool(yfinance_tools.get_insider_trades)
    mcp.add_tool(yfinance_tools.get_ticker_news_tool)


    # Register CNN Fear & Greed resources and tools
    mcp.resource("cnn://fng/current")(cnn_fng_tools.get_current_fng)
    mcp.resource("cnn://fng/history")(cnn_fng_tools.get_historical_fng)

    mcp.add_tool(cnn_fng_tools.get_current_fng_tool)
    mcp.add_tool(cnn_fng_tools.get_historical_fng_tool)
    mcp.add_tool(cnn_fng_tools.analyze_fng_trend)

    # Register calculation tools
    mcp.add_tool(calc_tools.calculate)
    # mcp.add_tool(calc_tools.calc_ta)

    # Register macro tools
    mcp.add_tool(macro_tools.get_current_time)
    mcp.add_tool(macro_tools.get_fred_series)
    mcp.add_tool(macro_tools.search_fred_series)
    mcp.add_tool(macro_tools.cnbc_news_feed)
    mcp.add_tool(macro_tools.social_media_feed)

    # Register prompts
    mcp.prompt()(prompts.chacteristics)
    mcp.prompt()(prompts.mode_instructions)
    mcp.prompt()(prompts.investment_principles)
    mcp.prompt()(prompts.portfolio_construction_prompt)

    return mcp
