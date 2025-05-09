

import logging
import sys

import argparse


from mcp.server.fastmcp import FastMCP

# Import the new tool and prompt modules
from . import option_analyze
from . import yfinance_tools
from . import cnn_fng_tools
from . import calc_tools
from . import macro_tools
from . import prompts
from . import sse_server

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Initialize MCP server
def main():
    # Initialize MCP server
    mcp = FastMCP("finance-tools-mcp", dependencies=["yfinance", "httpx", "pandas","ta-lib-easy"])

    # Register yfinance tools
    mcp.add_tool(yfinance_tools.get_ticker_data)
    # mcp.add_tool(yfinance_tools.get_options)
    mcp.add_tool(yfinance_tools.get_price_history)
    mcp.add_tool(yfinance_tools.get_financial_statements)
    mcp.add_tool(yfinance_tools.get_institutional_holders)
    mcp.add_tool(yfinance_tools.get_earnings_history)
    mcp.add_tool(yfinance_tools.get_insider_trades)
    mcp.add_tool(yfinance_tools.get_ticker_news_tool)

    # Register Option analyze
    mcp.add_tool(option_analyze.analyze_options_v2)

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

    # Add argument parsing
    parser = argparse.ArgumentParser(description="Run the Finance Tools MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol to use (stdio or sse)",
    )

    # Parse arguments and run the server
    args = parser.parse_args()
    if args.transport == "sse":
        sse_server.run_sse_server(mcp)
    else:
        mcp.run(transport=args.transport)

if __name__ == "__main__":
    main()

