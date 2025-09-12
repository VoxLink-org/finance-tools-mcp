

import logging
import sys

from starlette.responses import JSONResponse
from starlette.requests import Request

from mcp.server.fastmcp import FastMCP

from packages.investor_agent_lib import prompts
from packages.investor_agent_lib.tools import holdings_tools, yfinance_tools
from packages.investor_agent_lib.tools import cnn_fng_tools
from packages.investor_agent_lib.tools import calculation_tools
from packages.investor_agent_lib.tools import macro_tools
from packages.investor_agent_lib.tools import option_tools
from packages.investor_agent_lib.tools import predict_tools

from mcp.server.auth.settings import AuthSettings

from apps.mcp_server.simple_token_verifier import SimpleTokenVerifier, check_context


# Create an instance of the SimpleTokenVerifier
token_verifier = SimpleTokenVerifier()

# Configure AuthSettings
auth_settings = AuthSettings(
    required_scopes=["read", "write"],
    issuer_url="https://www.unkey.com",
    resource_server_url="https://example.com/api",
)


def health_check(request: Request, **kwargs)->JSONResponse :
    print(kwargs)
    token = request.headers.get("Authorization")
    return JSONResponse(
        {"status": "ok", "token": token},
    )


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Initialize MCP server
def create_mcp_application(port=8000):
    # Initialize MCP server
    mcp = FastMCP("finance-tools-mcp", dependencies=["yfinance", "httpx", "pandas","ta-lib-easy"],
                  token_verifier=token_verifier,
                  auth=auth_settings,
                  port=port
                  )

    print(mcp.settings)
    
    # Register yfinance tools
    mcp.add_tool(check_context(['read','write'])(yfinance_tools.get_ticker_data))
    mcp.add_tool(yfinance_tools.get_price_history)
    mcp.add_tool(yfinance_tools.get_financial_statements)
    mcp.add_tool(yfinance_tools.get_earnings_history)
    mcp.add_tool(yfinance_tools.get_ticker_news_tool)

    # Register option tools
    mcp.add_tool(option_tools.super_option_tool)

    # Register prediction tools
    mcp.add_tool(predict_tools.price_prediction)

    # Register holdings analysis tools
    mcp.add_tool(holdings_tools.get_top25_holders)
    mcp.add_tool(holdings_tools.get_insider_trades)

    # Register CNN Fear & Greed tools

    mcp.add_tool(cnn_fng_tools.get_overall_sentiment_tool)
    # mcp.add_tool(cnn_fng_tools.get_historical_fng_tool)
    # mcp.add_tool(cnn_fng_tools.analyze_fng_trend)

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
    
    
    # Register other routes
    try:
        from . import train_service
        mcp.custom_route("/train", methods=["GET","POST","OPTIONS"])(train_service.trigger_train_model)
        mcp.custom_route("/health", methods=["GET","HEAD","OPTIONS"])(health_check)
        logger.info("Registered train service routes at /train and billing routes at /billing")
    except:
        logger.error("Failed to register train service routes")
        pass
    
    return mcp
