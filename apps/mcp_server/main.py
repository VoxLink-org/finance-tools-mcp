

import logging
import sys

from starlette.responses import JSONResponse
from starlette.requests import Request

from mcp.server.fastmcp import FastMCP

from mcp.server.auth.settings import AuthSettings

from apps.mcp_server.simple_token_verifier import SimpleTokenVerifier, cost_extra_credit
from apps.mcp_server.tools import calculation_tools, cnn_fng_tools, holdings_tools, macro_tools, option_tools, predict_tools, yfinance_tools
from packages.investor_agent_lib import prompts


# Create an instance of the SimpleTokenVerifier
token_verifier = SimpleTokenVerifier()

# Configure AuthSettings
auth_settings = AuthSettings(
    required_scopes=["normal", "advanced"],
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
def create_mcp_application():
    # Initialize MCP server
    mcp = FastMCP("finance-tools-mcp", dependencies=["yfinance", "httpx", "pandas","ta-lib-easy"],
                  token_verifier=token_verifier,
                  auth=auth_settings,
                  )
    
    # Register yfinance tools
    mcp.add_tool(cost_extra_credit(extra_credit=1)(yfinance_tools.get_ticker_data))
    mcp.add_tool(cost_extra_credit(extra_credit=2)(yfinance_tools.get_price_history))
    mcp.add_tool(cost_extra_credit(extra_credit=1)(yfinance_tools.get_financial_statements))
    mcp.add_tool(cost_extra_credit(extra_credit=1)(yfinance_tools.get_earnings_history))
    mcp.add_tool(cost_extra_credit(extra_credit=1)(yfinance_tools.get_ticker_news_tool))

    # Register option tools
    mcp.add_tool(cost_extra_credit(extra_credit=2)(option_tools.super_option_tool))

    # Register prediction tools
    mcp.add_tool(cost_extra_credit(extra_credit=3)(predict_tools.price_prediction))

    # Register holdings analysis tools
    mcp.add_tool(cost_extra_credit(extra_credit=1)(holdings_tools.get_top25_holders))
    mcp.add_tool(cost_extra_credit(extra_credit=1)(holdings_tools.get_insider_trades))

    # Register CNN Fear & Greed tools
    mcp.add_tool(cost_extra_credit(extra_credit=1)(cnn_fng_tools.get_overall_sentiment_tool))
    
    # Register calculation tools
    mcp.add_tool(calculation_tools.calculate)

    # Register macro tools and resources
    mcp.add_tool(macro_tools.get_current_time)
    mcp.add_tool(cost_extra_credit(extra_credit=1)(macro_tools.get_fred_series))
    mcp.add_tool(cost_extra_credit(extra_credit=1)(macro_tools.search_fred_series))
    mcp.add_tool(cost_extra_credit(extra_credit=2)(macro_tools.cnbc_news_feed))
    mcp.add_tool(cost_extra_credit(extra_credit=2)(macro_tools.social_media_feed))

    # Register prompts
    mcp.prompt()(prompts.chacteristics)

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

mcp_app = create_mcp_application()