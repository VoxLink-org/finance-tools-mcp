import argparse
import logging

# Import the application factory
from apps.mcp_server import sse_server
from apps.mcp_server.main import mcp_app
import apps.mcp_server.tools.calculation_tools  # Ensure tools are registered
import apps.mcp_server.tools.cnn_fng_tools  # Ensure tools are registered
import apps.mcp_server.tools.holdings_tools  # Ensure tools are registered
import apps.mcp_server.tools.macro_tools  # Ensure tools are registered
import apps.mcp_server.tools.option_tools  # Ensure tools are registered
import apps.mcp_server.tools.predict_tools  # Ensure tools are registered
import apps.mcp_server.tools.yfinance_tools  # Ensure tools are registered


logger = logging.getLogger(__name__) # Use the logger from main.py or define a new one

def main():

    # Add argument parsing
    parser = argparse.ArgumentParser(description="Run the Finance Tools MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (stdio or sse)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    

    # Parse arguments and run the server
    args = parser.parse_args()
    
    mcp_app.settings.port = args.port
    
    logger.info(f"Starting server on port {args.port}")
    
    if args.transport != "stdio":
        sse_server.run_server(mcp_app, transport=args.transport)
    else:
        mcp_app.run(transport=args.transport)

if __name__ == "__main__":
    main()

