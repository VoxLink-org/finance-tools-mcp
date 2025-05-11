import argparse
import logging

# Import the application factory
from apps.mcp_server.main import create_mcp_application
# Import the sse_server run function from its new location
from . import sse_server

logger = logging.getLogger(__name__) # Use the logger from main.py or define a new one

def main():


    mcp_app = create_mcp_application()

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
        sse_server.run_sse_server(mcp_app)
    else:
        mcp_app.run(transport=args.transport)

if __name__ == "__main__":
    main()