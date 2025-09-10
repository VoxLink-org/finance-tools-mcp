import argparse
import logging

# Import the application factory
from apps.mcp_server import sse_server
from apps.mcp_server.main import create_mcp_application

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
    
    mcp_app = create_mcp_application(port=args.port)

    
    if args.transport != "stdio":
        sse_server.run_server(mcp_app, transport=args.transport)
    else:
        mcp_app.run(transport=args.transport)

if __name__ == "__main__":
    main()

