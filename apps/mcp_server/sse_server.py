from typing import Literal
import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.responses import JSONResponse
from starlette.requests import Request

from mcp.server.fastmcp import FastMCP


def run_server(
    mcp_server: FastMCP, transport: Literal["sse", "streamable-http"] = "sse"
):
    """Run the SSE server with global CORS middleware by mounting mcp.sse_app()."""

    # Define CORS configuration
    cors_middleware = Middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )

    if transport == "sse":
        # For SSE, we can mount the app directly
        app = Starlette(
            routes=[
                Mount("/", app=mcp_server.sse_app()),  # Mount the FastMCP SSE app
            ],
            middleware=[cors_middleware],
        )
    else:
        # For streamable HTTP, use the app directly as it includes proper lifespan management
        app = mcp_server.streamable_http_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )

    # Run the Starlette app with Uvicorn
    config = uvicorn.Config(
        app,
        host=mcp_server.settings.host,
        port=mcp_server.settings.port,
        log_level=mcp_server.settings.log_level.lower(),
        timeout_graceful_shutdown=3,  # Force shutdown after 3 seconds
    )
    server = uvicorn.Server(config)
    anyio.run(server.serve)
