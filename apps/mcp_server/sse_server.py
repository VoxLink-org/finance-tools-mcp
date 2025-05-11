import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.responses import JSONResponse
from starlette.requests import Request

from mcp.server.fastmcp import FastMCP

def run_sse_server(mcp_server: FastMCP):
    """Run the SSE server with global CORS middleware by mounting mcp.sse_app()."""

    # Define CORS configuration
    cors_middleware = Middleware(
        CORSMiddleware,
        allow_origins=["*"], # Allow all origins
        allow_methods=["*"], # Allow all methods
        allow_headers=["*"], # Allow all headers
    )
    # Add health check

    @mcp_server.custom_route("/", methods=["GET","HEAD","OPTIONS"])
    def health_check(request: Request)->JSONResponse :
        return JSONResponse(
            {"status": "ok"}
        )


    # Create a top-level Starlette app with CORS middleware and mount mcp.sse_app()
    app = Starlette(
        routes=[
            Mount('/', app=mcp_server.sse_app()), # Mount the FastMCP SSE app
        ],
        middleware=[cors_middleware]
    )

    # Run the Starlette app with Uvicorn
    config = uvicorn.Config(
        app,
        host=mcp_server.settings.host,
        port=mcp_server.settings.port,
        log_level=mcp_server.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    anyio.run(server.serve)