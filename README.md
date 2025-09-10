# finance-tools-mcp: A Financial Analysis MCP Server
> https://github.com/VoxLink-org/finance-tools-mcp

## Overview

The **finance-tools-mcp** is a Model Context Protocol (MCP) server designed to provide comprehensive financial insights and analysis capabilities to Large Language Models (LLMs). Modified from [investor-agent](https://github.com/ferdousbhai/investor-agent), it integrates with various data sources and analytical libraries to offer a suite of tools for detailed financial research and analysis.

<a href="https://glama.ai/mcp/servers/@VoxLink-org/finance-tools-mcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@VoxLink-org/finance-tools-mcp/badge" alt="Finance Tools MCP server" />
</a>

## Prerequisites

*   **Python:** 3.10 or higher is required.
*   **Package Manager:** [uv](https://docs.astral.sh/uv/) is the recommended package installer and resolver for this project.

## Installation

First, install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ensure `uv` is in your system's PATH. You might need to restart your terminal or add `~/.cargo/bin` to your PATH.

Then, you can run the `finance-tools-mcp` MCP server using `uvx` (which executes a package without explicitly installing it into your environment):

```bash
uvx finance-tools-mcp
```

To run the server with a FRED API key (for enhanced macroeconomic data access), set it as an environment variable:

```bash
FRED_API_KEY=YOUR_API_KEY uvx finance-tools-mcp
```

You can also run the server using Server-Sent Events (SSE) transport, which might be preferred by some MCP clients:

```bash
uvx finance-tools-mcp --transport sse
```

Or with both the FRED API key and SSE transport:

```bash
FRED_API_KEY=YOUR_API_KEY uvx finance-tools-mcp --transport sse
```

## Usage with MCP Clients

To integrate **finance-tools-mcp** with an MCP client (for example, Claude Desktop), add the following configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "investor": {
        "command": "path/to/uvx/command/uvx",
        "args": ["finance-tools-mcp"],
    }
  }
}
```

## Debugging

You can leverage the MCP inspector to debug the server:

```bash
npx @modelcontextprotocol/inspector uvx finance-tools-mcp
```

or

```bash
npx @modelcontextprotocol/inspector uv --directory  ./ run finance-tools-mcp
```

For log monitoring, check the following directories:

*   macOS: `~/Library/Logs/Claude/mcp*.log`
*   Windows: `%APPDATA%\Claude\logs\mcp*.log`

## Development

For local development and testing:

1.  Use the MCP inspector as described in the [Debugging](#debugging) section.
2.  Test using Claude Desktop with this configuration:

```json
{
  "mcpServers": {
    "investor": {
      "command": "path/to/uv/command/uv",
      "args": ["--directory", "path/to/finance-tools-mcp", "run", "finance-tools-mcp"],
    }
  }
}
```
