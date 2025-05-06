# finance-tools-mcp: A Financial Analysis MCP Server
> https://github.com/VoxLink-org/finance-tools-mcp

## Overview

The **finance-tools-mcp** , modified from [investor-agent](https://github.com/ferdousbhai/investor-agent), is a Model Context Protocol (MCP) server that provides comprehensive financial insights and analysis to Large Language Models. It leverages real-time market data, news, and advanced analytics to help users obtain:

- Detailed ticker reports including company overview, news, key metrics, performance, dates, analyst recommendations, and upgrades/downgrades.
- Options data highlighting high open interest.
- Historical price trends for stocks.
- Essential financial statements (income, balance sheet, cash flow) formatted in millions USD.
- Up-to-date institutional ownership and mutual fund holdings.
- Current and historical CNN Fear & Greed Index data and trend analysis.
- Prompts related to core investment principles and portfolio construction strategies.
- Earnings history and insider trading activity.
- Breaking world news from CNBC.
- FRED series search results.
- Technical indicators using ta-lib-python.

<a href="https://glama.ai/mcp/servers/@VoxLink-org/finance-tools-mcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@VoxLink-org/finance-tools-mcp/badge" alt="Finance Tools MCP server" />
</a>

The server integrates with [yfinance](https://pypi.org/project/yfinance/) for market data retrieval and fetches Fear & Greed data from CNN.

It also includes tools for calculating mathematical expressions and technical indicators using [ta-lib-python](https://pypi.org/project/ta-lib-python/).

And Macro economic indicators using [fredapi](https://pypi.org/project/fredapi/).

And breaking world news from [cnbc.com](https://www.cnbc.com/).

Make sure to also enable web search functionality if you would like to incoporate latest news in your analysis.

## Sample Report
![alt text](image.png)


## Prerequisites

- **Python:** 3.10 or higher
- **Package Manager:** [uv](https://docs.astral.sh/uv/)

## Installation

First, install **uv** if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, you can run the **finance-tools-mcp** MCP server using `uvx`:

```bash
uvx finance-tools-mcp
```

If you want to use your own FRED API key, you can set it as an environment variable:

```bash
FRED_API_KEY=YOUR_API_KEY uvx finance-tools-mcp
```

You can also run the server using Server-Sent Events (SSE) transport:

```bash
uvx finance-tools-mcp --transport sse
```

Or with the FRED API key and SSE transport:

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

- macOS: `~/Library/Logs/Claude/mcp*.log`
- Windows: `%APPDATA%\Claude\logs\mcp*.log`

## Development

For local development and testing:

1. Use the MCP inspector as described in the [Debugging](#debugging) section.
2. Test using Claude Desktop with this configuration:

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

## License

This MCP server is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Samples
- [carvana_analysis.md](reports/carvana_analysis.md)
- [palantir_analysis.md](reports/palantir_analysis.md)
- [pdd_analysis_20250503.md](reports/pdd_analysis_20250503.md)
- [meli_se_shop_comparison_20250504.md](reports/meli_se_shop_comparison_20250504.md)


## Todo
- [ ] Add supporting levels and resistance levels for stocks
- [ ] Add Fibonacci retracement levels for stocks
- [ ] Add moving average confluence levels for stocks
- [-] Add option model for prediction 