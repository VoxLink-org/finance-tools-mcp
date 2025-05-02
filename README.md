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


The server integrates with [yfinance](https://pypi.org/project/yfinance/) for market data retrieval and fetches Fear & Greed data from CNN.

It also includes tools for calculating mathematical expressions and technical indicators using [ta-lib-python](https://pypi.org/project/ta-lib-python/).

And Macro economic indicators using [fredapi](https://pypi.org/project/fredapi/).

And breaking world news from [cnbc.com](https://www.cnbc.com/).

Combine this with an MCP server for placing trades on a brokerage platform such as [tasty-agent](https://github.com/ferdousbhai/tasty-agent) to place trades on tastytrade platform. Make sure to also enable web search functionality if you would like to incoporate latest news in your analysis.

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

## Tools

The **finance-tools-mcp** server comes with several tools to support financial analysis:

### Ticker Information

1. **`get_ticker_data`**
   - **Description:** Retrieves a comprehensive report for a given ticker symbol, including company overview, news, key metrics, performance, dates, analyst recommendations, and upgrades/downgrades.
   - **Input:**
     - `ticker` (string): Stock ticker symbol (e.g., `"AAPL"`).
   - **Return:** A formatted multi-section report.

2. **`get_available_options`**
   - **Description:** Provides a list of stock options with the highest open interest.
   - **Inputs:**
     - `ticker_symbol` (string): Stock ticker symbol.
     - `num_options` (int, optional): Number of options to return (default: 10).
     - `start_date` & `end_date` (string, optional): Date range in `YYYY-MM-DD` format.
     - `strike_lower` & `strike_upper` (float, optional): Desired strike price range.
     - `option_type` (string, optional): Option type (`"C"` for calls, `"P"` for puts).
   - **Return:** A formatted table of options data.

3. **`get_price_history`**
   - **Description:** Retrieves historical price data for a specific ticker.
   - **Inputs:**
     - `ticker` (string): Stock ticker symbol.
     - `period` (string): Time period (choose from `"1d"`, `"5d"`, `"1mo"`, `"3mo"`, `"6mo"`, `"1y"`, `"2y"`, `"5y"`, `"10y"`, `"ytd"`, `"max"`).
   - **Return:** A table showing price history.

### Financial Data Tools

1. **`get_financial_statements`**
   - **Description:** Fetches financial statements (income, balance, or cash flow) formatted in millions USD.
   - **Inputs:**
     - `ticker` (string): Stock ticker symbol.
     - `statement_type` (string): `"income"`, `"balance"`, or `"cash"`.
     - `frequency` (string): `"quarterly"` or `"annual"`.
   - **Return:** A formatted financial statement.

2. **`get_institutional_holders`**
   - **Description:** Retrieves details about major institutional and mutual fund holders.
   - **Input:**
     - `ticker` (string): Stock ticker symbol.
   - **Return:** Two formatted tables listing institutional and mutual fund holders.

3. **`get_earnings_history`**
   - **Description:** Retrieves a formatted table of earnings history.
   - **Input:**
     - `ticker` (string): Stock ticker symbol.
   - **Return:** A table displaying historical earnings data.

4. **`get_insider_trades`**
   - **Description:** Fetches the recent insider trading activity for a given ticker.
   - **Input:**
     - `ticker` (string): Stock ticker symbol.
   - **Return:** A formatted table showing insider trades.

### CNN Fear & Greed Index Tools

1. **`get_current_fng_tool`**
   - **Description:** Retrieves the current CNN Fear & Greed Index score, rating, and classification.
   - **Inputs:** None
   - **Return:** A string containing the current index details.

2. **`get_historical_fng_tool`**
   - **Description:** Fetches historical CNN Fear & Greed Index data for a specified number of days.
   - **Inputs:**
     - `days` (int): Number of days of historical data to retrieve.
   - **Return:** A string listing historical scores and classifications.

3. **`analyze_fng_trend`**
   - **Description:** Analyzes the trend of the CNN Fear & Greed Index over a specified number of days.
   - **Inputs:**
     - `days` (int): Number of days to include in the trend analysis.
   - **Return:** A summary string including the latest value, average, range, trend direction, and classification.

### Utility Tools

1. **`calculate`**
   - **Description:** Calculate the result of a mathematical expression. Supports Python math syntax and NumPy.
   - **Input:**
     - `expression` (string): The mathematical expression to evaluate.
   - **Return:** A string containing a JSON object with the calculation result (e.g., `{'result': 10}`).

2. **`calc_ta`**
   - **Description:** Calculate technical indicators using ta-lib-python (TA-lib) and NumPy.
   - **Input:**
     - `ta_lib_expression` (string): The TA-Lib expression to evaluate.
   - **Return:** A string containing the result of the TA-Lib calculation.

3. **`get_current_time`**
   - **Description:** Get the current time in ISO 8601 format.
   - **Inputs:** None
   - **Return:** A string representing the current time in ISO 8601 format.

### Macro Tools

1. **`get_fred_series`**
   - **Description:** Fetches a FRED series by its ID.
   - **Input:**
     - `series_id` (string): FRED series ID.  
   - **Return:** A string containing the FRED series data.

2. **`search_fred_series`**
   - **Description:** Searches for the most popular FRED series by keyword.
   - **Input:**
     - `query` (string): Keyword to search for.
   - **Return:** A string containing the search results.

3. **`cnbc_news_feed`**
   - **Description:** Retrieves the latest breaking world news from CNBC and others.
   - **Inputs:** None
   - **Return:** A string containing CNBC news feed.

### Informational Prompts

1. **`investment_principles`**
   - **Description:** Provides a set of core investment principles and guidelines.
   - **Inputs:** None
   - **Return:** A string outlining several investment principles.

2. **`portfolio_construction_prompt`**
   - **Description:** Outlines a portfolio construction strategy incorporating tail-hedging.
   - **Inputs:** None
   - **Return:** A detailed prompt guiding the construction of a hedged portfolio.

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
