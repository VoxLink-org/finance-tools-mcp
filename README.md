# finance-tools-mcp: A Financial Analysis MCP Server
> https://github.com/VoxLink-org/finance-tools-mcp

## Overview

The **finance-tools-mcp** is a Model Context Protocol (MCP) server designed to provide comprehensive financial insights and analysis capabilities to Large Language Models (LLMs). Modified from [investor-agent](https://github.com/ferdousbhai/investor-agent), it integrates with various data sources and analytical libraries to offer a suite of tools for detailed financial research and analysis.

<a href="https://glama.ai/mcp/servers/@VoxLink-org/finance-tools-mcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@VoxLink-org/finance-tools-mcp/badge" alt="Finance Tools MCP server" />
</a>

## Tools Offered

The `finance-tools-mcp` server exposes a variety of tools via the Model Context Protocol (MCP), allowing connected clients (like LLMs) to access specific financial data and perform analyses. These tools are categorized for easier navigation:

### 1. Ticker Data & Analysis Tools (yfinance_tools)
These tools leverage `yfinance` and `finviz` to provide comprehensive data and insights for individual stock tickers.
*   `get_ticker_data(ticker: str)`: Provides a comprehensive report for a given ticker, including company overview, key metrics, sector/industry valuation, performance metrics, analyst coverage, important dates (earnings, dividends), and recent upgrades/downgrades.
*   `get_price_history(ticker: str, period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "6mo", start_date: str = '', end_date: str = '')`: Fetches historical price data. It can generate a structured digest for LLM consumption (including OHLCV samples, Technical Indicators, Risk Metrics, and quantitative analysis) for periods like 3 months, 6 months, or more. Alternatively, if `start_date` and `end_date` are specified (YYYY-MM-DD), it provides raw OHLCV data for a specific short date range.
*   `get_financial_statements(ticker: str, statement_type: Literal["income", "balance", "cash"] = "income", frequency: Literal["quarterly", "annual"] = "quarterly")`: Accesses financial statements (income, balance sheet, cash flow) for a ticker, available quarterly or annually. Values are formatted in billions/millions USD.
*   `get_earnings_history(ticker: str)`: Provides a detailed earnings history for a ticker, including estimated EPS, actual EPS, difference, and surprise percentage.
*   `get_ticker_news_tool(ticker: str)`: Fetches the latest Yahoo Finance news articles relevant to a specific ticker, useful for deep research.

### 2. Holdings Analysis Tools (holdings_tools)
These tools provide insights into institutional and insider trading activities.
*   `get_top25_holders(ticker: str)`: Retrieves the top 25 institutional holders for a given stock ticker, including their shares, value, percentage held, date reported, and percentage change in holdings.
*   `get_insider_trades(ticker: str)`: Fetches recent insider trading activity for a ticker, including transaction dates, insider names, titles, transaction text, and shares involved. It also lists company officers.

### 3. Option Analysis Tools (option_tools)
*   `super_option_tool(ticker: str)`: Analyzes and summarizes option data for a given ticker. It retrieves option indicators and Greeks, generates a digest summarizing key metrics, and formats a table of key option data including last trade date, strike, option type, open interest, volume, and implied volatility.

### 4. Market Sentiment & Macro Data Tools (cnn_fng_tools, macro_tools)
These tools provide broader market sentiment and macroeconomic indicators.
*   `get_overall_sentiment_tool()`: Gets comprehensive market sentiment indicators including the current CNN Fear & Greed Index (score and rating), Market RSI (Relative Strength Index), and VIX (Volatility Index).
*   `get_historical_fng_tool(days: int)`: Retrieves historical CNN Fear & Greed Index data for a specified number of days, including the score and its classification (e.g., Extreme Fear, Fear, Neutral, Greed, Extreme Greed).
*   `analyze_fng_trend(days: int)`: Analyzes trends in the CNN Fear & Greed Index over a specified number of days, providing the latest value, average value, range, and trend direction (rising, falling, stable).
*   `get_current_time()`: Provides the current time in ISO 8601 format.
*   `get_fred_series(series_id: str)`: Retrieves data for a specific FRED (Federal Reserve Economic Data) series by its ID. (Note: Data may not always be the latest).
*   `search_fred_series(query: str)`: Searches for popular FRED series by keyword (e.g., "GDP", "CPI"). (Note: Data may not always be the latest).
*   `cnbc_news_feed()`: Fetches the latest breaking stock market news from CNBC. It also includes real-time Fed rate predictions from CME Group 30-Day Fed Fund futures prices and key macro indicators from stlouisfed.org.
*   `social_media_feed(keywords: list[str] = None)`: Gets the most discussed stocks and investment opinions from Reddit. Optional `keywords` can be provided to filter for specific topics (e.g., `['tsla', 'tesla']`). If no keywords are provided, it returns general trending discussions.

### 5. Calculation Tools (calculation_tools)
*   `calculate(expression: str)`: Evaluates mathematical expressions using Python's `math` syntax and `NumPy`. Examples: `"2 * 3 + 4"`, `"sin(pi/2)"`, `"sqrt(16)"`, `"np.mean([1, 2, 3])"`.

### 6. Registered Prompts
The server also registers several prompts to guide LLM behavior and provide context:
*   `prompts.chacteristics`: Defines the characteristics of the investment agent.
*   `prompts.mode_instructions`: Provides instructions for different operational modes.
*   `prompts.investment_principles`: Outlines core investment principles.
*   `prompts.portfolio_construction_prompt`: Guides the LLM on portfolio construction strategies.


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

## Contributing

We welcome contributions to the `finance-tools-mcp` project! If you're interested in improving this server, please consider:

*   **Reporting Bugs:** If you find any issues, please open an issue on the GitHub repository.
*   **Suggesting Features:** Have an idea for a new financial tool or improvement? Let us know by opening an issue.
*   **Submitting Pull Requests:**
    1.  Fork the repository.
    2.  Create a new branch (`git checkout -b feature/your-feature-name`).
    3.  Make your changes and ensure your code adheres to the project's coding standards.
    4.  Write clear, concise commit messages.
    5.  Submit a pull request with a detailed description of your changes.

## License

This MCP server is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Samples
- [carvana_analysis.md](reports/carvana_analysis.md)
- [palantir_analysis.md](reports/palantir_analysis.md)
- [pdd_analysis_20250503.md](reports/pdd_analysis_20250503.md)
- [meli_se_shop_comparison_20250504.md](reports/meli_se_shop_comparison_20250504.md)
- [GLD_analysis_20250508.md](reports/GLD_analysis_20250508.md)

## Future Enhancements (Todo)

*   [x] Add supporting levels and resistance levels for stocks
*   [x] Add Fibonacci retracement levels for stocks
*   [x] Add moving average confluence levels for stocks
*   [ ] Add option model for prediction
*   [ ] Add predictive model by using finance sheets and other features

## Data Sources

The `finance-tools-mcp` server integrates with various data sources to provide comprehensive financial information:

*   **Yahoo Finance:** For ticker data, price history, financial statements, earnings, institutional holders, insider trades, and news.
*   **Finviz:** For sector and industry valuation, and additional insider trading data.
*   **CNN Business:** For the Fear & Greed Index.
*   **FRED (Federal Reserve Economic Data):** For macroeconomic time series data.
*   **CNBC, BBC, SCMP:** For breaking world news.
*   **Reddit:** For social media sentiment and trending investment discussions.
*   **CME Group:** For real-time Fed rate predictions.
