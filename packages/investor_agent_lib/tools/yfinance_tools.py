import logging
from datetime import datetime
from typing import Literal

import pandas as pd
from tabulate import tabulate

from packages.investor_agent_lib.services import yfinance_service, finviz_service
from packages.investor_agent_lib.digests import time_series_digest

logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

def get_ticker_data(ticker: str) -> str:
    """Get comprehensive report for ticker: overview, news, metrics, sector / industry valuation, performance, dates, analyst recommendations, and upgrades/downgrades."""
    try:
        info = yfinance_service.get_ticker_info(ticker)
        if not info:
            return f"No information available for {ticker}"

        sections = []

        # Company overview
        overview = [
            ["Company Name", info.get('longName', 'N/A')],
            ["Sector", info.get('sector', 'N/A')],
            ["Industry", info.get('industry', 'N/A')],
            ["Market Cap", f"${info.get('marketCap', 0):,.2f}" if info.get('marketCap') else "N/A"],
            ["Employees", f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else "N/A"],
            ["Beta", f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A"]
        ]
        sections.extend(["COMPANY OVERVIEW", tabulate(overview, tablefmt="plain")])

        # Key metrics
        metrics = [
            ["Current Price", f"${info.get('currentPrice', 0):.2f}" if info.get('currentPrice') else "N/A"],
            ["52-Week Range", f"${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekLow') and info.get('fiftyTwoWeekHigh') else "N/A"],
            ["Market Cap", f"${info.get('marketCap', 0):,.2f}" if info.get('marketCap') else "N/A"],
            ["Trailing P/E", info.get('trailingPE', 'N/A')],
            ["Forward P/E", info.get('forwardPE', 'N/A')],
            ["PEG Ratio", info.get('trailingPegRatio', 'N/A')],
            ["Price/Book", f"{info.get('priceToBook', 0):.2f}" if info.get('priceToBook') else "N/A"],
            ["Dividend Yield", f"{info.get('dividendYield', 0)*1:.2f}%" if info.get('dividendYield') else "N/A"],
            ["Short % of Float", f"{info.get('shortPercentOfFloat', 0)*100:.2f}%" if info.get('shortPercentOfFloat') else "N/A"]
        ]
        sections.extend(["\nKEY METRICS", tabulate(metrics, tablefmt="plain")])

        if average:= finviz_service.get_sector_and_industry_valuation(ticker):
            # use tabulate to format the dict
            sector_average = [[k, v] for k, v in average['sector_valuation'].items()]
            industry_average = [[k, v] for k, v in average['industry_valuation'].items()]
            sections.extend(["\nSECTOR AVERAGE VALUATION", tabulate(sector_average, tablefmt="plain")])
            sections.extend(["\nINDUSTRY AVERAGE VALUATION", tabulate(industry_average, tablefmt="plain")])

        # Performance metrics
        performance = [
            ["Return on Equity", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A"],
            ["Return on Assets", f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else "N/A"],
            ["Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A"],
            ["Operating Margin", f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else "N/A"],
            ["Debt to Equity", f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else "N/A"],
            ["Current Ratio", f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else "N/A"]
        ]
        sections.extend(["\nPERFORMANCE METRICS", tabulate(performance, tablefmt="plain")])

        # Analyst coverage
        analyst = [
            ["Analyst Count", str(info.get('numberOfAnalystOpinions', 'N/A'))],
            ["Mean Target", f"${info.get('targetMeanPrice', 0):.2f}" if info.get('targetMeanPrice') else "N/A"],
            ["High Target", f"${info.get('targetHighPrice', 0):.2f}" if info.get('targetHighPrice') else "N/A"],
            ["Low Target", f"${info.get('targetLowPrice', 0):.2f}" if info.get('targetLowPrice') else "N/A"],
            ["Recommendation", info.get('recommendationKey', 'N/A').title()]
        ]
        sections.extend(["\nANALYST COVERAGE", tabulate(analyst, tablefmt="plain")])

        # Calendar dates
        if calendar := yfinance_service.get_calendar(ticker):
            dates_data = []
                        
            # Process earnings dates
            if calendar.get('Earnings Date'):
                if len(calendar['Earnings Date']) == 1:
                    dates_data.append(['Earnings Date',
                                     calendar['Earnings Date'][0].strftime("%Y-%m-%d")])
                else:
                    start = calendar['Earnings Date'][0].strftime("%Y-%m-%d")
                    end = calendar['Earnings Date'][1].strftime("%Y-%m-%d")
                    dates_data.append(['Earnings Date', f"{start}-{end}"])
            
            # Add earnings estimates if available
            if any(calendar.get(k) for k in ['Earnings Average', 'Revenue Average']):
                estimates = [
                    ['Earnings Estimate',
                     f"{calendar.get('Earnings Low')} - {calendar.get('Earnings High')} "
                     f"(Avg: {calendar.get('Earnings Average')})"],
                    ['Revenue Estimate',
                     f"{calendar.get('Revenue Low'):,} - {calendar.get('Revenue High'):,} "
                     f"(Avg: {calendar.get('Revenue Average'):,})"]
                ]
                dates_data.extend(estimates)

            if dates_data:
                sections.extend([
                    "\nIMPORTANT DATES & ESTIMATES",
                    tabulate(dates_data, headers=["Event", "Details"], tablefmt="plain")
                ])

        # Recent recommendations
        if (recommendations := yfinance_service.get_recommendations(ticker)) is not None and not recommendations.empty:
            rec_data = [
                [
                    row['period'],  # Use the period column directly
                    row['strongBuy'],
                    row['buy'],
                    row['hold'],
                    row['sell'],
                    row['strongSell']
                ]
                for _, row in recommendations.iterrows()
                if not all(pd.isna(val) for val in row.values)
            ]
            if rec_data:
                sections.extend(["\nRECENT ANALYST RECOMMENDATIONS",
                               tabulate(rec_data,
                                      headers=["Period", "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                                      tablefmt="plain")])

        # Recent upgrades/downgrades
        if (upgrades := yfinance_service.get_upgrades_downgrades(ticker)) is not None and not upgrades.empty:
            upg_data = [
                [
                    pd.to_datetime(row.name).strftime('%Y-%m-%d'),
                    row.get('Firm', 'N/A'),
                    f"{row.get('FromGrade', 'N/A')} → {row.get('ToGrade', 'N/A')}"
                ]
                for _, row in upgrades.iterrows()
                if not all(pd.isna(val) for val in row.values)
            ]
            if upg_data:
                sections.extend(["\nRECENT UPGRADES/DOWNGRADES",
                               tabulate(upg_data, headers=["Date", "Firm", "Change"], tablefmt="plain")])

        return "\n".join(sections)

    except Exception as e:
        logger.error(f"Error getting ticker data for {ticker}: {e}")
        return f"Failed to retrieve data for {ticker}: {str(e)}"

def get_options(
    ticker_symbol: str,
    num_options: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    strike_lower: float | None = None,
    strike_upper: float | None = None,
    option_type: Literal["C", "P"] | None = None,
) -> str:
    """Get options with highest open interest. Dates: YYYY-MM-DD. Type: C=calls, P=puts."""
    try:
        df, error = yfinance_service.get_filtered_options(
            ticker_symbol, start_date, end_date, strike_lower, strike_upper, option_type
        )
        if error:
            return error

        options_data = [
            [
                "C" if "C" in row['contractSymbol'] else "P",
                f"${row['strike']:.2f}",
                row['expiryDate'],
                int(row['openInterest']) if pd.notnull(row['openInterest']) else 0,
                int(row['volume']) if pd.notnull(row['volume']) and row['volume'] > 0 else "N/A",
                f"{row['impliedVolatility']*100:.1f}%" if pd.notnull(row['impliedVolatility']) else "N/A"
            ]
            for _, row in df.head(num_options).iterrows()
        ]

        return tabulate(options_data, headers=["Type", "Strike", "Expiry", "OI", "Vol", "IV"], tablefmt="plain")

    except Exception as e:
        logger.error(f"Error getting options data for {ticker_symbol}: {e}")
        return f"Failed to retrieve options data for {ticker_symbol}: {str(e)}"

def get_price_history(
    ticker: str,
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "6mo",
    start_date: str = '',
    end_date: str  = ''
) -> str:
    """Get historical price data digest for specified period. There're two kinds of response mode:
    1. The period mode. It will generate a digest for LLM consumption. Usually get at least 3 months, 6 months or more.
    The response includes OCHLCV samples, Technical Indicators (by ta-lib) , Risk Metrics, and other quantitative analysis.
    2. The start_date and end_date mode. Once the start_date (yyyy-mm-dd) and end_date (yyyy-mm-dd) are specified, it will generate a raw OCHLCV data in the slot.
    And no digest will be generated in this mode. Useful for checking the price history of a specific short date range.
    """
    if start_date and end_date:
        return yfinance_service.get_price_slot(ticker, start_date, end_date)

    history = yfinance_service.get_price_history(ticker, period, raw=True)
    if history is None or type(history) == str or history.empty:
        return f"No historical data found for {ticker} {history}"

    digest_mode = True
    if digest_mode:
        return time_series_digest.generate_time_series_digest_for_LLM(history)


    price_data = [
        [
            idx.strftime('%Y-%m-%d'),  # exclude timestamp
            f"${row['Open']:.2f}",
            f"${row['Close']:.2f}",
            f"{row['Volume']:,.0f}",
            f"${row['Dividends']:.4f}" if row['Dividends'] > 0 else "-",
            f"{row['Stock Splits']:.0f}:1" if row['Stock Splits'] > 0 else "-"
        ]
        for idx, row in history.iterrows()
    ]

    return (f"PRICE HISTORY FOR {ticker} ({period}):\n" +
            tabulate(price_data, headers=["Date", "Open", "Close", "Volume", "Dividends", "Splits"], tablefmt="plain"))

def get_financial_statements(
    ticker: str,
    statement_type: Literal["income", "balance", "cash"] = "income",
    frequency: Literal["quarterly", "annual"] = "quarterly",
) -> str:
    """Get financial statements. Types: income, balance, cash. Frequency: quarterly, annual."""
    data = yfinance_service.get_financial_statements(ticker, statement_type, frequency)

    if data is None or data.empty:
        return f"No {statement_type} statement data found for {ticker}"

    statement_data = [
        [metric] + [
            "N/A" if pd.isna(value) else
            f"${value/1e9:.1f}B" if abs(value) >= 1e9 else
            f"${value/1e6:.1f}M"
            for value in data.loc[metric]
        ]
        for metric in data.index
    ]

    headers = ["Metric"] + [date.strftime("%Y-%m-%d") for date in data.columns]
    title = (f"{frequency.upper()} {statement_type.upper()} STATEMENT FOR {ticker}:\n"
             "(Values in billions/millions USD)")

    return title + "\n" + tabulate(statement_data, headers=headers, tablefmt="plain")

def get_institutional_holders(ticker: str, top_n: int = 20) -> str:
    """Get major institutional and mutual fund holders."""
    inst_holders, fund_holders = yfinance_service.get_institutional_holders(ticker)

    if (inst_holders is None or inst_holders.empty) and (fund_holders is None or fund_holders.empty):
        return f"No institutional holder data found for {ticker}"

    def format_holder_data(df: pd.DataFrame) -> list:
        return [
            [
                row['Holder'],
                f"{row['Shares']:,.0f}",
                f"${row['Value']:,.0f}",
                f"{row['pctHeld']*100:.2f}%",
                pd.to_datetime(row['Date Reported']).strftime('%Y-%m-%d'),
                f"{row['pctChange']*100:+.2f}%" if pd.notnull(row['pctChange']) else "N/A"
            ]
            for _, row in df.iterrows()
        ]

    headers = ["Holder", "Shares", "Value", "% Held", "Date Reported", "% Change"]
    sections = []

    if inst_holders is not None and not inst_holders.empty:
        sections.extend(["INSTITUTIONAL HOLDERS:",
                        tabulate(format_holder_data(inst_holders), headers=headers, tablefmt="plain")])

    if fund_holders is not None and not fund_holders.empty:
        sections.extend(["\nMUTUAL FUND HOLDERS:",
                        tabulate(format_holder_data(fund_holders), headers=headers, tablefmt="plain")])

    return "\n".join(sections)

def get_earnings_history(ticker: str) -> str:
    """Get earnings history with estimates and surprises."""
    earnings_history = yfinance_service.get_earnings_history(ticker)

    if earnings_history is None or earnings_history.empty:
        return f"No earnings history data found for {ticker}"

    earnings_data = [
        [
            date.strftime('%Y-%m-%d'),
            f"${row['epsEstimate']:.2f}" if pd.notnull(row['epsEstimate']) else "N/A",
            f"${row['epsActual']:.2f}" if pd.notnull(row['epsActual']) else "N/A",
            f"${row['epsDifference']:.2f}" if pd.notnull(row['epsDifference']) else "N/A",
            f"{row['surprisePercent']:.1f}%" if pd.notnull(row['surprisePercent']) else "N/A"
        ]
        for date, row in earnings_history.iterrows()
    ]

    return (f"EARNINGS HISTORY FOR {ticker}:\n" +
            tabulate(earnings_data, headers=["Date", "EPS Est", "EPS Act", "Surprise", "Surprise %"], tablefmt="plain"))

def get_insider_trades(ticker: str) -> str:
    """Get recent insider trading activity."""
    trades = yfinance_service.get_insider_trades(ticker)
    if trades is None or trades.empty:
        return f"No insider trading data found for {ticker}"

    trades_data = [
        [
            pd.to_datetime(row['Start Date']).strftime('%Y-%m-%d'),
            row.get('Insider', 'N/A'),
            row.get('Position', 'N/A'),
            row.get('Text', 'N/A'),
            f"{row.get('Shares', 0):,.0f}",
            # f"${row.get('Value', 0):,.0f}" if pd.notnull(row.get('Value')) else "N/A"
        ]
        for _, row in trades.iterrows()
    ]

    return (f"INSIDER TRADES FOR {ticker}:\n" +
            tabulate(trades_data, headers=["Date", "Insider", "Title", "Text", "Shares"], tablefmt="plain"))

def get_ticker_news_tool(ticker: str) -> str:
    """For getting yahoo financial news of a ticker. Useful for getting latest news, especially for doing deep research."""
    news = yfinance_service.get_ticker_news(ticker)

    if news is None or len(news) == 0:
        return f"No news found for {ticker}"

    
    return news