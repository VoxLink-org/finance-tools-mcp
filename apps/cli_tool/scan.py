from typing import Literal
from packages.investor_agent_lib.services import yfinance_service
from config.paths import DATA_DIR
from tabulate import tabulate
import pandas as pd
import math

def scan_watch_list(period: Literal["1d", "5d", "10d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "5d", end_date: pd.Timestamp=None):
    watch_list_path = DATA_DIR / "tickers.txt"

    with watch_list_path.open() as f:
        tickers = [line.strip() for line in f]

    tickers = tickers
    end_date = pd.Timestamp.utcnow() if end_date is None else end_date

    if period == "1d":
        start_date = end_date - pd.Timedelta(days=1)
    elif period == "5d":
        start_date = end_date - pd.Timedelta(days=5)
    elif period == "10d":
        start_date = end_date - pd.Timedelta(days=10)
    elif period == "1mo":
        start_date = end_date - pd.Timedelta(days=30)
    elif period == "3mo":
        start_date = end_date - pd.Timedelta(days=90)
    elif period == "6mo":
        start_date = end_date - pd.Timedelta(days=180)
    elif period == "1y":
        start_date = end_date - pd.Timedelta(days=365)
    elif period == "2y":
        start_date = end_date - pd.Timedelta(days=730)
    elif period == "5y":
        start_date = end_date - pd.Timedelta(days=1825)
    elif period == "10y":
        start_date = end_date - pd.Timedelta(days=3650)
    elif period == "ytd":
        start_date = pd.Timestamp.utcnow().replace(month=1, day=1)
    elif period == "max":
        start_date = pd.Timestamp.min

    data = yfinance_service.download_history(
        tickers,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1d"
    )
    
    if data is None or data.empty:
        return "No data available for the given period"
    
    # Calculate price changes
    results = []
    for ticker in tickers:
        if ticker in data.columns.get_level_values(1):
            ticker_data = data.xs(ticker, level=1, axis=1)
            if not ticker_data.empty:
                start_price = ticker_data['Close'].iloc[0]
                end_price = ticker_data['Close'].iloc[-1]
                pct_change = ((end_price - start_price) / start_price) * 100
                results.append({
                    'Ticker': ticker,
                    'Start Date': ticker_data.index[0].strftime('%Y-%m-%d'),
                    'End Date': ticker_data.index[-1].strftime('%Y-%m-%d'),
                    'Start Price': start_price,
                    'End Price': end_price,
                    'Change %': pct_change
                })
    
    if not results:
        return "No valid ticker data available"

    # remove NaN rows
    results = [result for result in results if not math.isnan(result['Change %'])]
        
    # Sort by percentage change
    results.sort(key=lambda x: x['Change %'], reverse=True)
    
    # Prepare output with top and bottom 20
    output = []
    if len(results) > 40:
        output.extend(results[:20])
        output.append({'Ticker': '...', 'Start Date': '', 'End Date': '', 'Start Price': '', 'End Price': '', 'Change %': ''})
        output.extend(results[-20:])
    else:
        output = results
    
    return tabulate(
        output,
        headers="keys",
        floatfmt=(None, None, ".2f", ".2f", ".2%"),
        tablefmt="psql"
    )



if __name__ == '__main__':
    print(scan_watch_list('10d'))
