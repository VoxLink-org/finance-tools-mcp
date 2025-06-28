from typing import Literal
from packages.investor_agent_lib.services import yfinance_service
import talib as ta
import pandas as pd # For pd.Timedelta if used, and general data handling
import numpy as np # For np.sum

import vectorbt as vbt

def RSI_Trading():
    cash = 100000  # Initial cash in USD
    # --- IMPORTANT: Using a historical date range for backtesting ---
    start = '2023-11-01 UTC'
    end = '2025-05-20 UTC'
    # --- You can change these dates to any other valid historical period ---

    print(f"Attempting to download SPY data from {start} to {end}")
    spy_price = vbt.YFData.download('spy', start=start, end=end).get('Close')

    if spy_price is None or spy_price.empty:
        print("Failed to download SPY data or data is empty. Check ticker, dates, and internet connection.")
        return None # Or handle error appropriately

    print(f"\nSPY Price Data (first 5 rows):\n{spy_price.head()}")
    print(f"SPY Price Data (last 5 rows):\n{spy_price.tail()}")
    print(f"Length of SPY price data: {len(spy_price)}")

    rsi_16 = vbt.RSI.run(spy_price, window=16)
    if rsi_16 is None or rsi_16.rsi.empty:
        print("Failed to calculate RSI or RSI data is empty.")
        return None # Or handle error appropriately
        
    print(f"\nRSI(16) values (first 10 after initial NaN period):\n{rsi_16.rsi.dropna().head(10)}")
    print(f"RSI(16) values (last 10):\n{rsi_16.rsi.tail(10)}")

    # Using scalar thresholds is often cleaner with vectorbt
    entries = rsi_16.rsi_crossed_above(30)
    # entries = rsi_16.rsi < 30 # Alternative way to define entries

    exits = rsi_16.rsi_crossed_above(80)
    # exits = rsi_16.rsi > 80 # Alternative way to define exits

    print(f"\nNumber of raw entry signals (RSI crossed below 50): {np.sum(entries)}")
    if np.sum(entries) > 0:
        print(f"Entry signal dates:\n{spy_price.index[entries]}")
    else:
        print("No entry signals generated.")

    print(f"\nNumber of raw exit signals (RSI crossed above 80): {np.sum(exits)}")
    if np.sum(exits) > 0:
        print(f"Exit signal dates:\n{spy_price.index[exits]}")
    else:
        print("No exit signals generated.")
        
    # Ensure signals are boolean arrays of the same length as price data
    if not (isinstance(entries, pd.Series) and entries.dtype == bool and len(entries) == len(spy_price)):
        print("Entries are not a valid boolean Series matching price data length.")
        return None
    if not (isinstance(exits, pd.Series) and exits.dtype == bool and len(exits) == len(spy_price)):
        print("Exits are not a valid boolean Series matching price data length.")
        return None

    pf = vbt.Portfolio.from_signals(spy_price, entries, exits, init_cash=cash, freq="1D")
    return pf

if __name__ == "__main__":
    portfolio = RSI_Trading()
    if portfolio is not None:
        print("\n--- Portfolio Stats ---")
        print(portfolio.stats())
        # You can also plot the results if you have a graphical environment
        # portfolio.plot().show()
    else:
        print("\nBacktest could not be completed due to errors.")

