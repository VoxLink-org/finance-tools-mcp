from packages.investor_agent_lib.services import yfinance_service
from config.my_paths import DATA_DIR
import pandas as pd
import talib as ta


def add_market_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add market indicators like VIX and SPY as additional features"""
    indicators = ["^VIX", "SPY"]
    # get the date range of the data
    start_date = data['date'].min()
    end_date = data['date'].max()
    
    # use cached data if available
    cache_name = DATA_DIR / f"market_data_{start_date}_{end_date}.pkl"
    if cache_name.exists():
        market_data = pd.read_pickle(cache_name)
    else:    
        print(f"Fetching market data from {start_date} to {end_date}...")
        raw = yfinance_service.download_history(
            indicators,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d",
        )

        if raw.empty:
            return data

        # calc some indicators for SPY
        spy_close = raw["Close"]["SPY"]
        spy_open = raw["Open"]["SPY"]
        spy_high = raw["High"]["SPY"]
        spy_low = raw["Low"]["SPY"]
        spy_volume = raw["Volume"]["SPY"]
        vix_close = raw["Close"]["^VIX"]
        vix_open = raw["Open"]["^VIX"]

        market_data = pd.DataFrame(index=raw.index)
        market_data["SPY_ADL"] = ta.AD(spy_high, spy_low, spy_close, spy_volume)
        market_data["SPY_ADL_5MA"] = market_data["SPY_ADL"].rolling(window=5).mean()
        # spy macd
        market_data['MACD'], market_data['MACD_Signal'], market_data['MACD_Hist'] = ta.MACD(
            spy_close, fastperiod=12, slowperiod=26, signalperiod=9)
        market_data['OPEN_CLOSE_RATIO'] = spy_open / spy_close
        

        market_data["SPY_ADL_CHG_PCT"] = (market_data["SPY_ADL"] - market_data["SPY_ADL_5MA"]) / market_data["SPY_ADL_5MA"]


        market_data["VIX_Change"] = vix_close.pct_change()
        market_data["VIX_Open_Close_Diff"] = vix_close - vix_open
        market_data["VIX_5MA"] = vix_close.rolling(window=5).mean()

        market_data.tail(100).to_csv("market_data.csv", index=False)
        # save to cache
        market_data.to_pickle(cache_name)

    # data['SPY_OPEN_CLOSE_RATIO'] = data['date'].map(market_data['OPEN_CLOSE_RATIO'])
    data['SPY_MACD_Hist'] = data['date'].map(market_data['MACD_Hist'])
    # data['SPY_ADL_CHG_PCT'] = data['date'].map(market_data['SPY_ADL_CHG_PCT'])
    # data['VIX_Change'] = data['date'].map(market_data['VIX_Change'])
    # data['VIX_Open_Close_Diff'] = data['date'].map(market_data['VIX_Open_Close_Diff'])
    # data['VIX_5MA'] = data['date'].map(market_data['VIX_5MA'])
    return data
