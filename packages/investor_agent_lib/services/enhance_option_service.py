import pandas as pd
from packages.investor_agent_lib.options import option_data, option_indicators

def get_option_greeks_and_ind(ticker):
    basic_indicators = option_indicators.calculate_indicators(ticker)
    greeks = option_indicators.calculate_greeks(ticker)
    # convert to dataframe

    merged_dict = {**basic_indicators, **greeks}
    current = pd.DataFrame([{k: merged_dict[k] for k in sorted(merged_dict)}])


    historical = option_data.get_historical_option_indicator_by_ticker(ticker)

    # concatenate the dataframes, drop duplicates and sort by date
    merged_df = pd.concat([historical, current], axis=0)
    merged_df["lastTradeDate"] = pd.to_datetime(merged_df["lastTradeDate"])
    merged_df = merged_df.sort_values(by='lastTradeDate', ascending=True)
    merged_df = merged_df.drop_duplicates(subset='lastTradeDate', keep='first')
    merged_df = merged_df.reset_index(drop=True)

    return merged_df