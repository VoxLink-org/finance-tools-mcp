import numpy as np
import pandas as pd
from tabulate import tabulate


def analyze_option_indicators_greeks(time_series_data: pd.DataFrame) -> str:
    """
    Analyzes the latest option indicators and Greeks from time series data
    and returns a concise digest string suitable for LLM consumption.

    Args:
        time_series_data: DataFrame containing time series of option indicators and Greeks.
                          Expected columns: atm_iv_avg, call_delta, call_rho, call_theta, date,
                          gamma, lastTradeDate, pc_ratio, put_delta, put_rho, put_theta,
                          skew_measure, term_structure_slope, ticker, underlyingPrice, vega.

    Returns:
        A string summarizing the key option metrics and their implications.
    """
    if time_series_data.empty:
        return "No option indicator data available."

    # Ensure data is sorted by date if multiple entries exist, take the latest
    if 'date' in time_series_data.columns:
        # Convert to datetime if not already, handling potential errors
        time_series_data['date'] = pd.to_datetime(time_series_data['date'], errors='coerce')
        # Drop rows where date conversion failed
        time_series_data = time_series_data.dropna(subset=['date'])
        if time_series_data.empty:
             return "No valid date entries found in option indicator data."
        latest_data = time_series_data.sort_values(by='date', ascending=False).iloc[0]
    else:
        # Fallback if no date column, assume the last row is the latest
        latest_data = time_series_data.iloc[-1]

    ticker = latest_data.get('ticker', 'N/A')
    report_date_obj = latest_data.get('date')
    report_date = report_date_obj.strftime('%Y-%m-%d') if pd.notna(report_date_obj) else 'N/A'
    underlying_price = latest_data.get('underlyingPrice', np.nan)

    # Key Indicators
    atm_iv = latest_data.get('atm_iv_avg', np.nan)
    pc_ratio = latest_data.get('pc_ratio', np.nan)
    skew = latest_data.get('skew_measure', np.nan)
    term_structure = latest_data.get('term_structure_slope', np.nan)

    # Greeks
    call_delta = latest_data.get('call_delta', np.nan)
    put_delta = latest_data.get('put_delta', np.nan) # Note: Put delta is typically negative
    gamma = latest_data.get('gamma', np.nan)
    vega = latest_data.get('vega', np.nan)
    call_theta = latest_data.get('call_theta', np.nan) # Note: Theta is typically negative
    put_theta = latest_data.get('put_theta', np.nan)   # Note: Theta is typically negative
    call_rho = latest_data.get('call_rho', np.nan)
    put_rho = latest_data.get('put_rho', np.nan)

    # --- Analysis Digest Generation ---
    digest_parts = []
    title = f"**Option Indicators & Greeks Analysis for {ticker} as of {report_date}"
    if pd.notna(underlying_price):
        title += f" (Underlying: ${underlying_price:.2f})**"
    else:
        title += "**"
    digest_parts.append(title + "\n")


    # Volatility Analysis
    digest_parts.append(f"- **Implied Volatility (ATM IV):** {atm_iv:.2%}" if pd.notna(atm_iv) else "- ATM IV: N/A")
    digest_parts.append(f"- **Put/Call Ratio:** {pc_ratio:.2f}" if pd.notna(pc_ratio) else "- P/C Ratio: N/A")
    if pd.notna(pc_ratio):
        if pc_ratio > 1.0:
            digest_parts[-1] += " (Suggests bearish sentiment or high hedging activity)"
        elif pc_ratio < 0.7:
            digest_parts[-1] += " (Suggests bullish sentiment)"
        else:
            digest_parts[-1] += " (Suggests relatively neutral sentiment)"

    # Skew and Term Structure
    digest_parts.append(f"- **Volatility Skew:** {skew:.4f}" if pd.notna(skew) else "- Skew: N/A")
    if pd.notna(skew):
         digest_parts[-1] += " (Measures relative IV of OTM puts vs OTM calls. Negative typical for equities)"
    digest_parts.append(f"- **Term Structure Slope:** {term_structure:.4f}" if pd.notna(term_structure) else "- Term Structure: N/A")
    if pd.notna(term_structure):
        if term_structure > 0:
             digest_parts[-1] += " (Contango: Longer-term IV > Shorter-term IV, typical market state)"
        else:
             digest_parts[-1] += " (Backwardation: Shorter-term IV > Longer-term IV, suggests near-term event/stress)"

    # Greeks Summary
    digest_parts.append("\n**Key Greeks (Latest Aggregate/Average):**")
    greeks_summary = []
    if pd.notna(call_delta) and pd.notna(put_delta):
        greeks_summary.append(f"  - Delta (Call/Put): {call_delta:.2f} / {put_delta:.2f} (Sensitivity to $1 underlying price change)")
    if pd.notna(gamma):
        greeks_summary.append(f"  - Gamma: {gamma:.4f} (Rate of change of Delta per $1 underlying move)")
    if pd.notna(vega):
        greeks_summary.append(f"  - Vega: {vega:.4f} (Sensitivity to 1% change in IV)")
    if pd.notna(call_theta) and pd.notna(put_theta):
        # Theta is often presented per day, ensure sign convention is clear (usually negative)
        greeks_summary.append(f"  - Theta (Call/Put): {call_theta:.4f} / {put_theta:.4f} (Daily value decay)")
    if pd.notna(call_rho) and pd.notna(put_rho):
        greeks_summary.append(f"  - Rho (Call/Put): {call_rho:.4f} / {put_rho:.4f} (Sensitivity to 1% interest rate change)")

    if not greeks_summary:
        digest_parts.append("  - No Greek data available.")
    else:
        digest_parts.extend(greeks_summary)

    # Potential Enhancements (Commented out for now, requires trend analysis)
    # - Trend analysis for IV, P/C ratio, Skew over the period
    # - Comparison to historical levels or sector averages

    return "\n".join(digest_parts)