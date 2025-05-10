import logging
from typing import Literal
import pandas as pd
from datetime import datetime, timedelta # Added imports

from .option_util import get_historical_options_by_ticker, get_options

logger = logging.getLogger(__name__)


def analyze_options(ticker_symbol: str) -> str:
    """Analyze historical and current options data for quantitative trading insights.
    
    Args:
        ticker_symbol: Stock ticker symbol to analyze
        
    Returns:
        str: Structured analysis containing:
        - Volume and open interest trends
        - Implied volatility patterns
        - Strike price distributions
        - Expiry date concentrations
        - Current vs historical comparisons
        - Option market factors (Put/Call ratios, activity/OI changes, IV dynamics)
    """
    historical = get_historical_options_by_ticker(ticker_symbol)
    current = get_options(ticker_symbol)
    
    if isinstance(historical, str) or isinstance(current, str):
        return f"Error analyzing options: {historical if isinstance(historical, str) else current}"
    
    # Track top 20 contracts by volume*OI
    tracked = current.nlargest(20, 'volume') if len(current) > 20 else current.copy()
    
    # 1. Volume and Open Interest Analysis
    vol_oi_analysis = f"""
    Volume & Open Interest Trends:
    - Current avg volume: {current['volume'].mean():.0f} vs historical {historical['volume'].mean():.0f}
    - Current avg OI: {current['openInterest'].mean():.0f} vs historical {historical['openInterest'].mean():.0f}
    - Volume/OI correlation: {historical[['volume', 'openInterest']].corr().iloc[0,1]:.2f}
    """
    
    # 2. Implied Volatility Analysis
    top_iv = current.nlargest(3, 'impliedVolatility')
    iv_analysis = f"""
    Implied Volatility Patterns:
    - Current avg IV: {current['impliedVolatility'].mean():.2%} vs historical {historical['impliedVolatility'].mean():.2%}
    - IV range: {current['impliedVolatility'].min():.2%}-{current['impliedVolatility'].max():.2%}
    - Highest IV contracts:
        * {top_iv.iloc[0]['contractSymbol']} (Exp: {top_iv.iloc[0]['expiryDate']}, Strike: {top_iv.iloc[0]['strike']:.2f}, IV: {top_iv.iloc[0]['impliedVolatility']:.2%})
        * {top_iv.iloc[1]['contractSymbol']} (Exp: {top_iv.iloc[1]['expiryDate']}, Strike: {top_iv.iloc[1]['strike']:.2f}, IV: {top_iv.iloc[1]['impliedVolatility']:.2%})
        * {top_iv.iloc[2]['contractSymbol']} (Exp: {top_iv.iloc[2]['expiryDate']}, Strike: {top_iv.iloc[2]['strike']:.2f}, IV: {top_iv.iloc[2]['impliedVolatility']:.2%})
    """
    
    # 3. Strike Price Distribution
    strikes = [f"{s:.2f}" for s in sorted(current['strike'].unique())]
    strike_analysis = f"""
    Strike Price Distribution:
    - Current strikes: {', '.join(strikes)}
    - Most concentrated strikes: {', '.join(f"{s:.2f}" for s in current['strike'].mode())}
    - Near-the-money concentration: {len(current[(abs(current['strike'] - current['underlyingPrice']) / current['underlyingPrice']) < 0.05])} contracts
    """
    
    # 4. Expiry Date Analysis
    expiry_analysis = f"""
    Expiry Date Concentrations:
    - Current expiries: {sorted(current['expiryDate'].unique())}
    - Most contracts expiring: {current['expiryDate'].mode().tolist()}
    - Days to nearest expiry: {(pd.to_datetime(current['expiryDate'].min()) - pd.to_datetime('today')).days}
    """
    
    # 5. Current vs Historical Comparison
    comparison = f"""
    Current vs Historical:
    - Volume change: {(current['volume'].mean() / historical['volume'].mean() - 1):.2%}
    - OI change: {(current['openInterest'].mean() / historical['openInterest'].mean() - 1):.2%}
    - IV change: {(current['impliedVolatility'].mean() / historical['impliedVolatility'].mean() - 1):.2%}
    """
    
    # 6. Option Market Factors
    # Put/Call ratios
    put_call_ratio = f"""
    Put/Call Ratios (Tracked Contracts):
    - Volume-based: {(tracked[tracked['optionType'] == 'P']['volume'].sum() / tracked[tracked['optionType'] == 'C']['volume'].sum()):.2f}
    - OI-based: {(tracked[tracked['optionType'] == 'P']['openInterest'].sum() / tracked[tracked['optionType'] == 'C']['openInterest'].sum()):.2f}
    """
    
    # Activity/OI changes
    activity_analysis = f"""
    Activity & OI Changes (Tracked Contracts):
    - Highest daily OI change: {tracked['openInterest'].pct_change().max():.2%}
    - Highest 5-day OI change: {tracked['openInterest'].pct_change(5).max():.2%}
    - Highest daily volume change: {tracked['volume'].pct_change().max():.2%}
    - Highest 5-day volume change: {tracked['volume'].pct_change(5).max():.2%}
    - Highest Volume/OI ratio: {(tracked['volume'] / tracked['openInterest']).max():.2f}
    """
    
    # IV dynamics
    iv_dynamics = f"""
    Implied Volatility Dynamics (Tracked Contracts):
    - IV trend (5-day): {tracked['impliedVolatility'].pct_change(5).mean():.2%}
    - IV/HV ratio: {(tracked['impliedVolatility'].mean() / historical['impliedVolatility'].mean()):.2f}
    - IV skew (5-day change): {(tracked[tracked['strike'] > tracked['underlyingPrice']]['impliedVolatility'].mean() - tracked[tracked['strike'] < tracked['underlyingPrice']]['impliedVolatility'].mean()):.2%}
    """
    
    # Hot contracts - calculate OI changes first
    tracked['oi_pct_change_5d'] = tracked['openInterest'].pct_change(5)
    hot_contracts = tracked.nlargest(3, 'volume')
    hot_analysis = f"""
    Hot Option Contracts (Highest Volume & OI Changes):
    - {hot_contracts.iloc[0]['contractSymbol']} (Type: {hot_contracts.iloc[0]['optionType']}, Volume: {hot_contracts.iloc[0]['volume']}, OI: {hot_contracts.iloc[0]['openInterest']}, 5d OI Δ: {hot_contracts.iloc[0]['oi_pct_change_5d']:.2%})
    - {hot_contracts.iloc[1]['contractSymbol']} (Type: {hot_contracts.iloc[1]['optionType']}, Volume: {hot_contracts.iloc[1]['volume']}, OI: {hot_contracts.iloc[1]['openInterest']}, 5d OI Δ: {hot_contracts.iloc[1]['oi_pct_change_5d']:.2%})
    - {hot_contracts.iloc[2]['contractSymbol']} (Type: {hot_contracts.iloc[2]['optionType']}, Volume: {hot_contracts.iloc[2]['volume']}, OI: {hot_contracts.iloc[2]['openInterest']}, 5d OI Δ: {hot_contracts.iloc[2]['oi_pct_change_5d']:.2%})
    """
    
    return "\n".join([
        f"Options Analysis for {ticker_symbol}",
        "="*40,
        vol_oi_analysis,
        iv_analysis,
        strike_analysis,
        expiry_analysis,
        comparison,
        put_call_ratio,
        activity_analysis,
        iv_dynamics,
        hot_analysis,
        "="*40
    ])

def contract_formatter(contract_symbol: str) -> str:
    """Format Yahoo Finance option contract symbol into human-readable string.
    
    Args:
        contract_symbol: Option contract symbol in Yahoo format (e.g. "AAPL220121C00150000")
        
    Returns:
        Formatted string like "C 150.0 2022-01-21"
        
    Raises:
        ValueError: If input is not a valid Yahoo option contract symbol
    """
    if not contract_symbol or len(contract_symbol) < 15:
        raise ValueError(f"Invalid contract symbol: {contract_symbol}")
        
    try:
        # Parse expiry (6 digits after ticker)
        expiry_str = contract_symbol[-15:-9]
        expiry = f"20{expiry_str[:2]}-{expiry_str[2:4]}-{expiry_str[4:6]}"
        
        # Parse option type (C/P)
        option_type = contract_symbol[-9]
        if option_type not in ('C', 'P'):
            raise ValueError(f"Invalid option type: {option_type}")
            
        # Parse strike price (remaining digits)
        strike_str = contract_symbol[-8:]
        strike = float(strike_str) / 1000  # Convert to decimal
        
        return f"{option_type} {strike} {expiry}"
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse contract symbol {contract_symbol}: {str(e)}")

def _calculate_skew(df: pd.DataFrame, underlying_price_col: str, iv_col: str, strike_col: str, option_type_col: str ='optionType') -> float:
    """Helper to calculate IV skew."""
    if df.empty or iv_col not in df.columns or underlying_price_col not in df.columns or strike_col not in df.columns:
        return float('nan')
    
    # Ensure underlying price is a single value for the df, or handle appropriately
    # For this use case, df is usually filtered to a single snapshot or contract's history where underlying_price_col is consistent for the skew calc
    
    otm_puts = df[(df[option_type_col] == 'P') & (df[strike_col] < df[underlying_price_col])]
    otm_calls = df[(df[option_type_col] == 'C') & (df[strike_col] > df[underlying_price_col])]
    
    avg_iv_otm_puts = otm_puts[iv_col].mean()
    avg_iv_otm_calls = otm_calls[iv_col].mean()
    
    if pd.isna(avg_iv_otm_puts) or pd.isna(avg_iv_otm_calls) or len(otm_puts) == 0 or len(otm_calls) == 0: # Ensure OTM options exist
        return float('nan')
    return avg_iv_otm_puts - avg_iv_otm_calls


def analyze_options_v2(ticker_symbol: str) -> str:
    """
    Analyzes option market factors for a given ticker symbol based on the top 30 contracts by volume.
    Focuses on:
    C. Option-based Factors (Based on 30 tracked contracts):
        - Overall option sentiment: Put/Call Ratios (Volume & OI) and their 5-day trends.
        - Activity and position changes: Significant daily/5-day OI & Volume changes, high Volume/OI ratio contracts.
        - Implied Volatility (IV) dynamics: IV trend for key contracts, IV vs. historical IV, 5-day IV Skew change.
        - "Hot" option contracts: Most significant Volume and OI changes.
    Args:
        ticker_symbol: The stock ticker symbol.
    Returns:
        A string containing the structured analysis.
    """
    analysis_parts = [f"Option Market Factors Analysis for {ticker_symbol} (Based on Top 30 Volume Contracts):"]
    analysis_parts.append("="*70)

    # 1. Fetch Data
    current_options_df = get_options(ticker_symbol)
    historical_options_df = get_historical_options_by_ticker(ticker_symbol)
    
    # Merge current and historical data, keeping most recent entries
    combined_df = pd.concat([current_options_df, historical_options_df])
    
    # Create unique key and drop duplicates (keep most recent lastTradeDate)
    combined_df['composite_key'] = combined_df['lastTradeDate'].astype(str) + combined_df['contractSymbol']
    combined_df = combined_df.sort_values('lastTradeDate', ascending=False)
    combined_df = combined_df.drop_duplicates('composite_key')
    
    # Filter out contracts with no open interest
    combined_df = combined_df[combined_df['openInterest'] > 0]
    

    if not isinstance(current_options_df, pd.DataFrame) or current_options_df.empty:
        return f"Error: Could not retrieve current options data for {ticker_symbol}."
    
    # Ensure date columns are datetime objects and handle potential errors
    try:
        current_options_df['snapshotDate'] = pd.to_datetime(current_options_df['snapshotDate'], errors='coerce')
        current_options_df['expiryDate'] = pd.to_datetime(current_options_df['expiryDate'], errors='coerce')
        current_options_df.dropna(subset=['snapshotDate', 'expiryDate'], inplace=True) # Drop rows where conversion failed
        if current_options_df.empty:
             return f"Error: No valid current options data after date conversion for {ticker_symbol}."

        if isinstance(historical_options_df, pd.DataFrame) and not historical_options_df.empty:
            historical_options_df['snapshotDate'] = pd.to_datetime(historical_options_df['snapshotDate'], errors='coerce')
            historical_options_df['expiryDate'] = pd.to_datetime(historical_options_df['expiryDate'], errors='coerce')
            historical_options_df.dropna(subset=['snapshotDate', 'expiryDate'], inplace=True)
    except Exception as e:
        logger.error(f"Error converting date columns for {ticker_symbol}: {e}")
        return f"Error: Data processing error during date conversion for {ticker_symbol}."


    # 2. Select Tracked Contracts
    if len(current_options_df) == 0:
        return f"Error: No current options data available for {ticker_symbol} to select tracked contracts."
    tracked_contracts_df = current_options_df.copy()
    if tracked_contracts_df.empty:
        analysis_parts.append("Warning: No contracts to track .")
        return "\n".join(analysis_parts)

    # --- Prepare historical data snapshots ---
    today_date = tracked_contracts_df['snapshotDate'].max()
    
    historical_data_5_days_ago = pd.DataFrame()
    closest_date_5_days_ago_actual = None
    if isinstance(historical_options_df, pd.DataFrame) and not historical_options_df.empty:
        date_5_days_ago_target = today_date - timedelta(days=5)
        historical_snapshot_dates = sorted(historical_options_df['snapshotDate'].dropna().unique())
        past_dates_5d = [d for d in historical_snapshot_dates if d <= date_5_days_ago_target]
        if past_dates_5d:
            closest_date_5_days_ago_actual = max(past_dates_5d)
            historical_data_5_days_ago = historical_options_df[
                historical_options_df['snapshotDate'] == closest_date_5_days_ago_actual
            ].copy()

    historical_data_1_day_ago = pd.DataFrame()
    closest_date_1_day_ago_actual = None
    if isinstance(historical_options_df, pd.DataFrame) and not historical_options_df.empty:
        date_1_day_ago_target = today_date - timedelta(days=1) # Target previous day
        # historical_snapshot_dates already sorted
        # Ensure we pick a date strictly before today_date, and closest to target
        past_dates_1d = [d for d in historical_snapshot_dates if d < today_date and d <= date_1_day_ago_target]
        if past_dates_1d: # If any such dates exist
            closest_date_1_day_ago_actual = max(past_dates_1d)
            historical_data_1_day_ago = historical_options_df[
                historical_options_df['snapshotDate'] == closest_date_1_day_ago_actual
            ].copy()
        elif historical_snapshot_dates and historical_snapshot_dates[-1] < today_date : # Fallback to most recent historical if no match for target
             # This case might be if today_date is far ahead of historical data
             # For "daily" change, we need the *immediately* preceding snapshot if target fails
             prev_dates = [d for d in historical_snapshot_dates if d < today_date]
             if prev_dates:
                closest_date_1_day_ago_actual = max(prev_dates)
                historical_data_1_day_ago = historical_options_df[
                    historical_options_df['snapshotDate'] == closest_date_1_day_ago_actual
                ].copy()


    # --- Create a combined DataFrame for analysis ---
    activity_df = tracked_contracts_df[[
        'contractSymbol', 'optionType', 'volume', 'openInterest',
        'impliedVolatility', 'strike', 'underlyingPrice', 'expiryDate', 'snapshotDate'
    ]].copy()
    activity_df.rename(columns={
        'volume': 'volume_current', 'openInterest': 'openInterest_current',
        'impliedVolatility': 'impliedVolatility_current', 'strike': 'strike_current',
        'underlyingPrice': 'underlyingPrice_current', 'expiryDate': 'expiryDate_current',
        'snapshotDate': 'snapshotDate_current'
    }, inplace=True)

    for hist_df, suffix, date_actual in [
        (historical_data_1_day_ago, '_1d_ago', closest_date_1_day_ago_actual),
        (historical_data_5_days_ago, '_5d_ago', closest_date_5_days_ago_actual)
    ]:
        if not hist_df.empty:
            activity_df = pd.merge(
                activity_df,
                hist_df[['contractSymbol', 'volume', 'openInterest', 'impliedVolatility', 'underlyingPrice', 'snapshotDate']],
                on='contractSymbol',
                how='left',
                suffixes=('', suffix) # Current df already has _current or no suffix yet for first merge
            )
            # Rename merged columns immediately to avoid conflict in next loop iteration
            rename_map = {}
            for col_base in ['volume', 'openInterest', 'impliedVolatility', 'underlyingPrice', 'snapshotDate']:
                if col_base + suffix not in activity_df.columns and col_base in hist_df.columns: # if merge created plain 'volume' from hist_df
                    rename_map[col_base] = col_base + suffix
                # if it's already suffixed by merge, fine.
            if rename_map: activity_df.rename(columns=rename_map, inplace=True)

        else: # Ensure columns exist even if historical data is missing for that period
            for col_base in ['volume', 'openInterest', 'impliedVolatility', 'underlyingPrice', 'snapshotDate']:
                activity_df[col_base + suffix] = float('nan')
        activity_df['date' + suffix] = pd.to_datetime(date_actual if date_actual else float('nan'))


    # Calculate changes
    for period_name, period_suffix in [('1d', '_1d_ago'), ('5d', '_5d_ago')]:
        for metric in ['openInterest', 'volume', 'impliedVolatility']:
            current_col = f"{metric}_current"
            prev_col = f"{metric}{period_suffix}"
            change_col = f"{metric}_change_{period_name}"
            pct_change_col = f"{metric}_pct_change_{period_name}"

            if current_col in activity_df.columns and prev_col in activity_df.columns:
                activity_df[change_col] = activity_df[current_col] - activity_df[prev_col]
                activity_df[pct_change_col] = (activity_df[change_col] / activity_df[prev_col].replace(0, float('nan'))).replace([float('inf'), -float('inf')], float('nan')) # Avoid div by zero
            else:
                activity_df[change_col] = float('nan')
                activity_df[pct_change_col] = float('nan')
    
    # --- 3. Overall Option Sentiment ---
    sentiment_parts = ["\n--- 1. Overall Option Sentiment (Tracked Contracts) ---"]
    current_puts_tracked = activity_df[activity_df['optionType'] == 'P']
    current_calls_tracked = activity_df[activity_df['optionType'] == 'C']

    current_pc_volume_ratio = float('nan')
    if not current_calls_tracked.empty and current_calls_tracked['volume_current'].sum() > 0:
        current_pc_volume_ratio = current_puts_tracked['volume_current'].sum() / current_calls_tracked['volume_current'].sum()
    
    current_pc_oi_ratio = float('nan')
    if not current_calls_tracked.empty and current_calls_tracked['openInterest_current'].sum() > 0:
        current_pc_oi_ratio = current_puts_tracked['openInterest_current'].sum() / current_calls_tracked['openInterest_current'].sum()

    sentiment_parts.append(f"  Put/Call Ratio (Current):")
    sentiment_parts.append(f"    - Based on Volume: {current_pc_volume_ratio:.2f}")
    sentiment_parts.append(f"    - Based on Open Interest: {current_pc_oi_ratio:.2f}")

    # 5-day trend of P/C Ratios for tracked contracts
    pc_vol_ratio_5d_ago = float('nan')
    pc_oi_ratio_5d_ago = float('nan')
    date_5d_label = f"on {activity_df['date_5d_ago'].dt.strftime('%Y-%m-%d').mode()[0]}" if not activity_df['date_5d_ago'].isna().all() else "N/A"

    if 'volume_5d_ago' in activity_df.columns:
        puts_5d_ago = activity_df[(activity_df['optionType'] == 'P') & activity_df['volume_5d_ago'].notna()]
        calls_5d_ago = activity_df[(activity_df['optionType'] == 'C') & activity_df['volume_5d_ago'].notna()]
        if not calls_5d_ago.empty and calls_5d_ago['volume_5d_ago'].sum() > 0:
            pc_vol_ratio_5d_ago = puts_5d_ago['volume_5d_ago'].sum() / calls_5d_ago['volume_5d_ago'].sum()
        
        if not calls_5d_ago.empty and calls_5d_ago['openInterest_5d_ago'].sum() > 0: # Assuming OI_5d_ago exists
             pc_oi_ratio_5d_ago = puts_5d_ago['openInterest_5d_ago'].sum() / calls_5d_ago['openInterest_5d_ago'].sum()


    if not pd.isna(pc_vol_ratio_5d_ago) and not pd.isna(current_pc_volume_ratio) and pc_vol_ratio_5d_ago != 0:
        pc_vol_trend = (current_pc_volume_ratio / pc_vol_ratio_5d_ago - 1)
        sentiment_parts.append(f"  5-Day Trend of Put/Call Volume Ratio: {pc_vol_trend:+.2%}")
        sentiment_parts.append(f"    (Current: {current_pc_volume_ratio:.2f}, 5 Days Ago ({date_5d_label}): {pc_vol_ratio_5d_ago:.2f})")
    else:
        sentiment_parts.append(f"  5-Day Trend of Put/Call Volume Ratio: Data unavailable or insufficient.")

    if not pd.isna(pc_oi_ratio_5d_ago) and not pd.isna(current_pc_oi_ratio) and pc_oi_ratio_5d_ago != 0:
        pc_oi_trend = (current_pc_oi_ratio / pc_oi_ratio_5d_ago - 1)
        sentiment_parts.append(f"  5-Day Trend of Put/Call OI Ratio: {pc_oi_trend:+.2%}")
        sentiment_parts.append(f"    (Current: {current_pc_oi_ratio:.2f}, 5 Days Ago ({date_5d_label}): {pc_oi_ratio_5d_ago:.2f})")
    else:
        sentiment_parts.append(f"  5-Day Trend of Put/Call OI Ratio: Data unavailable or insufficient.")
    analysis_parts.extend(sentiment_parts)

    # --- 4. Activity and Position Changes ---
    activity_section_parts = ["\n--- 2. Activity and Position Changes (Tracked Contracts) ---"]
    for period_name, period_label, date_col_suffix in [('Daily', '1d', '_1d_ago'), ('5-Day', '5d', '_5d_ago')]:
        date_label = f"on {activity_df[f'date{date_col_suffix}'].dt.strftime('%Y-%m-%d').mode()[0]}" if not activity_df[f'date{date_col_suffix}'].isna().all() else "N/A"
        
        activity_section_parts.append(f"  Significant {period_name} OI Changes (Top 3 by abs % change, then abs change), from {date_label}:")
        oi_change_col = f'openInterest_change_{period_label}'
        oi_pct_change_col = f'openInterest_pct_change_{period_label}'
        if oi_change_col in activity_df.columns and not activity_df[oi_change_col].isna().all():
            # Sort by abs pct change, then by abs change for tie-breaking or if pct change is NaN (e.g. from zero base)
            sorted_oi_changes = activity_df.copy()
            sorted_oi_changes['abs_oi_pct_change'] = sorted_oi_changes[oi_pct_change_col].abs()
            sorted_oi_changes['abs_oi_change'] = sorted_oi_changes[oi_change_col].abs()
            sorted_oi_changes = sorted_oi_changes.sort_values(by=['abs_oi_pct_change', 'abs_oi_change'], ascending=[False, False]).dropna(subset=[oi_change_col])
            
            for _, row in sorted_oi_changes.head(3).iterrows():
                activity_section_parts.append(f"    - {contract_formatter(row['contractSymbol'])} ({row['optionType']}): OI Chg: {row[oi_change_col]:+.0f} ({row[oi_pct_change_col]:+.2%}), Curr OI: {row['openInterest_current']:.0f}")
        else:
            activity_section_parts.append(f"    - Data unavailable for {period_name} OI changes.")

        activity_section_parts.append(f"  Significant {period_name} Volume Changes (Top 3 by abs % change, then abs change), from {date_label}:")
        vol_change_col = f'volume_change_{period_label}'
        vol_pct_change_col = f'volume_pct_change_{period_label}'
        if vol_change_col in activity_df.columns and not activity_df[vol_change_col].isna().all():
            sorted_vol_changes = activity_df.copy()
            sorted_vol_changes['abs_vol_pct_change'] = sorted_vol_changes[vol_pct_change_col].abs()
            sorted_vol_changes['abs_vol_change'] = sorted_vol_changes[vol_change_col].abs()
            sorted_vol_changes = sorted_vol_changes.sort_values(by=['abs_vol_pct_change', 'abs_vol_change'], ascending=[False, False]).dropna(subset=[vol_change_col])

            for _, row in sorted_vol_changes.head(3).iterrows():
                activity_section_parts.append(f"    - {contract_formatter(row['contractSymbol'])} ({row['optionType']}): Vol Chg: {row[vol_change_col]:+.0f} ({row[vol_pct_change_col]:+.2%}), Curr Vol: {row['volume_current']:.0f}")
        else:
            activity_section_parts.append(f"    - Data unavailable for {period_name} volume changes.")

    activity_df['volume_oi_ratio'] = (activity_df['volume_current'] / activity_df['openInterest_current'].replace(0, float('nan'))).replace([float('inf'), -float('inf')], float('nan'))
    activity_section_parts.append("  Contracts with High Volume/OI Ratio (Top 3):")
    if not activity_df['volume_oi_ratio'].isna().all():
        top_vol_oi_ratio = activity_df.sort_values(by='volume_oi_ratio', ascending=False).dropna(subset=['volume_oi_ratio']).head(3)
        for _, row in top_vol_oi_ratio.iterrows():
            activity_section_parts.append(f"    - {contract_formatter(row['contractSymbol'])} ({row['optionType']}): Ratio: {row['volume_oi_ratio']:.2f} (Vol: {row['volume_current']:.0f}, OI: {row['openInterest_current']:.0f})")
    else:
        activity_section_parts.append("    - Data unavailable.")
    analysis_parts.extend(activity_section_parts)

    # --- 5. Implied Volatility (IV) Dynamics ---
    iv_dynamics_parts = ["\n--- 3. Implied Volatility (IV) Dynamics (Tracked Contracts) ---"]
    # IV Trend for Key Contracts (near-month ATM among tracked)
    if not activity_df.empty and 'impliedVolatility_current' in activity_df.columns:
        key_contract_df = activity_df.copy()
        key_contract_df['moneyness_abs'] = (key_contract_df['strike_current'] - key_contract_df['underlyingPrice_current']).abs()
        key_contract_df_sorted = key_contract_df.sort_values(by=['expiryDate_current', 'moneyness_abs'])
        
        if not key_contract_df_sorted.empty:
            key_contract = key_contract_df_sorted.iloc[0]
            iv_dynamics_parts.append(f"  IV Trend for a Key Contract (Nearest Expiry, Closest to ATM among Tracked - {key_contract['contractSymbol']}):")
            iv_dynamics_parts.append(f"    - Current IV: {key_contract['impliedVolatility_current']:.4f}")
            if not pd.isna(key_contract['impliedVolatility_change_1d']):
                iv_dynamics_parts.append(f"    - Daily IV Change: {key_contract['impliedVolatility_change_1d']:+.4f} ({key_contract['impliedVolatility_pct_change_1d']:+.2%})")
            if not pd.isna(key_contract['impliedVolatility_change_5d']):
                iv_dynamics_parts.append(f"    - 5-Day IV Change: {key_contract['impliedVolatility_change_5d']:+.4f} ({key_contract['impliedVolatility_pct_change_5d']:+.2%})")
        else:
            iv_dynamics_parts.append("  Could not identify a key contract for IV trend analysis.")
    
    # IV vs Historical IV (Market average from 5 days ago as baseline)
    current_avg_iv_tracked = activity_df['impliedVolatility_current'].mean()
    market_avg_iv_5d_ago = historical_data_5_days_ago['impliedVolatility'].mean() if not historical_data_5_days_ago.empty else float('nan')
    date_5d_label_market = f"on {closest_date_5_days_ago_actual.strftime('%Y-%m-%d')}" if closest_date_5_days_ago_actual else "N/A"

    if not pd.isna(current_avg_iv_tracked) and not pd.isna(market_avg_iv_5d_ago) and market_avg_iv_5d_ago != 0:
        iv_vs_hist_iv_ratio = current_avg_iv_tracked / market_avg_iv_5d_ago
        iv_dynamics_parts.append(f"  Current Avg IV (Tracked) vs Market Avg IV (5 Days Ago - {date_5d_label_market}):")
        iv_dynamics_parts.append(f"    - Ratio: {iv_vs_hist_iv_ratio:.2f} (Current Tracked Avg IV: {current_avg_iv_tracked:.4f}, Market Avg IV 5D Ago: {market_avg_iv_5d_ago:.4f})")
        iv_dynamics_parts.append(f"    (Note: Using market avg IV from 5 days ago as a baseline, not true HV of underlying.)")
    else:
        iv_dynamics_parts.append("  IV vs Historical Market IV comparison data unavailable.")

    # IV Skew (5-day change for tracked contracts)
    current_skew = _calculate_skew(activity_df, 'underlyingPrice_current', 'impliedVolatility_current', 'strike_current')
    skew_5d_ago = _calculate_skew(activity_df, 'underlyingPrice_5d_ago', 'impliedVolatility_5d_ago', 'strike_current') if 'underlyingPrice_5d_ago' in activity_df else float('nan')

    iv_dynamics_parts.append(f"  IV Skew (OTM Puts IV - OTM Calls IV, based on Tracked Contracts):")
    if not pd.isna(current_skew):
        iv_dynamics_parts.append(f"    - Current Skew: {current_skew:+.4f}")
    else:
        iv_dynamics_parts.append("    - Current Skew: Data unavailable (ensure enough OTM Puts/Calls in tracked set).")
    
    if not pd.isna(skew_5d_ago):
        iv_dynamics_parts.append(f"    - Skew 5 Days Ago ({date_5d_label}): {skew_5d_ago:+.4f}")
        if not pd.isna(current_skew):
            skew_change_5d = current_skew - skew_5d_ago
            iv_dynamics_parts.append(f"    - 5-Day Change in Skew: {skew_change_5d:+.4f}")
        else:
            iv_dynamics_parts.append("    - 5-Day Change in Skew: Current skew needed for change calculation.")
    else:
        iv_dynamics_parts.append(f"    - Skew 5 Days Ago ({date_5d_label}): Data unavailable.")
        iv_dynamics_parts.append("    - 5-Day Change in Skew: Historical skew needed for change calculation.")
    analysis_parts.extend(iv_dynamics_parts)

    # --- 6. "Hot" Option Contracts ---
    hot_contracts_parts = ["\n--- 4. 'Hot' Option Contracts (Tracked Contracts) ---"]
    # Prioritize 1-day % changes if available, else 5-day for "hotness"
    oi_chg_period = '1d' if not activity_df[f'openInterest_pct_change_1d'].isna().all() else '5d'
    vol_chg_period = '1d' if not activity_df[f'volume_pct_change_1d'].isna().all() else '5d'

    hot_contracts_parts.append(f"  (Hotness based on {'Daily' if oi_chg_period == '1d' else '5-Day'} OI % Chg & {'Daily' if vol_chg_period == '1d' else '5-Day'} Vol % Chg)")

    activity_df['abs_oi_pct_change_hot'] = activity_df[f'openInterest_pct_change_{oi_chg_period}'].abs()
    activity_df['abs_vol_pct_change_hot'] = activity_df[f'volume_pct_change_{vol_chg_period}'].abs()

    hot_by_oi = activity_df.sort_values(by='abs_oi_pct_change_hot', ascending=False).dropna(subset=['abs_oi_pct_change_hot']).head(3)
    hot_contracts_parts.append(f"  Top 3 by OI % Change ({oi_chg_period}):")
    if not hot_by_oi.empty:
        for _, row in hot_by_oi.iterrows():
            hot_contracts_parts.append(f"    - {contract_formatter(row['contractSymbol'])} ({row['optionType']}): OI %Δ ({oi_chg_period}): {row[f'openInterest_pct_change_{oi_chg_period}']:.2%}, Curr OI: {row['openInterest_current']:.0f}")
    else:
        hot_contracts_parts.append("    - Data unavailable.")
            
    hot_by_volume = activity_df.sort_values(by='abs_vol_pct_change_hot', ascending=False).dropna(subset=['abs_vol_pct_change_hot']).head(3)
    hot_contracts_parts.append(f"  Top 3 by Volume % Change ({vol_chg_period}):")
    if not hot_by_volume.empty:
        for _, row in hot_by_volume.iterrows():
            hot_contracts_parts.append(f"    - {contract_formatter(row['contractSymbol'])} ({row['optionType']}): Vol %Δ ({vol_chg_period}): {row[f'volume_pct_change_{vol_chg_period}']:.2%}, Curr Vol: {row['volume_current']:.0f}")
    else:
        hot_contracts_parts.append("    - Data unavailable.")
    analysis_parts.extend(hot_contracts_parts)
    
    analysis_parts.append("\n" + "="*70)
    return "\n".join(analysis_parts)


if __name__ == "__main__":
    print(analyze_options_v2("AAPL"))
    # Example for v2, assuming you have data for these tickers
    # print("\n\n--- AAPL V2 ---")
    # print(analyze_options_v2("AAPL"))
    # print("\n\n--- TSLA V2 ---")
    # print(analyze_options_v2("TSLA"))