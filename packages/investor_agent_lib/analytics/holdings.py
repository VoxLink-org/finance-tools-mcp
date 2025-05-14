from tabulate import tabulate
import pandas as pd
from packages.investor_agent_lib.services import whalewisdom_service


def analyze_institutional_holdings(ticker: str, top_n: int = 5) -> str:
    """Analyze 13F 13D/G to get institutional holdings data from the last 6 months and return a formatted digest.

    Args:
        ticker: Stock ticker symbol.
        top_n: The number of institutions to list for each category.

    Returns:
        str: Formatted analysis of institutional holdings.
    """
    try:
        df = whalewisdom_service.get_whalewisdom_holdings(ticker)
    except Exception as e:
        return f"Error fetching holdings data for {ticker.upper()}: {str(e)}"

    if df.empty:
        return f"No institutional holdings data available for {ticker.upper()} in the last 6 months."

    # Ensure 'position_change_type' is string and handle potential NaN values for safe string operations
    df['position_change_type'] = df['position_change_type'].astype(str).fillna('')
    # Ensure numeric columns are indeed numeric
    df['shares_change'] = pd.to_numeric(df['shares_change'], errors='coerce').fillna(0)
    df['percent_ownership'] = pd.to_numeric(df['percent_ownership'], errors='coerce').fillna(0)
    df['percent_change'] = pd.to_numeric(df['percent_change'], errors='coerce').fillna(0)


    result_parts = [f"Institutional Holdings Analysis for {ticker.upper()} (Last 6 Months, Top {top_n})\n"]

    # 1. Top N Institutions with Largest Net Increase in Shares
    increased_df = df[df['shares_change'] > 0].sort_values(by='shares_change', ascending=False)
    result_parts.append(f"\n--- Top {top_n} Institutions with Largest Net Increase in Shares ---\n")
    if not increased_df.empty:
        table_data = increased_df[['name', 'shares_change', 'percent_ownership', 'source_date']].head(top_n).copy()
        table_data.columns = ["Institution", "Net Shares Added", "Current Stake (%)", "Source Date"]
        result_parts.append(tabulate(table_data, headers="keys", tablefmt="pipe", floatfmt=(None, ",.0f", ".2%", None), showindex=False))
    else:
        result_parts.append("No institutions found with a net increase in shares.")
    result_parts.append("\n")

    # 2. Top N Institutions with Largest Net Decrease in Shares
    decreased_df = df[df['shares_change'] < 0].sort_values(by='shares_change', ascending=True)  # most negative first
    result_parts.append(f"\n--- Top {top_n} Institutions with Largest Net Decrease in Shares ---\n")
    if not decreased_df.empty:
        table_data = decreased_df[['name', 'shares_change', 'percent_ownership', 'source_date']].head(top_n).copy()
        table_data.columns = ["Institution", "Net Shares Sold", "Current Stake (%)", "Source Date"]
        result_parts.append(tabulate(table_data, headers="keys", tablefmt="pipe", floatfmt=(None, ",.0f", ".2%", None), showindex=False))
    else:
        result_parts.append("No institutions found with a net decrease in shares.")
    result_parts.append("\n")

    # 3. Top N Institutions with Largest Percentage Increase in Ownership (% of Stock)
    perc_increase_df = df[df['percent_change'] > 0].sort_values(by='percent_change', ascending=False)
    result_parts.append(f"\n--- Top {top_n} Institutions with Largest Percentage Increase in Ownership ---\n")
    if not perc_increase_df.empty:
        table_data = perc_increase_df[['name', 'percent_change', 'percent_ownership', 'source_date']].head(top_n).copy()
        table_data.columns = ["Institution", "% Change in Ownership", "Current Stake (%)", "Source Date"]
        result_parts.append(tabulate(table_data, headers="keys", tablefmt="pipe", floatfmt=(None, ",.0f", ".2%", None), showindex=False))
    else:
        result_parts.append("No institutions found with a percentage increase in ownership.")
    result_parts.append("\n")

    # 4. Top N New Institutions with Largest Holdings
    new_holders_df = df[df['position_change_type'].str.lower() == 'new'].sort_values(by='shares_change', ascending=False)
    result_parts.append(f"\n--- Top {top_n} New Institutions with Largest Holdings ---\n")
    if not new_holders_df.empty:
        # For new positions, 'shares_change' is their total current holding.
        table_data = new_holders_df[['name', 'shares_change', 'percent_ownership', 'source_date']].head(top_n).copy()
        table_data.columns = ["Institution", "Shares Held (New Position)", "Current Stake (%)", "Source Date"]
        result_parts.append(tabulate(table_data, headers="keys", tablefmt="pipe", floatfmt=(None, ",.0f", ".2%", None), showindex=False))
    else:
        result_parts.append("No new institutions found (this category relies on 'new' as position_change_type).")
    result_parts.append("\n")

    # 5. Top N Institutions That Sold Out (Largest Prior Holdings)
    sold_out_filter = df['position_change_type'].str.lower().isin(['sold out', 'sold_out'])
    sold_out_df = df[sold_out_filter].copy()
    result_parts.append(f"\n--- Top {top_n} Institutions That Sold Out (Largest Prior Holdings) ---\n")
    if not sold_out_df.empty:
        sold_out_df['prior_shares_held'] = sold_out_df['shares_change'].abs()
        # Sort by the absolute value of shares changed (prior holding size)
        sold_out_df = sold_out_df.sort_values(by='prior_shares_held', ascending=False)
        table_data = sold_out_df[['name', 'prior_shares_held']].head(top_n).copy()
        table_data.columns = ["Institution", "Prior Shares Held"]
        result_parts.append(tabulate(table_data, headers="keys", tablefmt="pipe", floatfmt=(None, ",.0f"), showindex=False))
    else:
        result_parts.append("No institutions found that fully sold out (this category relies on 'sold out' or 'sold_out' as position_change_type).")
    result_parts.append("\n")

    return "".join(result_parts)

if __name__ == '__main__':
    print(analyze_institutional_holdings('se'))