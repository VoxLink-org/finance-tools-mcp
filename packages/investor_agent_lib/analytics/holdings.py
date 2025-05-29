from tabulate import tabulate
import pandas as pd
from packages.investor_agent_lib.services import institution_service


def analyze_institutional_holdings_v2(ticker: str) -> str:
    """Analyze institutional holdings data and return structured summary.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        str: Structured summary with:
            - summary: General holdings overview
            - institutions: Top institutions
            - activists: Activist investor activity
    """
    data = institution_service.get_digest_from_fintel(ticker)
    investors: pd.DataFrame = data['investors']
    activists: pd.DataFrame = data['activists']
    
    # Process investors data
    investors['value_change_pct'] = pd.to_numeric(investors['ΔValue (%)'], errors='coerce')
    
    # Get top 10 decreasing positions
    top_investors = (
        investors
        .sort_values('Shares (MM)', ascending=False)
        .rename(columns={
            'Owner': 'investor',
            'Shares (MM)': 'shares_millions',
            'ΔShares (%)': 'share_change_pct'
        })
        
    )
        
    # Process activists data
    activists_summary = activists if not activists.empty else []
    
    return f"""
{data['summary_text']}

------Top institutions------
{tabulate(top_investors[['investor', 'shares_millions', 'share_change_pct']], headers="keys", tablefmt="simple", showindex=False)}


------Activists------
{tabulate(activists_summary, headers="keys", tablefmt="simple", showindex=False)}
"""


def get_top25_holder(ticker: str) -> str:
    df = institution_service.get_whalewisdom_holdings(ticker)
    return f"""
=============Top 25 Holders============
{df[["name", "current_shares", "position_change_type", "source_date", "shares_change", "percent_change_in_%"]].to_markdown(index=False)}
"""

if __name__ == '__main__':
    print(get_top25_holder('googl'))