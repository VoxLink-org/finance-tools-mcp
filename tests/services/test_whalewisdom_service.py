import pytest
import pandas as pd
from packages.investor_agent_lib.services.whalewisdom_service import get_whalewisdom_holdings


def test_get_whalewisdom_holdings_returns_list():
    """Test that get_whalewisdom_holdings returns a list for valid ticker."""
    ticker = "NVDA"
    result = get_whalewisdom_holdings(ticker)
    
    assert isinstance(result, pd.DataFrame), "Should return a DataFrame"
    assert not result.empty, "Should not return an empty DataFrame"
    assert "name" in result.columns, "Should have 'name' column"
    assert "percent_ownership" in result.columns, "Should have 'percent_ownership' column"
    assert "position_change_type" in result.columns, "Should have 'position_change_type' column"
    assert "percent_change" in result.columns, "Should have 'percent_change' column"
    assert "source_date" in result.columns, "Should have 'source_date' column"
    assert "filing_date" in result.columns, "Should have 'filing_date' column"

def test_get_whalewisdom_holdings_invalid_ticker():
    """Test that invalid ticker returns empty list or raises appropriate error."""
    with pytest.raises(Exception):
        get_whalewisdom_holdings("INVALIDTICKER")