import pytest
import pandas as pd
from packages.investor_agent_lib.services.enhance_option_service import get_option_greeks_and_ind

def test_get_option_greeks_and_ind():
    """Test get_option_greeks_and_ind with TQQQ ticker."""
    # Call the function with TQQQ
    result = get_option_greeks_and_ind("TQQQ")
    
    print(result)

    # Verify return type is DataFrame
    assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
    
    # Verify DataFrame is not empty
    assert not result.empty, "DataFrame should not be empty"
    
    # Verify expected columns exist (adjust based on actual columns)
    expected_cols = ['date', 'atm_iv_avg', 'call_delta', 'call_rho', 'skew_measure', 'ticker']
    for col in expected_cols:
        assert col in result.columns, f"Missing expected column: {col}"
    
    assert result['date'].is_monotonic_increasing, "Data should be sorted by date ascending"
    

if __name__ == "__main__":
    pytest.main([__file__])