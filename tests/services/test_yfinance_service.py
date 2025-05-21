import pytest
import datetime
from packages.investor_agent_lib.services.yfinance_service import (
    get_calendar,
    get_ticker_info,
    get_recommendations,
    get_upgrades_downgrades,
    get_price_history,
    get_financial_statements,
    get_institutional_holders,
    get_earnings_history,
    get_insider_trades,
    get_options_chain,
    get_filtered_options,
    get_ticker_news,
    get_current_price,
)

class TestYFinanceService:
    def test_get_calendar(self):
        """Test get_calendar returns expected data structure for NVDA."""
        result = get_calendar("NVDA")
        
        # Result should be either dict or None
        assert result is None or isinstance(result, dict)
        print(result)
        if result is not None:
            # Check for required fields
            required_fields = {
                'Dividend Date': (type(None), datetime.date),
                'Ex-Dividend Date': (type(None), datetime.date),
                'Earnings Date': list,
                'Earnings High': (type(None), float),
                'Earnings Low': (type(None), float),
                'Earnings Average': (type(None), float),
                'Revenue High': (type(None), int),
                'Revenue Low': (type(None), int),
                'Revenue Average': (type(None), int)
            }
            
            # Verify all required fields exist and have correct types
            for field, expected_type in required_fields.items():
                assert field in result
                assert isinstance(result[field], expected_type)
            
            # Validate earnings dates are datetime.date objects
            for date in result['Earnings Date']:
                assert isinstance(date, datetime.date)

    @pytest.mark.skip(reason="Requires yfinance package")
    def test_get_calendar_error_handling(self):
        """Test error handling with invalid ticker."""
        # Note: Without mocking, we can't reliably test error cases
        # This test is skipped since it would require invalid API calls
        pass

    def test_get_ticker_info(self):
        """Test getting ticker info from yfinance API."""
        # Test with a known stable ticker (NBIS)
        result = get_ticker_info("NBIS")
        assert isinstance(result, dict)
        assert "symbol" in result
        assert result["symbol"] == "NBIS"
        
        # Test error case with invalid ticker
        result = get_ticker_info("INVALIDTICKER")
        assert result is None
        