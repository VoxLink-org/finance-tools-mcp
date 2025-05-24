import pytest
import pandas as pd
from packages.investor_agent_lib.services.institution_service import get_digest_from_fintel

def test_get_digest_from_fintel():
    df = get_digest_from_fintel('AAPL')
    assert isinstance(df, dict)
    assert 'summary_text' in df
    assert 'activists' in df
    assert 'investors' in df


