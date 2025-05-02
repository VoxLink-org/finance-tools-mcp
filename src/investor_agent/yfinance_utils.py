from concurrent.futures import ThreadPoolExecutor
import logging
from requests import Session
from typing import Literal
from datetime import datetime

from pyrate_limiter import Duration, RequestRate, Limiter
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# Create a session class that combines caching and rate limiting
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

# Create a session with rate limiting and caching
session = CachedLimiterSession(
    limiter=Limiter(RequestRate(5, Duration.SECOND)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache", expire_after=3600),
)

def get_ticker_info(ticker: str) -> dict | None:
    try:
        return yf.Ticker(ticker, session=session).get_info()
    except Exception as e:
        logger.error(f"Error retrieving ticker info for {ticker}: {e}", exc_info=True)
        return None

def get_calendar(ticker: str) -> dict | None:
    """Get calendar events including earnings and dividend dates."""
    try:
        return yf.Ticker(ticker, session=session).get_calendar()
    except Exception as e:
        logger.error(f"Error retrieving calendar for {ticker}: {e}", exc_info=True)
        return None

def get_recommendations(ticker: str, limit: int = 5) -> pd.DataFrame | None:
    """Get analyst recommendations.
    Returns DataFrame with columns: Firm, To Grade, From Grade, Action
    Limited to most recent entries by default.
    """
    try:
        df = yf.Ticker(ticker, session=session).get_recommendations()
        return df.head(limit) if df is not None else None
    except Exception as e:
        logger.error(f"Error retrieving recommendations for {ticker}: {e}")
        return None

def get_upgrades_downgrades(ticker: str, limit: int = 5) -> pd.DataFrame | None:
    """Get upgrades/downgrades history.
    Returns DataFrame with columns: firm, toGrade, fromGrade, action
    Limited to most recent entries by default.
    """
    try:
        df = yf.Ticker(ticker, session=session).get_upgrades_downgrades()
        return df.sort_index(ascending=False).head(limit) if df is not None else None
    except Exception as e:
        logger.error(f"Error retrieving upgrades/downgrades for {ticker}: {e}")
        return None

def get_price_history(
    ticker: str,
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "1mo"
) -> pd.DataFrame | None:
    try:
        res = yf.Ticker(ticker, session=session).history(period=period)
        # get the sample and append the tail to reduce the size
        # cal the frac according to the size
        # control the size less than 90
        frac = min(len(res) / 90, 0.5)

        # 假设我们要保留最近的 20 个交易日
        recent_count = 20
        # 分割 DataFrame
        # 取前 n-recent_count 条记录作为较早的条目
        early_df = res.iloc[:-recent_count]
        # 取最后 recent_count 条记录作为最近的条目
        recent_df = res.iloc[-recent_count:]
        # 对较早的条目进行抽样（例如，抽样 50%）
        sampled_early_df = early_df.sample(frac=frac)
        # 将抽样结果与最近的条目合并
        result_df = pd.concat([sampled_early_df, recent_df])
        # 重新排序
        result_df_sorted = result_df.sort_values(by='Date')

        return result_df_sorted
    except Exception as e:
        logger.error(f"Error retrieving price history for {ticker}: {e}")
        return str(e)

def get_financial_statements(
    ticker: str,
    statement_type: Literal["income", "balance", "cash"] = "income",
    frequency: Literal["quarterly", "annual"] = "quarterly"
) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker, session=session)
        statements = {
            "income": {"annual": t.income_stmt, "quarterly": t.quarterly_income_stmt},
            "balance": {"annual": t.balance_sheet, "quarterly": t.quarterly_balance_sheet},
            "cash": {"annual": t.cashflow, "quarterly": t.quarterly_cashflow}
        }
        return statements[statement_type][frequency]
    except Exception as e:
        logger.error(f"Error retrieving {frequency} {statement_type} statement for {ticker}: {e}")
        return None

def get_institutional_holders(ticker: str, top_n: int = 20) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        t = yf.Ticker(ticker, session=session)
        inst = t.get_institutional_holders()
        fund = t.get_mutualfund_holders()
        return (inst.head(top_n) if inst is not None else None,
                fund.head(top_n) if fund is not None else None)
    except Exception as e:
        logger.error(f"Error retrieving institutional holders for {ticker}: {e}")
        return None, None

def get_earnings_history(ticker: str, limit: int = 12) -> pd.DataFrame | None:
    """Get raw earnings history data.
    Default limit of 12 shows 3 years of quarterly earnings.
    """
    try:
        df = yf.Ticker(ticker, session=session).get_earnings_history()
        return df.head(limit) if df is not None else None
    except Exception as e:
        logger.error(f"Error retrieving earnings history for {ticker}: {e}")
        return None

def get_insider_trades(ticker: str, limit: int = 30) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker, session=session).get_insider_transactions()
        return df.head(limit) if df is not None else None
    except Exception as e:
        logger.error(f"Error retrieving insider trades for {ticker}: {e}")
        return None

def get_options_chain(
    ticker: str,
    expiry: str | None = None,
    option_type: Literal["C", "P"] | None = None
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Helper function to get raw options chain data for a specific expiry.
    Args:
        ticker: Stock ticker symbol
        expiry: Expiration date
        option_type: "C" for calls, "P" for puts, None for both
    """
    try:
        if not expiry:
            return None, "No expiry date provided"

        chain = yf.Ticker(ticker, session=session).option_chain(expiry)

        if option_type == "C":
            return chain.calls, None
        elif option_type == "P":
            return chain.puts, None
        return pd.concat([chain.calls, chain.puts]), None

    except Exception as e:
        logger.error(f"Error retrieving options chain for {ticker}: {e}")
        return None, str(e)

def get_filtered_options(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    strike_lower: float | None = None,
    strike_upper: float | None = None,
    option_type: Literal["C", "P"] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Get filtered options data efficiently."""
    try:
        # Validate date formats before processing
        if start_date:
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                return None, f"Invalid start_date format. Use YYYY-MM-DD"
                
        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                return None, f"Invalid end_date format. Use YYYY-MM-DD"

        t = yf.Ticker(ticker, session=session)
        expirations = t.options

        if not expirations:
            return None, f"No options available for {ticker}"

        # Convert date strings to datetime objects once
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

        # Filter expiration dates before making API calls
        valid_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            if ((not start_date_obj or exp_date >= start_date_obj) and
                (not end_date_obj or exp_date <= end_date_obj)):
                valid_expirations.append(exp)

        if not valid_expirations:
            return None, f"No options found for {ticker} within specified date range"

        # Parallel fetch options using ThreadPoolExecutor
        filtered_option_chains = []
        with ThreadPoolExecutor() as executor:
            options_results = list(executor.map(
                lambda exp: get_options_chain(ticker, exp, option_type),
                valid_expirations
            ))

        for (chain, error), expiry in zip(options_results, valid_expirations):
            if error:
                logger.warning(f"Error fetching options for expiry {expiry}: {error}")
                continue
            if chain is not None:
                filtered_option_chains.append(chain.assign(expiryDate=expiry))

        if not filtered_option_chains:
            return None, f"No options found for {ticker} matching criteria"

        df = pd.concat(filtered_option_chains, ignore_index=True)

        # Apply strike price filters
        if strike_lower is not None or strike_upper is not None:
            mask = pd.Series(True, index=df.index)
            if strike_lower is not None:
                mask &= df['strike'] >= strike_lower
            if strike_upper is not None:
                mask &= df['strike'] <= strike_upper
            df = df[mask]

        return df.sort_values(['openInterest', 'volume'], ascending=[False, False]), None

    except Exception as e:
        logger.error(f"Error in get_filtered_options: {str(e)}", exc_info=True)
        return None, f"Failed to retrieve options data: {str(e)}"