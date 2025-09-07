import asyncio
import pandas as pd
import curl_cffi
import io
from typing import List, Tuple
import time


async def get_option_history_async(ticker: str) -> Tuple[str, pd.DataFrame]:
    """
    Async version of get_option_history that fetches option history data for a given ticker.
    Returns a tuple of (ticker, DataFrame) to maintain ticker association.
    """
    url = (
        f"https://optioncharts.io/async/option_history_table?ticker={ticker}&period=max"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        async with curl_cffi.requests.AsyncSession() as session:
            response = await session.get(url, headers=headers)
            response.raise_for_status()

            # The response content is an HTML table, pandas can read it directly
            df = pd.read_html(response.content, match="Put-Call Ratio")[0].drop(
                columns=["IV 30d", "IV Rank", "Max Pain 7d", "GEX OI", "DEX OI"]
            )
            return (ticker, df)
    except curl_cffi.requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return (ticker, pd.DataFrame())
    except ValueError as e:
        print(f"Error parsing HTML for {ticker}: {e}")
        return (ticker, pd.DataFrame())
    except Exception as e:
        print(f"Unexpected error for {ticker}: {e}")
        return (ticker, pd.DataFrame())


async def process_batch(
    tickers: List[str], batch_num: int, total_batches: int
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Process a single batch of tickers with progress indicator.
    """
    print(
        f"Processing batch {batch_num}/{total_batches} ({len(tickers)} tickers): {', '.join(tickers)}"
    )

    tasks = [get_option_history_async(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    print(f"✓ Completed batch {batch_num}/{total_batches}")
    return results


async def batch_download_option_history_async(tickers: List[str]) -> pd.DataFrame:
    """
    Async batch processing of option history data with max 5 tickers per batch and 5-second sleep between batches.
    Includes progress indicator in console.
    """
    if not tickers:
        return pd.DataFrame()

    batch_size = 5
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    total_batches = len(batches)

    print(
        f"Starting batch download of option history for {len(tickers)} tickers in {total_batches} batches"
    )
    print("=" * 60)

    all_results = []

    for i, batch in enumerate(batches, 1):
        batch_results = await process_batch(batch, i, total_batches)
        all_results.extend(batch_results)

        # Sleep 5 seconds between batches, except after the last batch
        if i < total_batches:
            print(f"⏳ Sleeping 2 seconds before next batch...")
            await asyncio.sleep(2)

    print("=" * 60)
    print(f"✅ All batches completed! Processing {len(all_results)} results...")

    # Combine all DataFrames with ticker column
    combined_dfs = []
    for ticker, df in all_results:
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df["ticker"] = ticker
            combined_dfs.append(df)

    if combined_dfs:
        final_df = pd.concat(combined_dfs, ignore_index=False).rename(columns={
            'Date': 'date',
        })
        print(
            f"📊 Final DataFrame created with {len(final_df)} rows across {len(combined_dfs)} tickers"
        )
        return final_df
    else:
        print("⚠️ No data retrieved")
        return pd.DataFrame()


def download_option_history(tickers: list) -> pd.DataFrame:
    """
    Synchronous wrapper for async batch processing.
    """
    return asyncio.run(batch_download_option_history_async(tickers))


if __name__ == "__main__":
    df = download_option_history(
        ["GOOG", "MSFT", "AAPL", "TSLA", "NVDA", "AMD", "META"],
    )
    df.to_csv("option_history.csv", index=False)
