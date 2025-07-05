from datetime import datetime
import time
from typing import List
from venv import logger
from prefect import task, flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.artifacts import create_link_artifact, create_markdown_artifact

from apps.data_sync_worker import best_30days_task
import apps.data_sync_worker.option_snapshot_task as option_snapshot_task 
import apps.data_sync_worker.option_indicator_task as option_indicator_task

@flow(task_runner=ConcurrentTaskRunner())
def option_snapshot_pipeline(tickers: List[str]):
    """Main data pipeline that gets, validates and saves options data"""
    logger = get_run_logger()
    
        
    if not tickers or len(tickers) == 0:
        logger.error("No tickers provided")
        return
        
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        
        # Get options data
        options_data = option_snapshot_task.get_options_task(ticker)
        
        # Validate data
        is_valid = option_snapshot_task.validate_options_task(options_data)
        
        if not is_valid:
            logger.error(f"Invalid options data for {ticker}")
            continue
            
        # Save valid data
        save_success = option_snapshot_task.save_options_task(options_data)
        
        if save_success:
            logger.info(f"Successfully processed {ticker}")
            
            # Clean up old data
            deleted_count = option_snapshot_task.clean_up_the_days_before_10days()
            logger.info(f"Cleaned up {deleted_count} old records for {ticker}")
        else:
            logger.error(f"Failed to save data for {ticker}")

    create_link_artifact(
        key="options-snapshot",
        link="https://prefect.findata-be.uk/link_artifact/options_data.db",
        description="## Highly variable data",
    )


@flow(task_runner=ConcurrentTaskRunner())
def option_indicator_pipeline(tickers: List[str]):
    """Main data pipeline that gets, validates and saves options data"""
    logger = get_run_logger()
    
        
    if not tickers or len(tickers) == 0:
        logger.error("No tickers provided")
        return
        
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        
        # get indicators
        data = option_indicator_task.get_options_indicator_task(ticker)
        
        # validate
        is_valid = option_indicator_task.validate_option_indicator_task(data)
        
        if not is_valid:
            logger.error(f"Invalid options data for {ticker}")
            continue
            
        # Save valid data
        save_success = option_indicator_task.save_option_indicator_task(data)

        if save_success:
            logger.info(f"Successfully processed {ticker}")
            
        else:
            logger.error(f"Failed to save data for {ticker}")

    create_link_artifact(
        key="options-indicator",
        link="https://prefect.findata-be.uk/link_artifact/options_indicator.db",
        description="## Highly variable data",
    )

@flow(task_runner=ConcurrentTaskRunner())
def best_30days_pipeline(tickers: List[str]):
    """get best 30days data"""
    best_tickers = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            logger.info("Sleeping for 10 second to avoid rate limiting")
            time.sleep(10)
        
        result = best_30days_task.get_best_30days_task(ticker)
        if result:
            best_tickers.append(ticker)

    total_tickers = len(tickers)
    best_tickers_count = len(best_tickers)
    
    tickers_str = "\n".join(best_tickers)
    markdown = f"""
# Market Summary

- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total tickers: {total_tickers}
- Tickers at 30-day high: {best_tickers_count}
- Percentage: {best_tickers_count / total_tickers:.2%}
- Tickers at 30-day high: 

{tickers_str}
"""
    print(markdown)

    create_markdown_artifact(
        markdown=markdown,
        description="## Market Summary",
    )

    


def main():
    best_30days_pipeline(["AAPL","QQQ"])



if __name__ == "__main__":
    main()
