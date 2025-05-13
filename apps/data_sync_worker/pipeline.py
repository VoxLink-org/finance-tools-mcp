from datetime import datetime
from typing import List
from prefect import task, flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.artifacts import create_link_artifact

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




def main():
    option_indicator_pipeline(["SVXY",".DJI"])



if __name__ == "__main__":
    main()
