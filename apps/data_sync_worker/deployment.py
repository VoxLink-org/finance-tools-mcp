from datetime import  datetime
from prefect import serve
from prefect.schedules import Cron
import datetime
from apps.data_sync_worker.pipeline import option_snapshot_pipeline, option_indicator_pipeline, best_30days_pipeline
from apps.data_sync_worker import get_ticker_pool

def main():

    tickers =get_ticker_pool.get_ticker_pool()
    # remove duplicates
    tickers = list(set(tickers))
    schedule=Cron(
            "0 17 * * 1,2,3,4,5",
            timezone="America/New_York"
        )


    f1 = option_snapshot_pipeline.to_deployment(
        name="option_snapshot_pipeline",
        schedule=schedule,
        parameters={"tickers": tickers},
    )

    f2 = option_indicator_pipeline.to_deployment(
        name="option_indicator_pipeline",
        schedule=schedule,
        parameters={"tickers": tickers},
    )

    f3 = best_30days_pipeline.to_deployment(
        name="best_30days_pipeline",
        schedule=schedule,
        parameters={"tickers": tickers},
    )
    

    serve(f1, f2, f3)

