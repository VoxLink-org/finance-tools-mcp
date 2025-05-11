from datetime import  datetime
import get_ticker_pool
from prefect import serve
from prefect.schedules import Cron
import datetime
from pipeline import stock_data_pipeline, option_indicator_pipeline

def main():

    tickers =get_ticker_pool.get_ticker_pool()
    # remove duplicates
    tickers = list(set(tickers))
    schedule=Cron(
            "0 17 * * 1,2,3,4,5",
            timezone="America/New_York"
        )


    f1 = stock_data_pipeline.to_deployment(
        name="stock_data_pipeline",
        schedule=schedule,
        parameters={"tickers": tickers},
    )

    f2 = option_indicator_pipeline.to_deployment(
        name="option_indicator_pipeline",
        schedule=schedule,
        parameters={"tickers": tickers},
    )

    serve(f1, f2)

