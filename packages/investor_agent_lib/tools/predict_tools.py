import logging

from packages.investor_agent_lib.services.predict_service import predict_next_day_chg
from packages.predict_lib.predict_down_v3 import get_prediction_result_lower_bound

def price_prediction(ticker: str) -> dict:
    """
    A well trained model that can predict whether a stock's price will be lower than its predicted thresohold price in the next 5 days based on historical data.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AVGO').

    Returns:
    dict: A dictionary with prediction details including current price, thresohold price and confusion matrix report,
    
    """
    try:
        prediction = get_prediction_result_lower_bound(ticker)
        if prediction is None:
            logging.error(f"Failed to retrieve prediction for {ticker}.")
            return None
        return prediction
    except Exception as e:
        logging.error(f"An error occurred while predicting next day change for {ticker}: {e}")
        return None
    

if __name__ == "__main__":
    ticker = 'AVGO'  # Example ticker
    prediction = price_prediction(ticker)
    if prediction:
        print(f"Prediction for {ticker}: {prediction}")
    else:
        print(f"No prediction available for {ticker}.")
