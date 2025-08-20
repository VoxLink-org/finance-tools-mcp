import logging

from packages.investor_agent_lib.services.predict_service import predict_next_day_chg

def get_next_day_prediction(ticker: str) -> dict:
    """
    Retrieves the next day's price movement prediction for a given stock ticker.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'QQQ').

    Returns:
    dict: A dictionary with prediction details including current price, daily change,
          and historical frequencies for up, down, or flat movements.
          Returns None if data retrieval fails or no prediction is available.
    """
    try:
        prediction = predict_next_day_chg(ticker)
        if prediction is None:
            logging.error(f"Failed to retrieve prediction for {ticker}.")
            return None
        return prediction
    except Exception as e:
        logging.error(f"An error occurred while predicting next day change for {ticker}: {e}")
        return None
    

if __name__ == "__main__":
    ticker = 'QQQ'  # Example ticker
    prediction = get_next_day_prediction(ticker)
    if prediction:
        print(f"Prediction for {ticker}: {prediction}")
    else:
        print(f"No prediction available for {ticker}.")
