import logging


from packages.investor_agent_lib.services.predict_service import profit_prob
from packages.predict_lib.predict_down_v3 import get_prediction_result_lower_bound


def price_prediction(ticker: str) -> dict:
    """
    A well trained model that can predict whether a stock's price will be lower than its predicted thresohold price in the next 5 days based on historical data.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AVGO').

    Returns:
    dict: A dictionary with prediction details including current price, thresohold price and confusion matrix report, and the market implementing probability.
    
    """
    try:
        prediction = get_prediction_result_lower_bound(ticker)
        if prediction is None:
            logging.error(f"Failed to retrieve prediction for {ticker}.")
            return None
        
        prob_below, prob_above = profit_prob(ticker, prediction['last_close'], prediction['threshold_price'])
        prediction['option_market_implement_prob_go_below'] = round(prob_below, 2)
        prediction['option_market_implement_prob_go_above'] = round(prob_above, 2)
        
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
