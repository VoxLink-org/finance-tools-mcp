import logging
from datetime import datetime, timezone  # Import timezone

import httpx

logger = logging.getLogger(__name__)

# CNN Fear & Greed API endpoint
API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


# Helper function to classify CNN Fear & Greed Index score
def get_classification(score: int) -> str:
    """Classify the CNN Fear & Greed Index score."""
    if score <= 25:
        return "Extreme Fear"
    elif score <= 40:
        return "Fear"
    elif score <= 60:
        return "Neutral"
    elif score <= 75:
        return "Greed"
    else:
        return "Extreme Greed"

# Helper function to fetch fear and greed data from CNN
async def fetch_fng_data() -> dict | None:
    """Fetch the raw Fear & Greed data from CNN and pre-process timestamps."""
    # Headers to mimic a browser to avoid bot detection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
        "Origin": "https://www.cnn.com",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(API_URL, headers=headers)
            response.raise_for_status()
            data = response.json()

            # --- Pre-process timestamps ---
            def parse_timestamp(ts_value):
                if isinstance(ts_value, (int, float)):
                    return int(ts_value) # Assume already in milliseconds if numeric
                elif isinstance(ts_value, str):
                    try:
                        # datetime.fromisoformat can handle ISO format strings with or without colon in timezone offset
                        # Formats: '2025-04-30T00:00:00+00:00' or '2025-05-02T23:59:56+0000'
                        dt_obj = datetime.fromisoformat(ts_value).astimezone(timezone.utc)
                        return int(dt_obj.timestamp() * 1000) # Convert to milliseconds
                    except ValueError:
                        logger.warning(f"Could not parse timestamp string: {ts_value}")
                        return None # Or handle error as needed
                    except TypeError:
                         logger.warning(f"Unexpected timestamp type: {type(ts_value)}, value: {ts_value}")
                         return None
                return None

            # Process current fear_and_greed timestamp
            if 'fear_and_greed' in data and isinstance(data['fear_and_greed'], dict) and 'timestamp' in data['fear_and_greed']:
                original_ts = data['fear_and_greed']['timestamp']
                parsed_ts = parse_timestamp(original_ts)
                if parsed_ts is not None:
                    data['fear_and_greed']['timestamp'] = parsed_ts
                else:
                    # Decide how to handle unparseable current timestamp (e.g., remove, set to default)
                    logger.error(f"Failed to parse current timestamp: {original_ts}. Removing field.")
                    del data['fear_and_greed']['timestamp']


            # Process historical timestamps
            if 'fear_and_greed_historical' in data and isinstance(data['fear_and_greed_historical'], dict) and 'data' in data['fear_and_greed_historical'] and isinstance(data['fear_and_greed_historical']['data'], list):
                 historical_entries = data['fear_and_greed_historical']['data'] # Access the nested list
                 processed_history = []
                 for entry in historical_entries:

                    # CNN uses 'x' for timestamp and 'y' for score in historical data
                    if isinstance(entry, dict) and 'x' in entry and 'y' in entry:
                        original_ts = entry['x'] # Use 'x' for timestamp
                        parsed_ts = parse_timestamp(original_ts)
                        if parsed_ts is not None:
                            # Keep original structure but ensure timestamp is processed
                            processed_entry = entry.copy()
                            processed_entry['timestamp'] = parsed_ts # Add a standard 'timestamp' key
                            processed_entry['score'] = entry['y'] # Add a standard 'score' key
                            # Optionally remove 'x' and 'y' if you prefer a unified structure
                            # del processed_entry['x']
                            # del processed_entry['y']
                            processed_history.append(processed_entry)
                        else:
                             logger.warning(f"Failed to parse historical timestamp: {original_ts}. Skipping entry.")
                    else:
                        logger.warning(f"Historical entry has unexpected structure or missing keys 'x'/'y': {entry}. Skipping entry.")

                 # Replace the original historical data structure with our processed list
                 # Or store it differently if needed, e.g., data['processed_historical_fng'] = processed_history
                 data['fear_and_greed_historical'] = processed_history
            elif 'fear_and_greed_historical' in data:
                 # Log if historical data exists but structure is not as expected
                 logger.warning("Historical F&G data found but has unexpected structure.")
                 # Decide how to handle this: remove it, leave it as is, etc.
                 data['fear_and_greed_historical'] = [] # Default to empty list if structure is wrong

            return data

    except Exception as e:
        logger.error(f"Error fetching or processing CNN Fear & Greed data: {str(e)}")
        return None