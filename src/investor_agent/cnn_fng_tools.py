import logging
from datetime import datetime


from . import cnn_fng_utils
from . import market_rsi_util

logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

# --- CNN Fear & Greed Index Resources and Tools ---

# Resource to get current Fear & Greed Index
async def get_current_fng() -> str:
    """Get the current CNN Fear & Greed Index as a resource. As well as the market RSI."""
    logger.info("Fetching current CNN Fear & Greed Index resource and market RSI")
    data = await cnn_fng_utils.fetch_fng_data()

    if not data:
        return "Error: Unable to fetch CNN Fear & Greed Index data."

    market_rsi = market_rsi_util.get_market_rsi()

    try:
        fear_and_greed = data.get("fear_and_greed", {})
        current_score = int(fear_and_greed.get("score", 0))
        current_rating = fear_and_greed.get("rating")
        timestamp = fear_and_greed.get("timestamp")

        if timestamp:
            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(int(timestamp) / 1000)  # CNN API uses milliseconds
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            date_str = "Unknown date"

        # Construct output with proper formatting
        result = (
            f"CNN Fear & Greed Index (as of {date_str}):\n"  # Escaped newline
            f"Score: {current_score}\n"  # Escaped newline
            f"Rating: {current_rating}\n"
            f"{market_rsi}"
        )
        return result
    except Exception as e:
        logger.error(f"Error processing CNN Fear & Greed data: {str(e)}")
        return f"Error processing CNN Fear & Greed data: {str(e)}"

# Resource to get historical Fear & Greed data
async def get_historical_fng() -> str:
    """Get historical CNN Fear & Greed Index data as a resource."""
    logger.info("Fetching historical CNN Fear & Greed Index resource")
    data = await cnn_fng_utils.fetch_fng_data()

    if not data:
        return "Error: Unable to fetch CNN Fear & Greed Index data."

    try:
        history = data.get("fear_and_greed_historical", [])
        if not history:
            return "No historical data available."

        # Format historical data
        lines = ["Historical CNN Fear & Greed Index:"]
        for entry in history:
            timestamp = entry.get("timestamp")
            score = entry.get("score")

            if timestamp and score:
                dt = datetime.fromtimestamp(int(timestamp) / 1000)  # CNN API uses milliseconds
                date_str = dt.strftime("%Y-%m-%d")
                classification = cnn_fng_utils.get_classification(int(score))
                lines.append(f"{date_str}: {score} ({classification})")

        return "\\n".join(lines)  # Corrected join method
    except Exception as e:
        logger.error(f"Error processing historical Fear & Greed data: {str(e)}")
        return f"Error processing historical Fear & Greed data: {str(e)}"

# Tool to get current Fear & Greed Index
async def get_current_fng_tool() -> str:
    """
    Get the current CNN Fear & Greed Index, as well as the market RSI.

    Returns:
        str: The current Fear & Greed Index & market RSI, with score and rating.
    """
    logger.info("Fetching current CNN Fear & Greed Index tool")
    data = await cnn_fng_utils.fetch_fng_data()
    market_rsi = market_rsi_util.get_market_rsi()
    market_vix = market_rsi_util.get_market_vix()

    if not data:
        return "Error: Unable to fetch CNN Fear & Greed Index data."

    try:
        fear_and_greed = data.get("fear_and_greed", {})
        current_score = int(fear_and_greed.get("score", 0))
        current_rating = fear_and_greed.get("rating", "Unknown")
        timestamp = fear_and_greed.get("timestamp")

        if timestamp:
            dt = datetime.fromtimestamp(int(timestamp) / 1000)  # CNN API uses milliseconds
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            date_str = "Unknown date"

        # Add our own classification based on the numeric score for additional context
        score_classification = cnn_fng_utils.get_classification(current_score)

        # Construct output with proper formatting
        result = (
            f"CNN Fear & Greed Index (as of {date_str}):\n"  # Escaped newline
            f"Score: {current_score}\n"  # Escaped newline
            f"CNN Rating: {current_rating}\n"  # Escaped newline
            f"Classification: {score_classification}\n"
            f"{market_rsi}\n"
            f"{market_vix}"
        )
        return result
    except Exception as e:
        logger.error(f"Error processing CNN Fear & Greed data: {str(e)}")
        return f"Error processing CNN Fear & Greed data: {str(e)}"

async def get_historical_fng_tool(days: int) -> str:
    """
    Get historical CNN Fear & Greed Index data for a specified number of days.

    Parameters:
        days (int): Number of days of historical data to retrieve (limited by the API).

    Returns:
        str: Historical Fear & Greed Index values for the specified period.
    """
    logger.info(f"Fetching historical CNN Fear & Greed Index for {days} days")

    if days <= 0:
        return "Error: Days must be a positive integer"

    data = await cnn_fng_utils.fetch_fng_data()

    if not data:
        return "Error: Unable to fetch CNN Fear & Greed Index data."

    try:
        history = data.get("fear_and_greed_historical", [])
        if not history:
            return "No historical data available."

        # Limit to the requested number of days
        # Note: The API may not provide data for every day
        limited_history = history[:min(days, len(history))]

        # Format historical data
        lines = [f"Historical CNN Fear & Greed Index (Last {len(limited_history)} days):"]
        for entry in limited_history:
            timestamp = entry.get("timestamp")
            score = entry.get("score")

            if timestamp and score:
                dt = datetime.fromtimestamp(int(timestamp) / 1000)  # CNN API uses milliseconds
                date_str = dt.strftime("%Y-%m-%d")
                score_num = int(score)
                classification = cnn_fng_utils.get_classification(score_num)
                lines.append(f"{date_str}: {score} ({classification})")

        return "\\n".join(lines)  # Corrected join method
    except Exception as e:
        logger.error(f"Error processing historical Fear & Greed data: {str(e)}")
        return f"Error processing historical Fear & Greed data: {str(e)}"

async def analyze_fng_trend(days: int) -> str:
    """
    Analyze trends in CNN Fear & Greed Index over specified days.

    Parameters:
        days (int): Number of days to analyze (limited by available data).

    Returns:
        str: A string containing the analysis results, including latest value,
             average value, trend direction, and number of data points analyzed.
    """
    logger.info(f"Analyzing CNN Fear & Greed trends over {days} days")

    if days <= 0:
        return "Error: Days must be a positive integer"

    data = await cnn_fng_utils.fetch_fng_data()

    if not data:
        return "Error: Unable to fetch CNN Fear & Greed Index data."

    try:
        # Get current data
        fear_and_greed = data.get("fear_and_greed", {})
        current_score = int(fear_and_greed.get("score", 0))
        current_rating = fear_and_greed.get("rating", "Unknown")
        current_timestamp = fear_and_greed.get("timestamp")

        # Get historical data
        history = data.get("fear_and_greed_historical", [])
        if not history:
            return "No historical data available for trend analysis."

        # Limit to the requested number of days
        limited_history = history[:min(days, len(history))]

        # Calculate statistics
        scores = [int(entry.get("score", 0)) for entry in limited_history if "score" in entry]

        if not scores:
            return "No valid scores found for trend analysis."

        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        # Determine trend direction
        trend = "insufficient data to determine trend"  # Default value
        if len(scores) > 1:
            # Use the most recent 'days' points for trend calculation
            trend_scores = scores[:min(days, len(scores))]  # Use scores from the beginning (most recent) up to 'days'
            if len(trend_scores) > 1:
                # Compare first available score (most recent) with the last available score (oldest in the period)
                first_score = trend_scores[0]
                last_score = trend_scores[-1]
                diff = first_score - last_score

                if diff < -5:
                     trend = "rising significantly"
                elif diff < -2:
                     trend = "rising"
                elif diff > 5:
                     trend = "falling significantly"
                elif diff > 2:
                     trend = "falling"
                else:
                     trend = "relatively stable"

        # Format current timestamp
        if current_timestamp:
            dt = datetime.fromtimestamp(int(current_timestamp) / 1000)  # CNN API uses milliseconds
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            date_str = "Unknown date"

        # Generate the analysis report
        result = [
            f"CNN Fear & Greed Index Analysis ({len(limited_history)} days):",
            f"Latest Value: {current_score} ({current_rating}) as of {date_str}",
            f"Average Value over period: {avg_score:.1f}",
            f"Range over period: {min_score} to {max_score}",
            f"Trend over period: {trend}",
            f"Current Classification: {cnn_fng_utils.get_classification(current_score)}",
            f"Data points analyzed: {len(scores)}"
        ]

        return "\\n".join(result)  # Corrected join method
    except Exception as e:
        logger.error(f"Error analyzing Fear & Greed trend: {str(e)}")