from . import yfinance_utils
import talib as ta

def get_market_rsi():
    spy_price = yfinance_utils.get_price_history('SPY', period='3mo', raw=True)
    qqq_price = yfinance_utils.get_price_history('QQQ', period='3mo', raw=True)

    spy_rsi = ta.RSI(spy_price['Close'], timeperiod=14)
    qqq_rsi = ta.RSI(qqq_price['Close'], timeperiod=14)

    # Current RSI values
    current_spy_rsi = spy_rsi[-1]
    current_qqq_rsi = qqq_rsi[-1]
    
    # Classify RSI conditions
    def classify_rsi(rsi_value):
        if rsi_value < 30:
            return "oversold"
        elif rsi_value > 70:
            return "overbought"
        return "neutral"

    # Balanced divergence detection with relaxed thresholds
    def check_divergence(prices, rsi_values, window=14):
        if len(prices) < window or len(rsi_values) < window:
            return "no_clear_divergence"
            
        # Find significant turning points
        def find_turns(series):
            turns = []
            for i in range(1, len(series)-1):
                # Less strict turning point detection
                if (series[i] <= series[i-1] and series[i] <= series[i+1]) or \
                   (series[i] >= series[i-1] and series[i] >= series[i+1]):
                    turns.append((i, series[i]))
            return turns
            
        price_turns = find_turns(prices[-window:])
        rsi_turns = find_turns(rsi_values[-window:])
        
        # Need at least 2 turns in each series
        if len(price_turns) < 2 or len(rsi_turns) < 2:
            return "no_clear_divergence"
            
        # Compare latest two turns
        price1, price2 = price_turns[-2:]
        rsi1, rsi2 = rsi_turns[-2:]
        
        # Check for opposing trends with relaxed thresholds
        price_dir = "up" if price2[1] > price1[1] else "down"
        rsi_dir = "up" if rsi2[1] > rsi1[1] else "down"
        
        if price_dir != rsi_dir:
            # Relaxed thresholds: 0.5% price move and 1% RSI move
            price_move = abs(price2[1] - price1[1]) / ((price1[1] + price2[1])/2)
            rsi_move = abs(rsi2[1] - rsi1[1]) / ((rsi1[1] + rsi2[1])/2)
            
            if price_move > 0.005 and rsi_move > 0.01:
                return f"potential_{'bearish' if price_dir == 'up' else 'bullish'}_divergence"
                
        return "no_clear_divergence"

    spy_condition = classify_rsi(current_spy_rsi)
    qqq_condition = classify_rsi(current_qqq_rsi)
    spy_divergence = check_divergence(spy_price['Close'], spy_rsi)
    qqq_divergence = check_divergence(qqq_price['Close'], qqq_rsi)
    
    return (
        f"SPY RSI: {current_spy_rsi:.1f} ({spy_condition}), {spy_divergence}\n"
        f"QQQ RSI: {current_qqq_rsi:.1f} ({qqq_condition}), {qqq_divergence}"
    )

def get_market_vix():
    """Get comprehensive VIX analysis including trend and sentiment interpretation.
    Returns structured analysis for LLM consumption."""
    vix = yfinance_utils.get_price_history('^VIX', period='1mo', raw=True)
    close_prices = vix['Close']
    
    current = close_prices[-1]
    week_ago = close_prices[-5]
    month_high = close_prices.max()
    month_low = close_prices.min()
    
    # Determine trend direction
    if current > week_ago * 1.1:
        trend = "rising sharply"
    elif current > week_ago * 1.05:
        trend = "rising"
    elif current < week_ago * 0.9:
        trend = "falling sharply"
    elif current < week_ago * 0.95:
        trend = "falling"
    else:
        trend = "stable"
    
    # Interpret sentiment based on VIX level
    if current > 30:
        sentiment = "extreme fear"
    elif current > 25:
        sentiment = "high fear" 
    elif current > 20:
        sentiment = "moderate fear"
    elif current > 12:  # More aligned with historical mean
        sentiment = "neutral"
    else:
        sentiment = "complacency"
    
    return (
        f"VIX Analysis:\n"
        f"- Current: {current:.2f}\n"
        f"- Trend: {trend} (from {week_ago:.2f} a week ago)\n"
        f"- Monthly Range:  {month_low:.2f} ~ {month_high:.2f}\n"
        f"- Market Sentiment: {sentiment}"
    )
