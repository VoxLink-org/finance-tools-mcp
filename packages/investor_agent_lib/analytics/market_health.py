import talib as ta

from packages.investor_agent_lib.services import yfinance_service

def get_market_rsi():
    spy_price = yfinance_service.get_price_history('SPY', period='3mo', raw=True)
    qqq_price = yfinance_service.get_price_history('QQQ', period='3mo', raw=True)

    spy_rsi_14 = ta.RSI(spy_price['Close'], timeperiod=14)
    qqq_rsi_14 = ta.RSI(qqq_price['Close'], timeperiod=14)
    spy_rsi_6 = ta.RSI(spy_price['Close'], timeperiod=6)
    qqq_rsi_6 = ta.RSI(qqq_price['Close'], timeperiod=6)

    # Current RSI values
    current_spy_rsi_14 = spy_rsi_14[-1]
    current_qqq_rsi_14 = qqq_rsi_14[-1]
    current_spy_rsi_6 = spy_rsi_6[-1]
    current_qqq_rsi_6 = qqq_rsi_6[-1]
    
    # Classify RSI conditions
    def classify_rsi(rsi_value):
        if rsi_value < 30:
            return "oversold"
        elif rsi_value > 70:
            return "overbought"
        return "neutral"

    # Balanced divergence detection with relaxed thresholds
    def check_divergence(prices, rsi_values, window=14): # Default window for 14-day RSI
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

    spy_condition_14 = classify_rsi(current_spy_rsi_14)
    qqq_condition_14 = classify_rsi(current_qqq_rsi_14)
    spy_divergence_14 = check_divergence(spy_price['Close'], spy_rsi_14)
    qqq_divergence_14 = check_divergence(qqq_price['Close'], qqq_rsi_14)

    spy_condition_6 = classify_rsi(current_spy_rsi_6)
    qqq_condition_6 = classify_rsi(current_qqq_rsi_6)
    # For 6-day RSI, use a smaller window for divergence, e.g., 6 or 7
    spy_divergence_6 = check_divergence(spy_price['Close'], spy_rsi_6, window=6)
    qqq_divergence_6 = check_divergence(qqq_price['Close'], qqq_rsi_6, window=6)
    
    return (
        f"SPY RSI (14-day): {current_spy_rsi_14:.1f} ({spy_condition_14}), {spy_divergence_14}\n"
        f"QQQ RSI (14-day): {current_qqq_rsi_14:.1f} ({qqq_condition_14}), {qqq_divergence_14}\n"
        f"SPY RSI (6-day): {current_spy_rsi_6:.1f} ({spy_condition_6}), {spy_divergence_6}\n"
        f"QQQ RSI (6-day): {current_qqq_rsi_6:.1f} ({qqq_condition_6}), {qqq_divergence_6}"
    )

def get_market_vix():
    """Get comprehensive VIX analysis including trend and sentiment interpretation.
    Returns structured analysis for LLM consumption."""
    vix = yfinance_service.get_price_history('^VIX', period='1mo', raw=True)
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

def get_market_chg():
    spy_price = yfinance_service.get_price_history('SPY', period='5d', raw=True)
    qqq_price = yfinance_service.get_price_history('QQQ', period='5d', raw=True)
    
    
    spy_chg_5d = (spy_price['Close'][-1] - spy_price['Close'][0]) / spy_price['Close'][0]
    qqq_chg_5d = (qqq_price['Close'][-1] - qqq_price['Close'][0]) / qqq_price['Close'][0]
    spy_chg_1d = (spy_price['Close'][-1] - spy_price['Close'][-2]) / spy_price['Close'][-2]
    qqq_chg_1d = (qqq_price['Close'][-1] - qqq_price['Close'][-2]) / qqq_price['Close'][-2]
    
    return (
        f"Market Change Analysis:\n"
        f"- SPY: 5-day change: {spy_chg_5d:.2%}, 1-day change: {spy_chg_1d:.2%}\n"
        f"- QQQ: 5-day change: {qqq_chg_5d:.2%}, 1-day change: {qqq_chg_1d:.2%}"
    )

