from . import yfinance_utils
import talib as ta

def get_market_rsi():
    spy_price = yfinance_utils.get_price_history('SPY', period='1mo', raw=True)
    qqq_price = yfinance_utils.get_price_history('QQQ', period='1mo', raw=True)

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

    # Simple divergence detection (last 5 days)
    def check_divergence(prices, rsi_values):
        last_prices = prices[-5:]
        last_rsi = rsi_values[-5:]
        
        price_trend = "up" if last_prices[-1] > last_prices[0] else "down"
        rsi_trend = "up" if last_rsi[-1] > last_rsi[0] else "down"
        
        if price_trend != rsi_trend:
            return f"potential_{'bearish' if price_trend == 'up' else 'bullish'}_divergence"
        return "no_clear_divergence"

    spy_condition = classify_rsi(current_spy_rsi)
    qqq_condition = classify_rsi(current_qqq_rsi)
    spy_divergence = check_divergence(spy_price['Close'], spy_rsi)
    qqq_divergence = check_divergence(qqq_price['Close'], qqq_rsi)
    
    return (
        f"SPY RSI: {current_spy_rsi:.1f} ({spy_condition}), {spy_divergence}\n"
        f"QQQ RSI: {current_qqq_rsi:.1f} ({qqq_condition}), {qqq_divergence}"
    )
