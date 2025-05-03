import math
import numpy as np
import talib as ta
from src.investor_agent.yfinance_utils import _price_data_cache

def calc(expression):
    try:
        # Safe evaluation of the expression
        result = eval(expression, {"__builtins__": {}}, {
            "math": math,
            "np": np,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e
        })
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}



def calc_ta(ta_lib_expression, ticker: str | None = None):
    try:
        # Create evaluation context with TA-Lib and numpy
        context = {
            "ta": ta,
            "np": np
        }
        
        # Add cached price data if ticker is provided and exists in cache
        if ticker and ticker in _price_data_cache:
            context.update({
                "close": _price_data_cache[ticker]['close'],
                "high": _price_data_cache[ticker]['high'],
                "low": _price_data_cache[ticker]['low'],
                "open": _price_data_cache[ticker]['open'],
                "date": _price_data_cache[ticker]['date']
            })
        
        result = eval(ta_lib_expression, {"__builtins__": {}}, context)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
