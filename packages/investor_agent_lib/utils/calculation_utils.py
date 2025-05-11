import math
import numpy as np
import talib as ta
from packages.investor_agent_lib.services.yfinance_service import _price_data_cache

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


