import math
import numpy as np
import talib as ta

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


def calc_ta(ta_lib_expression):
    
    try:
        result = eval(ta_lib_expression, {"__builtins__": {}}, {
            "ta": ta,
            "np": np
        })
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
