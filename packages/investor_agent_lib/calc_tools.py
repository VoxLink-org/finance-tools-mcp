import logging
from typing import Literal

import pandas as pd
from mcp.server.fastmcp import FastMCP
from tabulate import tabulate

from . import calc_utils

logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression. Support python math syntax and numpy.
        > calculate("2 * 3 + 4")
        {'result': 10}
        > calculate("sin(pi/2)")
        {'result': 1.0}
        > calculate("sqrt(16)")
        {'result': 4.0}
        > calculate("np.mean([1, 2, 3])")
        {'result': 2.0}
    """
    return calc_utils.calc(expression)

def calc_ta(ta_lib_expression: str, ticker: str | None = None) -> str:
    """
    Calculate technical indicators using ta-lib-python (TA-lib) and numpy.
    This tool evaluates a given expression string using the ta-lib-python library.
    The expression should follow ta-lib-python syntax, for example:
    - 'ta.SMA(close, timeperiod=30)' with the ticker 'AAPL'
    - 'ta.ROC(close, timeperiod=30)' with the ticker 'MSFT'
    - 'ta.RSI(close, timeperiod=14)[-1]' with the ticker 'NVDA'
    You must specify a ticker to use the cached price data if you've used the 'get_price_data' tool with the same ticker.
    If not, the expression will be evaluated in a context where only ta-lib-python and numpy are available.

    The expression string is evaluated in a context where the ta-lib-python library is available as 'ta' and numpy is available as 'np'.
    """
    return calc_utils.calc_ta(ta_lib_expression, ticker)