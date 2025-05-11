import logging

from packages.investor_agent_lib.utils import calculation_utils



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
    return calculation_utils.calc(expression)

