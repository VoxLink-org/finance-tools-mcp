import logging

from packages.investor_agent_lib.utils import calculation_utils



logger = logging.getLogger(__name__)

# Note: MCP server initialization and registration will happen in server.py

def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression. Support python math syntax and numpy.
    > "2 * 3 + 4"
    > "sin(pi/2)"
    > "sqrt(16)"
    > "np.mean([1, 2, 3])"
    """
    return calculation_utils.calc(expression)

