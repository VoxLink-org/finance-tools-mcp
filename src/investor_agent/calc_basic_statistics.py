import numpy as np
import pandas as pd
from tabulate import tabulate


def calculate_basic_statistics(time_series_data: pd.DataFrame) -> dict:
    """Calculates basic price and volume statistics."""
    stats = {
        'Period': f"{time_series_data.index.min().strftime('%Y-%m-%d')} to {time_series_data.index.max().strftime('%Y-%m-%d')}",
        'Trading Days': len(time_series_data),
        'Close Price': {
            'Min': np.min(time_series_data['Close']),
            'Max': np.max(time_series_data['Close']),
            'Mean': np.mean(time_series_data['Close']),
            'Last': time_series_data['Close'].iloc[-1],
            'Change': (time_series_data['Close'].iloc[-1] - time_series_data['Close'].iloc[0]) / time_series_data['Close'].iloc[0]
        },
        'Volume': {
            'Total': np.sum(time_series_data['Volume']),
            'Avg': np.mean(time_series_data['Volume']),
            'Max': np.max(time_series_data['Volume'])
        }
    }

    text = "Basic Statistics:\n"
    text += f"  Period: {stats['Period']}\n"
    text += f"  Trading Days: {stats['Trading Days']}\n"
    text += "  Close Price:\n"
    text += f"    Min: {stats['Close Price']['Min']:.2f}\n"
    text += f"    Max: {stats['Close Price']['Max']:.2f}\n"
    text += f"    Mean: {stats['Close Price']['Mean']:.2f}\n"
    text += f"    Last: {stats['Close Price']['Last']:.2f}\n"
    text += f"    Change: {stats['Close Price']['Change']:.2%}\n"
    text += "  Volume:\n"
    text += f"    Total: {stats['Volume']['Total']:,}\n"
    text += f"    Avg: {stats['Volume']['Avg']:,}\n"
    text += f"    Max: {stats['Volume']['Max']:,}\n"

    return text