
from datetime import datetime
import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.stats

from packages.investor_agent_lib.options.option_selection import get_raw_options



def calculate_indicators(ticker:str)->dict:
    """Calculate key option indicators for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        dict: Dictionary containing:
            - atm_iv_avg: Average implied volatility of at-the-money options
            - skew_measure: Difference between OTM put and call IVs
            - term_structure_slope: IV slope between near and far expirations
            - pc_ratio: Put/Call volume ratio
    """
    options_data = get_raw_options(ticker)
    
    if not isinstance(options_data, pd.DataFrame):
        return {
            'atm_iv_avg': None,
            'skew_measure': None,
            'term_structure_slope': None,
            'pc_ratio': None
        }
    
    # Calculate moneyness (absolute distance from current price)
    options_data['moneyness'] = abs(options_data['strike'] - options_data['underlyingPrice'])
    
    # Get ATM options (closest to money)
    atm_options = options_data.nsmallest(10, 'moneyness')
    atm_iv_avg = atm_options['impliedVolatility'].mean()
    
    # Calculate skew measure (OTM puts IV - OTM calls IV)
    otm_puts = options_data[
        (options_data['optionType'] == 'P') &
        (options_data['strike'] < options_data['underlyingPrice'])
    ]
    otm_calls = options_data[
        (options_data['optionType'] == 'C') &
        (options_data['strike'] > options_data['underlyingPrice'])
    ]
    skew_measure = otm_puts['impliedVolatility'].mean() - otm_calls['impliedVolatility'].mean()
    
    # Calculate term structure slope
    options_data['days_to_expiry'] = (pd.to_datetime(options_data['expiryDate']) - pd.to_datetime('today')).dt.days
    near_term = options_data[options_data['days_to_expiry'] <= 30]
    far_term = options_data[options_data['days_to_expiry'] > 30]
    term_structure_slope = far_term['impliedVolatility'].mean() - near_term['impliedVolatility'].mean()
    
    # Calculate P/C ratio
    puts = options_data[options_data['optionType'] == 'P']
    calls = options_data[options_data['optionType'] == 'C']
    pc_ratio = puts['volume'].sum() / calls['volume'].sum()
    
    return {
        'atm_iv_avg': atm_iv_avg,
        'skew_measure': skew_measure,
        'term_structure_slope': term_structure_slope,
        'pc_ratio': pc_ratio,
        'ticker': ticker,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'lastTradeDate': options_data['lastTradeDate'].max(),
        'underlyingPrice': options_data['underlyingPrice'].max()
    }



def calculate_greeks(ticker):
    """Calculate option greeks for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        dict: Dictionary containing:
            - call_delta: Delta for call options
            - put_delta: Delta for put options
            - gamma: Gamma (same for calls and puts)
            - call_theta: Theta for call options
            - put_theta: Theta for put options
            - vega: Vega (same for calls and puts)
            - call_rho: Rho for call options
            - put_rho: Rho for put options
    """
    options_data = get_raw_options(ticker)
    
    if not isinstance(options_data, pd.DataFrame):
        return {
            'call_delta': None,
            'put_delta': None,
            'gamma': None,
            'call_theta': None,
            'put_theta': None,
            'vega': None,
            'call_rho': None,
            'put_rho': None
        }
    
    # Get ATM options (closest to money)
    options_data['moneyness'] = abs(options_data['strike'] - options_data['underlyingPrice'])
    atm_options = options_data.nsmallest(10, 'moneyness')
    
    # Calculate time to expiry in years
    expiry_date = pd.to_datetime(atm_options['expiryDate'].iloc[0])
    time_to_expiry = (expiry_date - pd.to_datetime('today')).days / 365.0
    
    # Get required parameters
    S = atm_options['underlyingPrice'].iloc[0]  # Current price
    K = atm_options['strike'].iloc[0]  # Strike price
    r = 0.05  # Risk-free rate (approximation)
    sigma = atm_options['impliedVolatility'].iloc[0]  # Volatility
    
    # Black-Scholes calculations
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
    d2 = d1 - sigma * np.sqrt(time_to_expiry)
    
    # Standard normal CDF and PDF
    N = scipy.stats.norm.cdf
    N_prime = scipy.stats.norm.pdf
    
    # Calculate Greeks
    call_delta = N(d1)
    put_delta = call_delta - 1
    gamma = N_prime(d1) / (S * sigma * np.sqrt(time_to_expiry))
    call_theta = (-S * N_prime(d1) * sigma / (2 * np.sqrt(time_to_expiry))
                  - r * K * np.exp(-r * time_to_expiry) * N(d2))
    put_theta = (-S * N_prime(d1) * sigma / (2 * np.sqrt(time_to_expiry))
                 + r * K * np.exp(-r * time_to_expiry) * N(-d2))
    vega = S * N_prime(d1) * np.sqrt(time_to_expiry) / 100  # per 1% change in vol
    call_rho = K * time_to_expiry * np.exp(-r * time_to_expiry) * N(d2) / 100  # per 1% change in rate
    put_rho = -K * time_to_expiry * np.exp(-r * time_to_expiry) * N(-d2) / 100
    
    return {
        'call_delta': call_delta,
        'put_delta': put_delta,
        'gamma': gamma,
        'call_theta': call_theta,
        'put_theta': put_theta,
        'vega': vega,
        'call_rho': call_rho,
        'put_rho': put_rho,
        'ticker': ticker,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'lastTradeDate': options_data['lastTradeDate'].max()
    }

