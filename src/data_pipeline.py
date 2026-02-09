# src/data_pipeline.py
"""
Data fetching and processing functions.
"""

import yfinance as yf
import pandas as pd
import datetime
from src.config import TICKERS, LOOKBACK_YEARS, TRADING_DAYS


def fetch_price_data(tickers=None, years=None):
    """Download historical adjusted close prices."""
    if tickers is None:
        tickers = TICKERS
    if years is None:
        years = LOOKBACK_YEARS

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years * 365)

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    prices = data["Close"]

    # Ensure ticker order matches input
    if isinstance(prices, pd.DataFrame):
        prices = prices[tickers]

    return prices


def calculate_returns(prices):
    """Calculate daily returns from prices."""
    daily_returns = prices.pct_change().dropna()
    return daily_returns


def calculate_annual_stats(daily_returns):
    """
    Calculate annualized return, volatility, and covariance.

    Returns:
        tuple: (annual_return, annual_volatility, annual_cov)
    """
    annual_return = daily_returns.mean() * TRADING_DAYS
    annual_volatility = daily_returns.std() * (TRADING_DAYS ** 0.5)
    annual_cov = daily_returns.cov() * TRADING_DAYS

    return annual_return, annual_volatility, annual_cov


def validate_tickers(tickers):
    """
    Validate that tickers exist and have sufficient data.

    Args:
        tickers: List of ticker symbols to validate

    Returns:
        list: Valid tickers only
    """
    import yfinance as yf

    valid = []
    invalid = []

    for ticker in tickers:
        try:
            # Try to fetch recent data
            test_data = yf.download(ticker, period="5d", progress=False)

            if not test_data.empty and len(test_data) > 0:
                valid.append(ticker)
            else:
                invalid.append(ticker)
        except Exception as e:
            invalid.append(ticker)

    if invalid:
        print(f"⚠️  Invalid tickers removed: {', '.join(invalid)}")

    if len(valid) < 2:
        raise ValueError(f"Need at least 2 valid tickers. Only found: {', '.join(valid)}")

    return valid