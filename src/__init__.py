# src/__init__.py
"""
Portfolio optimization package.
"""

from src.data_pipeline import fetch_price_data, calculate_returns, calculate_annual_stats
from src.portfolio_engine import portfolio_stats, generate_random_portfolios, monte_carlo_portfolio
from src import config