# src/config.py
"""
Configuration constants for portfolio optimizer.
"""

# Portfolio settings
TICKERS = ["SPY", "AAPL", "MSFT", "AMD", "NVDA"]
LOOKBACK_YEARS = 10
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0  # Simplified for now

# Position constraints
MAX_POSITION = 0.40  # 40% max per stock
MIN_POSITION = 0.05  # 5% min per stock

# Random portfolio settings
NUM_PORTFOLIOS = 10000