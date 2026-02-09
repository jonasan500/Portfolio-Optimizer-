# src/portfolio_engine.py
"""
Portfolio calculation and optimization functions.
"""

import numpy as np
from src.config import RISK_FREE_RATE, MIN_POSITION, MAX_POSITION


def portfolio_stats(weights, annual_return, annual_cov, rf=None):
    """
    Calculate portfolio return, volatility, and Sharpe ratio.

    Args:
        weights: np.array of portfolio weights (must sum to 1)
        annual_return: pd.Series of annualized expected returns
        annual_cov: pd.DataFrame annualized covariance matrix
        rf: risk-free rate (optional, defaults to config)

    Returns:
        tuple: (portfolio_return, portfolio_volatility, sharpe_ratio)
    """
    if rf is None:
        rf = RISK_FREE_RATE

    # Convert to numpy arrays if needed
    ret_values = annual_return.values if hasattr(annual_return, 'values') else annual_return
    cov_values = annual_cov.values if hasattr(annual_cov, 'values') else annual_cov

    port_return = np.dot(weights, ret_values)
    port_variance = np.dot(weights, np.dot(cov_values, weights))
    port_volatility = np.sqrt(port_variance)
    sharpe = (port_return - rf) / port_volatility if port_volatility > 0 else 0

    return port_return, port_volatility, sharpe


def generate_random_portfolios(annual_return, annual_cov, num_portfolios,
                               min_weight=None, max_weight=None):
    """
    Generate random portfolios with optional position constraints.

    Returns:
        dict: Contains 'returns', 'volatility', 'sharpe', 'weights' arrays
    """
    if min_weight is None:
        min_weight = MIN_POSITION
    if max_weight is None:
        max_weight = MAX_POSITION

    n_assets = len(annual_return)

    results = {
        'returns': [],
        'volatility': [],
        'sharpe': [],
        'weights': []
    }

    for _ in range(num_portfolios):
        # Generate valid random weights
        valid = False
        while not valid:
            weights = np.random.random(n_assets)
            weights /= weights.sum()

            if weights.max() <= max_weight and weights.min() >= min_weight:
                valid = True

        ret, vol, sharpe = portfolio_stats(weights, annual_return, annual_cov)

        results['returns'].append(ret)
        results['volatility'].append(vol)
        results['sharpe'].append(sharpe)
        results['weights'].append(weights)

    # Convert lists to numpy arrays
    for key in ['returns', 'volatility', 'sharpe']:
        results[key] = np.array(results[key])

    return results


def monte_carlo_portfolio(weights, annual_return, annual_cov, time_horizon=1,
                          num_simulations=10000, initial_investment=10000):
    """
    Run Monte Carlo simulation for portfolio returns.
    Now includes path storage for visualization.
    """

    port_return, port_volatility, _ = portfolio_stats(weights, annual_return, annual_cov)

    daily_return = port_return / 252
    daily_volatility = port_volatility / np.sqrt(252)

    # Number of trading days to simulate
    num_days = int(time_horizon * 252)

    # Storage for all paths (NEW)
    all_paths = np.zeros((num_simulations, num_days))

    # Run simulations
    simulated_returns = []
    final_values = []

    for sim in range(num_simulations):
        # Generate random daily returns
        daily_returns = np.random.normal(daily_return, daily_volatility, num_days)

        # Calculate portfolio value path (NEW)
        portfolio_values = np.zeros(num_days)
        portfolio_values[0] = initial_investment * (1 + daily_returns[0])

        for day in range(1, num_days):
            portfolio_values[day] = portfolio_values[day - 1] * (1 + daily_returns[day])

        all_paths[sim] = portfolio_values

        # Calculate cumulative return
        cumulative_return = (portfolio_values[-1] / initial_investment) - 1

        simulated_returns.append(cumulative_return)
        final_values.append(portfolio_values[-1])

    simulated_returns = np.array(simulated_returns)
    final_values = np.array(final_values)

    # Calculate statistics
    results = {
        'returns': simulated_returns,
        'final_values': final_values,
        'paths': all_paths,  # NEW: Store all paths for plotting
        'mean_return': np.mean(simulated_returns),
        'median_return': np.median(simulated_returns),
        'std_return': np.std(simulated_returns),
        'percentile_5': np.percentile(simulated_returns, 5),
        'percentile_10': np.percentile(simulated_returns, 10),
        'percentile_25': np.percentile(simulated_returns, 25),
        'percentile_75': np.percentile(simulated_returns, 75),
        'percentile_90': np.percentile(simulated_returns, 90),
        'percentile_95': np.percentile(simulated_returns, 95),
        'prob_loss': np.sum(simulated_returns < 0) / num_simulations,
        'prob_gain': np.sum(simulated_returns > 0) / num_simulations,
        'var_95': np.percentile(simulated_returns, 5),
        'best_case': np.max(simulated_returns),
        'worst_case': np.min(simulated_returns)
    }

    return results
