# src/visualizations.py
"""
Visualization functions for portfolio analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_monte_carlo_paths(mc_results, weights, tickers, num_paths_to_plot=100,
                           initial_investment=10000, save_path=None):
    """
    Plot Monte Carlo simulation paths showing portfolio value over time.

    Args:
        mc_results: Results from monte_carlo_portfolio_paths()
        weights: Portfolio weights
        tickers: List of ticker symbols
        num_paths_to_plot: How many paths to display (default 100 for clarity)
        initial_investment: Starting value
        save_path: If provided, saves plot to this path
    """
    paths = mc_results['paths']
    num_simulations, num_days = paths.shape

    # Select random subset of paths to plot (for clarity)
    indices = np.random.choice(num_simulations, size=min(num_paths_to_plot, num_simulations), replace=False)

    plt.figure(figsize=(12, 7))

    # Plot individual paths with transparency
    for idx in indices:
        plt.plot(paths[idx], alpha=0.3, linewidth=0.8)

    # Plot median path in bold
    median_path = np.median(paths, axis=0)
    plt.plot(median_path, color='black', linewidth=2.5, label='Median Path', zorder=100)

    # Plot percentile bands
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    plt.fill_between(range(num_days), p5, p95, alpha=0.2, color='gray', label='90% Confidence Band')

    plt.axhline(y=initial_investment, color='red', linestyle='--', linewidth=1.5, label='Initial Investment')

    plt.title('Monte Carlo Simulation: Portfolio Value Paths\n' +
              f'({num_simulations:,} simulations, {num_days} trading days)', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add portfolio composition as text
    portfolio_text = "Portfolio: " + ", ".join([f"{t}: {w * 100:.1f}%" for t, w in zip(tickers, weights)])
    plt.figtext(0.5, 0.02, portfolio_text, ha='center', fontsize=9, style='italic')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_return_distribution(mc_results, save_path=None):
    """
    Plot histogram of Monte Carlo return distribution.
    """
    returns = mc_results['returns'] * 100  # Convert to percentage

    plt.figure(figsize=(12, 6))

    # Histogram
    n, bins, patches = plt.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    # Add percentile lines
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'green', 'orange', 'red']

    for p, color in zip(percentiles, colors):
        value = np.percentile(returns, p)
        plt.axvline(value, color=color, linestyle='--', linewidth=2,
                    label=f'{p}th percentile: {value:.1f}%')

    plt.title('Monte Carlo Return Distribution\n1-Year Horizon', fontsize=14, fontweight='bold')
    plt.xlabel('Return (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics box
    stats_text = f"Mean: {mc_results['mean_return'] * 100:.1f}%\n"
    stats_text += f"Median: {mc_results['median_return'] * 100:.1f}%\n"
    stats_text += f"Std Dev: {mc_results['std_return'] * 100:.1f}%\n"
    stats_text += f"P(Loss): {mc_results['prob_loss'] * 100:.1f}%"

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_efficient_frontier(results, max_sharpe_idx, min_vol_idx, save_path=None):
    """
    Plot efficient frontier from random portfolio results.

    Args:
        results: Dictionary with 'returns', 'volatility', 'sharpe' arrays
        max_sharpe_idx: Index of maximum Sharpe portfolio
        min_vol_idx: Index of minimum volatility portfolio
        save_path: Optional path to save figure
    """
    returns = results['returns'] * 100
    volatility = results['volatility'] * 100
    sharpe = results['sharpe']

    plt.figure(figsize=(12, 8))

    # Scatter plot colored by Sharpe ratio
    scatter = plt.scatter(volatility, returns, c=sharpe, cmap='viridis',
                          alpha=0.5, s=10, edgecolors='none')
    plt.colorbar(scatter, label='Sharpe Ratio')

    # Highlight special portfolios
    plt.scatter(volatility[max_sharpe_idx], returns[max_sharpe_idx],
                color='red', s=200, marker='*', edgecolors='black', linewidths=2,
                label='Max Sharpe', zorder=100)

    plt.scatter(volatility[min_vol_idx], returns[min_vol_idx],
                color='green', s=200, marker='s', edgecolors='black', linewidths=2,
                label='Min Volatility', zorder=100)

    plt.title('Efficient Frontier\nRisk vs Return Trade-off', fontsize=14, fontweight='bold')
    plt.xlabel('Volatility (Annual Std Dev) %', fontsize=12)
    plt.ylabel('Expected Return (Annual) %', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")

    plt.tight_layout()
    plt.show()