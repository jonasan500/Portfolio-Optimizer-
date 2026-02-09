# app.py
"""
Portfolio Optimizer - Main execution script
"""
import numpy as np
from src.data_pipeline import fetch_price_data, calculate_returns, calculate_annual_stats
from src.portfolio_engine import generate_random_portfolios, monte_carlo_portfolio
from src.config import TICKERS, NUM_PORTFOLIOS, MIN_POSITION, MAX_POSITION


def main():
    """Run complete portfolio analysis."""

    print("=" * 70)
    print("PORTFOLIO OPTIMIZER".center(70))
    print("=" * 70)

    # Fetch and process data
    print("\nStep 1: Fetching historical data...")
    prices = fetch_price_data()
    print(f"✓ Loaded {len(prices)} days of price data for {len(TICKERS)} assets")

    print("\nStep 2: Calculating returns and statistics...")
    daily_returns = calculate_returns(prices)
    annual_return, annual_volatility, annual_cov = calculate_annual_stats(daily_returns)
    print(f"✓ Calculated statistics from {len(daily_returns)} days of returns")

    # Display individual asset stats
    print("\n" + "=" * 70)
    print("INDIVIDUAL ASSET STATISTICS".center(70))
    print("=" * 70)

    print("\nAnnualized Return:")
    for ticker in TICKERS:
        print(f"  {ticker:5s}: {annual_return[ticker] * 100:6.2f}%")

    print("\nAnnualized Volatility:")
    for ticker in TICKERS:
        print(f"  {ticker:5s}: {annual_volatility[ticker] * 100:6.2f}%")

    print("\nSharpe Ratio:")
    for ticker in TICKERS:
        sharpe = annual_return[ticker] / annual_volatility[ticker]
        print(f"  {ticker:5s}: {sharpe:6.3f}")

    # Generate random portfolios
    print(f"\nStep 3: Generating {NUM_PORTFOLIOS:,} random portfolios...")
    print(f"         Position limits: {MIN_POSITION * 100:.0f}%-{MAX_POSITION * 100:.0f}% per stock")

    results = generate_random_portfolios(
        annual_return,
        annual_cov,
        NUM_PORTFOLIOS,
        MIN_POSITION,
        MAX_POSITION
    )
    print(f"✓ Generated {NUM_PORTFOLIOS:,} portfolio combinations")

    # Find optimal portfolios
    max_sharpe_idx = np.argmax(results['sharpe'])
    min_vol_idx = np.argmin(results['volatility'])

        # After finding optimal portfolios, add this:
    print("\nGenerating efficient frontier visualization...")

    from src.visualizations import plot_efficient_frontier

    plot_efficient_frontier(results, max_sharpe_idx, min_vol_idx)

    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZED PORTFOLIOS".center(70))
    print("=" * 70)

    print("\n1) MAXIMUM SHARPE RATIO PORTFOLIO")
    print("-" * 70)
    print(f"   Expected Return:  {results['returns'][max_sharpe_idx] * 100:6.2f}%")
    print(f"   Volatility:       {results['volatility'][max_sharpe_idx] * 100:6.2f}%")
    print(f"   Sharpe Ratio:     {results['sharpe'][max_sharpe_idx]:6.3f}")
    print("\n   Allocation:")
    for i, ticker in enumerate(TICKERS):
        weight = results['weights'][max_sharpe_idx][i]
        print(f"     {ticker:5s}: {weight * 100:5.1f}%")

    print("\n2) MINIMUM VOLATILITY PORTFOLIO")
    print("-" * 70)
    print(f"   Expected Return:  {results['returns'][min_vol_idx] * 100:6.2f}%")
    print(f"   Volatility:       {results['volatility'][min_vol_idx] * 100:6.2f}%")
    print(f"   Sharpe Ratio:     {results['sharpe'][min_vol_idx]:6.3f}")
    print("\n   Allocation:")
    for i, ticker in enumerate(TICKERS):
        weight = results['weights'][min_vol_idx][i]
        print(f"     {ticker:5s}: {weight * 100:5.1f}%")

    print("\n" + "=" * 70)
    print(f"Analysis complete: {NUM_PORTFOLIOS:,} portfolios tested")
    print("=" * 70)


def test_monte_carlo():
    """Test Monte Carlo simulation on max Sharpe portfolio."""

    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION".center(70))
    print("=" * 70)

    # Get data
    print("\nFetching data and calculating statistics...")
    prices = fetch_price_data()
    daily_returns = calculate_returns(prices)
    annual_return, annual_volatility, annual_cov = calculate_annual_stats(daily_returns)

    # Generate portfolios to find max Sharpe
    print("Finding optimal portfolio...")
    results = generate_random_portfolios(annual_return, annual_cov, NUM_PORTFOLIOS,
                                         MIN_POSITION, MAX_POSITION)
    max_sharpe_idx = np.argmax(results['sharpe'])
    optimal_weights = results['weights'][max_sharpe_idx]

    print(f"\nOptimal Portfolio (Max Sharpe: {results['sharpe'][max_sharpe_idx]:.3f}):")
    for i, ticker in enumerate(TICKERS):
        print(f"  {ticker}: {optimal_weights[i] * 100:.1f}%")

    # Run Monte Carlo
    print(f"\nRunning Monte Carlo simulation...")
    print(f"  Simulations: 10,000")
    print(f"  Time Horizon: 1 year")
    print(f"  Initial Investment: $10,000")

    mc_results = monte_carlo_portfolio(
        optimal_weights,
        annual_return,
        annual_cov,
        time_horizon=1,
        num_simulations=10000,
        initial_investment=10000
    )

    # Display results
    print("\n" + "-" * 70)
    print("SIMULATION RESULTS")
    print("-" * 70)

    print(f"\nExpected Outcomes (1 Year):")
    print(f"  Mean Return:      {mc_results['mean_return'] * 100:6.2f}%")
    print(f"  Median Return:    {mc_results['median_return'] * 100:6.2f}%")
    print(f"  Std Deviation:    {mc_results['std_return'] * 100:6.2f}%")

    print(f"\nPercentile Analysis:")
    print(f"  95th percentile:  {mc_results['percentile_95'] * 100:6.2f}%  (best 5%)")
    print(f"  90th percentile:  {mc_results['percentile_90'] * 100:6.2f}%")
    print(f"  75th percentile:  {mc_results['percentile_75'] * 100:6.2f}%")
    print(f"  50th percentile:  {mc_results['median_return'] * 100:6.2f}%  (median)")
    print(f"  25th percentile:  {mc_results['percentile_25'] * 100:6.2f}%")
    print(f"  10th percentile:  {mc_results['percentile_10'] * 100:6.2f}%")
    print(f"  5th percentile:   {mc_results['percentile_5'] * 100:6.2f}%   (worst 5%)")

    print(f"\nRisk Metrics:")
    print(f"  Probability of Gain:  {mc_results['prob_gain'] * 100:.1f}%")
    print(f"  Probability of Loss:  {mc_results['prob_loss'] * 100:.1f}%")
    print(f"  Value at Risk (95%):  {mc_results['var_95'] * 100:6.2f}%")

    print(f"\nExtreme Scenarios:")
    print(f"  Best Case (max):   {mc_results['best_case'] * 100:6.2f}%")
    print(f"  Worst Case (min):  {mc_results['worst_case'] * 100:6.2f}%")

    print(f"\nFinal Portfolio Value Distribution:")
    print(f"  Expected:  ${np.mean(mc_results['final_values']):,.0f}")
    print(f"  Median:    ${np.median(mc_results['final_values']):,.0f}")
    print(f"  Best 5%:   ${np.percentile(mc_results['final_values'], 95):,.0f}")
    print(f"  Worst 5%:  ${np.percentile(mc_results['final_values'], 5):,.0f}")
    # Generate visualizations
    print("\nGenerating visualizations...")

    from src.visualizations import plot_monte_carlo_paths, plot_return_distribution

    # Plot Monte Carlo paths
    plot_monte_carlo_paths(mc_results, optimal_weights, TICKERS,
                           num_paths_to_plot=200, initial_investment=10000)

    # Plot return distribution
    plot_return_distribution(mc_results)

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()