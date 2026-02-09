# streamlit_app.py
"""
Portfolio Optimizer - Interactive Web Interface
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_pipeline import (
    fetch_price_data,
    calculate_returns,
    calculate_annual_stats,
    validate_tickers,
)
from src.portfolio_engine import generate_random_portfolios, monte_carlo_portfolio
from src.config import TICKERS, LOOKBACK_YEARS, NUM_PORTFOLIOS


# ---------------------------
# Session state
# ---------------------------
if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None

if "panel_open" not in st.session_state:
    st.session_state.panel_open = False

# Persist inputs so panel can be closed without losing values
if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = ", ".join(TICKERS)
if "min_position" not in st.session_state:
    st.session_state.min_position = 0.05
if "max_position" not in st.session_state:
    st.session_state.max_position = 0.40
if "lookback_years" not in st.session_state:
    st.session_state.lookback_years = LOOKBACK_YEARS
if "num_portfolios" not in st.session_state:
    st.session_state.num_portfolios = NUM_PORTFOLIOS


# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Portfolio Optimizer by Jonathan Sanchez",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# CSS
# ---------------------------
st.markdown(
    """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Scope+One&display=swap');

    * { font-family: 'Scope One', serif !important; border-radius: 0px !important; }

    .stApp { background-color: #000000; }
    .stApp, p, span, label, div, li { color: #ffffff !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 400 !important; }

    /* Hide Streamlit sidebar visually (we won't use it) */
    [data-testid="stSidebar"] { display: none !important; }

    .creator-badge {
        position: fixed;
        bottom: 15px;
        right: 15px;
        background-color: #1a1a1a;
        color: #999999;
        padding: 8px 16px;
        border: 1px solid #333333;
        font-size: 12px;
        z-index: 999;
    }

    .stButton>button {
        background-color: #540001 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 24px;
        font-size: 14px;
        transition: all 0.2s;
    }
    .stButton>button:hover { background-color: #6B0F1A !important; }

    [data-testid="stMetricValue"] { font-size: 28px; color: #540001 !important; }
    [data-testid="stMetricLabel"] { color: #ffffff !important; font-size: 13px; }

    input, textarea, select {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    [data-baseweb="select"] { background-color: #1a1a1a !important; }
    [data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #540001 !important;
    }
    [role="listbox"] { background-color: #1a1a1a !important; border: 1px solid #540001 !important; }
    [role="option"] { background-color: #1a1a1a !important; color: #ffffff !important; padding: 8px !important; }
    [role="option"]:hover { background-color: #540001 !important; color: #ffffff !important; }

    input[type="number"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #540001 !important;
    }

    table, [data-testid="stDataFrame"] { background-color: #1a1a1a !important; color: #ffffff !important; }

    .stSuccess, .stWarning, .stError, .stInfo {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
    }

    .stTabs [data-baseweb="tab"] { color: #999999 !important; }
    .stTabs [aria-selected="true"] { color: #540001 !important; border-bottom: 2px solid #540001 !important; }

    #MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }

    /* In-page control panel */
    .control-panel {
        background-color: #000000;
        border: 1px solid #333333;
        padding: 14px;
        position: sticky;
        top: 10px;
    }

    @media only screen and (max-width: 768px) {
        .main .block-container { padding: 0.5rem !important; max-width: 100vw !important; }
        .stButton>button {
            width: 100% !important;
            padding: 14px 20px !important;
            font-size: 16px !important;
            margin: 8px 0 !important;
        }
        [data-testid="stMetricValue"] { font-size: 20px !important; }
        [data-testid="stMetricLabel"] { font-size: 11px !important; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        p, div, span, label, li { font-size: 13px !important; }
        .creator-badge { display: none !important; }
    }
</style>

<div class="creator-badge">Jonathan Sanchez</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Header + BEGIN toggle
# ---------------------------
st.markdown(
    """
<div style='border-bottom: 1px solid #333333; padding-bottom: 20px; margin-bottom: 14px;'>
    <h1 style='font-size: 32px; margin-bottom: 8px;'>Sharpe Ratio Portfolio Optimizer</h1>
    <p style='color: #999999; font-size: 14px; margin: 0;'>
        Quantitative portfolio optimization using Modern Portfolio Theory
    </p>
</div>
""",
    unsafe_allow_html=True,
)

bcol1, bcol2 = st.columns([1, 6])
with bcol1:
    if st.button("🚀 BEGIN", type="primary"):
        st.session_state.panel_open = not st.session_state.panel_open
with bcol2:
    st.caption("Tap **BEGIN** to open/close settings. (This replaces the sidebar so it works everywhere.)")

st.markdown("---")

# ---------------------------
# Layout: left panel + main
# ---------------------------
if st.session_state.panel_open:
    left, main = st.columns([1, 2.2], gap="large")
else:
    left, main = st.columns([0.0001, 1], gap="large")

# ---------------------------
# Control Panel (PURE STREAMLIT)
# ---------------------------
optimize_button = False  # default

with left:
    if st.session_state.panel_open:
        st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
        st.header("⚙️ Portfolio Settings")

        st.subheader("1. Select Assets")
        st.text_input(
            "Enter tickers (comma-separated)",
            key="ticker_input",
        )

        st.subheader("2. Position Limits")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.min_position = st.slider(
                "Min %",
                0,
                50,
                int(st.session_state.min_position * 100),
                5,
            ) / 100
        with c2:
            st.session_state.max_position = st.slider(
                "Max %",
                10,
                100,
                int(st.session_state.max_position * 100),
                5,
            ) / 100

        st.subheader("3. Historical Data")
        st.session_state.lookback_years = st.slider("Years of history", 3, 20, int(st.session_state.lookback_years))
        st.session_state.num_portfolios = st.slider(
            "Portfolios to test",
            1000,
            50000,
            int(st.session_state.num_portfolios),
            1000,
        )

        optimize_button = st.button("🚀 Optimize Portfolio", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

# Parse tickers from stored input (works even when panel is closed)
tickers_list = [t.strip().upper() for t in st.session_state.ticker_input.split(",") if t.strip()]

# ---------------------------
# Optimization logic (runs when Optimize clicked)
# ---------------------------
if optimize_button:
    if len(tickers_list) < 2:
        with main:
            st.error("⚠️ Please enter at least 2 tickers")
    else:
        with main:
            progress_bar = st.progress(0)
            status_text = st.empty()

        try:
            with main:
                status_text.text("🔍 Validating tickers...")
                progress_bar.progress(10)
            valid_tickers = validate_tickers(tickers_list)

            if len(valid_tickers) < 2:
                with main:
                    st.error("⚠️ Not enough valid tickers")
                st.stop()

            with main:
                st.success(f"✅ Valid tickers: {', '.join(valid_tickers)}")
                status_text.text("📥 Fetching data...")
                progress_bar.progress(30)

            prices = fetch_price_data(tickers=valid_tickers, years=st.session_state.lookback_years)

            with main:
                status_text.text("📊 Calculating...")
                progress_bar.progress(50)

            daily_returns = calculate_returns(prices)
            annual_return, annual_volatility, annual_cov = calculate_annual_stats(daily_returns)

            with main:
                status_text.text(f"🔄 Testing {st.session_state.num_portfolios:,} portfolios...")
                progress_bar.progress(70)

            results = generate_random_portfolios(
                annual_return,
                annual_cov,
                st.session_state.num_portfolios,
                st.session_state.min_position,
                st.session_state.max_position,
            )

            max_sharpe_idx = int(np.argmax(results["sharpe"]))
            min_vol_idx = int(np.argmin(results["volatility"]))

            with main:
                progress_bar.progress(100)
                status_text.text("✅ Complete!")

            st.session_state.optimization_results = {
                "results": results,
                "max_sharpe_idx": max_sharpe_idx,
                "min_vol_idx": min_vol_idx,
                "annual_return": annual_return,
                "annual_cov": annual_cov,
                "annual_volatility": annual_volatility,
                "valid_tickers": valid_tickers,
                "lookback_years": st.session_state.lookback_years,
                "num_portfolios": st.session_state.num_portfolios,
                "min_position": st.session_state.min_position,
                "max_position": st.session_state.max_position,
            }

        except Exception as e:
            with main:
                st.error(f"❌ Error: {str(e)}")

# ---------------------------
# Main content (results / tabs)
# ---------------------------
with main:
    if st.session_state.optimization_results is None:
        st.info("👈 Tap **BEGIN** to open settings, then click **Optimize Portfolio**.")
    else:
        stored = st.session_state.optimization_results
        results = stored["results"]
        max_sharpe_idx = stored["max_sharpe_idx"]
        min_vol_idx = stored["min_vol_idx"]
        annual_return = stored["annual_return"]
        annual_cov = stored["annual_cov"]
        annual_volatility = stored["annual_volatility"]
        valid_tickers = stored["valid_tickers"]

        st.success("🎯 Optimization Complete!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Return", f"{results['returns'][max_sharpe_idx] * 100:.2f}%")
        with col2:
            st.metric("Portfolio Risk", f"{results['volatility'][max_sharpe_idx] * 100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{results['sharpe'][max_sharpe_idx]:.3f}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📈 Efficient Frontier", "💼 Portfolio", "🎲 Monte Carlo", "📊 Assets", "ℹ️ About"]
        )

        with tab1:
            st.subheader("Efficient Frontier")
            fig, ax = plt.subplots(figsize=(10, 6), facecolor="#000000")
            ax.set_facecolor("#000000")

            returns_pct = results["returns"] * 100
            volatility_pct = results["volatility"] * 100

            scatter = ax.scatter(volatility_pct, returns_pct, c=results["sharpe"], cmap="plasma", alpha=0.7, s=15)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Sharpe", color="#fff")
            cbar.ax.tick_params(colors="#fff")

            ax.scatter(
                volatility_pct[max_sharpe_idx],
                returns_pct[max_sharpe_idx],
                color="#0f0",
                s=400,
                marker="*",
                edgecolors="#fff",
                linewidths=2,
                label="Max Sharpe",
            )
            ax.scatter(
                volatility_pct[min_vol_idx],
                returns_pct[min_vol_idx],
                color="#0bf",
                s=250,
                marker="D",
                edgecolors="#fff",
                linewidths=2,
                label="Min Vol",
            )

            ax.set_title("Efficient Frontier", fontsize=16, color="#fff", pad=20)
            ax.set_xlabel("Volatility (%)", fontsize=13, color="#fff")
            ax.set_ylabel("Return (%)", fontsize=13, color="#fff")
            ax.tick_params(colors="#fff", labelsize=11)
            ax.legend(loc="lower right", facecolor="#1a1a1a", edgecolor="#fff", labelcolor="#fff")
            ax.grid(True, alpha=0.3, color="#444")
            st.pyplot(fig)

        with tab2:
            st.subheader("Portfolio Allocation")

            optimal_weights = results["weights"][max_sharpe_idx]
            allocation_df = pd.DataFrame({"Ticker": valid_tickers, "Weight": optimal_weights * 100}).sort_values(
                "Weight", ascending=False
            )
            st.dataframe(allocation_df.style.format({"Weight": "{:.1f}%"}), hide_index=True)

            fig, ax = plt.subplots(figsize=(7, 7), facecolor="#000")
            ax.set_facecolor("#000")
            colors = ["#F66", "#4EC", "#45B", "#FA7", "#98D", "#F7D"]
            ax.pie(
                optimal_weights,
                labels=valid_tickers,
                autopct="%1.1f%%",
                colors=colors[: len(valid_tickers)],
                textprops={"color": "#fff", "fontsize": 12},
            )
            ax.set_title("Portfolio Allocation", fontsize=16, color="#fff", pad=20)
            st.pyplot(fig)

        with tab3:
            st.subheader("🎲 Monte Carlo Analysis")

            c1, c2, c3 = st.columns(3)
            with c1:
                mc_sims = st.selectbox("Simulations", [1000, 5000, 10000], index=2)
            with c2:
                horizon = st.selectbox("Years", [1, 3, 5, 10], index=0)
            with c3:
                invest = st.number_input("Investment ($)", min_value=1000, value=10000, step=1000)

            if st.button("🎲 Run Simulation", type="primary"):
                with st.spinner("Running..."):
                    mc_res = monte_carlo_portfolio(
                        results["weights"][max_sharpe_idx],
                        annual_return,
                        annual_cov,
                        time_horizon=horizon,
                        num_simulations=mc_sims,
                        initial_investment=invest,
                    )

                st.success("✅ Complete!")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Median Return", f"{mc_res['median_return'] * 100:.1f}%")
                c2.metric("Prob Gain", f"{mc_res['prob_gain'] * 100:.1f}%")
                c3.metric("VaR 95%", f"{mc_res['var_95'] * 100:.1f}%")
                c4.metric("Expected", f"${np.mean(mc_res['final_values']):,.0f}")

                fig, ax = plt.subplots(figsize=(10, 6), facecolor="#000")
                ax.set_facecolor("#000")
                for i in range(min(200, len(mc_res["paths"]))):
                    ax.plot(mc_res["paths"][i], alpha=0.3, linewidth=0.5, color="#fff")
                median = np.median(mc_res["paths"], axis=0)
                ax.plot(median, color="#0f0", linewidth=3.5, label="Median")
                ax.axhline(invest, color="#fd0", linestyle="--", linewidth=2.5, label="Initial")
                ax.set_title(f"Monte Carlo: {mc_sims:,} Simulations ({horizon}Y)", fontsize=15, color="#fff", pad=15)
                ax.set_xlabel("Days", fontsize=12, color="#fff")
                ax.set_ylabel("Value ($)", fontsize=12, color="#fff")
                ax.tick_params(colors="#fff")
                ax.legend(facecolor="#1a1a1a", labelcolor="#fff")
                ax.grid(True, alpha=0.3, color="#444")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10, 5), facecolor="#000")
                ax.set_facecolor("#000")
                rets = mc_res["returns"] * 100
                ax.hist(rets, bins=40, alpha=0.8, color="#4EC", edgecolor="#fff")

                percs = [5, 25, 50, 75, 95]
                cols = ["#F66", "#FA7", "#fd0", "#9e9", "#0f0"]
                for p, col in zip(percs, cols):
                    val = np.percentile(rets, p)
                    ax.axvline(val, color=col, linestyle="--", linewidth=2.5, label=f"{p}th: {val:.1f}%")

                ax.set_title("Return Distribution", fontsize=15, color="#fff", pad=15)
                ax.set_xlabel("Return (%)", fontsize=12, color="#fff")
                ax.set_ylabel("Frequency", fontsize=12, color="#fff")
                ax.tick_params(colors="#fff")
                ax.legend(facecolor="#1a1a1a", labelcolor="#fff", fontsize=9)
                ax.grid(True, alpha=0.25, axis="y", color="#444")
                st.pyplot(fig)

        with tab4:
            st.subheader("Individual Assets")

            assets_df = pd.DataFrame(
                {
                    "Ticker": valid_tickers,
                    "Return (%)": annual_return.values * 100,
                    "Volatility (%)": annual_volatility.values * 100,
                    "Sharpe": annual_return.values / annual_volatility.values,
                }
            ).sort_values("Sharpe", ascending=False)

            st.dataframe(
                assets_df.style.format({"Return (%)": "{:.2f}%", "Volatility (%)": "{:.2f}%", "Sharpe": "{:.3f}"}),
                hide_index=True,
            )

            fig, ax = plt.subplots(figsize=(9, 5), facecolor="#000")
            ax.set_facecolor("#000")
            x = np.arange(len(valid_tickers))
            ax.bar(x - 0.175, annual_return.values * 100, 0.35, label="Return", color="#0f0", alpha=0.8)
            ax.bar(x + 0.175, annual_volatility.values * 100, 0.35, label="Volatility", color="#F66", alpha=0.8)
            ax.set_xlabel("Asset", color="#fff")
            ax.set_ylabel("Percentage (%)", color="#fff")
            ax.set_title("Risk vs Return", fontsize=15, color="#fff", pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(valid_tickers, color="#fff")
            ax.tick_params(colors="#fff")
            ax.legend(facecolor="#1a1a1a", labelcolor="#fff")
            ax.grid(True, alpha=0.25, axis="y", color="#444")
            st.pyplot(fig)

        with tab5:
            st.subheader("About This Optimizer")
            st.markdown(
                """
**What this website does**

This web app builds and evaluates portfolios using **Modern Portfolio Theory**. You enter a list of tickers and constraints (min/max weight per asset),
choose a history window, and select how many random portfolios to test. The optimizer:

- Downloads **historical price data** for the selected tickers over your chosen lookback period  
- Converts prices into **returns** and estimates each asset’s:
  - expected annual return  
  - annualized volatility  
  - covariance matrix (how assets move together)
- Generates thousands of **random portfolios** that respect your position limits
- Computes portfolio **expected return, risk (volatility), and Sharpe ratio**
- Highlights the portfolio with the **maximum Sharpe ratio** and the portfolio with **minimum volatility**
- Runs an optional **Monte Carlo simulation** to model a range of potential future outcomes based on historical return/volatility relationships

**Important limitations & realism notes**

- **More data = more time.** Increasing “Years of history” or “Portfolios to test” can significantly increase runtime.
- **Historical returns can be skewed.** A single strong year (or crash) can heavily influence average returns and volatility estimates.
- **Regime changes happen.** Correlations and volatilities change over time — relationships observed in the past may not hold in the future.
- **Data quality matters.** Missing price points, short histories, and corporate actions can distort results.
- **This is a model, not a guarantee.** “Optimal” here means optimal under the assumptions + window you selected — not a prediction.

**Disclaimer**

This tool is provided **for educational and research purposes only** and is **not financial advice**. Nothing on this page is a recommendation to buy or sell any security.
Use at your own risk.
"""
            )
            st.error("**NOT financial advice. Educational purposes only.**")
            st.markdown("---")
            st.markdown(
                "<p style='text-align: center; color: #999;'>Created by Jonathan Sanchez • 2026</p>",
                unsafe_allow_html=True,
            )
