# streamlit_app.py
"""
Portfolio Optimizer - Interactive Web Interface
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_pipeline import fetch_price_data, calculate_returns, calculate_annual_stats, validate_tickers
from src.portfolio_engine import generate_random_portfolios, portfolio_stats, monte_carlo_portfolio
from src.config import TICKERS, LOOKBACK_YEARS, NUM_PORTFOLIOS

# Initialize session state
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer by Jonathan Sanchez",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Scope+One&display=swap');

    * {
        font-family: 'Scope One', serif !important;
        border-radius: 0px !important;
    }

    .stApp {
        background-color: #000000;
    }

    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333333;
    }

    .stApp, p, span, label, div, li {
        color: #ffffff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 400 !important;
    }

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

    .stButton>button:hover {
        background-color: #6B0F1A !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #540001 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 13px;
    }

    input, textarea, select {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Fix dropdown visibility - CRITICAL FIX */
    [data-baseweb="select"] {
        background-color: #1a1a1a !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #540001 !important;
    }
    
    /* Dropdown menu items */
    [role="listbox"] {
        background-color: #1a1a1a !important;
        border: 1px solid #540001 !important;
    }
    
    [role="option"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        padding: 8px !important;
    }
    
    [role="option"]:hover {
        background-color: #540001 !important;
        color: #ffffff !important;
    }
    
    /* Number input visibility */
    input[type="number"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #540001 !important;
    }

    table, [data-testid="stDataFrame"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }

    .stSuccess, .stWarning, .stError, .stInfo {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #999999 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #540001 !important;
        border-bottom: 2px solid #540001 !important;
    }

    #MainMenu, footer, [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    
    /* ULTRA BRIGHT RED SIDEBAR TOGGLE - IMPOSSIBLE TO MISS */
    button[kind="header"] {
        background-color: #FF0000 !important;
        color: #ffffff !important;
        border: 4px solid #ffffff !important;
        padding: 14px 20px !important;
        font-size: 26px !important;
        font-weight: 900 !important;
        box-shadow: 0 0 30px rgba(255, 0, 0, 1), 0 0 60px rgba(255, 0, 0, 0.6) !important;
        cursor: pointer !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    button[kind="header"]:hover {
        background-color: #CC0000 !important;
        transform: scale(1.15) !important;
        box-shadow: 0 0 40px rgba(255, 0, 0, 1), 0 0 80px rgba(255, 0, 0, 0.8) !important;
    }

    /* Mobile Responsive Fixes */
    @media only screen and (max-width: 768px) {
        /* Force proper mobile layout */
        .main .block-container {
            padding: 0.5rem !important;
            max-width: 100vw !important;
        }
        
        /* Full width buttons on mobile */
        .stButton>button {
            width: 100% !important;
            padding: 14px 20px !important;
            font-size: 16px !important;
            margin: 10px 0 !important;
        }
        
        /* Readable metrics on mobile */
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 11px !important;
        }
        
        /* Smaller headings on mobile */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        
        /* Readable text on mobile */
        p, div, span, label, li { font-size: 13px !important; }
        
        /* Fit charts to screen */
        canvas, [data-testid="stImage"] {
            max-width: 100% !important;
            width: 100% !important;
            height: auto !important;
        }
        
        /* Better sidebar on mobile */
        [data-testid="stSidebar"] {
            width: 100% !important;
        }
        
        /* Scrollable tables on mobile */
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
            font-size: 11px !important;
        }
        
        /* Full width inputs on mobile */
        input, textarea, select {
            width: 100% !important;
            font-size: 16px !important;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        
        /* Hide creator badge on mobile */
        .creator-badge {
            display: none !important;
        }
        
        /* Better info boxes on mobile */
        .stInfo, .stSuccess, .stWarning, .stError {
            font-size: 13px !important;
            padding: 10px !important;
        }
        
        /* ULTRA VISIBLE RED GLOWING SIDEBAR TOGGLE ON MOBILE */
        button[kind="header"] {
            display: block !important;
            position: fixed !important;
            top: 65px !important;
            left: 15px !important;
            z-index: 99999 !important;
            background-color: #FF0000 !important;
            color: #ffffff !important;
            border: 4px solid #ffffff !important;
            padding: 16px 22px !important;
            font-size: 24px !important;
            font-weight: 900 !important;
            box-shadow: 0 0 35px rgba(255, 0, 0, 1), 0 0 70px rgba(255, 0, 0, 0.8) !important;
            border-radius: 8px !important;
            animation: pulse-red 2s infinite !important;
        }
        
        button[kind="header"]:hover {
            transform: scale(1.2) !important;
            box-shadow: 0 0 50px rgba(255, 0, 0, 1), 0 0 100px rgba(255, 0, 0, 1) !important;
        }
        
        @keyframes pulse-red {
            0%, 100% {
                box-shadow: 0 0 35px rgba(255, 0, 0, 1), 0 0 70px rgba(255, 0, 0, 0.8);
            }
            50% {
                box-shadow: 0 0 50px rgba(255, 0, 0, 1), 0 0 100px rgba(255, 0, 0, 1);
            }
        }
    }
</style>

<div class="creator-badge">Jonathan Sanchez</div>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div style='border-bottom: 1px solid #333333; padding-bottom: 20px; margin-bottom: 30px;'>
    <h1 style='font-size: 32px; margin-bottom: 8px;'>
        Sharpe Ratio Portfolio Optimizer
    </h1>
    <p style='color: #999999; font-size: 14px; margin: 0;'>
        Quantitative portfolio optimization using Modern Portfolio Theory
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Portfolio Settings")
st.sidebar.subheader("1. Select Assets")
ticker_input = st.sidebar.text_input("Enter tickers (comma-separated)", value=", ".join(TICKERS))
tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

st.sidebar.subheader("2. Position Limits")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_position = st.slider("Min %", 0, 50, 5, 5) / 100
with col2:
    max_position = st.slider("Max %", 10, 100, 40, 5) / 100

st.sidebar.subheader("3. Historical Data")
lookback_years = st.sidebar.slider("Years of history", 3, 20, LOOKBACK_YEARS)
num_portfolios = st.sidebar.slider("Portfolios to test", 1000, 50000, NUM_PORTFOLIOS, 1000)

optimize_button = st.sidebar.button("🚀 Optimize Portfolio", type="primary")

# Optimization
if optimize_button:
    if len(tickers_list) < 2:
        st.error("⚠️ Please enter at least 2 tickers")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔍 Validating tickers...")
            progress_bar.progress(10)
            valid_tickers = validate_tickers(tickers_list)

            if len(valid_tickers) < 2:
                st.error("⚠️ Not enough valid tickers")
                st.stop()

            st.success(f"✅ Valid tickers: {', '.join(valid_tickers)}")

            status_text.text("📥 Fetching data...")
            progress_bar.progress(30)
            prices = fetch_price_data(tickers=valid_tickers, years=lookback_years)

            status_text.text("📊 Calculating...")
            progress_bar.progress(50)
            daily_returns = calculate_returns(prices)
            annual_return, annual_volatility, annual_cov = calculate_annual_stats(daily_returns)

            status_text.text(f"🔄 Testing {num_portfolios:,} portfolios...")
            progress_bar.progress(70)
            results = generate_random_portfolios(annual_return, annual_cov, num_portfolios,
                                                 min_position, max_position)

            max_sharpe_idx = np.argmax(results['sharpe'])
            min_vol_idx = np.argmin(results['volatility'])

            progress_bar.progress(100)
            status_text.text("✅ Complete!")

            st.session_state.optimization_results = {
                'results': results,
                'max_sharpe_idx': max_sharpe_idx,
                'min_vol_idx': min_vol_idx,
                'annual_return': annual_return,
                'annual_cov': annual_cov,
                'annual_volatility': annual_volatility,
                'valid_tickers': valid_tickers,
                'lookback_years': lookback_years,
                'num_portfolios': num_portfolios,
                'min_position': min_position,
                'max_position': max_position
            }

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Results
if st.session_state.optimization_results is not None:
    stored = st.session_state.optimization_results
    results = stored['results']
    max_sharpe_idx = stored['max_sharpe_idx']
    min_vol_idx = stored['min_vol_idx']
    annual_return = stored['annual_return']
    annual_cov = stored['annual_cov']
    annual_volatility = stored['annual_volatility']
    valid_tickers = stored['valid_tickers']

    st.success("🎯 Optimization Complete!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Return", f"{results['returns'][max_sharpe_idx] * 100:.2f}%")
    with col2:
        st.metric("Portfolio Risk", f"{results['volatility'][max_sharpe_idx] * 100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{results['sharpe'][max_sharpe_idx]:.3f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Efficient Frontier", "💼 Portfolio", "🎲 Monte Carlo", "📊 Assets", "ℹ️ About"])

    with tab1:
        st.subheader("Efficient Frontier")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#000000')
        ax.set_facecolor('#000000')

        returns_pct = results['returns'] * 100
        volatility_pct = results['volatility'] * 100

        scatter = ax.scatter(volatility_pct, returns_pct, c=results['sharpe'], cmap='plasma',
                             alpha=0.7, s=15)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe', color='#fff')
        cbar.ax.tick_params(colors='#fff')

        ax.scatter(volatility_pct[max_sharpe_idx], returns_pct[max_sharpe_idx],
                   color='#0f0', s=400, marker='*', edgecolors='#fff', linewidths=2, label='Max Sharpe')
        ax.scatter(volatility_pct[min_vol_idx], returns_pct[min_vol_idx],
                   color='#0bf', s=250, marker='D', edgecolors='#fff', linewidths=2, label='Min Vol')

        ax.set_title('Efficient Frontier', fontsize=16, color='#fff', pad=20)
        ax.set_xlabel('Volatility (%)', fontsize=13, color='#fff')
        ax.set_ylabel('Return (%)', fontsize=13, color='#fff')
        ax.tick_params(colors='#fff', labelsize=11)
        ax.legend(loc='lower right', facecolor='#1a1a1a', edgecolor='#fff', labelcolor='#fff')
        ax.grid(True, alpha=0.3, color='#444')

        st.pyplot(fig)

    with tab2:
        st.subheader("Portfolio Allocation")

        optimal_weights = results['weights'][max_sharpe_idx]
        allocation_df = pd.DataFrame({
            'Ticker': valid_tickers,
            'Weight': optimal_weights * 100
        }).sort_values('Weight', ascending=False)

        st.dataframe(allocation_df.style.format({'Weight': '{:.1f}%'}), hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 7), facecolor='#000')
        ax.set_facecolor('#000')
        colors = ['#F66', '#4EC', '#45B', '#FA7', '#98D', '#F7D']
        ax.pie(optimal_weights, labels=valid_tickers, autopct='%1.1f%%',
               colors=colors[:len(valid_tickers)], textprops={'color': '#fff', 'fontsize': 12})
        ax.set_title('Portfolio Allocation', fontsize=16, color='#fff', pad=20)
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
                mc_res = monte_carlo_portfolio(results['weights'][max_sharpe_idx],
                                               annual_return, annual_cov,
                                               time_horizon=horizon,
                                               num_simulations=mc_sims,
                                               initial_investment=invest)

                st.success("✅ Complete!")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Median Return", f"{mc_res['median_return'] * 100:.1f}%")
                c2.metric("Prob Gain", f"{mc_res['prob_gain'] * 100:.1f}%")
                c3.metric("VaR 95%", f"{mc_res['var_95'] * 100:.1f}%")
                c4.metric("Expected", f"${np.mean(mc_res['final_values']):,.0f}")

                # Paths Chart
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#000')
                ax.set_facecolor('#000')

                for i in range(min(200, len(mc_res['paths']))):
                    ax.plot(mc_res['paths'][i], alpha=0.3, linewidth=0.5, color='#fff')

                median = np.median(mc_res['paths'], axis=0)
                ax.plot(median, color='#0f0', linewidth=3.5, label='Median')
                ax.axhline(invest, color='#fd0', linestyle='--', linewidth=2.5, label='Initial')

                ax.set_title(f'Monte Carlo: {mc_sims:,} Simulations ({horizon}Y)',
                             fontsize=15, color='#fff', pad=15)
                ax.set_xlabel('Days', fontsize=12, color='#fff')
                ax.set_ylabel('Value ($)', fontsize=12, color='#fff')
                ax.tick_params(colors='#fff')
                ax.legend(facecolor='#1a1a1a', labelcolor='#fff')
                ax.grid(True, alpha=0.3, color='#444')
                st.pyplot(fig)

                # Distribution
                fig, ax = plt.subplots(figsize=(10, 5), facecolor='#000')
                ax.set_facecolor('#000')

                rets = mc_res['returns'] * 100
                ax.hist(rets, bins=40, alpha=0.8, color='#4EC', edgecolor='#fff')

                percs = [5, 25, 50, 75, 95]
                cols = ['#F66', '#FA7', '#fd0', '#9e9', '#0f0']

                for p, col in zip(percs, cols):
                    val = np.percentile(rets, p)
                    ax.axvline(val, color=col, linestyle='--', linewidth=2.5,
                               label=f'{p}th: {val:.1f}%')

                ax.set_title('Return Distribution', fontsize=15, color='#fff', pad=15)
                ax.set_xlabel('Return (%)', fontsize=12, color='#fff')
                ax.set_ylabel('Frequency', fontsize=12, color='#fff')
                ax.tick_params(colors='#fff')
                ax.legend(facecolor='#1a1a1a', labelcolor='#fff', fontsize=9)
                ax.grid(True, alpha=0.25, axis='y', color='#444')
                st.pyplot(fig)

    with tab4:
        st.subheader("Individual Assets")

        assets_df = pd.DataFrame({
            'Ticker': valid_tickers,
            'Return (%)': annual_return.values * 100,
            'Volatility (%)': annual_volatility.values * 100,
            'Sharpe': annual_return.values / annual_volatility.values
        }).sort_values('Sharpe', ascending=False)

        st.dataframe(assets_df.style.format({
            'Return (%)': '{:.2f}%',
            'Volatility (%)': '{:.2f}%',
            'Sharpe': '{:.3f}'
        }), hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 5), facecolor='#000')
        ax.set_facecolor('#000')

        x = np.arange(len(valid_tickers))
        ax.bar(x - 0.175, annual_return.values * 100, 0.35, label='Return', color='#0f0', alpha=0.8)
        ax.bar(x + 0.175, annual_volatility.values * 100, 0.35, label='Volatility', color='#F66', alpha=0.8)

        ax.set_xlabel('Asset', color='#fff')
        ax.set_ylabel('Percentage (%)', color='#fff')
        ax.set_title('Risk vs Return', fontsize=15, color='#fff', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_tickers, color='#fff')
        ax.tick_params(colors='#fff')
        ax.legend(facecolor='#1a1a1a', labelcolor='#fff')
        ax.grid(True, alpha=0.25, axis='y', color='#444')
        st.pyplot(fig)

    with tab5:
        st.subheader("About")
        st.write("Modern Portfolio Theory optimization for maximum Sharpe ratio.")
        st.error("**NOT financial advice.** Educational purposes only.")
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #999;'>Created by Jonathan Sanchez • 2026</p>",
                    unsafe_allow_html=True)

else:
    st.info("👈 Configure settings and click 'Optimize Portfolio'")