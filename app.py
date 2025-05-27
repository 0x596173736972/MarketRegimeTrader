import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from models.hmm_model import HMMRegimeDetector
from strategies.regime_strategies import RegimeStrategies
from strategies.strategy_factory import StrategyFactory
from backtesting.backtest_engine import BacktestEngine
from risk.risk_engine import RiskEngine
from risk.trade_diagnostics import TradeDiagnostics
from backtesting.walk_forward_analyzer import WalkForwardAnalyzer
from utils.data_loader import DataLoader
from utils.plotting import PlotlyVisualizer
from strategies.strategy_evolution_ui import display_strategy_evolution_interface

# Page configuration
st.set_page_config(
    page_title="HMM Regime Detection & Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-message {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-message {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample data for testing the application."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # Price simulation with different regimes
    price = 100
    prices = []
    volumes = []

    for i in range(len(dates)):
        # Regime change every 150-300 days
        if i % 200 < 50:  # Bearish regime
            drift = -0.0005
            volatility = 0.025
        elif i % 200 < 150:  # Bullish regime
            drift = 0.001
            volatility = 0.015
        else:  # Range regime
            drift = 0.0001
            volatility = 0.008

        # Price with geometric Brownian motion
        return_rate = drift + volatility * np.random.normal()
        price *= (1 + return_rate)
        prices.append(price)

        # Random volume
        base_volume = 1000000
        volume = base_volume * (1 + 0.5 * np.random.normal())
        volumes.append(max(volume, 100000))

    # Create OHLCV
    data = pd.DataFrame({
        'Close': prices
    }, index=dates)

    # Approximation for Open, High, Low
    data['Open'] = data['Close'].shift(1).fillna(data['Close'].iloc[0])
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
    data['Volume'] = volumes

    return data

def display_metrics_cards(metrics):
    """Display metrics in styled cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Total Return</h4>
            <h2>{metrics.get('total_return', 0):.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Sharpe Ratio</h4>
            <h2>{metrics.get('sharpe_ratio', 0):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 Max Drawdown</h4>
            <h2>{metrics.get('max_drawdown', 0):.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Win Rate</h4>
            <h2>{metrics.get('win_rate', 0):.1%}</h2>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Main title with styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 3rem; font-weight: bold;">
            🔬 HMM Market Regime Detection
        </h1>
        <p style="font-size: 1.2rem; color: #666;">
            Hidden Markov Models for financial regime analysis and trading strategies
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'hmm_model' not in st.session_state:
        st.session_state.hmm_model = None
    if 'regimes' not in st.session_state:
        st.session_state.regimes = None
    if 'regime_probs' not in st.session_state:
        st.session_state.regime_probs = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None

    # Enhanced sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Data section
        st.markdown("### 📁 Data")

        # Option for sample data
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Use Sample Data"):
                # Complete state reset
                st.session_state.data = create_sample_data()
                st.session_state.hmm_model = None
                st.session_state.regimes = None
                st.session_state.regime_probs = None
                st.session_state.backtest_results = None
                
                # Reset walk-forward analysis results
                if hasattr(st.session_state, 'wf_results'):
                    del st.session_state.wf_results
                
                # Reset TDA components
                if hasattr(st.session_state, 'tda_extractor'):
                    del st.session_state.tda_extractor
                if hasattr(st.session_state, 'tda_features'):
                    del st.session_state.tda_features
                if hasattr(st.session_state, 'tda_analysis_results'):
                    del st.session_state.tda_analysis_results
                
                st.markdown('<div class="success-message">✅ Sample data loaded!</div>', unsafe_allow_html=True)
                st.rerun()

        with col2:
            if st.button("🔄 Reset Application"):
                # Complete reset - clear all state variables
                keys_to_clear = [
                    'data', 'hmm_model', 'regimes', 'regime_probs', 'backtest_results',
                    'wf_results', 'tda_extractor', 'tda_features', 'tda_analysis_results'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Application reset!")
                st.rerun()

        st.markdown("**OR**")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload OHLCV Dataset",
            type=['csv', 'parquet'],
            help="Expected format: Date, Open, High, Low, Close, Volume"
        )

        if uploaded_file is not None:
            try:
                data_loader = DataLoader()
                new_data = data_loader.load_data(uploaded_file)

                # Reset state if new dataset
                if (st.session_state.data is None or 
                    len(new_data) != len(st.session_state.data) or 
                    not new_data.index.equals(st.session_state.data.index)):

                    # Complete state reset for new dataset
                    st.session_state.data = new_data
                    st.session_state.hmm_model = None
                    st.session_state.regimes = None
                    st.session_state.regime_probs = None
                    st.session_state.backtest_results = None
                    
                    # Reset walk-forward analysis results
                    if hasattr(st.session_state, 'wf_results'):
                        del st.session_state.wf_results
                    
                    # Reset TDA components
                    if hasattr(st.session_state, 'tda_extractor'):
                        del st.session_state.tda_extractor
                    if hasattr(st.session_state, 'tda_features'):
                        del st.session_state.tda_features
                    if hasattr(st.session_state, 'tda_analysis_results'):
                        del st.session_state.tda_analysis_results
                    
                    # Force rerun to clear UI
                    st.rerun()

                st.markdown(f'<div class="success-message">✅ Data loaded: {len(st.session_state.data)} rows</div>', unsafe_allow_html=True)

                # Data info
                st.write(f"**Period:** {st.session_state.data.index[0].strftime('%Y-%m-%d')} → {st.session_state.data.index[-1].strftime('%Y-%m-%d')}")
                st.write(f"**Columns:** {', '.join(st.session_state.data.columns)}")

            except Exception as e:
                st.markdown(f'<div class="warning-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

        # HMM configuration
        if st.session_state.data is not None:
            st.markdown("### 🔬 HMM Parameters")

            n_regimes = st.selectbox("Number of Regimes", [2, 3, 4], index=1)

            # Multiple feature selection
            st.write("**Features to Use:**")
            available_features = {
                'returns': 'Returns',
                'log_returns': 'Log-Returns', 
                'volatility': 'Volatility',
                'momentum': 'Momentum',
                'rsi': 'RSI',
                'volume_ratio': 'Volume Ratio',
                'hl_spread': 'H-L Spread',
                'price_position': 'Price Position',
                'sma_ratio': 'SMA Ratio'
            }

            selected_features = []
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.checkbox('Returns', value=True, key='returns_cb'):
                    selected_features.append('returns')
                if st.checkbox('Volatility', value=True, key='volatility_cb'):
                    selected_features.append('volatility')
                if st.checkbox('RSI', value=True, key='rsi_cb'):
                    selected_features.append('rsi')

            with col2:
                if st.checkbox('Log-Returns', key='log_returns_cb'):
                    selected_features.append('log_returns')
                if st.checkbox('Momentum', value=True, key='momentum_cb'):
                    selected_features.append('momentum')
                if st.checkbox('Volume Ratio', key='volume_ratio_cb'):
                    selected_features.append('volume_ratio')

            with col3:
                if st.checkbox('H-L Spread', key='hl_spread_cb'):
                    selected_features.append('hl_spread')
                if st.checkbox('Price Position', key='price_position_cb'):
                    selected_features.append('price_position')
                if st.checkbox('SMA Ratio', key='sma_ratio_cb'):
                    selected_features.append('sma_ratio')
                if st.checkbox('TDA Features', value=True, key='tda_features_cb'):
                    selected_features.append('tda_features')

            if not selected_features:
                selected_features = ['returns']  # Default fallback

            lookback_window = st.slider("Calculation Window", 5, 50, 20)
            n_iterations = st.slider("Max Iterations (faster = less)", 20, 100, 50)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Train Model"):
                    with st.spinner("Training in progress..."):
                        try:
                            hmm_detector = HMMRegimeDetector(n_regimes=n_regimes, random_state=42)
                            regimes, regime_probs = hmm_detector.fit_predict(
                                st.session_state.data,
                                selected_features=selected_features,
                                lookback_window=lookback_window,
                                n_iter=n_iterations
                            )

                            st.session_state.hmm_model = hmm_detector
                            st.session_state.regimes = regimes
                            st.session_state.regime_probs = regime_probs

                            st.markdown('<div class="success-message">✅ HMM model trained!</div>', unsafe_allow_html=True)
                            st.rerun()

                        except Exception as e:
                            st.markdown(f'<div class="warning-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

            with col2:
                if st.session_state.regimes is not None:
                    st.markdown("✅ **Model ready**")
                else:
                    st.markdown("⏳ **Waiting**")

        # Strategy configuration
        if st.session_state.regimes is not None:
            st.markdown("### 💼 Trading Strategy")

            strategy_type = st.selectbox(
                "Strategy Type",
                StrategyFactory.get_all_strategies(),
                format_func=lambda x: StrategyFactory.get_strategy_info(x).get('name', x)
            )

            # Show strategy description
            strategy_info = StrategyFactory.get_strategy_info(strategy_type)
            if strategy_info:
                st.info(f"**Description:** {strategy_info.get('description', '')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Best for:** {strategy_info.get('best_for', 'N/A')}")
                with col2:
                    st.write(f"**Risk level:** {strategy_info.get('risk_level', 'N/A')}")

            # Strategy recommendation
            if hasattr(st.session_state, 'regimes') and st.session_state.regimes is not None:
                regime_dist = st.session_state.regimes.value_counts(normalize=True).to_dict()
                data_vol = st.session_state.data['Close'].pct_change().std()
                vol_level = 'high' if data_vol > 0.03 else 'low' if data_vol < 0.015 else 'medium'

                recommended = StrategyFactory.recommend_strategy(regime_dist, vol_level)
                if recommended != strategy_type:
                    rec_info = StrategyFactory.get_strategy_info(recommended)
                    st.warning(f"💡 **Recommendation:** {rec_info.get('name', recommended)} might be better suited for your data.")

            initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
            transaction_cost = st.slider("Transaction Costs (bps)", 0, 50, 10) / 10000
            max_position_size = st.slider("Max Position Size (%)", 10, 100, 50) / 100

            if st.button("📊 Run Backtest"):
                with st.spinner("Backtest in progress..."):
                    try:
                        # Initialize strategy
                        strategy = StrategyFactory.create_strategy(
                            strategy_type=strategy_type,
                            initial_capital=initial_capital,
                            transaction_cost=transaction_cost,
                            max_position_size=max_position_size
                        )

                        # Generate signals
                        signals = strategy.generate_signals(
                            st.session_state.data,
                            st.session_state.regimes,
                            st.session_state.regime_probs
                        )

                        # Run backtest
                        backtest_engine = BacktestEngine(
                            initial_capital=initial_capital,
                            transaction_cost=transaction_cost
                        )

                        st.session_state.backtest_results = backtest_engine.run_backtest(
                            st.session_state.data, signals
                        )

                        st.markdown('<div class="success-message">✅ Backtest completed!</div>', unsafe_allow_html=True)
                        st.rerun()

                    except Exception as e:
                        st.markdown(f'<div class="warning-message">❌ Backtest error: {str(e)}</div>', unsafe_allow_html=True)

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "🧭 Navigation",
        ["📊 Home", "🔬 HMM Analysis", "🔬 TDA Analysis"]
    )

    # Main content
    if page == "📊 Home":
        # Home page
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2>🚀 Start Your Analysis</h2>
            <p style="font-size: 1.1rem; color: #666; margin: 2rem 0;">
                Use sample data or upload your own OHLCV dataset
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 📋 Expected Data Format
            Your CSV or Parquet file should contain:
            - **Date/DateTime**: Timestamp
            - **Open**: Opening price
            - **High**: Highest price
            - **Low**: Lowest price  
            - **Close**: Closing price
            - **Volume**: Trading volume

            Data should be sorted chronologically for optimal results.
            """)

    elif page == "🔬 TDA Analysis":
        st.title("🔬 Topological Data Analysis")
        
        if st.session_state.data is not None:
            try:
                from tda.tda_analysis import TDAAnalysisUI
                
                tda_ui = TDAAnalysisUI()
                regimes = getattr(st.session_state, 'regimes', None)
                tda_ui.create_tda_analysis_section(st.session_state.data, regimes)
                
            except ImportError as e:
                st.error(f"❌ TDA modules not available: {e}")
                st.info("💡 Please ensure all TDA dependencies are installed")
            except Exception as e:
                st.error(f"❌ Error in TDA analysis: {e}")
        else:
            st.warning("Please load data first from the Data & Features page.")

    elif page == "🔬 HMM Analysis":
        st.title("🔬 HMM Regime Detection")

        if st.session_state.data is not None:
            data = st.session_state.data

            # Tabs to organize content
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Price Analysis", "🔬 Detected Regimes", "📊 Performance", "⚠️ Risk Management", "🎯 Walk-Forward Analysis", "🧬 Automated Strategy Evolution"])

            with tab1:
                st.subheader("📈 Price and Volume Analysis")

                # Price chart
                fig = go.Figure()

                # Closing price
                fig.add_trace(go.Scatter(
                    x=st.session_state.data.index,
                    y=st.session_state.data['Close'],
                    mode='lines',
                    name='Closing Price',
                    line=dict(color='#1f77b4', width=2)
                ))

                if st.session_state.regimes is not None:
                    # Create safer mask for alignment
                    try:
                        # Extend regimes to cover all data
                        regime_data_aligned = pd.DataFrame(index=st.session_state.data.index)
                        regime_data_aligned['regime'] = st.session_state.regimes.reindex(
                            st.session_state.data.index, method='ffill'
                        )

                        # Add regime coloring
                        for regime in regime_data_aligned['regime'].dropna().unique():
                            if not pd.isna(regime):
                                mask = regime_data_aligned['regime'] == regime
                                regime_prices = st.session_state.data.loc[mask, 'Close']

                                regime_name = "Bearish" if regime == 0 else ("Range" if regime == 1 else "Bullish")
                                color = '#ef5350' if regime == 0 else ('#ffa726' if regime == 1 else '#26a69a')

                                if len(regime_prices) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=regime_prices.index,
                                        y=regime_prices.values,
                                        mode='markers',
                                        name=f'{regime_name} Regime',
                                        marker=dict(color=color, size=4, opacity=0.7)
                                    ))
                    except Exception as e:
                        st.warning(f"Error displaying regimes: {str(e)}")

                fig.update_layout(
                    title="Price Evolution with Detected Regimes",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=600,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Data overview
                with st.expander("📋 Data Overview"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**First rows:**")
                        st.dataframe(st.session_state.data.head())
                    with col2:
                        st.write("**Statistics:**")
                        st.dataframe(st.session_state.data.describe())

            with tab2:
                if st.session_state.regimes is not None:
                    st.subheader("🔬 Detected Market Regimes")

                    # Calculate comprehensive statistics
                    comprehensive_stats = st.session_state.hmm_model.get_regime_statistics(
                        st.session_state.data, st.session_state.regimes
                    )

                    # Regime overview
                    st.subheader("📊 Regime Overview")
                    cols = st.columns(len(comprehensive_stats))

                    for i, (regime, stats) in enumerate(comprehensive_stats.items()):
                        with cols[i]:
                            regime_name = stats['name']
                            color = '#ef5350' if 'Bearish' in regime_name else ('#ffa726' if 'Range' in regime_name else '#26a69a')

                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                                        padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                                <h4 style="color: {color}; margin: 0;">{regime_name}</h4>
                                <p style="font-size: 0.9rem; margin: 5px 0;"><strong>{stats['total_days']}</strong> days ({stats['frequency']:.1%})</p>
                                <p style="font-size: 0.9rem; margin: 5px 0;">Return: <strong>{stats['mean_return']:.3%}</strong></p>
                                <p style="font-size: 0.9rem; margin: 5px 0;">Volatility: <strong>{stats['std_return']:.3%}</strong></p>
                                <p style="font-size: 0.9rem; margin: 5px 0;">Sharpe: <strong>{stats['sharpe_ratio']:.2f}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Detailed statistics by regime
                    st.subheader("📈 Detailed Analysis by Regime")

                    regime_tabs = st.tabs([stats['name'] for stats in comprehensive_stats.values()])

                    for i, (regime, stats) in enumerate(comprehensive_stats.items()):
                        with regime_tabs[i]:
                            col1, col2, col3 = st.columns(3)

                            # Return metrics
                            with col1:
                                st.markdown("#### 💰 Return Metrics")
                                st.metric("Mean Return", f"{stats['mean_return']:.3%}")
                                st.metric("Median Return", f"{stats['median_return']:.3%}")
                                st.metric("Annualized Return", f"{stats['annualized_return']:.2%}")
                                st.metric("Largest Gain", f"{stats['largest_gain']:.3%}")
                                st.metric("Largest Loss", f"{stats['largest_loss']:.3%}")

                            # Risk metrics
                            with col2:
                                st.markdown("#### ⚠️ Risk Metrics")
                                st.metric("Volatility", f"{stats['std_return']:.3%}")
                                st.metric("Annualized Volatility", f"{stats['annualized_volatility']:.2%}")
                                st.metric("VaR 95%", f"{stats['var_95']:.3%}")
                                st.metric("VaR 99%", f"{stats['var_99']:.3%}")
                                st.metric("Max Drawdown", f"{stats['max_drawdown']:.3%}")

                            # Regime characteristics
                            with col3:
                                st.markdown("#### 📊 Characteristics")
                                st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.3f}")
                                st.metric("Sortino Ratio", f"{stats['sortino_ratio']:.3f}")
                                st.metric("% Positive Returns", f"{stats['positive_return_ratio']:.1%}")
                                st.metric("Skewness", f"{stats['skewness']:.3f}")
                                st.metric("Kurtosis", f"{stats['kurtosis']:.3f}")

                            # Duration statistics
                            st.markdown("#### ⏱️ Period Duration Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Average Duration", f"{stats['avg_duration']:.1f} days")
                            with col2:
                                st.metric("Median Duration", f"{stats['median_duration']:.1f} days")
                            with col3:
                                st.metric("Max Duration", f"{stats['max_duration']:.0f} days")
                            with col4:
                                st.metric("Number of Periods", f"{stats['total_periods']:.0f}")

                            # Return distribution chart
                            st.markdown("#### 📈 Return Distribution")
                            aligned_regimes = st.session_state.regimes.reindex(st.session_state.data.index, method='ffill')
                            regime_returns = st.session_state.data['Close'].pct_change()[aligned_regimes == regime]

                            if len(regime_returns.dropna()) > 0:
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=regime_returns.dropna() * 100,
                                    nbinsx=30,
                                    name=f'{stats["name"]} Returns',
                                    opacity=0.7,
                                    marker_color='#ef5350' if 'Bearish' in stats['name'] else ('#ffa726' if 'Range' in stats['name'] else '#26a69a')
                                ))

                                # Add statistical lines
                                fig.add_vline(x=stats['mean_return']*100, line_dash="dash", line_color="red", 
                                            annotation_text="Mean")
                                fig.add_vline(x=stats['median_return']*100, line_dash="dash", line_color="blue", 
                                            annotation_text="Median")

                                fig.update_layout(
                                    title=f"Return Distribution - {stats['name']}",
                                    xaxis_title="Return (%)",
                                    yaxis_title="Frequency",
                                    template='plotly_white',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    # Regime comparison
                    st.subheader("🔄 Regime Comparison")

                    # Create comparison table
                    comparison_data = []
                    for regime, stats in comprehensive_stats.items():
                        comparison_data.append({
                            'Regime': stats['name'],
                            'Frequency': f"{stats['frequency']:.1%}",
                            'Mean Return': f"{stats['mean_return']:.3%}",
                            'Volatility': f"{stats['std_return']:.3%}",
                            'Sharpe': f"{stats['sharpe_ratio']:.3f}",
                            'VaR 95%': f"{stats['var_95']:.3%}",
                            'Max Drawdown': f"{stats['max_drawdown']:.3%}",
                            'Average Duration': f"{stats['avg_duration']:.1f} days"
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                    # Radar chart of characteristics
                    st.subheader("🕸️ Regime Profile (Radar Chart)")

                    categories = ['Return', 'Volatility', 'Sharpe', 'Stability', 'Persistence']

                    fig = go.Figure()

                    for regime, stats in comprehensive_stats.items():
                        # Normalize metrics for radar
                        normalized_values = [
                            (stats['mean_return'] + 0.01) * 50,  # Return
                            stats['std_return'] * 100,  # Volatility
                            max(0, stats['sharpe_ratio'] * 10),  # Sharpe
                            stats['avg_duration'] / 10,  # Stability (duration)
                            stats['volatility_persistence'] * 100  # Persistence
                        ]

                        color = '#ef5350' if 'Bearish' in stats['name'] else ('#ffa726' if 'Range' in stats['name'] else '#26a69a')

                        fig.add_trace(go.Scatterpolar(
                            r=normalized_values,
                            theta=categories,
                            fill='toself',
                            name=stats['name'],
                            line_color=color
                        ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 50]
                            )),
                        showlegend=True,
                        title="Comparative Regime Profile",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Transition matrix
                    if hasattr(st.session_state.hmm_model, 'model') and st.session_state.hmm_model.model is not None:
                        st.subheader("🔄 Regime Transition Matrix")

                        transition_matrix = st.session_state.hmm_model.model.transmat_
                        regime_names = [comprehensive_stats[i]['name'] for i in range(len(transition_matrix))]

                        # Create heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=transition_matrix,
                            x=regime_names,
                            y=regime_names,
                            colorscale='Blues',
                            text=np.round(transition_matrix, 3),
                            texttemplate="%{text}",
                            textfont={"size": 12},
                            hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.3f}<extra></extra>'
                        ))

                        fig.update_layout(
                            title="Transition Probabilities between Regimes",
                            xaxis_title="Next Regime",
                            yaxis_title="Current Regime",
                            height=500,
                            width=600
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Matrix interpretation
                        st.markdown("#### 💡 Transition Matrix Interpretation")
                        for i, from_regime in enumerate(regime_names):
                            most_likely_next = np.argmax(transition_matrix[i])
                            prob = transition_matrix[i][most_likely_next]
                            to_regime = regime_names[most_likely_next]

                            if i == most_likely_next:
                                st.info(f"The **{from_regime}** regime tends to persist ({prob:.1%} chance of continuing)")
                            else:
                                st.warning(f"The **{from_regime}** regime tends to evolve to **{to_regime}** ({prob:.1%} probability)")
                else:
                    st.info("👆 Train the HMM model first to see detected regimes")

            with tab3:
                if st.session_state.backtest_results is not None:
                    st.subheader("📊 Strategy Performance")

                    # Main metrics
                    metrics = st.session_state.backtest_results['metrics']
                    display_metrics_cards(metrics)

                    # Equity curve
                    st.subheader("💰 Portfolio Evolution")
                    equity_data = st.session_state.backtest_results['equity_curve']

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_data.index,
                        y=equity_data['portfolio_value'],
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='green', width=3)
                    ))

                    fig.update_layout(
                        title="Portfolio Performance",
                        xaxis_title="Date",
                        yaxis_title="Value ($)",
                        height=500,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Trade details
                    with st.expander("📋 Trade History"):
                        trades_df = st.session_state.backtest_results['trades']
                        if not trades_df.empty:
                            st.dataframe(trades_df)
                        else:
                            st.info("No trades executed in this backtest")
                else:
                    st.info("👆 Run a backtest first to see performance")

            with tab5:
                st.subheader("🎯 Walk-Forward Analysis")

                if st.session_state.data is not None:
                    st.markdown("""
                    **Walk-Forward Analysis** systematically tests strategy robustness by:
                    - Rolling training and testing windows
                    - Hyperparameter optimization on each window
                    - Comprehensive trade diagnostics and explainability
                    """)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 📋 Analysis Configuration")

                        wf_strategy = st.selectbox(
                            "Strategy for Walk-Forward",
                            options=StrategyFactory.get_available_strategies(),
                            index=0,
                            key="wf_strategy"
                        )

                        train_months = st.slider("Training Window (months)", 6, 24, 12)
                        test_months = st.slider("Testing Window (months)", 1, 6, 3)
                        step_months = st.slider("Step Size (months)", 1, 6, 1)

                        # Data length estimation
                        if st.session_state.data is not None:
                            total_days = len(st.session_state.data)
                            estimated_windows = max(0, (total_days - train_months * 30) // (step_months * 30))
                            st.info(f"💡 Estimated {estimated_windows} analysis windows with current settings")
                            if estimated_windows < 5:
                                st.warning("⚠️ Consider reducing window sizes for more analysis windows")

                        optimize_params = st.checkbox("Optimize Hyperparameters", value=True)
                        n_trials = st.slider("Optimization Trials", 10, 100, 30) if optimize_params else 10

                    with col2:
                        st.markdown("#### ⚙️ Advanced Options")

                        include_diagnostics = st.checkbox("Trade Diagnostics", value=True)
                        include_clustering = st.checkbox("Trade Clustering", value=True)
                        min_samples = st.number_input("Min Training Samples", 100, 1000, 252)

                    if st.button("🚀 Start Walk-Forward Analysis", key="run_wf"):
                        with st.spinner("Running walk-forward analysis... This may take several minutes."):
                            try:
                                # Initialize walk-forward analyzer
                                wf_analyzer = WalkForwardAnalyzer(
                                    training_window_months=train_months,
                                    testing_window_months=test_months,
                                    step_months=step_months,
                                    min_training_samples=min_samples
                                )

                                # Run analysis
                                wf_results = wf_analyzer.run_walk_forward_analysis(
                                    st.session_state.data,
                                    strategy_type=wf_strategy,
                                    optimize_params=optimize_params,
                                    n_trials=n_trials
                                )

                                st.session_state.wf_results = wf_results

                                st.success(f"✅ Walk-forward analysis completed! Analyzed {len(wf_results['window_results'])} windows.")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Walk-forward analysis error: {str(e)}")

                    # Display results if available
                    if hasattr(st.session_state, 'wf_results') and st.session_state.wf_results:
                        st.markdown("---")
                        st.markdown("#### 📊 Walk-Forward Results")

                        wf_results = st.session_state.wf_results
                        summary = wf_results['summary']

                        # Check if analysis failed
                        if 'error_message' in summary:
                            st.error(f"❌ Walk-forward analysis failed: {summary['error_message']}")
                            st.info("💡 Try reducing the lookback window or using a longer dataset with more data points.")
                            return

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Avg Sharpe Ratio",
                                f"{summary['avg_metrics']['sharpe_ratio']:.3f}",
                                f"±{summary['std_metrics']['sharpe_ratio']:.3f}"
                            )

                        with col2:
                            st.metric(
                                "Avg Annual Return",
                                f"{summary['avg_metrics']['total_return']:.2%}",
                                f"±{summary['std_metrics']['total_return']:.2%}"
                            )

                        with col3:
                            st.metric(
                                "Avg Max Drawdown",
                                f"{summary['avg_metrics']['max_drawdown']:.2%}",
                                f"±{summary['std_metrics']['max_drawdown']:.2%}"
                            )

                        with col4:
                            st.metric(
                                "Consistency",
                                f"{summary['consistency']['positive_returns']:.1%}",
                                "Positive Windows"
                            )

                        # Window-by-window results
                        st.markdown("#### 📈 Window-by-Window Performance")

                        window_df = pd.DataFrame([
                            {
                                'Window': i+1,
                                'Test Period': f"{result['test_start'].strftime('%Y-%m')} to {result['test_end'].strftime('%Y-%m')}",
                                'Sharpe Ratio': result['metrics']['sharpe_ratio'],
                                'Total Return': result['metrics']['total_return'],
                                'Max Drawdown': result['metrics']['max_drawdown'],
                                'Win Rate': result['metrics']['win_rate'],
                                'Total Trades': result['metrics']['total_trades']
                            }
                            for i, result in enumerate(wf_results['window_results'])
                        ])

                        st.dataframe(window_df, use_container_width=True)

                        # Performance chart
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(wf_results['window_results'])+1)),
                            y=[r['metrics']['sharpe_ratio'] for r in wf_results['window_results']],
                            mode='lines+markers',
                            name='Sharpe Ratio',
                            line=dict(color='blue', width=2)
                        ))

                        fig.update_layout(
                            title="Walk-Forward Sharpe Ratio by Window",
                            xaxis_title="Window Number",
                            yaxis_title="Sharpe Ratio",
                            height=400,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Trade diagnostics
                        if include_diagnostics and 'trade_summary' in summary:
                            st.markdown("#### 🔍 Trade Diagnostics Summary")

                            trade_summary = summary['trade_summary']

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Total Trades Analyzed", trade_summary.get('total_trades_analyzed', 0))

                            with col2:
                                st.metric("Overall Win Rate", f"{trade_summary.get('overall_win_rate', 0):.2%}")

                            with col3:
                                st.metric("Winning Trades", trade_summary.get('total_winning_trades', 0))

                            # Regime performance
                            if 'regime_performance' in trade_summary:
                                st.markdown("##### Performance by Regime")
                                regime_perf_df = pd.DataFrame([
                                    {
                                        'Regime': f"Regime {regime_id}",
                                        'Total Trades': perf['trades'],
                                        'Total Value': f"${perf['total_value']:,.2f}"
                                    }
                                    for regime_id, perf in trade_summary['regime_performance'].items()
                                    if regime_id != -1  # Exclude invalid regimes
                                ])

                                if not regime_perf_df.empty:
                                    st.dataframe(regime_perf_df, use_container_width=True)

                        # Parameter effectiveness
                        if 'parameter_analysis' in summary:
                            with st.expander("🎛️ Parameter Effectiveness Analysis"):
                                param_analysis = summary['parameter_analysis']

                                st.markdown("**Most Effective Parameters:**")

                                # N regimes effectiveness
                                if 'n_regimes' in param_analysis:
                                    best_n_regimes = max(param_analysis['n_regimes'].items(), 
                                                       key=lambda x: x[1]['avg_sharpe'])
                                    st.write(f"• **Number of Regimes**: {best_n_regimes[0]} "
                                            f"(Avg Sharpe: {best_n_regimes[1]['avg_sharpe']:.3f})")

                                # Lookback window effectiveness
                                if 'lookback_window' in param_analysis:
                                    best_lookback = max(param_analysis['lookback_window'].items(),
                                                      key=lambda x: x[1]['avg_sharpe'])
                                    st.write(f"• **Lookback Window**: {best_lookback[0]} days "
                                            f"(Avg Sharpe: {best_lookback[1]['avg_sharpe']:.3f})")

                        # Download results
                        if st.button("📥 Download Walk-Forward Results"):
                            # Create downloadable summary
                            results_summary = {
                                'summary_metrics': summary['avg_metrics'],
                                'window_results': [
                                    {
                                        'window': i+1,
                                        'test_start': str(r['test_start']),
                                        'test_end': str(r['test_end']),
                                        **r['metrics']
                                    }
                                    for i, r in enumerate(wf_results['window_results'])
                                ]
                            }

                            st.download_button(
                                "Download Results (JSON)",
                                data=pd.Series(results_summary).to_json(indent=2),
                                file_name=f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

                else:
                    st.info("👆 Upload data first to run walk-forward analysis")

            with tab6:
                display_strategy_evolution_interface()

            with tab4:
                if st.session_state.backtest_results is not None:
                    st.subheader("⚠️ Risk Analysis")

                    # Calculate risk metrics
                    risk_engine = RiskEngine()
                    portfolio_returns = st.session_state.backtest_results['returns']
                    risk_metrics = risk_engine.calculate_risk_metrics(portfolio_returns)

                    # Risk metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📉 VaR (95%)</h4>
                            <h2>{risk_metrics.get('var_95', 0):.2%}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>⚡ Volatility</h4>
                            <h2>{risk_metrics.get('volatility', 0):.2%}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📊 Sortino Ratio</h4>
                            <h2>{risk_metrics.get('sortino_ratio', 0):.3f}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    # Return distribution
                    st.subheader("📈 Return Distribution")
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=portfolio_returns * 100,
                        nbinsx=50,
                        name='Returns',
                        opacity=0.7
                    ))
                    fig.update_layout(
                        title="Daily Return Distribution",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("👆 Run a backtest first to see risk analysis")

    else:
        # Home page
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2>🚀 Advanced Trading System with HMM & TDA</h2>
            <p style="font-size: 1.1rem; color: #666; margin: 2rem 0;">
                Combine Hidden Markov Models with Topological Data Analysis for sophisticated market regime detection
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🎯 System Features
            - **🔬 HMM Analysis**: Regime detection with strategy testing and risk analysis
            - **🔬 TDA Analysis**: Topological market structure analysis with anomaly detection
            - **💼 Integrated Trading**: Strategy evolution and backtesting built-in
            
            ### 📋 Expected Data Format
            Your CSV or Parquet file should contain:
            - **Date/DateTime**: Timestamp
            - **Open**: Opening price
            - **High**: Highest price
            - **Low**: Lowest price  
            - **Close**: Closing price
            - **Volume**: Trading volume

            Data should be sorted chronologically for optimal results.
            """)

if __name__ == "__main__":
    main()
def display_strategy_evolution_interface():
    st.subheader("🧬 Strategy Evolution")
    st.write("Under construction: Automated strategy generation module using symbolic regression and genetic programming.")