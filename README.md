
# HMM Regime Detection & Trading System

A sophisticated financial application that uses Hidden Markov Models (HMM) to detect market regimes and implement regime-based trading strategies with comprehensive backtesting, risk management, and automated strategy evolution.

## 🎯 Overview

This application combines advanced statistical modeling with automated strategy generation to:
- **Detect Market Regimes**: Identify different market states (bullish, bearish, range-bound) using HMM
- **Generate Trading Strategies**: Automatically evolve optimal trading strategies using genetic programming
- **Backtest Performance**: Simulate trading with realistic costs and slippage
- **Analyze Risk**: Comprehensive risk analysis and portfolio management
- **Walk-Forward Analysis**: Robust out-of-sample testing with rolling windows
- **Topological Data Analysis**: Advanced market structure analysis using TDA

## 🚀 Latest Updates & Fixes

### ✅ Recent Fixes
- **Dataset Reset**: Fixed dataset change not resetting all dependent state variables
- **Strategy Cleanup**: Removed problematic strategies that caused timestamp comparison errors
- **TDA Integration**: Fixed duplicate TDA components and analysis errors
- **State Management**: Complete application state reset when changing datasets
- **Connection Stability**: Improved Streamlit configuration for better connection handling

### 🔧 Current Working Features
- ✅ HMM Model Training with 2-4 regimes
- ✅ Multiple Technical Indicators (9 features available)
- ✅ Strategy Backtesting with realistic costs
- ✅ Risk Analysis and Performance Metrics
- ✅ Walk-Forward Analysis with parameter optimization
- ✅ TDA Market Structure Analysis
- ✅ Sample Data Generation for testing

## 🧮 Mathematical Foundations

### Hidden Markov Models (HMM)

Hidden Markov Models detect unobservable market regimes from observable price movements:

**Core Components:**
- **States (S)**: Hidden market regimes (Bearish, Range-bound, Bullish)
- **Observations (O)**: Observable features (returns, volatility, momentum, RSI, volume)
- **Transition Matrix (A)**: `A[i,j] = P(S_t+1 = j | S_t = i)`
- **Emission Matrix (B)**: `B[i,k] = P(O_t = k | S_t = i)`
- **Initial Distribution (π)**: `π[i] = P(S_1 = i)`

**Parameter Estimation** via Expectation-Maximization:
```
E-Step: γ_t(i) = P(S_t = i | O_1:T, λ)
M-Step: A[i,j] = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
```

### Topological Data Analysis (TDA)

Advanced market structure analysis using persistent homology:

**Key Metrics:**
- **Topological Complexity**: Captures market structure complexity
- **Persistence Score**: Measures pattern stability over time
- **Market Structure Index**: Overall structural health indicator
- **Betti Numbers**: Counts topological features (0-dimensional and 1-dimensional holes)

### Technical Indicators & Features

**Returns & Volatility**:
```
R_t = (P_t - P_t-1) / P_t-1
σ_t = √(1/n Σ_(i=t-n+1)^t (R_i - μ)²)
```

**RSI (Relative Strength Index)**:
```
RS = Average_Gain / Average_Loss
RSI = 100 - (100 / (1 + RS))
```

**Bollinger Bands**:
```
BB_upper = SMA + (2 × σ)
BB_position = (Price - BB_lower) / (BB_upper - BB_lower)
```

**MACD**:
```
MACD = EMA_12 - EMA_26
Signal = EMA_9(MACD)
```

### Risk Metrics

**Value at Risk (VaR)**:
```
VaR_α = -inf{x ∈ ℝ : P(X ≤ x) ≥ α}
```

**Conditional Value at Risk (CVaR)**:
```
CVaR_α = E[X | X ≤ VaR_α]
```

**Sharpe Ratio**:
```
SR = (E[R_p] - R_f) / σ_p
```

**Maximum Drawdown**:
```
MDD = max_t (max_s≤t X_s - X_t) / max_s≤t X_s
```

## 🏗️ Architecture

### Core Components

1. **`models/hmm_model.py`**: HMM implementation with regime detection
2. **`strategies/strategy_factory.py`**: Trading strategy factory and management
3. **`backtesting/backtest_engine.py`**: Realistic backtesting engine
4. **`backtesting/walk_forward_analyzer.py`**: Rolling window analysis
5. **`risk/risk_engine.py`**: Comprehensive risk analysis
6. **`risk/trade_diagnostics.py`**: Trade-level analytics
7. **`tda/tda_analysis.py`**: Topological data analysis implementation
8. **`tda/topological_features.py`**: TDA feature extraction
9. **`utils/`**: Data loading and visualization utilities

### Available Strategies

**Current Working Strategies**:
- **Regime Momentum**: Long bullish, short bearish regimes
- **Mean Reversion**: Contrarian positions within regimes
- **Adaptive Volatility**: Position sizing based on regime volatility
- **Contrarian**: Fade extreme movements within regimes
- **Trend Following**: Enhanced trend detection
- **TDA Enhanced Momentum**: Momentum with topological features
- **TDA Enhanced Contrarian**: Contrarian with topological features

## 🚀 Usage Guide

### 1. Load Data
```python
# Upload CSV/Parquet with OHLCV data
# Or use sample data for testing
```

### 2. Configure HMM Model
- **Number of Regimes**: 2-4 states (recommended: 3)
- **Features**: Select from 9 technical indicators + TDA features
- **Lookback Window**: Historical period for calculations (5-50 days)
- **Max Iterations**: Model convergence iterations (20-100)

### 3. Train Model & Detect Regimes
```python
hmm_detector = HMMRegimeDetector(n_regimes=3)
regimes, probs = hmm_detector.fit_predict(data, features)
```

### 4. Strategy Implementation

**Pre-built Strategies**:
```python
strategy = StrategyFactory.create_strategy(
    'regime_momentum',
    initial_capital=100000,
    transaction_cost=0.001
)
signals = strategy.generate_signals(data, regimes)
```

### 5. Comprehensive Analysis

**Backtesting**:
```python
backtest_engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost=0.001
)
results = backtest_engine.run_backtest(data, signals)
```

**Walk-Forward Analysis**:
```python
wf_analyzer = WalkForwardAnalyzer()
wf_results = wf_analyzer.analyze(
    data, strategy_name, 
    train_months=12, test_months=3
)
```

## 📊 Key Features

### 🔬 HMM Analysis
- **Automatic Regime Detection**: Statistical identification of market states
- **Transition Probabilities**: Regime change likelihood analysis
- **Duration Statistics**: Regime persistence metrics
- **Performance Attribution**: Returns analysis by regime
- **Comprehensive Statistics**: 30+ metrics per regime

### 🌀 Topological Data Analysis
- **Market Structure Analysis**: Persistent homology of price data
- **Anomaly Detection**: Topological outlier identification
- **Pattern Persistence**: Stability analysis of market patterns
- **Feature Integration**: TDA features combined with traditional indicators

### ⚠️ Risk Management
- **Portfolio VaR/CVaR**: Multiple confidence levels
- **Stress Testing**: Scenario analysis for extreme events
- **Rolling Risk Metrics**: Time-varying risk assessment
- **Drawdown Analysis**: Detailed loss period examination

### 📈 Performance Analytics
- **Multi-Strategy Comparison**: Side-by-side analysis
- **Trade-Level Diagnostics**: Individual trade analysis
- **Rolling Performance**: Time-varying metrics
- **Regime Attribution**: Performance by market state

### 🔄 Walk-Forward Analysis
- **Rolling Windows**: Out-of-sample validation
- **Parameter Optimization**: Hyperparameter tuning per window
- **Stability Analysis**: Performance consistency over time
- **Overfitting Detection**: Train vs test performance gaps

## 🛠️ Configuration Options

### HMM Parameters
- **Regimes**: 2-4 market states
- **Features**: 9 technical indicators + TDA features
- **Lookback Window**: 5-50 periods
- **Convergence**: EM algorithm settings (20-100 iterations)

### TDA Parameters
- **Window Size**: Analysis window for topological features
- **Dimension**: Maximum homology dimension to compute
- **Metric**: Distance metric for point cloud analysis

### Backtesting
- **Transaction Costs**: 0.01-1% per trade
- **Initial Capital**: Starting portfolio value
- **Max Position Size**: Maximum position as % of capital
- **Rebalancing**: Daily or regime-triggered

### Walk-Forward
- **Training Window**: 6-24 months
- **Testing Window**: 1-6 months
- **Step Size**: 1-6 months
- **Optimization Trials**: Hyperparameter search

## 📋 Data Requirements

Your dataset should include:
- **Date**: Timestamp column (index)
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price
- **Volume**: Trading volume (optional)

**Supported formats**: CSV, Parquet  
**Minimum data points**: 500+ recommended for stable analysis

## 🔬 Technical Implementation

### Dependencies
- **Core**: pandas, numpy, scikit-learn
- **Visualization**: plotly, streamlit
- **HMM**: hmmlearn
- **TDA**: gudhi, ripser (for topological analysis)
- **Optimization**: optuna (for parameter optimization)

### Performance Considerations
- **Memory Usage**: Efficient data handling for large datasets
- **Computation**: Optimized algorithms for real-time analysis
- **Caching**: State management for responsive UI

## 🎯 Use Cases

- **Quantitative Trading**: Systematic strategy development
- **Risk Management**: Portfolio regime analysis
- **Academic Research**: Market dynamics studies
- **Strategy Optimization**: Automated parameter tuning
- **Educational**: Understanding HMM and TDA in finance

## 🚀 Getting Started

1. **Launch Application**: Click the Run button in Replit
2. **Load Data**: Use "🎲 Use Sample Data" or upload your CSV/Parquet file
3. **Configure HMM**: Select features and parameters in the sidebar
4. **Train Model**: Click "🚀 Train Model" to detect regimes
5. **Choose Strategy**: Select from available trading strategies
6. **Run Analysis**: Execute backtests and walk-forward analysis
7. **Explore Results**: Navigate between analysis tabs

## 📊 Performance Metrics (30+ calculated)

**Return Metrics**: Total, annualized, rolling returns  
**Risk Metrics**: Volatility, VaR, CVaR, maximum drawdown  
**Risk-Adjusted**: Sharpe, Sortino, Calmar ratios  
**Trade Statistics**: Win rate, profit factor, avg win/loss  
**Regime Analysis**: Performance by market state  
**TDA Metrics**: Topological complexity, persistence, structure indices

## 🐛 Troubleshooting

### Common Issues
- **Connection Drops**: The app auto-reconnects; refresh if needed
- **Data Upload Errors**: Ensure CSV has proper OHLCV columns
- **Training Failures**: Reduce lookback window or increase data size
- **Empty Results**: Check that data has sufficient history

### Performance Tips
- Use sample data first to test functionality
- Start with 3 regimes for most datasets
- Reduce max iterations for faster training
- Use smaller walk-forward windows for quick testing

## 🔄 Version History

**Latest Version Features:**
- ✅ Fixed dataset reset issues
- ✅ Improved TDA integration
- ✅ Enhanced state management
- ✅ Streamlined strategy factory
- ✅ Better error handling
- ✅ Connection stability improvements

---

**Built with Python, Streamlit, and advanced AI/ML for comprehensive quantitative finance analysis on Replit.**

**🚀 Ready to start?** Click "🎲 Use Sample Data" to explore with synthetic market data, or upload your own OHLCV dataset to begin your analysis!
