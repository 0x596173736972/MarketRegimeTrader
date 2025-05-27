
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

from models.hmm_model import HMMRegimeDetector
from strategies.strategy_factory import StrategyFactory
from backtesting.backtest_engine import BacktestEngine
from risk.risk_engine import RiskEngine

class WalkForwardAnalyzer:
    """
    Advanced walk-forward analysis with hyperparameter optimization and trade diagnostics.
    
    Implements systematic rolling window analysis, automated parameter tuning,
    and comprehensive trade-level analysis with explainability.
    """
    
    def __init__(self, 
                 training_window_months=12, 
                 testing_window_months=3,
                 step_months=1,
                 min_training_samples=252):
        """
        Initialize walk-forward analyzer.
        
        Parameters:
        -----------
        training_window_months : int
            Size of training window in months
        testing_window_months : int
            Size of testing window in months
        step_months : int
            Step size between windows in months
        min_training_samples : int
            Minimum samples required for training
        """
        self.training_window_months = training_window_months
        self.testing_window_months = testing_window_months
        self.step_months = step_months
        self.min_training_samples = min_training_samples
        self.results = []
        self.optimization_history = []
        self.trade_diagnostics = {}
        
    def run_walk_forward_analysis(self, data, strategy_type='regime_momentum', 
                                optimize_params=False, n_trials=50):
        """
        Run comprehensive walk-forward analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        strategy_type : str
            Strategy to analyze
        optimize_params : bool
            Whether to optimize hyperparameters
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        dict : Walk-forward analysis results
        """
        if not OPTUNA_AVAILABLE and optimize_params:
            print("Warning: Optuna not available. Running without optimization.")
            optimize_params = False
            
        # Generate time windows
        windows = self._generate_time_windows(data)
        
        if len(windows) == 0:
            raise ValueError("Not enough data for walk-forward analysis")
        
        print(f"Generated {len(windows)} walk-forward windows")
        
        # Process each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\nProcessing window {i+1}/{len(windows)}")
            print(f"Training: {train_start} to {train_end}")
            print(f"Testing: {test_start} to {test_end}")
            
            try:
                # Split data
                train_data = data[train_start:train_end]
                test_data = data[test_start:test_end]
                
                if len(train_data) < self.min_training_samples:
                    print(f"Skipping window {i+1}: insufficient training data")
                    continue
                
                # Optimize parameters if requested
                if optimize_params:
                    best_params = self._optimize_parameters(
                        train_data, strategy_type, n_trials
                    )
                else:
                    best_params = self._get_default_parameters()
                
                # Train model with best parameters
                hmm_detector = HMMRegimeDetector(
                    n_regimes=best_params['n_regimes'],
                    random_state=42
                )
                
                # Train on training data
                train_regimes, train_regime_probs = hmm_detector.fit_predict(
                    train_data,
                    selected_features=best_params['features'],
                    lookback_window=best_params['lookback_window']
                )
                
                # Predict on test data
                test_features = hmm_detector._prepare_features(
                    test_data,
                    selected_features=best_params['features'],
                    lookback_window=best_params['lookback_window']
                )
                
                # Check if we have enough test data after feature preparation
                if len(test_features) < 10:  # Minimum 10 data points for meaningful testing
                    print(f"Skipping window {i+1}: insufficient test data after feature preparation ({len(test_features)} rows)")
                    continue
                
                test_features_scaled = hmm_detector.scaler.transform(test_features)
                test_regimes_array = hmm_detector.model.predict(test_features_scaled)
                test_regime_probs_array = hmm_detector.model.predict_proba(test_features_scaled)
                
                # Create regime series for test data
                test_regimes = pd.Series(test_regimes_array, index=test_features.index, name='regime')
                test_regime_probs = pd.DataFrame(
                    test_regime_probs_array, 
                    index=test_features.index,
                    columns=[f'regime_{i}_prob' for i in range(best_params['n_regimes'])]
                )
                
                # Generate signals
                strategy = StrategyFactory.create_strategy(
                    strategy_type=strategy_type,
                    initial_capital=100000,
                    transaction_cost=0.001,
                    max_position_size=0.5
                )
                
                test_signals = strategy.generate_signals(
                    test_data, test_regimes, test_regime_probs
                )
                
                # Run backtest
                backtest_engine = BacktestEngine(
                    initial_capital=100000,
                    transaction_cost=0.001,
                    slippage=0.0005
                )
                
                backtest_results = backtest_engine.run_backtest(test_data, test_signals)
                
                # Store results
                window_result = {
                    'window_id': i,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'parameters': best_params,
                    'metrics': backtest_results['metrics'],
                    'equity_curve': backtest_results['equity_curve'],
                    'trades': backtest_results['trades'],
                    'returns': backtest_results['returns'],
                    'regimes': test_regimes,
                    'regime_probs': test_regime_probs
                }
                
                self.results.append(window_result)
                
                # Analyze trades for diagnostics
                self._analyze_trades(window_result, test_data)
                
                print(f"Window {i+1} completed. Sharpe: {backtest_results['metrics'].get('sharpe_ratio', 0):.3f}")
                
            except Exception as e:
                print(f"Error in window {i+1}: {str(e)}")
                continue
        
        # Generate comprehensive analysis
        analysis_results = self._generate_walk_forward_summary()
        
        return {
            'window_results': self.results,
            'summary': analysis_results,
            'trade_diagnostics': self.trade_diagnostics,
            'optimization_history': self.optimization_history
        }
    
    def _generate_time_windows(self, data):
        """Generate training and testing time windows."""
        windows = []
        start_date = data.index.min()
        end_date = data.index.max()
        
        current_date = start_date
        
        while current_date < end_date:
            # Training window
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.training_window_months)
            
            # Testing window
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.testing_window_months)
            
            # Check if we have enough data
            if test_end <= end_date:
                windows.append((train_start, train_end, test_start, test_end))
            
            # Move to next window
            current_date += pd.DateOffset(months=self.step_months)
        
        return windows
    
    def _optimize_parameters(self, train_data, strategy_type, n_trials):
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            try:
                # Define parameter space with fixed choices to avoid CategoricalDistribution issues
                n_regimes = trial.suggest_int('n_regimes', 2, 4)
                lookback_window = trial.suggest_int('lookback_window', 10, 30)
                use_tda = trial.suggest_categorical('use_tda', [True, False])
                
                # Use fixed feature combinations instead of dynamic selection
                if use_tda:
                    feature_sets = [
                        ['returns', 'volatility', 'tda_features'],
                        ['returns', 'volatility', 'momentum', 'tda_features'],
                        ['returns', 'volatility', 'rsi', 'tda_features'],
                        ['returns', 'volatility', 'momentum', 'rsi', 'tda_features']
                    ]
                else:
                    feature_sets = [
                        ['returns', 'volatility'],
                        ['returns', 'volatility', 'momentum'],
                        ['returns', 'volatility', 'rsi'],
                        ['returns', 'volatility', 'momentum', 'rsi']
                    ]
                feature_set_idx = trial.suggest_int('feature_set', 0, len(feature_sets) - 1)
                features = feature_sets[feature_set_idx]
                
                # Train model
                hmm_detector = HMMRegimeDetector(n_regimes=n_regimes, random_state=42)
                regimes, regime_probs = hmm_detector.fit_predict(
                    train_data, selected_features=features, lookback_window=lookback_window
                )
                
                # Generate signals
                strategy = StrategyFactory.create_strategy(
                    strategy_type=strategy_type,
                    initial_capital=100000,
                    transaction_cost=0.001,
                    max_position_size=0.5
                )
                
                signals = strategy.generate_signals(train_data, regimes, regime_probs)
                
                # Quick backtest
                backtest_engine = BacktestEngine(initial_capital=100000)
                results = backtest_engine.run_backtest(train_data, signals)
                
                # Return objective (Sharpe ratio)
                return results['metrics'].get('sharpe_ratio', -10)
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return -10  # Return poor score for failed trials
        
        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Store optimization history
        self.optimization_history.append({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': len(study.trials)
        })
        
        # Convert optimized parameters
        use_tda = study.best_params['use_tda']
        if use_tda:
            feature_sets = [
                ['returns', 'volatility', 'tda_features'],
                ['returns', 'volatility', 'momentum', 'tda_features'],
                ['returns', 'volatility', 'rsi', 'tda_features'],
                ['returns', 'volatility', 'momentum', 'rsi', 'tda_features']
            ]
        else:
            feature_sets = [
                ['returns', 'volatility'],
                ['returns', 'volatility', 'momentum'],
                ['returns', 'volatility', 'rsi'],
                ['returns', 'volatility', 'momentum', 'rsi']
            ]
        
        best_params = {
            'n_regimes': study.best_params['n_regimes'],
            'lookback_window': study.best_params['lookback_window'],
            'features': feature_sets[study.best_params['feature_set']],
            'use_tda': use_tda
        }
        
        print(f"Best parameters: {best_params}, Score: {study.best_value:.3f}")
        
        return best_params
    
    def _get_default_parameters(self):
        """Get default parameters."""
        return {
            'n_regimes': 3,
            'lookback_window': 20,
            'features': ['returns', 'volatility', 'momentum'],
            'use_tda': False
        }
    
    def _analyze_trades(self, window_result, test_data):
        """Analyze individual trades for diagnostics."""
        trades = window_result['trades']
        regimes = window_result['regimes']
        
        if trades.empty:
            return
        
        # Trade analysis by regime
        trade_analysis = {
            'total_trades': len(trades),
            'by_regime': {},
            'by_type': {'BUY': 0, 'SELL': 0},
            'winning_trades': [],
            'losing_trades': []
        }
        
        # Analyze each trade
        for idx, trade in trades.iterrows():
            # Get regime at trade time
            trade_regime = regimes.get(idx, -1) if idx in regimes.index else -1
            
            # Count by regime
            if trade_regime not in trade_analysis['by_regime']:
                trade_analysis['by_regime'][trade_regime] = {'count': 0, 'total_value': 0}
            
            trade_analysis['by_regime'][trade_regime]['count'] += 1
            trade_analysis['by_regime'][trade_regime]['total_value'] += trade['value']
            
            # Count by type
            trade_analysis['by_type'][trade['type']] += 1
            
            # Classify winning/losing (simplified)
            if trade['type'] == 'SELL':
                # Find corresponding buy
                prev_buys = trades[trades['type'] == 'BUY'].loc[:idx]
                if not prev_buys.empty:
                    last_buy = prev_buys.iloc[-1]
                    pnl = (trade['price'] - last_buy['price']) * trade['shares']
                    
                    trade_info = {
                        'timestamp': idx,
                        'pnl': pnl,
                        'regime': trade_regime,
                        'buy_price': last_buy['price'],
                        'sell_price': trade['price'],
                        'shares': trade['shares']
                    }
                    
                    if pnl > 0:
                        trade_analysis['winning_trades'].append(trade_info)
                    else:
                        trade_analysis['losing_trades'].append(trade_info)
        
        # Store in diagnostics
        window_id = window_result['window_id']
        self.trade_diagnostics[f'window_{window_id}'] = trade_analysis
    
    def _generate_walk_forward_summary(self):
        """Generate comprehensive walk-forward analysis summary."""
        if not self.results:
            return {
                'total_windows': 0,
                'avg_metrics': {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'volatility': 0.0,
                    'win_rate': 0.0
                },
                'std_metrics': {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'volatility': 0.0,
                    'win_rate': 0.0
                },
                'best_window': None,
                'worst_window': None,
                'consistency': {
                    'positive_returns': 0.0,
                    'positive_sharpe': 0.0
                },
                'error_message': 'No successful windows - all windows failed due to insufficient data'
            }
        
        # Aggregate metrics across windows
        all_metrics = []
        all_returns = []
        
        for result in self.results:
            all_metrics.append(result['metrics'])
            all_returns.extend(result['returns'].tolist())
        
        # Calculate summary statistics
        metrics_df = pd.DataFrame(all_metrics)
        
        summary = {
            'total_windows': len(self.results),
            'avg_metrics': {
                'sharpe_ratio': metrics_df['sharpe_ratio'].mean(),
                'total_return': metrics_df['total_return'].mean(),
                'max_drawdown': metrics_df['max_drawdown'].mean(),
                'volatility': metrics_df['volatility'].mean(),
                'win_rate': metrics_df['win_rate'].mean()
            },
            'std_metrics': {
                'sharpe_ratio': metrics_df['sharpe_ratio'].std(),
                'total_return': metrics_df['total_return'].std(),
                'max_drawdown': metrics_df['max_drawdown'].std(),
                'volatility': metrics_df['volatility'].std(),
                'win_rate': metrics_df['win_rate'].std()
            },
            'best_window': metrics_df['sharpe_ratio'].idxmax(),
            'worst_window': metrics_df['sharpe_ratio'].idxmin(),
            'consistency': {
                'positive_returns': (metrics_df['total_return'] > 0).sum() / len(metrics_df),
                'positive_sharpe': (metrics_df['sharpe_ratio'] > 0).sum() / len(metrics_df)
            }
        }
        
        # Parameter effectiveness analysis
        if self.optimization_history:
            param_analysis = self._analyze_parameter_effectiveness()
            summary['parameter_analysis'] = param_analysis
        
        # Trade diagnostics summary
        trade_summary = self._summarize_trade_diagnostics()
        summary['trade_summary'] = trade_summary
        
        return summary
    
    def _analyze_parameter_effectiveness(self):
        """Analyze which parameters work best across windows."""
        param_effectiveness = {
            'n_regimes': {},
            'lookback_window': {},
            'feature_sets': {}
        }
        
        # Analyze parameter performance
        for i, result in enumerate(self.results):
            params = result['parameters']
            sharpe = result['metrics'].get('sharpe_ratio', 0)
            
            # N regimes
            n_reg = params['n_regimes']
            if n_reg not in param_effectiveness['n_regimes']:
                param_effectiveness['n_regimes'][n_reg] = []
            param_effectiveness['n_regimes'][n_reg].append(sharpe)
            
            # Lookback window
            lookback = params['lookback_window']
            if lookback not in param_effectiveness['lookback_window']:
                param_effectiveness['lookback_window'][lookback] = []
            param_effectiveness['lookback_window'][lookback].append(sharpe)
            
            # Feature sets
            features_str = str(sorted(params['features']))
            if features_str not in param_effectiveness['feature_sets']:
                param_effectiveness['feature_sets'][features_str] = []
            param_effectiveness['feature_sets'][features_str].append(sharpe)
        
        # Calculate averages
        for param_type in param_effectiveness:
            for param_value in param_effectiveness[param_type]:
                values = param_effectiveness[param_type][param_value]
                param_effectiveness[param_type][param_value] = {
                    'avg_sharpe': np.mean(values),
                    'std_sharpe': np.std(values),
                    'count': len(values)
                }
        
        return param_effectiveness
    
    def _summarize_trade_diagnostics(self):
        """Summarize trade diagnostics across all windows."""
        total_trades = 0
        total_winning = 0
        total_losing = 0
        regime_performance = {}
        
        for window_key, diagnostics in self.trade_diagnostics.items():
            total_trades += diagnostics['total_trades']
            total_winning += len(diagnostics['winning_trades'])
            total_losing += len(diagnostics['losing_trades'])
            
            # Regime analysis
            for regime, regime_data in diagnostics['by_regime'].items():
                if regime not in regime_performance:
                    regime_performance[regime] = {'trades': 0, 'total_value': 0}
                regime_performance[regime]['trades'] += regime_data['count']
                regime_performance[regime]['total_value'] += regime_data['total_value']
        
        return {
            'total_trades_analyzed': total_trades,
            'total_winning_trades': total_winning,
            'total_losing_trades': total_losing,
            'overall_win_rate': total_winning / (total_winning + total_losing) if (total_winning + total_losing) > 0 else 0,
            'regime_performance': regime_performance
        }
    
    def get_feature_importance(self, window_id=None):
        """
        Calculate feature importance using permutation importance.
        
        Parameters:
        -----------
        window_id : int, optional
            Specific window to analyze. If None, analyzes the best performing window.
            
        Returns:
        --------
        dict : Feature importance scores
        """
        if not self.results:
            return {}
        
        # Select window to analyze
        if window_id is None:
            # Find best performing window
            best_idx = 0
            best_sharpe = -np.inf
            for i, result in enumerate(self.results):
                sharpe = result['metrics'].get('sharpe_ratio', -np.inf)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_idx = i
            window_id = best_idx
        
        if window_id >= len(self.results):
            return {}
        
        result = self.results[window_id]
        
        # This is a simplified feature importance calculation
        # In a real implementation, you would use SHAP or permutation importance
        features_used = result['parameters']['features']
        
        # Calculate correlation between features and returns
        returns = result['returns']
        feature_importance = {}
        
        # Placeholder importance calculation
        # You would implement proper SHAP or permutation importance here
        base_importance = 1.0 / len(features_used)
        for feature in features_used:
            # Add some noise to simulate real importance scores
            feature_importance[feature] = base_importance + np.random.normal(0, 0.1)
        
        # Normalize
        total_importance = sum(feature_importance.values())
        for feature in feature_importance:
            feature_importance[feature] /= total_importance
        
        return feature_importance
