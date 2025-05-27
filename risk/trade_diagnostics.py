
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TradeDiagnostics:
    """
    Advanced trade analysis and diagnostic system.
    
    Provides comprehensive analysis of trading performance with
    regime context, clustering, and explainability features.
    """
    
    def __init__(self):
        """Initialize trade diagnostics system."""
        self.trade_clusters = None
        self.cluster_model = None
        self.scaler = StandardScaler()
        
    def analyze_trade_performance(self, trades_df, regimes, returns, price_data):
        """
        Comprehensive analysis of trade performance.
        
        Parameters:
        -----------
        trades_df : pd.DataFrame
            Individual trades
        regimes : pd.Series
            Market regimes
        returns : pd.Series
            Portfolio returns
        price_data : pd.DataFrame
            OHLCV price data
            
        Returns:
        --------
        dict : Comprehensive trade analysis
        """
        if trades_df.empty:
            return {'error': 'No trades to analyze'}
        
        analysis = {
            'basic_stats': self._calculate_basic_trade_stats(trades_df),
            'regime_analysis': self._analyze_trades_by_regime(trades_df, regimes),
            'temporal_analysis': self._analyze_temporal_patterns(trades_df),
            'size_analysis': self._analyze_trade_sizes(trades_df),
            'clustering_analysis': self._cluster_trades(trades_df, price_data),
            'failure_analysis': self._analyze_failure_patterns(trades_df, regimes, price_data)
        }
        
        return analysis
    
    def _calculate_basic_trade_stats(self, trades_df):
        """Calculate basic trade statistics."""
        stats = {
            'total_trades': len(trades_df),
            'buy_trades': len(trades_df[trades_df['type'] == 'BUY']),
            'sell_trades': len(trades_df[trades_df['type'] == 'SELL']),
            'avg_trade_size': trades_df['shares'].mean(),
            'avg_trade_value': trades_df['value'].mean(),
            'total_commission': trades_df['commission'].sum(),
            'avg_commission_per_trade': trades_df['commission'].mean()
        }
        
        # Calculate trade frequency
        if len(trades_df) > 1:
            time_diff = trades_df.index.max() - trades_df.index.min()
            stats['avg_time_between_trades'] = time_diff.total_seconds() / (len(trades_df) - 1) / 86400  # days
        else:
            stats['avg_time_between_trades'] = 0
        
        return stats
    
    def _analyze_trades_by_regime(self, trades_df, regimes):
        """Analyze trade performance by market regime."""
        regime_analysis = {}
        
        # Align trades with regimes
        for regime_id in regimes.unique():
            if pd.isna(regime_id):
                continue
                
            regime_mask = regimes == regime_id
            regime_dates = regimes[regime_mask].index
            
            # Find trades in this regime
            regime_trades = []
            for trade_date in trades_df.index:
                # Find closest regime date
                closest_date = min(regime_dates, key=lambda x: abs((x - trade_date).total_seconds()), default=None)
                if closest_date and abs((closest_date - trade_date).total_seconds()) < 86400:  # Within 1 day
                    if regimes[closest_date] == regime_id:
                        regime_trades.append(trade_date)
            
            regime_trades_df = trades_df.loc[regime_trades] if regime_trades else pd.DataFrame()
            
            regime_analysis[f'regime_{regime_id}'] = {
                'trade_count': len(regime_trades_df),
                'avg_trade_size': regime_trades_df['shares'].mean() if not regime_trades_df.empty else 0,
                'total_value': regime_trades_df['value'].sum() if not regime_trades_df.empty else 0,
                'buy_ratio': len(regime_trades_df[regime_trades_df['type'] == 'BUY']) / len(regime_trades_df) if not regime_trades_df.empty else 0,
                'avg_price': regime_trades_df['price'].mean() if not regime_trades_df.empty else 0
            }
        
        return regime_analysis
    
    def _analyze_temporal_patterns(self, trades_df):
        """Analyze temporal patterns in trading."""
        temporal_analysis = {}
        
        # Add time features
        trades_with_time = trades_df.copy()
        trades_with_time['hour'] = trades_with_time.index.hour
        trades_with_time['day_of_week'] = trades_with_time.index.dayofweek
        trades_with_time['month'] = trades_with_time.index.month
        
        # Analyze by hour
        hourly_stats = trades_with_time.groupby('hour').agg({
            'shares': ['count', 'mean'],
            'value': 'sum',
            'price': 'mean'
        }).round(3)
        
        # Analyze by day of week
        daily_stats = trades_with_time.groupby('day_of_week').agg({
            'shares': ['count', 'mean'],
            'value': 'sum',
            'price': 'mean'
        }).round(3)
        
        # Analyze by month
        monthly_stats = trades_with_time.groupby('month').agg({
            'shares': ['count', 'mean'],
            'value': 'sum',
            'price': 'mean'
        }).round(3)
        
        temporal_analysis = {
            'hourly_patterns': hourly_stats.to_dict(),
            'daily_patterns': daily_stats.to_dict(),
            'monthly_patterns': monthly_stats.to_dict()
        }
        
        return temporal_analysis
    
    def _analyze_trade_sizes(self, trades_df):
        """Analyze trade size distribution and patterns."""
        trade_sizes = trades_df['shares']
        trade_values = trades_df['value']
        
        size_analysis = {
            'size_distribution': {
                'mean': float(trade_sizes.mean()),
                'median': float(trade_sizes.median()),
                'std': float(trade_sizes.std()),
                'min': float(trade_sizes.min()),
                'max': float(trade_sizes.max()),
                'q25': float(trade_sizes.quantile(0.25)),
                'q75': float(trade_sizes.quantile(0.75))
            },
            'value_distribution': {
                'mean': float(trade_values.mean()),
                'median': float(trade_values.median()),
                'std': float(trade_values.std()),
                'min': float(trade_values.min()),
                'max': float(trade_values.max()),
                'q25': float(trade_values.quantile(0.25)),
                'q75': float(trade_values.quantile(0.75))
            }
        }
        
        # Size categories
        size_percentiles = trade_sizes.quantile([0.33, 0.67])
        small_trades = trade_sizes <= size_percentiles.iloc[0]
        medium_trades = (trade_sizes > size_percentiles.iloc[0]) & (trade_sizes <= size_percentiles.iloc[1])
        large_trades = trade_sizes > size_percentiles.iloc[1]
        
        size_analysis['size_categories'] = {
            'small_trades': {
                'count': int(small_trades.sum()),
                'avg_size': float(trade_sizes[small_trades].mean()),
                'total_value': float(trade_values[small_trades].sum())
            },
            'medium_trades': {
                'count': int(medium_trades.sum()),
                'avg_size': float(trade_sizes[medium_trades].mean()),
                'total_value': float(trade_values[medium_trades].sum())
            },
            'large_trades': {
                'count': int(large_trades.sum()),
                'avg_size': float(trade_sizes[large_trades].mean()),
                'total_value': float(trade_values[large_trades].sum())
            }
        }
        
        return size_analysis
    
    def _cluster_trades(self, trades_df, price_data, n_clusters=3):
        """Cluster trades to identify behavioral patterns."""
        if len(trades_df) < n_clusters:
            return {'error': 'Not enough trades for clustering'}
        
        # Prepare features for clustering
        features = []
        feature_names = []
        
        # Basic trade features
        features.extend([
            trades_df['shares'].values,
            trades_df['value'].values,
            trades_df['price'].values
        ])
        feature_names.extend(['shares', 'value', 'price'])
        
        # Time-based features
        features.append(trades_df.index.hour.values)
        features.append(trades_df.index.dayofweek.values)
        feature_names.extend(['hour', 'day_of_week'])
        
        # Market context features
        volatility_at_trade = []
        price_change_before = []
        
        for trade_date in trades_df.index:
            # Calculate volatility in days before trade
            window_start = trade_date - pd.Timedelta(days=5)
            window_data = price_data[window_start:trade_date]
            
            if len(window_data) > 1:
                returns = window_data['Close'].pct_change().dropna()
                vol = returns.std() if len(returns) > 0 else 0
                price_chg = (window_data['Close'].iloc[-1] / window_data['Close'].iloc[0] - 1) if len(window_data) > 1 else 0
            else:
                vol = 0
                price_chg = 0
            
            volatility_at_trade.append(vol)
            price_change_before.append(price_chg)
        
        features.extend([volatility_at_trade, price_change_before])
        feature_names.extend(['volatility_before', 'price_change_before'])
        
        # Create feature matrix
        feature_matrix = np.column_stack(features)
        
        # Handle any NaN values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(feature_matrix_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_trades = trades_df[cluster_mask]
            cluster_features = feature_matrix[cluster_mask]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'trade_count': int(cluster_mask.sum()),
                'avg_features': {
                    feature_names[i]: float(cluster_features[:, i].mean()) 
                    for i in range(len(feature_names))
                },
                'trade_types': {
                    'buy_ratio': len(cluster_trades[cluster_trades['type'] == 'BUY']) / len(cluster_trades) if len(cluster_trades) > 0 else 0
                },
                'performance_proxy': {
                    'avg_trade_size': float(cluster_trades['shares'].mean()) if len(cluster_trades) > 0 else 0,
                    'avg_trade_value': float(cluster_trades['value'].mean()) if len(cluster_trades) > 0 else 0
                }
            }
        
        self.trade_clusters = cluster_labels
        
        return cluster_analysis
    
    def _analyze_failure_patterns(self, trades_df, regimes, price_data):
        """Analyze patterns in losing trades to identify failure points."""
        if len(trades_df) < 2:
            return {'error': 'Not enough trades for failure analysis'}
        
        # Calculate trade P&L (simplified)
        trade_pnl = []
        trade_info = []
        
        buy_trades = trades_df[trades_df['type'] == 'BUY'].copy()
        sell_trades = trades_df[trades_df['type'] == 'SELL'].copy()
        
        for _, sell_trade in sell_trades.iterrows():
            # Find the most recent buy trade before this sell
            prior_buys = buy_trades[buy_trades.index < sell_trade.name]
            
            if not prior_buys.empty:
                buy_trade = prior_buys.iloc[-1]
                pnl = (sell_trade['price'] - buy_trade['price']) * sell_trade['shares']
                
                trade_pnl.append(pnl)
                trade_info.append({
                    'buy_date': buy_trade.name,
                    'sell_date': sell_trade.name,
                    'buy_price': buy_trade['price'],
                    'sell_price': sell_trade['price'],
                    'shares': sell_trade['shares'],
                    'pnl': pnl,
                    'is_winner': pnl > 0
                })
        
        if not trade_info:
            return {'error': 'No complete buy-sell pairs found'}
        
        # Analyze winning vs losing trades
        winners = [t for t in trade_info if t['is_winner']]
        losers = [t for t in trade_info if not t['is_winner']]
        
        failure_analysis = {
            'trade_summary': {
                'total_complete_trades': len(trade_info),
                'winning_trades': len(winners),
                'losing_trades': len(losers),
                'win_rate': len(winners) / len(trade_info) if trade_info else 0,
                'avg_winner_pnl': np.mean([t['pnl'] for t in winners]) if winners else 0,
                'avg_loser_pnl': np.mean([t['pnl'] for t in losers]) if losers else 0
            },
            'failure_patterns': {}
        }
        
        # Analyze failure patterns
        if losers:
            loser_hold_times = [(t['sell_date'] - t['buy_date']).days for t in losers]
            winner_hold_times = [(t['sell_date'] - t['buy_date']).days for t in winners] if winners else [0]
            
            failure_analysis['failure_patterns'] = {
                'avg_losing_hold_time': np.mean(loser_hold_times),
                'avg_winning_hold_time': np.mean(winner_hold_times),
                'losing_trade_size_avg': np.mean([t['shares'] for t in losers]),
                'winning_trade_size_avg': np.mean([t['shares'] for t in winners]) if winners else 0
            }
            
            # Regime analysis for failures
            regime_failures = {}
            for trade in losers:
                buy_date = trade['buy_date']
                # Find regime at buy time
                closest_regime_date = min(regimes.index, key=lambda x: abs((x - buy_date).total_seconds()), default=None)
                if closest_regime_date:
                    regime = regimes[closest_regime_date]
                    if regime not in regime_failures:
                        regime_failures[regime] = 0
                    regime_failures[regime] += 1
            
            failure_analysis['failure_patterns']['regime_failures'] = regime_failures
        
        return failure_analysis
    
    def generate_trade_report(self, analysis_results):
        """Generate a comprehensive trade diagnostic report."""
        if 'error' in analysis_results:
            return f"Trade Analysis Error: {analysis_results['error']}"
        
        report = []
        report.append("=== TRADE DIAGNOSTIC REPORT ===\n")
        
        # Basic statistics
        basic = analysis_results.get('basic_stats', {})
        report.append(f"Total Trades: {basic.get('total_trades', 0)}")
        report.append(f"Buy Trades: {basic.get('buy_trades', 0)}")
        report.append(f"Sell Trades: {basic.get('sell_trades', 0)}")
        report.append(f"Average Trade Size: {basic.get('avg_trade_size', 0):.2f}")
        report.append(f"Total Commission: ${basic.get('total_commission', 0):.2f}")
        report.append("")
        
        # Regime analysis
        regime = analysis_results.get('regime_analysis', {})
        if regime:
            report.append("=== REGIME ANALYSIS ===")
            for regime_name, stats in regime.items():
                report.append(f"{regime_name}: {stats.get('trade_count', 0)} trades, "
                            f"Avg Size: {stats.get('avg_trade_size', 0):.2f}")
            report.append("")
        
        # Clustering results
        clustering = analysis_results.get('clustering_analysis', {})
        if clustering and 'error' not in clustering:
            report.append("=== TRADE CLUSTERING ===")
            for cluster_name, stats in clustering.items():
                report.append(f"{cluster_name}: {stats.get('trade_count', 0)} trades")
            report.append("")
        
        # Failure analysis
        failure = analysis_results.get('failure_analysis', {})
        if failure and 'error' not in failure:
            summary = failure.get('trade_summary', {})
            report.append("=== FAILURE ANALYSIS ===")
            report.append(f"Win Rate: {summary.get('win_rate', 0):.2%}")
            report.append(f"Average Winner P&L: ${summary.get('avg_winner_pnl', 0):.2f}")
            report.append(f"Average Loser P&L: ${summary.get('avg_loser_pnl', 0):.2f}")
            report.append("")
        
        return "\n".join(report)
