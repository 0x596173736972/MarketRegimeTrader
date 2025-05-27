import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskEngine:
    """
    Comprehensive risk management engine for portfolio analysis.
    
    Calculates various risk metrics including VaR, CVaR, volatility measures,
    and provides risk-adjusted performance analysis.
    """
    
    def __init__(self, confidence_levels=[0.95, 0.99]):
        """
        Initialize risk engine.
        
        Parameters:
        -----------
        confidence_levels : list
            Confidence levels for VaR/CVaR calculations
        """
        self.confidence_levels = confidence_levels
        
    def calculate_risk_metrics(self, returns, benchmark_returns=None):
        """
        Calculate comprehensive risk metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series, optional
            Benchmark returns for relative risk metrics
            
        Returns:
        --------
        dict : Risk metrics
        """
        if returns.empty or len(returns) < 2:
            return {}
        
        metrics = {}
        
        # Basic volatility measures
        metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
        metrics['downside_volatility'] = self._calculate_downside_volatility(returns)
        
        # Value at Risk (VaR)
        for conf_level in self.confidence_levels:
            var_key = f'var_{int(conf_level*100)}'
            cvar_key = f'cvar_{int(conf_level*100)}'
            
            metrics[var_key] = self._calculate_var(returns, conf_level)
            metrics[cvar_key] = self._calculate_cvar(returns, conf_level)
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Tail Risk Measures
        metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
        metrics['up_capture'] = self._calculate_capture_ratio(returns, benchmark_returns, 'up')
        metrics['down_capture'] = self._calculate_capture_ratio(returns, benchmark_returns, 'down')
        
        # Risk-Adjusted Returns
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)
        
        # Stress Testing
        metrics['stress_test'] = self._perform_stress_test(returns)
        
        # Rolling Risk Metrics
        metrics['rolling_volatility'] = self._calculate_rolling_risk(returns)
        
        return metrics
    
    def _calculate_downside_volatility(self, returns, target_return=0):
        """Calculate downside volatility (volatility of negative returns)."""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_var(self, returns, confidence_level=0.95):
        """
        Calculate Value at Risk using historical simulation.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
        --------
        float : VaR value
        """
        if len(returns) == 0:
            return 0
        
        # Historical simulation VaR
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var
    
    def _calculate_cvar(self, returns, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        confidence_level : float
            Confidence level
            
        Returns:
        --------
        float : CVaR value
        """
        if len(returns) == 0:
            return 0
        
        var = self._calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar if not pd.isna(cvar) else var
    
    def _calculate_tail_ratio(self, returns, percentile=95):
        """
        Calculate tail ratio (ratio of average returns in top/bottom percentiles).
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        percentile : float
            Percentile for tail definition
            
        Returns:
        --------
        float : Tail ratio
        """
        if len(returns) == 0:
            return 0
        
        upper_tail = returns[returns >= np.percentile(returns, percentile)].mean()
        lower_tail = returns[returns <= np.percentile(returns, 100 - percentile)].mean()
        
        if lower_tail == 0:
            return 0
        
        return abs(upper_tail / lower_tail)
    
    def _calculate_capture_ratio(self, returns, benchmark_returns, direction='up'):
        """
        Calculate upside/downside capture ratio vs benchmark.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
        direction : str
            'up' for upside capture, 'down' for downside capture
            
        Returns:
        --------
        float : Capture ratio
        """
        if benchmark_returns is None or len(returns) == 0:
            return 1.0
        
        # Align returns
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        if aligned_returns.empty:
            return 1.0
        
        portfolio_rets = aligned_returns.iloc[:, 0]
        benchmark_rets = aligned_returns.iloc[:, 1]
        
        if direction == 'up':
            mask = benchmark_rets > 0
        else:  # down
            mask = benchmark_rets < 0
        
        if mask.sum() == 0:
            return 1.0
        
        portfolio_performance = portfolio_rets[mask].mean()
        benchmark_performance = benchmark_rets[mask].mean()
        
        if benchmark_performance == 0:
            return 1.0
        
        return portfolio_performance / benchmark_performance
    
    def _calculate_sortino_ratio(self, returns, target_return=0):
        """
        Calculate Sortino ratio (return vs downside volatility).
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        target_return : float
            Target return for downside calculation
            
        Returns:
        --------
        float : Sortino ratio
        """
        if len(returns) == 0:
            return 0
        
        excess_return = returns.mean() * 252 - target_return  # Annualized
        downside_vol = self._calculate_downside_volatility(returns, target_return)
        
        if downside_vol == 0:
            return 0
        
        return excess_return / downside_vol
    
    def _calculate_calmar_ratio(self, returns):
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        float : Calmar ratio
        """
        if len(returns) == 0:
            return 0
        
        annualized_return = returns.mean() * 252
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown == 0:
            return 0
        
        return annualized_return / max_drawdown
    
    def _perform_stress_test(self, returns):
        """
        Perform stress testing scenarios.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        dict : Stress test results
        """
        if len(returns) == 0:
            return {}
        
        stress_scenarios = {
            'market_crash_2008': -0.30,  # 30% market decline
            'flash_crash': -0.10,        # 10% single-day decline
            'high_volatility': returns.std() * 3,  # 3-sigma volatility shock
            'tail_risk': np.percentile(returns, 1)  # 1st percentile return
        }
        
        stress_results = {}
        current_value = 100  # Assume $100 portfolio value
        
        for scenario, shock in stress_scenarios.items():
            stressed_value = current_value * (1 + shock)
            stress_results[scenario] = {
                'shock': shock,
                'portfolio_value': stressed_value,
                'loss_amount': current_value - stressed_value,
                'loss_percentage': shock
            }
        
        return stress_results
    
    def _calculate_rolling_risk(self, returns, window=252):
        """
        Calculate rolling risk metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        window : int
            Rolling window size
            
        Returns:
        --------
        pd.DataFrame : Rolling risk metrics
        """
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling volatility
        rolling_metrics['volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Rolling VaR
        rolling_metrics['var_95'] = returns.rolling(window=window).quantile(0.05)
        rolling_metrics['var_99'] = returns.rolling(window=window).quantile(0.01)
        
        # Rolling Sharpe ratio
        rolling_return = returns.rolling(window=window).mean() * 252
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_metrics['sharpe_ratio'] = rolling_return / rolling_vol
        
        return rolling_metrics.dropna()
    
    def calculate_portfolio_var(self, positions, returns_matrix, confidence_level=0.95):
        """
        Calculate portfolio VaR using correlation matrix.
        
        Parameters:
        -----------
        positions : dict
            Asset positions {asset: weight}
        returns_matrix : pd.DataFrame
            Historical returns for all assets
        confidence_level : float
            Confidence level
            
        Returns:
        --------
        dict : Portfolio VaR analysis
        """
        if returns_matrix.empty:
            return {}
        
        # Calculate portfolio returns
        weights = pd.Series(positions)
        aligned_returns = returns_matrix[weights.index].dropna()
        
        if aligned_returns.empty:
            return {}
        
        portfolio_returns = (aligned_returns * weights).sum(axis=1)
        
        # Portfolio VaR
        portfolio_var = self._calculate_var(portfolio_returns, confidence_level)
        
        # Component VaR (marginal contribution to portfolio VaR)
        component_vars = {}
        portfolio_vol = portfolio_returns.std()
        
        for asset in weights.index:
            if asset in aligned_returns.columns:
                asset_returns = aligned_returns[asset]
                correlation = portfolio_returns.corr(asset_returns)
                asset_vol = asset_returns.std()
                
                # Marginal VaR
                marginal_var = correlation * asset_vol / portfolio_vol if portfolio_vol > 0 else 0
                component_vars[asset] = marginal_var * weights[asset]
        
        return {
            'portfolio_var': portfolio_var,
            'portfolio_volatility': portfolio_vol * np.sqrt(252),
            'component_vars': component_vars,
            'diversification_ratio': sum(component_vars.values()) / portfolio_var if portfolio_var != 0 else 0
        }
    
    def generate_risk_report(self, returns, benchmark_returns=None):
        """
        Generate comprehensive risk report.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series, optional
            Benchmark returns
            
        Returns:
        --------
        dict : Comprehensive risk report
        """
        risk_metrics = self.calculate_risk_metrics(returns, benchmark_returns)
        
        # Risk assessment
        risk_level = "Low"
        if risk_metrics.get('volatility', 0) > 0.20:
            risk_level = "High"
        elif risk_metrics.get('volatility', 0) > 0.15:
            risk_level = "Medium"
        
        # Risk-return efficiency
        sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
        efficiency = "Poor"
        if sharpe_ratio > 1.0:
            efficiency = "Excellent"
        elif sharpe_ratio > 0.5:
            efficiency = "Good"
        elif sharpe_ratio > 0.0:
            efficiency = "Fair"
        
        report = {
            'summary': {
                'risk_level': risk_level,
                'efficiency': efficiency,
                'total_observations': len(returns)
            },
            'metrics': risk_metrics,
            'recommendations': self._generate_risk_recommendations(risk_metrics)
        }
        
        return report
    
    def _generate_risk_recommendations(self, risk_metrics):
        """Generate risk management recommendations based on metrics."""
        recommendations = []
        
        # High volatility warning
        if risk_metrics.get('volatility', 0) > 0.25:
            recommendations.append("Consider reducing position sizes due to high volatility")
        
        # High drawdown warning
        if risk_metrics.get('max_drawdown', 0) < -0.20:
            recommendations.append("Implement stop-loss mechanisms to limit drawdowns")
        
        # Poor risk-adjusted returns
        if risk_metrics.get('sortino_ratio', 0) < 0.5:
            recommendations.append("Review strategy performance - poor risk-adjusted returns")
        
        # Tail risk concerns
        if risk_metrics.get('var_95', 0) < -0.05:
            recommendations.append("High tail risk detected - consider hedging strategies")
        
        # Positive skewness
        if risk_metrics.get('skewness', 0) < -1:
            recommendations.append("Negative skew detected - beware of large losses")
        
        if not recommendations:
            recommendations.append("Risk profile appears reasonable")
        
        return recommendations
