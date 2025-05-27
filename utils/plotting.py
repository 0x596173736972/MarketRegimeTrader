import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px

class PlotlyVisualizer:
    """
    Comprehensive plotting utility for financial data visualization.
    
    Provides interactive Plotly charts for price data, regime detection,
    performance analysis, and risk metrics visualization.
    """
    
    def __init__(self):
        """Initialize visualizer with default styling."""
        self.color_palette = {
            'bullish': '#26a69a',      # Teal green
            'bearish': '#ef5350',      # Red
            'range_bound': '#ffa726',  # Orange
            'neutral': '#78909c',      # Blue grey
            'background': '#fafafa',   # Light grey
            'grid': '#e0e0e0'         # Light grey for grid
        }
        
        # Regime color mapping
        self.regime_colors = {
            0: self.color_palette['bearish'],     # Bearish
            1: self.color_palette['range_bound'], # Range-bound
            2: self.color_palette['bullish']      # Bullish
        }
    
    def plot_price_with_regimes(self, data, regimes, regime_names=None, height=600):
        """
        Plot price chart with regime coloring.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        regimes : pd.Series
            Regime assignments
        regime_names : dict, optional
            Mapping of regime numbers to names
        height : int
            Chart height
            
        Returns:
        --------
        plotly.graph_objects.Figure : Interactive plot
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price Chart with Regimes', 'Volume', 'Regime Probabilities'),
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2],
            shared_xaxes=True
        )
        
        # Align regimes with data
        aligned_regimes = regimes.reindex(data.index, method='ffill')
        
        # Plot price line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add regime background coloring
        self._add_regime_backgrounds(fig, data.index, aligned_regimes, regime_names, row=1)
        
        # Add volume bars
        if 'Volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(128, 128, 128, 0.5)',
                    hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add regime indicator
        regime_values = aligned_regimes.fillna(-1)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=regime_values,
                mode='lines',
                name='Regime',
                line=dict(color='purple', width=2),
                hovertemplate='Date: %{x}<br>Regime: %{y}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Market Regime Detection Analysis',
            height=height,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Regime", row=3, col=1)
        
        return fig
    
    def _add_regime_backgrounds(self, fig, dates, regimes, regime_names=None, row=1):
        """Add regime background coloring to plot."""
        if regimes.empty:
            return
        
        # Group consecutive regimes
        regime_changes = regimes.diff().fillna(1) != 0
        regime_groups = regime_changes.cumsum()
        
        for group_id in regime_groups.unique():
            group_mask = regime_groups == group_id
            group_dates = dates[group_mask]
            
            if len(group_dates) == 0:
                continue
                
            regime_value = regimes[group_mask].iloc[0]
            
            if pd.isna(regime_value):
                continue
            
            # Get regime color
            color = self.regime_colors.get(regime_value, self.color_palette['neutral'])
            
            # Get regime name
            if regime_names and regime_value in regime_names:
                name = regime_names[regime_value]
            else:
                name = f'Regime {int(regime_value)}'
            
            # Add vertical rectangle
            fig.add_vrect(
                x0=group_dates.min(),
                x1=group_dates.max(),
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                row=row, col=1
            )
    
    def plot_regime_statistics(self, regime_stats):
        """
        Plot regime statistics as bar charts.
        
        Parameters:
        -----------
        regime_stats : dict
            Regime statistics from HMM model
            
        Returns:
        --------
        plotly.graph_objects.Figure : Bar chart of regime statistics
        """
        if not regime_stats:
            return go.Figure()
        
        # Prepare data
        regimes = list(regime_stats.keys())
        regime_names = [regime_stats[r]['name'] for r in regimes]
        frequencies = [regime_stats[r]['frequency'] * 100 for r in regimes]
        mean_returns = [regime_stats[r]['mean_return'] * 100 for r in regimes]
        volatilities = [regime_stats[r]['volatility'] * 100 for r in regimes]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Regime Frequency (%)', 'Mean Return (%)', 'Volatility (%)'),
            horizontal_spacing=0.1
        )
        
        # Frequency chart
        fig.add_trace(
            go.Bar(
                x=regime_names,
                y=frequencies,
                name='Frequency',
                marker_color=[self.regime_colors.get(r, '#666666') for r in regimes],
                text=[f'{f:.1f}%' for f in frequencies],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Mean returns chart
        fig.add_trace(
            go.Bar(
                x=regime_names,
                y=mean_returns,
                name='Mean Return',
                marker_color=[self.regime_colors.get(r, '#666666') for r in regimes],
                text=[f'{r:.2f}%' for r in mean_returns],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Volatility chart
        fig.add_trace(
            go.Bar(
                x=regime_names,
                y=volatilities,
                name='Volatility',
                marker_color=[self.regime_colors.get(r, '#666666') for r in regimes],
                text=[f'{v:.2f}%' for v in volatilities],
                textposition='auto'
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Regime Statistics Summary',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_equity_curve(self, equity_curve, height=500):
        """
        Plot portfolio equity curve.
        
        Parameters:
        -----------
        equity_curve : pd.DataFrame
            Equity curve data
        height : int
            Chart height
            
        Returns:
        --------
        plotly.graph_objects.Figure : Equity curve plot
        """
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['portfolio_value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            )
        )
        
        # Calculate and plot drawdown
        running_max = equity_curve['portfolio_value'].expanding().max()
        drawdown = (equity_curve['portfolio_value'] - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=drawdown,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                yaxis='y2',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='Portfolio Performance vs Benchmark',
            height=height,
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(
                title='Portfolio Value ($)',
                side='left'
            ),
            yaxis2=dict(
                title='Drawdown (%)',
                side='right',
                overlaying='y',
                range=[drawdown.min() * 1.1, 5]
            )
        )
        
        fig.update_xaxes(title='Date')
        
        return fig
    
    def plot_risk_metrics(self, risk_metrics, height=400):
        """
        Plot risk metrics visualization.
        
        Parameters:
        -----------
        risk_metrics : dict
            Risk metrics from risk engine
        height : int
            Chart height
            
        Returns:
        --------
        plotly.graph_objects.Figure : Risk metrics plot
        """
        # Filter numeric metrics
        numeric_metrics = {
            k: v for k, v in risk_metrics.items() 
            if isinstance(v, (int, float)) and not pd.isna(v)
        }
        
        if not numeric_metrics:
            return go.Figure()
        
        # Categorize metrics
        risk_categories = {
            'Volatility': ['volatility', 'downside_volatility'],
            'Value at Risk': ['var_95', 'var_99', 'cvar_95', 'cvar_99'],
            'Performance': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
            'Other': ['max_drawdown', 'skewness', 'kurtosis', 'tail_ratio']
        }
        
        # Create subplots
        num_categories = len(risk_categories)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(risk_categories.keys()),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (category, metrics) in enumerate(risk_categories.items()):
            if i >= len(positions):
                break
                
            row, col = positions[i]
            
            # Filter metrics for this category
            category_metrics = {k: v for k, v in numeric_metrics.items() if k in metrics}
            
            if not category_metrics:
                continue
            
            # Create bar chart for this category
            metric_names = list(category_metrics.keys())
            metric_values = list(category_metrics.values())
            
            # Color bars based on values (red for negative, green for positive)
            colors = ['red' if v < 0 else 'green' for v in metric_values]
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name=category,
                    marker_color=colors,
                    text=[f'{v:.3f}' for v in metric_values],
                    textposition='auto',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Risk Metrics Dashboard',
            height=height,
            template='plotly_white'
        )
        
        return fig
    
    def plot_returns_distribution(self, returns, height=400):
        """
        Plot returns distribution with normal overlay.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        height : int
            Chart height
            
        Returns:
        --------
        plotly.graph_objects.Figure : Returns distribution plot
        """
        if returns.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Distribution', 'Q-Q Plot vs Normal'),
            horizontal_spacing=0.1
        )
        
        # Histogram of returns
        fig.add_trace(
            go.Histogram(
                x=returns * 100,  # Convert to percentage
                nbinsx=50,
                name='Returns',
                marker_color='lightblue',
                opacity=0.7,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        # Overlay normal distribution
        x_range = np.linspace(returns.min(), returns.max(), 100) * 100
        normal_dist = stats.norm.pdf(x_range/100, returns.mean(), returns.std()) / 100
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Q-Q plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.sort(returns)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles * 100,
                y=sample_quantiles * 100,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='blue', size=4),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add reference line for Q-Q plot
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min()) * 100
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max()) * 100
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Normal',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Returns Distribution Analysis',
            height=height,
            template='plotly_white'
        )
        
        fig.update_xaxes(title='Returns (%)', row=1, col=1)
        fig.update_yaxes(title='Density', row=1, col=1)
        fig.update_xaxes(title='Theoretical Quantiles (%)', row=1, col=2)
        fig.update_yaxes(title='Sample Quantiles (%)', row=1, col=2)
        
        return fig
    
    def plot_rolling_metrics(self, rolling_metrics, height=500):
        """
        Plot rolling risk metrics over time.
        
        Parameters:
        -----------
        rolling_metrics : pd.DataFrame
            Rolling metrics from risk engine
        height : int
            Chart height
            
        Returns:
        --------
        plotly.graph_objects.Figure : Rolling metrics plot
        """
        if rolling_metrics.empty:
            return go.Figure()
        
        # Create subplots for different metrics
        available_metrics = rolling_metrics.columns.tolist()
        num_metrics = len(available_metrics)
        
        fig = make_subplots(
            rows=min(num_metrics, 3), cols=1,
            subplot_titles=available_metrics[:3],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, metric in enumerate(available_metrics[:3]):
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_metrics.index,
                    y=rolling_metrics[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    hovertemplate=f'Date: %{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>'
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title='Rolling Risk Metrics',
            height=height,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title='Date', row=min(num_metrics, 3), col=1)
        
        return fig
