from .regime_strategies import RegimeStrategies
from .advanced_regime_strategies import AdvancedRegimeStrategies
from .auto_strategy_generator import AutoStrategyGenerator, AutoStrategyAdapter

class StrategyFactory:
    """
    Factory class for creating and managing trading strategies.

    Provides easy access to all available regime-based strategies
    with standardized configuration and validation.
    """

    STRATEGY_DESCRIPTIONS = {
        'regime_momentum': {
            'name': '📈 Regime Momentum',
            'description': 'Follows momentum in bullish/bearish regimes',
            'best_for': 'Markets with clear trends',
            'risk_level': 'Medium'
        },
        'regime_mean_reversion': {
            'name': '🔄 Mean Reversion',
            'description': 'Exploits mean reversion in range-bound phases',
            'best_for': 'Sideways and volatile markets',
            'risk_level': 'Low'
        },
        'regime_adaptive_volatility': {
            'name': '⚡ Adaptive Volatility',
            'description': 'Adapts positions according to regime volatility',
            'best_for': 'All market types',
            'risk_level': 'Medium'
        },
        
        'regime_contrarian': {
            'name': '↩️ Contrarian',
            'description': 'Fades extreme movements within each regime',
            'best_for': 'Markets with frequent reversals',
            'risk_level': 'High'
        },
        'regime_trend_following': {
            'name': '📊 Trend Following',
            'description': 'Enhanced trend detection with multiple timeframes',
            'best_for': 'Markets with persistent trends',
            'risk_level': 'Medium'
        },
        
        'tda_enhanced_momentum': {
            'name': '🌀 TDA Enhanced Momentum',
            'description': 'Momentum strategy enhanced with Topological Data Analysis',
            'best_for': 'Markets with complex, persistent trends',
            'risk_level': 'Medium'
        },
        'tda_enhanced_contrarian': {
            'name': '➿ TDA Enhanced Contrarian',
            'description': 'Contrarian strategy enhanced with Topological Data Analysis',
            'best_for': 'Markets with frequent reversals and complex dynamics',
            'risk_level': 'High'
        },
        
    }

    @classmethod
    def create_strategy(cls, strategy_type, **kwargs):
        """
        Create a strategy instance.

        Parameters:
        -----------
        strategy_type : str
            Type of strategy to create
        **kwargs : dict
            Strategy parameters

        Returns:
        --------
        BaseStrategy : Strategy instance
        """
        if strategy_type in ['regime_momentum', 'regime_mean_reversion', 'regime_adaptive_volatility']:
            return RegimeStrategies(strategy_type=strategy_type, **kwargs)
        elif strategy_type in ['regime_contrarian', 'regime_trend_following']:
            return AdvancedRegimeStrategies(strategy_type=strategy_type, **kwargs)
        elif strategy_type == 'auto_generated':
            # For auto-generated strategies, we need to provide the expression and feature generator
            # This should be called after evolution is complete with actual evolved strategies
            if 'strategy_expression' not in kwargs or 'feature_generator_func' not in kwargs:
                raise ValueError("Auto-generated strategy requires 'strategy_expression' and 'feature_generator_func' parameters")
            strategy_expression = kwargs.pop('strategy_expression')
            feature_generator_func = kwargs.pop('feature_generator_func')
            return cls.create_auto_strategy(strategy_expression, feature_generator_func, **kwargs)
        elif strategy_type == 'tda_enhanced_momentum':
            from .tda_enhanced_strategies import TopologicalMomentumStrategy
            return TopologicalMomentumStrategy(**kwargs)
        elif strategy_type == 'tda_enhanced_contrarian':
            from .tda_enhanced_strategies import TopologicalContrarianStrategy
            return TopologicalContrarianStrategy(**kwargs)
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    @staticmethod
    def get_strategy_info(strategy_type):
        """Get information about a strategy type."""
        return StrategyFactory.STRATEGY_DESCRIPTIONS.get(strategy_type, {})

    @staticmethod
    def create_auto_strategy(strategy_expression, feature_generator_func, **kwargs):
        """
        Create an auto-generated strategy from evolved expression.

        Parameters:
        -----------
        strategy_expression : StrategyExpression
            Evolved strategy expression
        feature_generator_func : callable
            Function to generate features
        **kwargs : dict
            Additional strategy parameters

        Returns:
        --------
        AutoStrategyAdapter : Adapted strategy instance
        """
        return AutoStrategyAdapter(
            strategy_expression=strategy_expression,
            feature_generator_func=feature_generator_func,
            **kwargs
        )

    @classmethod
    def get_all_strategies(cls):
        """Get list of all available strategies."""
        return list(cls.STRATEGY_DESCRIPTIONS.keys())

    @classmethod
    def get_available_strategies(cls):
        """Get list of available strategies (alias for get_all_strategies)."""
        return cls.get_all_strategies()

    @classmethod
    def get_strategies_by_risk_level(cls, risk_level):
        """Get strategies filtered by risk level."""
        return [
            strategy for strategy, info in cls.STRATEGY_DESCRIPTIONS.items()
            if info.get('risk_level') == risk_level
        ]

    @classmethod
    def recommend_strategy(cls, regime_distribution, volatility_level):
        """
        Recommend a strategy based on market characteristics.

        Parameters:
        -----------
        regime_distribution : dict
            Distribution of regimes {regime_id: frequency}
        volatility_level : str
            'low', 'medium', 'high'

        Returns:
        --------
        str : Recommended strategy type
        """
        # Calculate regime stability
        max_regime_freq = max(regime_distribution.values()) if regime_distribution else 0

        # Recommend based on characteristics
        if max_regime_freq > 0.6:  # Stable regime
            if volatility_level == 'low':
                return 'regime_trend_following'
            else:
                return 'regime_momentum'
        elif max_regime_freq < 0.4:  # Frequent regime changes
            return 'regime_adaptive_combo'
        else:  # Moderate stability
            if volatility_level == 'high':
                return 'regime_volatility_timing'
            else:
                return 'regime_mean_reversion'