import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class RegimeStrategies(BaseStrategy):
    """
    Collection of trading strategies that exploit detected market regimes.

    Implements multiple regime-based strategies with comprehensive
    money management and risk controls.
    """

    def __init__(self, strategy_type='regime_momentum', **kwargs):
        """
        Initialize regime-based strategy.

        Parameters:
        -----------
        strategy_type : str
            Type of strategy ('regime_momentum', 'regime_mean_reversion', 'regime_volatility',
                            'regime_adaptive_combo', 'regime_breakout', 'regime_contrarian',
                            'regime_trend_following', 'regime_volatility_timing')
        **kwargs : dict
            Additional parameters passed to base strategy
        """
        super().__init__(**kwargs)
        self.strategy_type = strategy_type

    def generate_signals(self, data, regimes, regime_probs=None):
        """
        Generate trading signals based on market regimes.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV market data
        regimes : pd.Series
            Detected market regimes
        regime_probs : pd.DataFrame, optional
            Regime probabilities

        Returns:
        --------
        pd.DataFrame : Trading signals
        """
        if self.strategy_type == 'regime_momentum':
            return self._regime_momentum_strategy(data, regimes, regime_probs)
        elif self.strategy_type == 'regime_mean_reversion':
            return self._regime_mean_reversion_strategy(data, regimes, regime_probs)
        elif self.strategy_type == 'regime_volatility':
            return self._regime_volatility_strategy(data, regimes, regime_probs)
        elif self.strategy_type in ['regime_adaptive_combo', 'regime_breakout', 'regime_contrarian', 
                                   'regime_trend_following', 'regime_volatility_timing']:
            # Import and use advanced strategies
            from .advanced_regime_strategies import AdvancedRegimeStrategies
            advanced_strategy = AdvancedRegimeStrategies(
                strategy_type=self.strategy_type,
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost,
                max_position_size=self.max_position_size
            )
            return advanced_strategy.generate_signals(data, regimes, regime_probs)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

    def _regime_momentum_strategy(self, data, regimes, regime_probs=None):
        """
        Momentum strategy based on market regimes.

        Strategy Logic:
        - Go long in bullish regimes
        - Go short in bearish regimes  
        - Stay neutral in range-bound regimes
        - Use regime probabilities for position sizing
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['position_size'] = 0.0

        # Align regimes with data
        aligned_regimes = regimes.reindex(data.index, method='ffill')

        # Calculate momentum indicators
        returns_5d = data['Close'].pct_change(5)
        returns_20d = data['Close'].pct_change(20)
        rsi = self._calculate_rsi(data['Close'], window=14)

        for i, idx in enumerate(signals.index):
            if idx not in aligned_regimes.index or pd.isna(aligned_regimes[idx]):
                continue

            current_regime = aligned_regimes[idx]
            signal_strength = 0.0

            # Base signal from regime
            if current_regime == 2:  # Bullish regime
                signal_strength = 0.8
            elif current_regime == 0:  # Bearish regime
                signal_strength = -0.8
            else:  # Range-bound regime
                signal_strength = 0.0

            # Adjust signal based on momentum
            if not pd.isna(returns_5d[idx]) and not pd.isna(returns_20d[idx]):
                momentum_factor = np.tanh(returns_5d[idx] * 10)  # Scale momentum
                signal_strength *= (1 + momentum_factor * 0.3)

            # RSI filter to avoid overbought/oversold conditions
            if not pd.isna(rsi[idx]):
                if signal_strength > 0 and rsi[idx] > 75:  # Overbought
                    signal_strength *= 0.5
                elif signal_strength < 0 and rsi[idx] < 25:  # Oversold
                    signal_strength *= 0.5

            # Use regime probabilities if available
            if regime_probs is not None and idx in regime_probs.index:
                regime_prob_cols = [col for col in regime_probs.columns if 'prob' in col]
                if len(regime_prob_cols) >= 3:
                    bullish_prob = regime_probs.loc[idx, regime_prob_cols[2]]
                    bearish_prob = regime_probs.loc[idx, regime_prob_cols[0]]

                    # Adjust signal based on regime confidence
                    prob_adjustment = bullish_prob - bearish_prob
                    signal_strength = signal_strength * 0.5 + prob_adjustment * 0.5

            signals.loc[idx, 'signal'] = np.clip(signal_strength, -1.0, 1.0)

        # Apply money management
        signals = self.apply_money_management(signals, data)

        return signals

    def _regime_mean_reversion_strategy(self, data, regimes, regime_probs=None):
        """
        Mean reversion strategy based on market regimes.

        Strategy Logic:
        - In range-bound regimes, trade mean reversion
        - In trending regimes, reduce position size or stay neutral
        - Use Bollinger Bands and RSI for entry/exit signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['position_size'] = 0.0

        # Align regimes with data
        aligned_regimes = regimes.reindex(data.index, method='ffill')

        # Calculate mean reversion indicators
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['Close'])
        rsi = self._calculate_rsi(data['Close'])

        for i, idx in enumerate(signals.index):
            if idx not in aligned_regimes.index or pd.isna(aligned_regimes[idx]):
                continue

            current_regime = aligned_regimes[idx]
            signal_strength = 0.0

            # Only trade in range-bound regimes
            if current_regime == 1:  # Range-bound regime
                price = data.loc[idx, 'Close']

                # Bollinger Bands signals
                if not pd.isna(bb_upper[idx]) and not pd.isna(bb_lower[idx]):
                    if price <= bb_lower[idx]:  # Price at lower band - buy signal
                        signal_strength = 0.6
                    elif price >= bb_upper[idx]:  # Price at upper band - sell signal
                        signal_strength = -0.6
                    else:
                        # Mean reversion towards middle band
                        distance_from_middle = (price - bb_middle[idx]) / bb_middle[idx]
                        signal_strength = -distance_from_middle * 2  # Revert to mean

                # RSI confirmation
                if not pd.isna(rsi[idx]):
                    if signal_strength > 0 and rsi[idx] < 35:  # Oversold confirmation
                        signal_strength *= 1.2
                    elif signal_strength < 0 and rsi[idx] > 65:  # Overbought confirmation
                        signal_strength *= 1.2
                    else:
                        signal_strength *= 0.8  # Reduce signal without confirmation

            # Reduce positions in trending regimes
            elif current_regime in [0, 2]:  # Trending regimes
                signal_strength *= 0.3  # Reduced exposure in trending markets

            signals.loc[idx, 'signal'] = np.clip(signal_strength, -1.0, 1.0)

        # Apply money management
        signals = self.apply_money_management(signals, data)

        return signals

    def _regime_volatility_strategy(self, data, regimes, regime_probs=None):
        """
        Volatility-based strategy that adapts to market regimes.

        Strategy Logic:
        - In high volatility regimes, reduce position sizes
        - In low volatility regimes, increase position sizes
        - Use volatility breakouts for signal generation
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['position_size'] = 0.0

        # Align regimes with data
        aligned_regimes = regimes.reindex(data.index, method='ffill')

        # Calculate volatility indicators
        returns = data['Close'].pct_change()
        vol_20d = returns.rolling(window=20).std()
        vol_5d = returns.rolling(window=5).std()
        atr = self.calculate_atr(data)

        for i, idx in enumerate(signals.index):
            if idx not in aligned_regimes.index or pd.isna(aligned_regimes[idx]):
                continue

            current_regime = aligned_regimes[idx]
            signal_strength = 0.0

            # Volatility breakout signals
            if not pd.isna(vol_5d[idx]) and not pd.isna(vol_20d[idx]):
                vol_ratio = vol_5d[idx] / vol_20d[idx]

                # Volatility expansion
                if vol_ratio > 1.5:
                    # Direction based on recent returns
                    recent_return = returns[idx-4:idx].sum() if idx >= 4 else 0
                    signal_strength = np.sign(recent_return) * 0.7

                # Volatility contraction
                elif vol_ratio < 0.7:
                    # Prepare for breakout
                    signal_strength = 0.3  # Small long bias during low volatility

            # Regime-based adjustments
            if current_regime == 0:  # Bearish regime
                signal_strength *= 1.2 if signal_strength < 0 else 0.5
            elif current_regime == 2:  # Bullish regime
                signal_strength *= 1.2 if signal_strength > 0 else 0.5
            else:  # Range-bound regime
                signal_strength *= 0.8  # Reduce signal in sideways markets

            # Volatility-based position sizing
            if not pd.isna(vol_20d[idx]) and vol_20d[idx] > 0:
                # Reduce position size for high volatility
                vol_adjustment = min(1.0, 0.02 / vol_20d[idx])
                signal_strength *= vol_adjustment

            signals.loc[idx, 'signal'] = np.clip(signal_strength, -1.0, 1.0)

        # Apply money management
        signals = self.apply_money_management(signals, data)

        return signals

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        return upper_band, lower_band, rolling_mean

    def generate_multi_timeframe_signals(self, data, regimes, regime_probs=None, 
                                       timeframes=['1D', '1W']):
        """
        Generate signals using multiple timeframes.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        regimes : pd.Series
            Detected regimes
        regime_probs : pd.DataFrame, optional
            Regime probabilities
        timeframes : list
            List of timeframes to consider

        Returns:
        --------
        pd.DataFrame : Multi-timeframe signals
        """
        signals_list = []

        for timeframe in timeframes:
            # Resample data to different timeframe
            resampled_data = data.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            # Resample regimes
            resampled_regimes = regimes.resample(timeframe).last()

            # Generate signals for this timeframe
            tf_signals = self.generate_signals(resampled_data, resampled_regimes, regime_probs)

            # Reindex to original frequency
            tf_signals_reindexed = tf_signals.reindex(data.index, method='ffill')
            tf_signals_reindexed.columns = [f'{col}_{timeframe}' for col in tf_signals.columns]

            signals_list.append(tf_signals_reindexed)

        # Combine signals from different timeframes
        combined_signals = pd.concat(signals_list, axis=1)

        # Create final signal as weighted average
        signal_cols = [col for col in combined_signals.columns if 'signal' in col]
        weights = [1.0, 0.7]  # Higher weight for shorter timeframe

        final_signals = pd.DataFrame(index=data.index)
        final_signals['signal'] = 0.0
        final_signals['position_size'] = 0.0

        for i, col in enumerate(signal_cols):
            weight = weights[i] if i < len(weights) else 0.5
            final_signals['signal'] += combined_signals[col] * weight

        # Normalize signals
        final_signals['signal'] = np.clip(final_signals['signal'], -1.0, 1.0)

        # Apply money management
        final_signals = self.apply_money_management(final_signals, data)

        return final_signals