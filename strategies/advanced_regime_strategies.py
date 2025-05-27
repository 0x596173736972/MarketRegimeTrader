import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class AdvancedRegimeStrategies(BaseStrategy):
    """
    Advanced trading strategies that exploit regime characteristics.

    Implements sophisticated strategies including pairs trading,
    momentum/mean-reversion combinations, and volatility clustering.
    """

    def __init__(self, strategy_type='regime_adaptive_combo', **kwargs):
        """
        Initialize advanced regime strategy.

        Parameters:
        -----------
        strategy_type : str
            Type of strategy ('regime_adaptive_combo', 'regime_breakout', 
                            'regime_contrarian', 'regime_trend_following')
        **kwargs : dict
            Additional parameters passed to base strategy
        """
        super().__init__(**kwargs)
        self.strategy_type = strategy_type

    def generate_signals(self, data, regimes, regime_probs=None):
        """Generate signals based on advanced regime strategies."""
        if self.strategy_type == 'regime_contrarian':
            return self._regime_contrarian_strategy(data, regimes, regime_probs)
        elif self.strategy_type == 'regime_trend_following':
            return self._regime_trend_following_strategy(data, regimes, regime_probs)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

    

    

    def _regime_contrarian_strategy(self, data, regimes, regime_probs=None):
        """
        Contrarian strategy that fades extreme regime moves.

        Strategy Logic:
        - Fade extreme moves within regimes
        - Look for exhaustion signals
        - Use regime probabilities for timing
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['position_size'] = 0.0

        aligned_regimes = regimes.reindex(data.index, method='ffill')

        # Calculate contrarian indicators
        rsi = self._calculate_rsi(data['Close'], window=14)
        stoch_k, stoch_d = self._calculate_stochastic(data)
        williams_r = self._calculate_williams_r(data)

        for i, idx in enumerate(signals.index):
            if idx not in aligned_regimes.index or pd.isna(aligned_regimes[idx]):
                continue

            current_regime = aligned_regimes[idx]
            signal_strength = 0.0

            # Contrarian signals based on oversold/overbought
            oversold_score = 0
            overbought_score = 0

            # RSI
            if not pd.isna(rsi[idx]):
                if rsi[idx] < 25:
                    oversold_score += 1
                elif rsi[idx] > 75:
                    overbought_score += 1

            # Stochastic
            if not pd.isna(stoch_k[idx]):
                if stoch_k[idx] < 20:
                    oversold_score += 1
                elif stoch_k[idx] > 80:
                    overbought_score += 1

            # Williams %R
            if not pd.isna(williams_r[idx]):
                if williams_r[idx] < -80:
                    oversold_score += 1
                elif williams_r[idx] > -20:
                    overbought_score += 1

            # Generate contrarian signals
            if oversold_score >= 2:  # Multiple oversold indicators
                signal_strength = 0.6
            elif overbought_score >= 2:  # Multiple overbought indicators
                signal_strength = -0.6

            # Regime-specific adjustments
            if current_regime == 2:  # Bullish regime
                # Be more selective with shorts
                if signal_strength < 0:
                    signal_strength *= 0.5
            elif current_regime == 0:  # Bearish regime
                # Be more selective with longs
                if signal_strength > 0:
                    signal_strength *= 0.5

            signals.loc[idx, 'signal'] = np.clip(signal_strength, -1.0, 1.0)

        signals = self.apply_money_management(signals, data)
        return signals

    def _regime_trend_following_strategy(self, data, regimes, regime_probs=None):
        """
        Pure trend following that adapts to regime characteristics.

        Strategy Logic:
        - Follow strong trends in trending regimes
        - Reduce exposure in range-bound regimes
        - Use multiple timeframe confirmation
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['position_size'] = 0.0

        aligned_regimes = regimes.reindex(data.index, method='ffill')

        # Calculate trend indicators
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        adx = self._calculate_adx(data)
        slope_20 = self._calculate_slope(data['Close'], window=20)

        for i, idx in enumerate(signals.index):
            if idx not in aligned_regimes.index or pd.isna(aligned_regimes[idx]):
                continue

            current_regime = aligned_regimes[idx]
            signal_strength = 0.0

            # EMA crossover
            if not pd.isna(ema_12[idx]) and not pd.isna(ema_26[idx]):
                if ema_12[idx] > ema_26[idx]:
                    signal_strength += 0.4
                else:
                    signal_strength -= 0.4

            # Trend strength (ADX)
            if not pd.isna(adx[idx]):
                if adx[idx] > 25:  # Strong trend
                    signal_strength *= 1.3
                elif adx[idx] < 15:  # Weak trend
                    signal_strength *= 0.5

            # Price slope
            if not pd.isna(slope_20[idx]):
                signal_strength += slope_20[idx] * 0.3

            # Regime-specific adjustments
            if current_regime in [0, 2]:  # Trending regimes
                signal_strength *= 1.2  # Amplify in trending markets
            else:  # Range-bound regime
                signal_strength *= 0.4  # Reduce in sideways markets

            signals.loc[idx, 'signal'] = np.clip(signal_strength, -1.0, 1.0)

        signals = self.apply_money_management(signals, data)
        return signals

    

    # Technical indicator helper methods
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band, rolling_mean

    def _calculate_stochastic(self, data, k_window=14, d_window=3):
        """Calculate Stochastic oscillator."""
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

    def _calculate_williams_r(self, data, window=14):
        """Calculate Williams %R."""
        high_max = data['High'].rolling(window=window).max()
        low_min = data['Low'].rolling(window=window).min()
        williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
        return williams_r

    def _calculate_adx(self, data, window=14):
        """Calculate Average Directional Index."""
        high = data['High']
        low = data['Low']
        close = data['Close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0

        # Smoothed values
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()

        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth

        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()

        return adx

    def _calculate_slope(self, prices, window=20):
        """Calculate price slope."""
        slopes = []
        for i in range(len(prices)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = prices.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0] / prices.iloc[i]
                slopes.append(slope)
        return pd.Series(slopes, index=prices.index)

    def _calculate_garch_volatility(self, returns, window=20):
        """Simple GARCH-like volatility estimate."""
        # Simplified GARCH(1,1) approximation
        vol = returns.rolling(window=window).std()
        ewm_vol = returns.ewm(span=window).std()
        garch_vol = 0.7 * vol + 0.3 * ewm_vol
        return garch_vol