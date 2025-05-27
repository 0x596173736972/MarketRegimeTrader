
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from tda.topological_features import TopologicalFeatureExtractor, TopologicalAnomalyDetector

class TDAEnhancedRegimeStrategy(BaseStrategy):
    """
    Trading strategy enhanced with Topological Data Analysis features.
    
    Combines traditional regime-based signals with topological indicators
    to improve regime detection accuracy and signal generation.
    """
    
    def __init__(self, strategy_type='tda_enhanced_momentum', tda_weight=0.3, **kwargs):
        """
        Initialize TDA-enhanced strategy.
        
        Parameters:
        -----------
        strategy_type : str
            Base strategy type
        tda_weight : float
            Weight for TDA features in signal generation (0-1)
        **kwargs : dict
            Additional parameters passed to base strategy
        """
        super().__init__(**kwargs)
        self.strategy_type = strategy_type
        self.tda_weight = tda_weight
        self.tda_extractor = None
        self.anomaly_detector = None
        
    def generate_signals(self, data, regimes, regime_probs=None):
        """
        Generate trading signals enhanced with TDA features.
        
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
        pd.DataFrame : Enhanced trading signals
        """
        # Initialize TDA components
        self.tda_extractor = TopologicalFeatureExtractor(
            max_dimension=2,
            window_size=min(50, len(data) // 4),
            overlap=0.5
        )
        
        self.anomaly_detector = TopologicalAnomalyDetector(
            sensitivity=2.0,
            min_anomaly_duration=3
        )
        
        # Compute TDA features
        try:
            tda_features = self.tda_extractor.extract_tda_features(data, embedding_dim=3, delay=1)
        except Exception as e:
            print(f"Warning: Could not compute TDA features: {e}")
            # Fallback to standard regime strategy
            return self._generate_standard_signals(data, regimes, regime_probs)
        
        # Generate base strategy signals
        base_signals = self._generate_base_strategy_signals(data, regimes, regime_probs)
        
        # Generate TDA-enhanced signals
        tda_signals = self._generate_tda_enhanced_signals(data, regimes, tda_features, base_signals)
        
        # Combine signals
        enhanced_signals = self._combine_signals(base_signals, tda_signals)
        
        # Apply money management
        enhanced_signals = self.apply_money_management(enhanced_signals, data)
        
        return enhanced_signals
    
    def _generate_base_strategy_signals(self, data, regimes, regime_probs):
        """Generate base strategy signals without TDA enhancement."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['position_size'] = 0.0
        
        # Align regimes with data
        aligned_regimes = regimes.reindex(data.index, method='ffill')
        
        # Calculate technical indicators
        returns = data['Close'].pct_change()
        rsi = self._calculate_rsi(data['Close'])
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['Close'])
        
        for idx in signals.index:
            if idx not in aligned_regimes.index or pd.isna(aligned_regimes[idx]):
                continue
                
            current_regime = aligned_regimes[idx]
            signal_strength = 0.0
            
            # Basic regime-based signal
            if self.strategy_type == 'tda_regime_momentum':
                if current_regime == 2:  # Bullish regime
                    signal_strength = 0.6
                elif current_regime == 0:  # Bearish regime
                    signal_strength = -0.6
                else:  # Range-bound regime
                    signal_strength = 0.0
                    
            elif self.strategy_type == 'tda_regime_mean_reversion':
                if current_regime == 1:  # Range-bound regime
                    price = data.loc[idx, 'Close']
                    if not pd.isna(bb_upper[idx]) and not pd.isna(bb_lower[idx]):
                        if price <= bb_lower[idx]:
                            signal_strength = 0.5
                        elif price >= bb_upper[idx]:
                            signal_strength = -0.5
                        
            
            
            signals.loc[idx, 'signal'] = np.clip(signal_strength, -1.0, 1.0)
        
        return signals
    
    def _generate_tda_enhanced_signals(self, data, regimes, tda_features, base_signals):
        """Generate TDA-enhanced signal adjustments."""
        tda_adjustments = pd.DataFrame(index=data.index)
        tda_adjustments['tda_signal'] = 0.0
        
        # Align TDA features with data
        aligned_tda = tda_features.reindex(data.index, method='ffill')
        
        # Anomaly detection
        anomaly_scores = self.tda_extractor.compute_topological_anomaly_score(tda_features)
        bifurcations = self.anomaly_detector.detect_bifurcations(tda_features)
        early_warnings = self.anomaly_detector.generate_early_warning_signals(tda_features)
        
        # Align anomaly indicators
        aligned_anomalies = anomaly_scores.reindex(data.index, method='ffill').fillna(0)
        aligned_bifurcations = bifurcations.reindex(data.index, method='ffill').fillna(0)
        
        for idx in tda_adjustments.index:
            if idx not in aligned_tda.index:
                continue
                
            tda_signal = 0.0
            
            # Topological complexity signal
            if 'topological_complexity' in aligned_tda.columns:
                complexity = aligned_tda.loc[idx, 'topological_complexity']
                complexity_ma = aligned_tda['topological_complexity'].rolling(20).mean()[idx]
                
                if not pd.isna(complexity_ma) and complexity_ma > 0:
                    complexity_ratio = complexity / complexity_ma
                    if complexity_ratio > 1.2:  # High complexity
                        tda_signal += 0.2  # Slight bullish bias
                    elif complexity_ratio < 0.8:  # Low complexity
                        tda_signal -= 0.1  # Slight bearish bias
            
            # Persistence score signal
            if 'persistence_score' in aligned_tda.columns:
                persistence = aligned_tda.loc[idx, 'persistence_score']
                persistence_ma = aligned_tda['persistence_score'].rolling(20).mean()[idx]
                
                if not pd.isna(persistence_ma) and persistence_ma > 0:
                    persistence_ratio = persistence / persistence_ma
                    if persistence_ratio > 1.3:  # High persistence
                        tda_signal += 0.3
                    elif persistence_ratio < 0.7:  # Low persistence
                        tda_signal -= 0.2
            
            # Market structure signal
            if 'market_structure_index' in aligned_tda.columns:
                structure = aligned_tda.loc[idx, 'market_structure_index']
                if structure > 0.7:  # High structure
                    tda_signal += 0.1
                elif structure < 0.3:  # Low structure
                    tda_signal -= 0.1
            
            # Anomaly-based adjustments
            anomaly_score = aligned_anomalies[idx] if idx in aligned_anomalies.index else 0
            
            if anomaly_score > 2.5:  # High anomaly - reduce positions
                tda_signal *= 0.5
            elif anomaly_score > 1.5:  # Medium anomaly - cautious
                tda_signal *= 0.8
            
            # Bifurcation signal - anticipate regime change
            if idx in aligned_bifurcations.index and aligned_bifurcations[idx] == 1:
                # Reverse or reduce current position before regime change
                base_signal = base_signals.loc[idx, 'signal'] if idx in base_signals.index else 0
                tda_signal += -0.5 * np.sign(base_signal)
            
            # Early warning adjustments
            early_warning_adjustment = 0.0
            if early_warnings:
                for warning_name, warning_series in early_warnings.items():
                    if idx in warning_series.index and warning_series[idx] == 1:
                        if 'complexity_increasing' in warning_name:
                            early_warning_adjustment += 0.1
                        elif 'stability_breakdown' in warning_name:
                            early_warning_adjustment -= 0.2
                        elif 'entropy_spike' in warning_name:
                            early_warning_adjustment += 0.05
            
            tda_signal += early_warning_adjustment
            
            tda_adjustments.loc[idx, 'tda_signal'] = np.clip(tda_signal, -1.0, 1.0)
        
        return tda_adjustments
    
    def _combine_signals(self, base_signals, tda_signals):
        """Combine base strategy signals with TDA enhancements."""
        combined_signals = pd.DataFrame(index=base_signals.index)
        
        # Weighted combination
        base_weight = 1.0 - self.tda_weight
        
        combined_signals['signal'] = (
            base_weight * base_signals['signal'] + 
            self.tda_weight * tda_signals['tda_signal']
        )
        
        # Ensure signals are within bounds
        combined_signals['signal'] = np.clip(combined_signals['signal'], -1.0, 1.0)
        
        # Position sizing based on signal strength and TDA confidence
        combined_signals['position_size'] = combined_signals['signal'] * self.max_position_size
        
        return combined_signals
    
    def _generate_standard_signals(self, data, regimes, regime_probs):
        """Fallback to standard regime strategy if TDA fails."""
        from .regime_strategies import RegimeStrategies
        
        fallback_strategy = RegimeStrategies(
            strategy_type='regime_momentum',
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            max_position_size=self.max_position_size
        )
        
        return fallback_strategy.generate_signals(data, regimes, regime_probs)
    
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

class TopologicalMomentumStrategy(TDAEnhancedRegimeStrategy):
    """
    Momentum strategy that uses topological features to identify trend strength
    and anticipate regime changes.
    """
    
    def __init__(self, **kwargs):
        super().__init__(strategy_type='tda_regime_momentum', tda_weight=0.4, **kwargs)
    
    def generate_signals(self, data, regimes, regime_probs=None):
        """Generate momentum signals enhanced with topological trend analysis."""
        signals = super().generate_signals(data, regimes, regime_probs)
        
        # Additional topological momentum logic
        try:
            tda_features = self.tda_extractor.extract_tda_features(data)
            
            # Use persistence score changes to identify momentum shifts
            if 'persistence_score' in tda_features.columns:
                persistence_momentum = tda_features['persistence_score'].diff(5)
                persistence_momentum_norm = np.tanh(persistence_momentum / persistence_momentum.std())
                
                # Enhance signals with persistence momentum
                aligned_persistence = persistence_momentum_norm.reindex(data.index, method='ffill').fillna(0)
                signals['signal'] += 0.2 * aligned_persistence
                signals['signal'] = np.clip(signals['signal'], -1.0, 1.0)
        
        except Exception:
            pass  # Fall back to base signals
        
        return signals

class TopologicalContrarianStrategy(TDAEnhancedRegimeStrategy):
    """
    Contrarian strategy that uses topological anomaly detection to identify
    reversal points and market exhaustion.
    """
    
    def __init__(self, **kwargs):
        super().__init__(strategy_type='tda_regime_mean_reversion', tda_weight=0.5, **kwargs)
    
    def generate_signals(self, data, regimes, regime_probs=None):
        """Generate contrarian signals based on topological anomalies."""
        signals = super().generate_signals(data, regimes, regime_probs)
        
        # Additional contrarian logic using TDA
        try:
            tda_features = self.tda_extractor.extract_tda_features(data)
            anomaly_scores = self.tda_extractor.compute_topological_anomaly_score(tda_features)
            
            # High anomaly scores suggest market exhaustion - contrarian signal
            aligned_anomalies = anomaly_scores.reindex(data.index, method='ffill').fillna(0)
            
            # Generate contrarian signals when anomalies are high
            contrarian_signals = pd.Series(0.0, index=data.index)
            
            for idx in data.index:
                if idx in aligned_anomalies.index:
                    anomaly = aligned_anomalies[idx]
                    if anomaly > 2.0:  # High anomaly
                        # Contrarian to recent price movement
                        recent_return = data['Close'].pct_change(5)[idx] if idx in data.index else 0
                        contrarian_signals[idx] = -np.sign(recent_return) * min(anomaly / 5.0, 0.5)
            
            # Combine with base signals
            signals['signal'] += contrarian_signals
            signals['signal'] = np.clip(signals['signal'], -1.0, 1.0)
        
        except Exception:
            pass  # Fall back to base signals
        
        return signals
