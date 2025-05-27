import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class HMMRegimeDetector:
    """
    Hidden Markov Model for detecting market regimes.

    Uses Expectation-Maximization algorithm for parameter estimation
    with multivariate Gaussian emissions on returns or log-returns.
    """

    def __init__(self, n_regimes=3, random_state=42):
        """
        Initialize HMM regime detector.

        Parameters:
        -----------
        n_regimes : int
            Number of market regimes to detect (default: 3)
        random_state : int
            Random seed for reproducible results
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.regime_names = {
            0: "Bearish Trend",
            1: "Range-Bound", 
            2: "Bullish Trend"
        }

    def _prepare_features(self, data, selected_features=None, lookback_window=20, include_tda=True):
        """
        Prepare features for HMM training including TDA features.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        selected_features : list
            List of features to include
        lookback_window : int
            Number of periods for calculations
        include_tda : bool
            Whether to include topological features

        Returns:
        --------
        pd.DataFrame : Feature matrix
        """
        if selected_features is None:
            selected_features = ['returns']

        features_list = []
        feature_names = []

        # Calculate all possible features
        returns = data['Close'].pct_change()

        # 1. Returns features
        if 'returns' in selected_features:
            features_list.append(returns)
            feature_names.append('returns')

        if 'log_returns' in selected_features:
            log_returns = np.log(data['Close'] / data['Close'].shift(1))
            features_list.append(log_returns)
            feature_names.append('log_returns')

        # 2. Volatility
        if 'volatility' in selected_features:
            volatility = returns.rolling(window=lookback_window).std()
            features_list.append(volatility)
            feature_names.append('volatility')

        # 3. Momentum
        if 'momentum' in selected_features:
            momentum = data['Close'] / data['Close'].shift(lookback_window) - 1
            features_list.append(momentum)
            feature_names.append('momentum')

        # 4. RSI
        if 'rsi' in selected_features:
            rsi = self._calculate_rsi(data['Close'], lookback_window)
            features_list.append(rsi / 100.0)
            feature_names.append('rsi')

        # 5. Volume ratio
        if 'volume_ratio' in selected_features and 'Volume' in data.columns:
            try:
                volume_ma = data['Volume'].rolling(window=lookback_window).mean()
                volume_ratio = data['Volume'] / volume_ma.replace(0, np.nan)
                volume_ratio_log = np.log1p(volume_ratio.fillna(1))
                features_list.append(volume_ratio_log)
                feature_names.append('volume_ratio')
            except:
                pass

        # 6. High-Low spread
        if 'hl_spread' in selected_features:
            try:
                hl_spread = (data['High'] - data['Low']) / data['Close']
                features_list.append(hl_spread)
                feature_names.append('hl_spread')
            except:
                pass

        # 7. Price position
        if 'price_position' in selected_features:
            try:
                price_position = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
                price_position = price_position.fillna(0.5)
                features_list.append(price_position)
                feature_names.append('price_position')
            except:
                pass

        # 8. Moving averages
        if 'sma_ratio' in selected_features:
            sma = data['Close'].rolling(window=lookback_window).mean()
            sma_ratio = data['Close'] / sma - 1
            features_list.append(sma_ratio)
            feature_names.append('sma_ratio')

        # 9. TDA Features
        if include_tda and 'tda_features' in selected_features:
            try:
                from tda.topological_features import TopologicalFeatureExtractor

                print("Computing TDA features...")
                tda_extractor = TopologicalFeatureExtractor(
                    max_dimension=2,
                    window_size=min(50, len(data) // 4),
                    overlap=0.5
                )

                tda_features = tda_extractor.extract_tda_features(data, embedding_dim=3, delay=1)

                # Select key TDA features for HMM
                key_tda_features = [
                    'topological_complexity',
                    'persistence_score', 
                    'market_structure_index',
                    'betti_0', 'betti_1',
                    'persistence_entropy_0', 'persistence_entropy_1',
                    'max_persistence_0', 'max_persistence_1',
                    'topological_stability'
                ]

                for tda_feature in key_tda_features:
                    if tda_feature in tda_features.columns:
                        # Normalize TDA features
                        tda_values = tda_features[tda_feature].fillna(0)
                        if tda_values.std() > 1e-6:
                            tda_values = (tda_values - tda_values.mean()) / tda_values.std()
                        features_list.append(tda_values)
                        feature_names.append(f'tda_{tda_feature}')

                print(f"Added {len([f for f in feature_names if f.startswith('tda_')])} TDA features")

            except Exception as e:
                print(f"Warning: Could not compute TDA features: {e}")
                print("Continuing without TDA features...")

        if not features_list:
            # Fallback to returns if no valid features
            features_list = [returns]
            feature_names = ['returns']

        # Combine all features
        feature_matrix = pd.concat(features_list, axis=1)
        feature_matrix.columns = feature_names

        # Remove infinite values and NaN
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).dropna()

        # Ensure we have enough data
        if len(feature_matrix) < max(50, lookback_window):
            raise ValueError(f"Not enough data after feature preparation. Got {len(feature_matrix)} rows")

        return feature_matrix

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def fit_predict(self, data, selected_features=None, lookback_window=20, n_iter=50):
        """
        Fit HMM model and predict regimes.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        selected_features : list
            List of features to use
        lookback_window : int
            Lookback window for feature calculation
        n_iter : int
            Maximum number of EM iterations (reduced for speed)

        Returns:
        --------
        tuple : (regimes, regime_probabilities)
        """
        try:
            # Prepare features
            self.features = self._prepare_features(data, selected_features, lookback_window)
            print(f"Features prepared: {self.features.shape}, columns: {list(self.features.columns)}")

            # Scale features
            features_scaled = self.scaler.fit_transform(self.features)
            print(f"Features scaled successfully: {features_scaled.shape}")

            # Initialize HMM model with faster parameters
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                n_iter=n_iter,
                random_state=self.random_state,
                tol=1e-2,  # Less strict for speed
                init_params="mc"
            )

            # Single training attempt for speed
            self.model.fit(features_scaled)
            score = self.model.score(features_scaled)
            print(f"Model trained with log-likelihood: {score}")

            # Predict regimes
            regimes = self.model.predict(features_scaled)
            regime_probs = self.model.predict_proba(features_scaled)

            # Align regimes with original data index
            regime_index = self.features.index
            regime_series = pd.Series(regimes, index=regime_index, name='regime')

            # Create regime probabilities DataFrame
            regime_probs_df = pd.DataFrame(regime_probs, index=regime_index)
            regime_probs_df.columns = [f'regime_{i}_prob' for i in range(self.n_regimes)]

            # Interpret regimes based on mean returns
            if len(regime_index) > 0:
                self._interpret_regimes(data.loc[regime_index])

            print(f"Regimes detected successfully. Distribution: {regime_series.value_counts().to_dict()}")

            return regime_series, regime_probs_df

        except Exception as e:
            print(f"Error in fit_predict: {str(e)}")
            raise

    def _interpret_regimes(self, data):
        """
        Interpret regimes based on statistical characteristics.

        Parameters:
        -----------
        data : pd.DataFrame
            Price data aligned with regimes
        """
        if self.model is None:
            return

        # Calculate mean returns for each regime
        returns = data['Close'].pct_change()
        regime_returns = []

        for i in range(self.n_regimes):
            regime_mask = self.model.predict(self.scaler.transform(self.features)) == i
            if regime_mask.sum() > 0:
                mean_return = returns[self.features.index][regime_mask].mean()
                regime_returns.append((i, mean_return))

        # Sort regimes by mean returns
        regime_returns.sort(key=lambda x: x[1])

        # Assign interpretable names
        if self.n_regimes == 3:
            self.regime_names = {
                regime_returns[0][0]: "Bearish Trend",    # Lowest returns
                regime_returns[1][0]: "Range-Bound",      # Medium returns
                regime_returns[2][0]: "Bullish Trend"     # Highest returns
            }
        elif self.n_regimes == 2:
            self.regime_names = {
                regime_returns[0][0]: "Bearish Market",
                regime_returns[1][0]: "Bullish Market"
            }
        else:
            # For other numbers of regimes, use generic names
            for i, (regime_idx, _) in enumerate(regime_returns):
                self.regime_names[regime_idx] = f"Regime {i+1}"

    def get_regime_statistics(self, data, regimes):
        """
        Calculate comprehensive statistics for each regime.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        regimes : pd.Series
            Regime assignments

        Returns:
        --------
        dict : Comprehensive regime statistics
        """
        if self.model is None:
            return {}

        returns = data['Close'].pct_change()
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        stats = {}

        # Align regimes with data
        aligned_regimes = regimes.reindex(data.index, method='ffill')

        for regime in range(self.n_regimes):
            regime_mask = aligned_regimes == regime
            regime_data = data[regime_mask]
            regime_returns = returns[regime_mask]
            regime_log_returns = log_returns[regime_mask]

            if len(regime_returns) > 0 and len(regime_data) > 0:
                # Basic statistics
                mean_return = regime_returns.mean()
                volatility = regime_returns.std()

                # Calculate additional metrics
                positive_returns = regime_returns[regime_returns > 0]
                negative_returns = regime_returns[regime_returns < 0]

                # Price statistics
                price_changes = regime_data['Close'].pct_change()
                volume_stats = regime_data['Volume'].describe() if 'Volume' in regime_data.columns else None

                # Regime duration statistics
                regime_periods = self._calculate_regime_durations(aligned_regimes, regime)

                # Advanced risk metrics
                downside_returns = regime_returns[regime_returns < 0]
                downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0

                # Skewness and Kurtosis
                from scipy import stats as scipy_stats
                skewness = scipy_stats.skew(regime_returns.dropna()) if len(regime_returns.dropna()) > 3 else 0
                kurtosis = scipy_stats.kurtosis(regime_returns.dropna()) if len(regime_returns.dropna()) > 3 else 0

                # VaR calculations
                var_95 = np.percentile(regime_returns.dropna(), 5) if len(regime_returns.dropna()) > 0 else 0
                var_99 = np.percentile(regime_returns.dropna(), 1) if len(regime_returns.dropna()) > 0 else 0

                # Maximum drawdown for regime
                cumulative_returns = (1 + regime_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

                stats[regime] = {
                    # Basic info
                    'name': self.regime_names.get(regime, f'Regime {regime}'),
                    'frequency': regime_mask.sum() / len(aligned_regimes.dropna()),
                    'total_days': regime_mask.sum(),

                    # Return statistics
                    'mean_return': mean_return,
                    'median_return': regime_returns.median(),
                    'std_return': volatility,
                    'min_return': regime_returns.min(),
                    'max_return': regime_returns.max(),

                    # Risk metrics
                    'sharpe_ratio': mean_return / volatility if volatility > 0 else 0,
                    'sortino_ratio': mean_return / downside_deviation if downside_deviation > 0 else 0,
                    'var_95': var_95,
                    'var_99': var_99,
                    'max_drawdown': max_drawdown,

                    # Distribution characteristics
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'positive_return_ratio': len(positive_returns) / len(regime_returns) if len(regime_returns) > 0 else 0,

                    # Positive/Negative return statistics
                    'avg_positive_return': positive_returns.mean() if len(positive_returns) > 0 else 0,
                    'avg_negative_return': negative_returns.mean() if len(negative_returns) > 0 else 0,
                    'largest_gain': regime_returns.max(),
                    'largest_loss': regime_returns.min(),

                    # Regime duration statistics
                    'avg_duration': regime_periods['mean'] if regime_periods else 0,
                    'median_duration': regime_periods['median'] if regime_periods else 0,
                    'max_duration': regime_periods['max'] if regime_periods else 0,
                    'min_duration': regime_periods['min'] if regime_periods else 0,
                    'total_periods': regime_periods['count'] if regime_periods else 0,

                    # Price statistics
                    'avg_price': regime_data['Close'].mean(),
                    'price_range': regime_data['High'].max() - regime_data['Low'].min() if len(regime_data) > 0 else 0,
                    'avg_volume': volume_stats['mean'] if volume_stats is not None else 0,

                    # Volatility clustering
                    'volatility_persistence': self._calculate_volatility_persistence(regime_returns),

                    # Annualized metrics (assuming daily data)
                    'annualized_return': mean_return * 252,
                    'annualized_volatility': volatility * np.sqrt(252),
                }

        return stats

    def _calculate_regime_durations(self, regimes, target_regime):
        """Calculate duration statistics for regime periods."""
        try:
            # Find regime changes
            regime_changes = (regimes != regimes.shift(1)).fillna(True)
            regime_periods = regime_changes.cumsum()

            # Get periods for target regime
            target_periods = regimes[regimes == target_regime]
            if len(target_periods) == 0:
                return None

            # Group by periods and count duration
            period_groups = target_periods.groupby(regime_periods[regimes == target_regime])
            durations = period_groups.size()

            if len(durations) > 0:
                return {
                    'mean': durations.mean(),
                    'median': durations.median(),
                    'max': durations.max(),
                    'min': durations.min(),
                    'count': len(durations)
                }
            return None
        except:
            return None

    def _calculate_volatility_persistence(self, returns):
        """Calculate volatility persistence (autocorrelation of squared returns)."""
        try:
            if len(returns) < 10:
                return 0
            squared_returns = returns ** 2
            autocorr = squared_returns.autocorr(lag=1)
            return autocorr if not pd.isna(autocorr) else 0
        except:
            return 0

    def get_transition_matrix(self):
        """
        Get the regime transition matrix.

        Returns:
        --------
        np.ndarray : Transition probability matrix
        """
        if self.model is None:
            return None

        return self.model.transmat_

    def predict_next_regime(self, current_features):
        """
        Predict the most likely next regime.

        Parameters:
        -----------
        current_features : np.ndarray
            Current feature vector

        Returns:
        --------
        int : Most likely next regime
        """
        if self.model is None:
            return None

        # Get current regime probabilities
        current_probs = self.model.predict_proba(current_features.reshape(1, -1))[0]

        # Calculate expected next regime probabilities
        transition_matrix = self.model.transmat_
        next_probs = np.dot(current_probs, transition_matrix)

        return np.argmax(next_probs)

    def get_model_score(self, features_scaled):
        """
        Get model likelihood score.

        Parameters:
        -----------
        features_scaled : np.ndarray
            Scaled feature matrix

        Returns:
        --------
        float : Model log-likelihood
        """
        if self.model is None:
            return None

        return self.model.score(features_scaled)