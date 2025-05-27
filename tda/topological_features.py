import numpy as np
import pandas as pd
from ripser import ripser
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Try to import persim functions, with fallbacks
try:
    from persim import plot_diagrams, sliced_wasserstein
except ImportError:
    plot_diagrams = None
    sliced_wasserstein = None

try:
    from persim import persistence_entropy
except ImportError:
    # Fallback implementation of persistence entropy
    def persistence_entropy(diagram):
        """Fallback implementation of persistence entropy."""
        if len(diagram) == 0:
            return 0.0

        finite_diagram = diagram[diagram[:, 1] != np.inf]
        if len(finite_diagram) == 0:
            return 0.0

        persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
        if len(persistences) == 0 or np.sum(persistences) == 0:
            return 0.0

        # Normalize to get probabilities
        probs = persistences / np.sum(persistences)

        # Calculate entropy
        return -np.sum(probs * np.log2(probs + 1e-10))

class TopologicalFeatureExtractor:
    """
    Advanced Topological Data Analysis (TDA) feature extractor for financial time series.

    Computes topological invariants including persistence diagrams, Betti numbers,
    persistence entropy, and composite topological indicators for market regime detection.
    """

    def __init__(self, max_dimension=2, window_size=50, overlap=0.5):
        """
        Initialize TDA feature extractor.

        Parameters:
        -----------
        max_dimension : int
            Maximum homology dimension to compute (0=components, 1=loops, 2=voids)
        window_size : int
            Sliding window size for time series embedding
        overlap : float
            Overlap ratio between consecutive windows
        """
        self.max_dimension = max_dimension
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.scaler = StandardScaler()

    def extract_tda_features(self, data, embedding_dim=3, delay=1):
        """
        Extract comprehensive TDA features from financial time series.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV market data
        embedding_dim : int
            Dimension for time delay embedding
        delay : int
            Time delay for embedding

        Returns:
        --------
        pd.DataFrame : TDA features aligned with input data
        """
        # Prepare price series
        prices = data['Close'].values
        returns = np.diff(np.log(prices))

        # Initialize feature storage
        tda_features = []

        # Sliding window analysis
        for i in range(0, len(returns) - self.window_size + 1, self.step_size):
            window_returns = returns[i:i + self.window_size]
            window_end_idx = i + self.window_size - 1

            # Time delay embedding
            embedded_data = self._time_delay_embedding(window_returns, embedding_dim, delay)

            # Compute persistence diagrams
            persistence_diagrams = self._compute_persistence_diagrams(embedded_data)

            # Extract topological features
            features = self._extract_features_from_diagrams(persistence_diagrams)
            features['window_end_idx'] = window_end_idx

            tda_features.append(features)

        # Convert to DataFrame and align with original data
        tda_df = pd.DataFrame(tda_features)

        # Create full feature matrix aligned with input data
        full_features = pd.DataFrame(index=data.index)

        # Fill TDA features using window end indices
        for _, row in tda_df.iterrows():
            idx = int(row['window_end_idx'])
            if idx < len(data):
                timestamp = data.index[idx]
                for col in tda_df.columns:
                    if col != 'window_end_idx':
                        full_features.loc[timestamp, col] = row[col]

        # Forward fill missing values
        full_features = full_features.fillna(method='ffill').fillna(0)

        # Add composite indicators
        full_features = self._compute_composite_indicators(full_features, data)

        return full_features

    def _time_delay_embedding(self, time_series, embedding_dim, delay):
        """Create time delay embedding of the time series."""
        n_points = len(time_series) - (embedding_dim - 1) * delay
        embedded = np.zeros((n_points, embedding_dim))

        for i in range(n_points):
            for j in range(embedding_dim):
                embedded[i, j] = time_series[i + j * delay]

        return embedded

    def _compute_persistence_diagrams(self, point_cloud):
        """Compute persistence diagrams using Rips filtration."""
        try:
            # Normalize point cloud
            point_cloud_scaled = self.scaler.fit_transform(point_cloud)

            # Compute persistence diagrams
            diagrams = ripser(point_cloud_scaled, maxdim=self.max_dimension)['dgms']

            return diagrams
        except Exception as e:
            # Return empty diagrams if computation fails
            return [np.array([]).reshape(0, 2) for _ in range(self.max_dimension + 1)]

    def _extract_features_from_diagrams(self, diagrams):
        """Extract numerical features from persistence diagrams."""
        features = {}

        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                # No topological features in this dimension
                features[f'betti_{dim}'] = 0
                features[f'persistence_entropy_{dim}'] = 0
                features[f'max_persistence_{dim}'] = 0
                features[f'mean_persistence_{dim}'] = 0
                features[f'std_persistence_{dim}'] = 0
                features[f'total_persistence_{dim}'] = 0
                features[f'persistence_variance_{dim}'] = 0
                continue

            # Remove infinite persistence points for computation
            finite_diagram = diagram[diagram[:, 1] != np.inf]

            if len(finite_diagram) == 0:
                features[f'betti_{dim}'] = len(diagram)  # Count infinite points
                features[f'persistence_entropy_{dim}'] = 0
                features[f'max_persistence_{dim}'] = 0
                features[f'mean_persistence_{dim}'] = 0
                features[f'std_persistence_{dim}'] = 0
                features[f'total_persistence_{dim}'] = 0
                features[f'persistence_variance_{dim}'] = 0
                continue

            # Betti numbers (number of topological features)
            features[f'betti_{dim}'] = len(diagram)

            # Persistence values (death - birth)
            persistences = finite_diagram[:, 1] - finite_diagram[:, 0]

            # Persistence statistics
            features[f'max_persistence_{dim}'] = np.max(persistences) if len(persistences) > 0 else 0
            features[f'mean_persistence_{dim}'] = np.mean(persistences) if len(persistences) > 0 else 0
            features[f'std_persistence_{dim}'] = np.std(persistences) if len(persistences) > 0 else 0
            features[f'total_persistence_{dim}'] = np.sum(persistences) if len(persistences) > 0 else 0
            features[f'persistence_variance_{dim}'] = np.var(persistences) if len(persistences) > 0 else 0

            # Persistence entropy
            try:
                pers_entropy = persistence_entropy(finite_diagram)
                features[f'persistence_entropy_{dim}'] = pers_entropy if not np.isnan(pers_entropy) else 0
            except:
                features[f'persistence_entropy_{dim}'] = 0

            # Birth and death statistics
            births = finite_diagram[:, 0]
            deaths = finite_diagram[:, 1]

            features[f'mean_birth_{dim}'] = np.mean(births) if len(births) > 0 else 0
            features[f'mean_death_{dim}'] = np.mean(deaths) if len(deaths) > 0 else 0
            features[f'birth_death_ratio_{dim}'] = np.mean(deaths) / (np.mean(births) + 1e-10) if len(births) > 0 else 0

            # Persistence landscape features
            features.update(self._compute_persistence_landscape_features(finite_diagram, dim))

        return features

    def _compute_persistence_landscape_features(self, diagram, dim):
        """Compute persistence landscape summary statistics."""
        features = {}

        if len(diagram) == 0:
            features[f'landscape_norm_{dim}'] = 0
            features[f'landscape_max_{dim}'] = 0
            return features

        # Simple persistence landscape approximation
        persistences = diagram[:, 1] - diagram[:, 0]
        births = diagram[:, 0]

        # Landscape norm (sum of persistences)
        features[f'landscape_norm_{dim}'] = np.sum(persistences)

        # Landscape maximum
        features[f'landscape_max_{dim}'] = np.max(persistences) if len(persistences) > 0 else 0

        return features

    def _compute_composite_indicators(self, tda_features, data):
        """Compute composite topological indicators."""
        # Topological Complexity Score
        complexity_components = []
        for dim in range(self.max_dimension + 1):
            if f'persistence_entropy_{dim}' in tda_features.columns:
                complexity_components.append(tda_features[f'persistence_entropy_{dim}'])
            if f'betti_{dim}' in tda_features.columns:
                complexity_components.append(tda_features[f'betti_{dim}'] / 10.0)  # Normalize

        if complexity_components:
            tda_features['topological_complexity'] = np.mean(complexity_components, axis=0)
        else:
            tda_features['topological_complexity'] = 0

        # Topological Stability Score (inverse of variability)
        stability_components = []
        for dim in range(self.max_dimension + 1):
            if f'std_persistence_{dim}' in tda_features.columns:
                stability_components.append(1.0 / (tda_features[f'std_persistence_{dim}'] + 1e-6))

        if stability_components:
            tda_features['topological_stability'] = np.mean(stability_components, axis=0)
        else:
            tda_features['topological_stability'] = 1.0

        # Market Structure Index (combining H0 and H1)
        if 'betti_0' in tda_features.columns and 'betti_1' in tda_features.columns:
            tda_features['market_structure_index'] = (
                tda_features['betti_0'] + 2 * tda_features['betti_1']
            ) / (tda_features['betti_0'] + tda_features['betti_1'] + 1)
        else:
            tda_features['market_structure_index'] = 0

        # Persistence Score (weighted persistence across dimensions)
        persistence_score_components = []
        weights = [1.0, 2.0, 1.5]  # Different weights for H0, H1, H2

        for dim in range(min(self.max_dimension + 1, len(weights))):
            if f'total_persistence_{dim}' in tda_features.columns:
                weighted_persistence = weights[dim] * tda_features[f'total_persistence_{dim}']
                persistence_score_components.append(weighted_persistence)

        if persistence_score_components:
            tda_features['persistence_score'] = np.sum(persistence_score_components, axis=0)
        else:
            tda_features['persistence_score'] = 0

        # Topological Volatility (rate of change in topological features)
        for feature_col in ['topological_complexity', 'persistence_score']:
            if feature_col in tda_features.columns:
                tda_features[f'{feature_col}_volatility'] = (
                    tda_features[feature_col].rolling(window=10).std().fillna(0)
                )

        return tda_features

    def compute_topological_anomaly_score(self, tda_features, lookback_window=20):
        """
        Compute anomaly scores based on topological features (optimized).

        Parameters:
        -----------
        tda_features : pd.DataFrame
            TDA features
        lookback_window : int
            Window for computing historical statistics (reduced for speed)

        Returns:
        --------
        pd.Series : Anomaly scores
        """
        # Key features for anomaly detection (reduced set for speed)
        key_features = [
            'topological_complexity',
            'persistence_score', 
            'market_structure_index'
        ]

        # Available features
        available_features = [f for f in key_features if f in tda_features.columns]

        if not available_features:
            return pd.Series(0, index=tda_features.index)

        # Vectorized computation for speed
        feature_data = tda_features[available_features].values
        anomaly_scores = pd.Series(0.0, index=tda_features.index)

        # Pre-compute rolling statistics for efficiency
        for i, feature in enumerate(available_features):
            feature_series = tda_features[feature]
            rolling_mean = feature_series.rolling(window=lookback_window, min_periods=1).mean()
            rolling_std = feature_series.rolling(window=lookback_window, min_periods=1).std()

            # Compute z-scores
            z_scores = (feature_series - rolling_mean) / (rolling_std + 1e-6)

            # Accumulate anomaly scores
            anomaly_scores += z_scores.abs()

        # Normalize by number of features
        anomaly_scores = anomaly_scores / len(available_features)

        return anomaly_scores

    def detect_topological_regime_changes(self, tda_features, threshold=2.0):
        """
        Detect regime changes based on topological features.

        Parameters:
        -----------
        tda_features : pd.DataFrame
            TDA features
        threshold : float
            Threshold for regime change detection

        Returns:
        --------
        pd.Series : Binary regime change signals
        """
        # Compute anomaly scores
        anomaly_scores = self.compute_topological_anomaly_score(tda_features)

        # Detect regime changes when anomaly score exceeds threshold
        regime_changes = (anomaly_scores > threshold).astype(int)

        # Additional logic: detect sudden changes in key topological features
        if 'topological_complexity' in tda_features.columns:
            complexity_changes = np.abs(tda_features['topological_complexity'].diff()) > threshold * 0.5
            regime_changes = regime_changes | complexity_changes.astype(int)

        return regime_changes

    def create_persistence_summary(self, diagrams):
        """Create summary statistics for persistence diagrams."""
        summary = {}

        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                summary[f'H{dim}'] = {'count': 0, 'max_persistence': 0, 'total_persistence': 0}
                continue

            finite_diagram = diagram[diagram[:, 1] != np.inf]

            if len(finite_diagram) > 0:
                persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
                summary[f'H{dim}'] = {
                    'count': len(diagram),
                    'max_persistence': np.max(persistences),
                    'total_persistence': np.sum(persistences),
                    'mean_persistence': np.mean(persistences)
                }
            else:
                summary[f'H{dim}'] = {
                    'count': len(diagram),
                    'max_persistence': 0,
                    'total_persistence': 0,
                    'mean_persistence': 0
                }

        return summary

class TopologicalAnomalyDetector:
    """
    Specialized class for detecting market anomalies using topological features.
    """

    def __init__(self, sensitivity=2.0, min_anomaly_duration=3):
        """
        Initialize anomaly detector.

        Parameters:
        -----------
        sensitivity : float
            Sensitivity threshold for anomaly detection
        min_anomaly_duration : int
            Minimum duration for persistent anomalies
        """
        self.sensitivity = sensitivity
        self.min_anomaly_duration = min_anomaly_duration

    def detect_bifurcations(self, tda_features, feature_col='betti_1'):
        """
        Detect topological bifurcations (sudden changes in topology).

        Parameters:
        -----------
        tda_features : pd.DataFrame
            TDA features
        feature_col : str
            Feature to monitor for bifurcations

        Returns:
        --------
        pd.Series : Bifurcation signals
        """
        if feature_col not in tda_features.columns:
            return pd.Series(0, index=tda_features.index)

        feature_values = tda_features[feature_col]

        # Detect sudden jumps
        diff_values = feature_values.diff().abs()
        threshold = diff_values.rolling(window=20).mean() + self.sensitivity * diff_values.rolling(window=20).std()

        bifurcations = (diff_values > threshold).astype(int)

        return bifurcations

    def generate_early_warning_signals(self, tda_features):
        """
        Generate early warning signals for regime changes.

        Parameters:
        -----------
        tda_features : pd.DataFrame
            TDA features

        Returns:
        --------
        dict : Various early warning signals
        """
        signals = {}

        # Increasing complexity signal
        if 'topological_complexity' in tda_features.columns:
            complexity_trend = tda_features['topological_complexity'].rolling(window=10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            signals['complexity_increasing'] = (complexity_trend > 0.01).astype(int)

        # Persistence entropy changes
        if 'persistence_entropy_1' in tda_features.columns:
            entropy_volatility = tda_features['persistence_entropy_1'].rolling(window=10).std()
            entropy_threshold = entropy_volatility.rolling(window=20).quantile(0.8)
            signals['entropy_spike'] = (entropy_volatility > entropy_threshold).astype(int)

        # Topological stability breakdown
        if 'topological_stability' in tda_features.columns:
            stability_declining = tda_features['topological_stability'].rolling(window=5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            ) < -0.01
            signals['stability_breakdown'] = stability_declining.astype(int)

        return signals