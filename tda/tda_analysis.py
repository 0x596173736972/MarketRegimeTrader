import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tda.topological_features import TopologicalFeatureExtractor, TopologicalAnomalyDetector

class TDAAnalysisUI:
    """
    Streamlit UI components for TDA analysis and visualization.
    """

    def __init__(self):
        self.tda_extractor = None
        self.anomaly_detector = None

    def create_tda_analysis_section(self, data, regimes=None):
        """
        Create comprehensive TDA analysis section.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV market data
        regimes : pd.Series, optional
            Market regimes
        """
        st.header("🔬 Topological Data Analysis")

        # TDA Configuration
        with st.expander("⚙️ TDA Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                max_dimension = st.selectbox("Max Homology Dimension", [1, 2], index=1, key="tda_max_dimension")
                window_size = st.slider("Window Size", 20, 100, 50, key="tda_window_size")

            with col2:
                embedding_dim = st.selectbox("Embedding Dimension", [2, 3, 4], index=1, key="tda_embedding_dim")
                overlap = st.slider("Window Overlap", 0.0, 0.8, 0.5, key="tda_overlap")

            with col3:
                anomaly_sensitivity = st.slider("Anomaly Sensitivity", 1.0, 5.0, 2.0, key="tda_anomaly_sensitivity")

        # Compute TDA features
        if st.button("🧮 Compute TDA Features", type="primary", key="compute_tda_features_button"):
            with st.spinner("Computing topological features..."):
                try:
                    # Clear any previous TDA data
                    if hasattr(st.session_state, 'tda_features'):
                        del st.session_state.tda_features
                    if hasattr(st.session_state, 'tda_extractor'):
                        del st.session_state.tda_extractor
                    
                    self.tda_extractor = TopologicalFeatureExtractor(
                        max_dimension=max_dimension,
                        window_size=min(window_size, len(data) // 4),  # Ensure reasonable window size
                        overlap=overlap
                    )

                    tda_features = self.tda_extractor.extract_tda_features(
                        data, 
                        embedding_dim=embedding_dim, 
                        delay=1
                    )

                    if tda_features is not None and not tda_features.empty:
                        # Store in session state
                        st.session_state.tda_features = tda_features
                        st.session_state.tda_extractor = self.tda_extractor
                        st.success("✅ TDA features computed successfully!")
                    else:
                        st.error("❌ TDA computation returned empty results")
                        return

                except Exception as e:
                    st.error(f"❌ Error computing TDA features: {str(e)}")
                    st.info("💡 Try adjusting the window size or other parameters")
                    return

        # Display TDA results if available
        if hasattr(st.session_state, 'tda_features') and st.session_state.tda_features is not None:
            tda_features = st.session_state.tda_features

            # TDA Feature Overview
            st.subheader("📊 TDA Feature Overview")

            # Key metrics with interpretation
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_complexity = tda_features['topological_complexity'].mean()
                complexity_level = "High" if avg_complexity > 0.6 else ("Medium" if avg_complexity > 0.3 else "Low")
                st.metric("Topological Complexity", f"{avg_complexity:.3f}", f"{complexity_level} market complexity")

            with col2:
                avg_persistence = tda_features['persistence_score'].mean()
                persistence_level = "Strong" if avg_persistence > 0.5 else ("Moderate" if avg_persistence > 0.2 else "Weak")
                st.metric("Persistence Score", f"{avg_persistence:.3f}", f"{persistence_level} trend persistence")

            with col3:
                avg_structure = tda_features['market_structure_index'].mean()
                structure_level = "Well-formed" if avg_structure > 0.4 else ("Developing" if avg_structure > 0.2 else "Fragmented")
                st.metric("Market Structure", f"{avg_structure:.3f}", f"{structure_level} structure")

            with col4:
                avg_stability = tda_features['topological_stability'].mean()
                stability_level = "Stable" if avg_stability > 0.7 else ("Moderate" if avg_stability > 0.4 else "Unstable")
                st.metric("Topological Stability", f"{avg_stability:.3f}", f"{stability_level} topology")

            # Add interpretation panel
            st.markdown("### 💡 TDA Interpretation Guide")

            interpretation_col1, interpretation_col2 = st.columns(2)

            with interpretation_col1:
                st.markdown("""
                **🔬 Topological Complexity**
                - **High (>0.6)**: Complex market structures, multiple patterns
                - **Medium (0.3-0.6)**: Moderate complexity, some structure
                - **Low (<0.3)**: Simple patterns, trending markets

                **📊 Persistence Score**  
                - **Strong (>0.5)**: Persistent patterns, strong trends
                - **Moderate (0.2-0.5)**: Some persistence, mixed signals
                - **Weak (<0.2)**: Little persistence, noisy market
                """)

            with interpretation_col2:
                st.markdown("""
                **🏗️ Market Structure**
                - **Well-formed (>0.4)**: Clear structural patterns
                - **Developing (0.2-0.4)**: Emerging structures  
                - **Fragmented (<0.2)**: Lack of clear structure

                **⚖️ Topological Stability**
                - **Stable (>0.7)**: Consistent topological features
                - **Moderate (0.4-0.7)**: Some fluctuation in topology
                - **Unstable (<0.4)**: Rapidly changing topology
                """)

            # TDA Feature Visualization
            self._plot_tda_features(tda_features, data)

            # Anomaly Detection
            st.subheader("🚨 Topological Anomaly Detection")

            # Initialize anomaly detector and extractor if needed
            if not hasattr(st.session_state, 'tda_extractor') or st.session_state.tda_extractor is None:
                self.tda_extractor = TopologicalFeatureExtractor(
                    max_dimension=max_dimension,
                    window_size=window_size,
                    overlap=overlap
                )
                st.session_state.tda_extractor = self.tda_extractor
            else:
                self.tda_extractor = st.session_state.tda_extractor

            self.anomaly_detector = TopologicalAnomalyDetector(
                sensitivity=anomaly_sensitivity,
                min_anomaly_duration=3
            )

            # Compute anomaly scores with progress
            with st.spinner("Computing anomaly scores..."):
                anomaly_scores = self.tda_extractor.compute_topological_anomaly_score(tda_features, lookback_window=20)  # Reduced window

            with st.spinner("Detecting topological bifurcations..."):
                bifurcations = self.anomaly_detector.detect_bifurcations(tda_features)

            with st.spinner("Generating early warnings..."):
                early_warnings = self.anomaly_detector.generate_early_warning_signals(tda_features)

            # Anomaly interpretation
            high_anomalies = (anomaly_scores > 2.0).sum()
            medium_anomalies = ((anomaly_scores > 1.0) & (anomaly_scores <= 2.0)).sum()

            st.markdown("#### 🚨 Anomaly Analysis Summary")
            anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)

            with anomaly_col1:
                st.metric("High Risk Anomalies", high_anomalies, "Score > 2.0")

            with anomaly_col2:
                st.metric("Medium Risk Anomalies", medium_anomalies, "Score 1.0-2.0")

            with anomaly_col3:
                bifurcation_count = bifurcations.sum()
                st.metric("Topological Bifurcations", bifurcation_count, "Structural changes")

            # Interpretation
            if high_anomalies > len(anomaly_scores) * 0.05:  # More than 5%
                st.warning("⚠️ **High anomaly activity detected** - Market showing unusual topological behavior")
            elif high_anomalies > 0:
                st.info("ℹ️ **Some anomalies detected** - Monitor for potential regime changes")
            else:
                st.success("✅ **Normal topological behavior** - Market topology is stable")

            # Plot anomaly detection
            self._plot_anomaly_detection(data, anomaly_scores, bifurcations, early_warnings)

            # Detailed anomaly interpretation
            st.markdown("#### 📖 Understanding Topological Anomalies")

            with st.expander("🔍 What do these anomalies mean?", expanded=False):
                st.markdown("""
                **🚨 High Anomaly Scores (>2.0)**
                - Indicate significant deviations from normal topological patterns
                - Often precede major market moves or regime changes
                - Suggest increased market stress or structural breakdown

                **⚠️ Medium Anomaly Scores (1.0-2.0)**  
                - Moderate deviations that warrant attention
                - May indicate emerging patterns or transitional periods
                - Useful for early warning of market shifts

                **🔄 Topological Bifurcations**
                - Sudden changes in market topology structure
                - Can signal regime transitions or volatility spikes
                - Important inflection points in market behavior

                **⏰ Early Warning Signals**
                - Complexity increasing: Market becoming more complex
                - Stability breakdown: Loss of structural coherence  
                - Entropy spikes: Increased randomness in patterns
                """)

            # Regime Analysis with TDA
            if regimes is not None:
                st.subheader("🎯 TDA-Regime Analysis")
                self._analyze_tda_regimes(tda_features, regimes, data)

            # TDA Feature Correlation
            st.subheader("🔗 TDA Feature Correlations")
            self._plot_tda_correlations(tda_features)

    def _plot_tda_features(self, tda_features, data):
        """Plot key TDA features over time."""
        # Main TDA features plot
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Topological Complexity', 'Persistence Score',
                'Market Structure Index', 'Topological Stability',
                'Betti Numbers (H0 & H1)', 'Persistence Entropy'
            ],
            vertical_spacing=0.08
        )

        # Plot 1: Topological Complexity
        fig.add_trace(
            go.Scatter(
                x=tda_features.index,
                y=tda_features['topological_complexity'],
                name='Complexity',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Plot 2: Persistence Score
        fig.add_trace(
            go.Scatter(
                x=tda_features.index,
                y=tda_features['persistence_score'],
                name='Persistence',
                line=dict(color='red')
            ),
            row=1, col=2
        )

        # Plot 3: Market Structure Index
        fig.add_trace(
            go.Scatter(
                x=tda_features.index,
                y=tda_features['market_structure_index'],
                name='Structure',
                line=dict(color='green')
            ),
            row=2, col=1
        )

        # Plot 4: Topological Stability
        fig.add_trace(
            go.Scatter(
                x=tda_features.index,
                y=tda_features['topological_stability'],
                name='Stability',
                line=dict(color='purple')
            ),
            row=2, col=2
        )

        # Plot 5: Betti Numbers
        if 'betti_0' in tda_features.columns:
            fig.add_trace(
                go.Scatter(
                    x=tda_features.index,
                    y=tda_features['betti_0'],
                    name='H0 (Components)',
                    line=dict(color='orange')
                ),
                row=3, col=1
            )

        if 'betti_1' in tda_features.columns:
            fig.add_trace(
                go.Scatter(
                    x=tda_features.index,
                    y=tda_features['betti_1'],
                    name='H1 (Loops)',
                    line=dict(color='brown')
                ),
                row=3, col=1
            )

        # Plot 6: Persistence Entropy
        if 'persistence_entropy_1' in tda_features.columns:
            fig.add_trace(
                go.Scatter(
                    x=tda_features.index,
                    y=tda_features['persistence_entropy_1'],
                    name='H1 Entropy',
                    line=dict(color='pink')
                ),
                row=3, col=2
            )

        fig.update_layout(
            title="TDA Features Over Time",
            height=800,
            showlegend=False,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_anomaly_detection(self, data, anomaly_scores, bifurcations, early_warnings):
        """Plot anomaly detection results."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Price & Anomaly Scores', 'Bifurcation Detection', 'Early Warning Signals'],
            specs=[[{"secondary_y": True}], [{}], [{}]],
            vertical_spacing=0.08
        )

        # Plot 1: Price and anomaly scores
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                line=dict(color='black')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=anomaly_scores.index,
                y=anomaly_scores.values,
                name='Anomaly Score',
                line=dict(color='red'),
                yaxis='y2'
            ),
            row=1, col=1
        )

        # Highlight high anomaly periods
        high_anomaly = anomaly_scores > 2.0
        if high_anomaly.any():
            anomaly_dates = anomaly_scores[high_anomaly].index
            for date in anomaly_dates:
                fig.add_vline(
                    x=date,
                    line=dict(color='red', width=1, dash='dash'),
                    row=1, col=1
                )

        # Plot 2: Bifurcations
        bifurcation_dates = bifurcations[bifurcations == 1].index
        if len(bifurcation_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=bifurcation_dates,
                    y=[1] * len(bifurcation_dates),
                    mode='markers',
                    name='Bifurcations',
                    marker=dict(color='orange', size=10, symbol='triangle-up')
                ),
                row=2, col=1
            )

        # Plot 3: Early warning signals
        if early_warnings:
            for signal_name, signal_series in early_warnings.items():
                signal_dates = signal_series[signal_series == 1].index
                if len(signal_dates) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=signal_dates,
                            y=[1] * len(signal_dates),
                            mode='markers',
                            name=signal_name.replace('_', ' ').title(),
                            marker=dict(size=8)
                        ),
                        row=3, col=1
                    )

        fig.update_layout(
            title="Topological Anomaly Detection",
            height=800,
            template='plotly_white'
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Anomaly Score", secondary_y=True, row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def _analyze_tda_regimes(self, tda_features, regimes, data):
        """Analyze TDA features by regime."""
        # Align regimes with TDA features
        aligned_regimes = regimes.reindex(tda_features.index, method='ffill')

        # Regime statistics
        regime_stats = {}
        for regime in aligned_regimes.dropna().unique():
            regime_mask = aligned_regimes == regime
            regime_tda = tda_features[regime_mask]

            if len(regime_tda) > 0:
                regime_stats[f'Regime {int(regime)}'] = {
                    'Avg Complexity': regime_tda['topological_complexity'].mean(),
                    'Avg Persistence': regime_tda['persistence_score'].mean(),
                    'Avg Structure': regime_tda['market_structure_index'].mean(),
                    'Avg Stability': regime_tda['topological_stability'].mean()
                }

        # Display regime comparison
        if regime_stats:
            regime_df = pd.DataFrame(regime_stats).T
            st.dataframe(regime_df.round(4))

            # Regime TDA visualization
            fig = go.Figure()

            for regime in aligned_regimes.dropna().unique():
                regime_mask = aligned_regimes == regime
                regime_data = tda_features[regime_mask]

                if len(regime_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_data['topological_complexity'],
                            y=regime_data['persistence_score'],
                            mode='markers',
                            name=f'Regime {int(regime)}',
                            marker=dict(size=8, opacity=0.7)
                        )
                    )

            fig.update_layout(
                title="TDA Features by Regime",
                xaxis_title="Topological Complexity",
                yaxis_title="Persistence Score",
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    def _plot_tda_correlations(self, tda_features):
        """Plot TDA feature correlations."""
        # Select key TDA features for correlation
        key_features = [
            'topological_complexity', 'persistence_score', 
            'market_structure_index', 'topological_stability'
        ]

        # Add Betti numbers if available
        betti_features = [col for col in tda_features.columns if col.startswith('betti_')]
        key_features.extend(betti_features[:2])  # H0 and H1

        # Filter available features
        available_features = [f for f in key_features if f in tda_features.columns]

        if len(available_features) > 1:
            corr_matrix = tda_features[available_features].corr()

            fig = px.imshow(
                corr_matrix,
                title="TDA Feature Correlations",
                color_continuous_scale='RdBu',
                aspect='auto'
            )

            st.plotly_chart(fig, use_container_width=True)

    def get_tda_summary_stats(self, tda_features):
        """Get summary statistics for TDA features."""
        if tda_features is None or tda_features.empty:
            return {}

        key_features = [
            'topological_complexity', 'persistence_score',
            'market_structure_index', 'topological_stability'
        ]

        stats = {}
        for feature in key_features:
            if feature in tda_features.columns:
                stats[feature] = {
                    'mean': tda_features[feature].mean(),
                    'std': tda_features[feature].std(),
                    'min': tda_features[feature].min(),
                    'max': tda_features[feature].max()
                }

        return stats