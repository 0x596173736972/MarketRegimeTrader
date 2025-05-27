
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime

from strategies.auto_strategy_generator import AutoStrategyGenerator, AutoStrategyAdapter
from strategies.strategy_factory import StrategyFactory
from backtesting.backtest_engine import BacktestEngine

def display_strategy_evolution_interface():
    """Display the strategy evolution interface in Streamlit."""
    
    st.markdown("## 🧬 Automated Strategy Evolution")
    st.markdown("Generate trading strategies using genetic programming and symbolic regression.")
    
    # Check if we have the required data
    if not hasattr(st.session_state, 'data') or st.session_state.data is None:
        st.warning("⚠️ Please load market data first.")
        return
    
    if not hasattr(st.session_state, 'regimes') or st.session_state.regimes is None:
        st.warning("⚠️ Please detect market regimes first.")
        return
    
    # Evolution parameters
    st.markdown("### ⚙️ Evolution Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        population_size = st.slider("Population Size", 20, 200, 50, step=10,
                                   help="Number of strategies in each generation")
        max_generations = st.slider("Max Generations", 10, 100, 30, step=5,
                                   help="Maximum number of evolution cycles")
        mutation_rate = st.slider("Mutation Rate", 0.05, 0.5, 0.15, step=0.05,
                                 help="Probability of random mutations")
    
    with col2:
        crossover_rate = st.slider("Crossover Rate", 0.5, 1.0, 0.8, step=0.05,
                                  help="Probability of combining strategies")
        max_depth = st.slider("Max Expression Depth", 3, 8, 6, step=1,
                             help="Maximum complexity of strategy expressions")
        tournament_size = st.slider("Tournament Size", 2, 10, 3, step=1,
                                   help="Selection pressure for evolution")
    
    with col3:
        objective = st.selectbox("Optimization Objective", 
                                ['sharpe_ratio', 'calmar_ratio', 'profit_factor', 'max_drawdown'],
                                help="What metric to optimize")
        validation_split = st.slider("Validation Split", 0.1, 0.5, 0.3, step=0.05,
                                    help="Fraction of data for out-of-sample testing")
        early_stopping_patience = st.slider("Early Stopping", 5, 20, 10, step=1,
                                           help="Generations without improvement before stopping")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        diversity_threshold = st.slider("Diversity Threshold", 0.8, 1.0, 0.95, step=0.01,
                                       help="Minimum population diversity to maintain")
        random_state = st.number_input("Random Seed", 1, 1000, 42,
                                      help="For reproducible results")
    
    # Start evolution
    if st.button("🚀 Start Evolution", type="primary"):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        generation_chart = st.empty()
        
        def update_progress(generation, max_gen, stats):
            progress = generation / max_gen
            progress_bar.progress(progress)
            status_text.text(f"Generation {generation+1}/{max_gen} - Best Fitness: {stats['best_fitness']:.4f}")
            
            # Update generation chart
            if hasattr(st.session_state, 'generation_stats'):
                gen_data = pd.DataFrame(st.session_state.generation_stats + [stats])
                if not gen_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=gen_data['generation'],
                        y=gen_data['best_fitness'],
                        name='Best Fitness',
                        line=dict(color='green')
                    ))
                    fig.add_trace(go.Scatter(
                        x=gen_data['generation'],
                        y=gen_data['avg_fitness'],
                        name='Average Fitness',
                        line=dict(color='blue', dash='dash')
                    ))
                    fig.update_layout(
                        title="Evolution Progress",
                        xaxis_title="Generation",
                        yaxis_title="Fitness",
                        height=300
                    )
                    generation_chart.plotly_chart(fig, use_container_width=True)
        
        with st.spinner("Running genetic programming evolution..."):
            try:
                # Initialize generator
                generator = AutoStrategyGenerator(
                    population_size=population_size,
                    max_generations=max_generations,
                    mutation_rate=mutation_rate,
                    crossover_rate=crossover_rate,
                    max_depth=max_depth,
                    tournament_size=tournament_size,
                    early_stopping_patience=early_stopping_patience,
                    diversity_threshold=diversity_threshold,
                    random_state=random_state
                )
                
                # Generate features
                features = generator.generate_features(
                    st.session_state.data,
                    st.session_state.regimes,
                    getattr(st.session_state, 'regime_probs', None)
                )
                
                # Run evolution
                results = generator.evolve_strategies(
                    features=features,
                    data=st.session_state.data,
                    regimes=st.session_state.regimes,
                    objective=objective,
                    validation_split=validation_split,
                    progress_callback=update_progress
                )
                
                # Store results
                st.session_state.evolution_results = results
                st.session_state.auto_generator = generator
                st.session_state.generation_stats = results['generation_stats']
                
                progress_bar.progress(1.0)
                status_text.text("✅ Evolution completed successfully!")
                
                st.success(f"🎉 Evolution completed! Found {len(results['best_strategies'])} candidate strategies.")
                
            except Exception as e:
                st.error(f"❌ Evolution failed: {str(e)}")
                return
    
    # Display results if available
    if hasattr(st.session_state, 'evolution_results') and st.session_state.evolution_results:
        st.markdown("---")
        st.markdown("### 📊 Evolution Results")
        
        results = st.session_state.evolution_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Strategy Fitness", 
                     f"{results['best_strategies'][0]['validation_fitness']:.4f}")
        
        with col2:
            st.metric("Convergence", 
                     "Yes" if results['convergence_info']['converged'] else "No")
        
        with col3:
            st.metric("Final Generation", 
                     results['final_generation'])
        
        with col4:
            st.metric("Strategies Found", 
                     len(results['best_strategies']))
        
        # Evolution charts
        st.markdown("#### 📈 Evolution Progress")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fitness evolution
            gen_data = pd.DataFrame(results['generation_stats'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gen_data['generation'],
                y=gen_data['best_fitness'],
                name='Best Fitness',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=gen_data['generation'],
                y=gen_data['avg_fitness'],
                name='Average Fitness',
                line=dict(color='blue', dash='dash')
            ))
            fig.update_layout(
                title="Fitness Evolution",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Diversity evolution
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gen_data['generation'],
                y=gen_data['diversity'],
                name='Population Diversity',
                line=dict(color='purple', width=2)
            ))
            fig.update_layout(
                title="Population Diversity",
                xaxis_title="Generation",
                yaxis_title="Diversity",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Best strategies table
        st.markdown("#### 🏆 Top Evolved Strategies")
        
        strategy_table = []
        for i, strategy in enumerate(results['best_strategies'][:10]):
            strategy_table.append({
                'Rank': i + 1,
                'Training Fitness': f"{strategy['train_fitness']:.4f}",
                'Validation Fitness': f"{strategy['validation_fitness']:.4f}",
                'Overfitting Score': f"{strategy['overfitting_score']:.4f}",
                'Complexity': strategy['complexity'],
                'Expression Preview': strategy['human_readable'][:50] + "..." if len(strategy['human_readable']) > 50 else strategy['human_readable']
            })
        
        strategy_df = pd.DataFrame(strategy_table)
        st.dataframe(strategy_df, use_container_width=True)
        
        # Strategy selection and analysis
        st.markdown("#### 🔍 Strategy Analysis")
        
        strategy_options = {i: f"Strategy {i+1} (Fitness: {s['validation_fitness']:.4f})" 
                           for i, s in enumerate(results['best_strategies'][:10])}
        
        selected_strategy_idx = st.selectbox("Select Strategy for Analysis", 
                                            list(strategy_options.keys()),
                                            format_func=lambda x: strategy_options[x])
        
        selected_strategy = results['best_strategies'][selected_strategy_idx]
        
        # Strategy details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📝 Strategy Expression")
            st.code(selected_strategy['human_readable'], language='text')
            
            st.markdown("#### 📊 Performance Metrics")
            metrics_df = pd.DataFrame([
                {'Metric': 'Training Fitness', 'Value': f"{selected_strategy['train_fitness']:.4f}"},
                {'Metric': 'Validation Fitness', 'Value': f"{selected_strategy['validation_fitness']:.4f}"},
                {'Metric': 'Overfitting Score', 'Value': f"{selected_strategy['overfitting_score']:.4f}"},
                {'Metric': 'Complexity', 'Value': selected_strategy['complexity']}
            ])
            st.dataframe(metrics_df, hide_index=True)
        
        with col2:
            st.markdown("#### ⚠️ Risk Assessment")
            
            # Risk indicators
            overfitting_risk = "🔴 High" if selected_strategy['overfitting_score'] > 0.2 else \
                              "🟡 Medium" if selected_strategy['overfitting_score'] > 0.1 else "🟢 Low"
            
            complexity_risk = "🔴 High" if selected_strategy['complexity'] > 20 else \
                             "🟡 Medium" if selected_strategy['complexity'] > 10 else "🟢 Low"
            
            fitness_diff = abs(selected_strategy['train_fitness'] - selected_strategy['validation_fitness'])
            stability_risk = "🔴 High" if fitness_diff > 0.5 else \
                            "🟡 Medium" if fitness_diff > 0.2 else "🟢 Low"
            
            st.write(f"**Overfitting Risk:** {overfitting_risk}")
            st.write(f"**Complexity Risk:** {complexity_risk}")
            st.write(f"**Stability Risk:** {stability_risk}")
        
        # Backtest selected strategy
        if st.button("🔍 Backtest Selected Strategy", type="secondary"):
            with st.spinner("Running comprehensive backtest..."):
                try:
                    # Create feature generator function
                    def feature_generator(data, regimes, regime_probs=None):
                        return st.session_state.auto_generator.generate_features(
                            data, regimes, regime_probs
                        )
                    
                    # Create strategy adapter
                    strategy = StrategyFactory.create_auto_strategy(
                        selected_strategy['expression'],
                        feature_generator,
                        max_position_size=0.1
                    )
                    
                    # Generate signals
                    signals = strategy.generate_signals(
                        st.session_state.data,
                        st.session_state.regimes,
                        getattr(st.session_state, 'regime_probs', None)
                    )
                    
                    # Run backtest
                    backtest_engine = BacktestEngine()
                    backtest_results = backtest_engine.run_backtest(st.session_state.data, signals)
                    
                    # Store results
                    st.session_state.auto_strategy_results = {
                        'strategy': strategy,
                        'signals': signals,
                        'backtest_results': backtest_results,
                        'expression': selected_strategy['human_readable']
                    }
                    
                    st.success("✅ Backtest completed! Check the results below.")
                    
                    # Display quick results
                    metrics = backtest_results['metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")
                    with col2:
                        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
                    with col3:
                        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
                    with col4:
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
                    
                except Exception as e:
                    st.error(f"❌ Backtest failed: {str(e)}")
        
        # Compare strategies
        st.markdown("### ⚖️ Strategy Comparison")
        
        if len(results['best_strategies']) > 1:
            comparison_df = pd.DataFrame([
                {
                    'Strategy': f"Strategy {i+1}",
                    'Training Fitness': strategy['train_fitness'],
                    'Validation Fitness': strategy['validation_fitness'],
                    'Overfitting Score': strategy['overfitting_score'],
                    'Complexity': strategy['complexity']
                }
                for i, strategy in enumerate(results['best_strategies'][:5])
            ])
            
            # Comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Training Fitness',
                x=comparison_df['Strategy'],
                y=comparison_df['Training Fitness'],
                opacity=0.7
            ))
            
            fig.add_trace(go.Bar(
                name='Validation Fitness', 
                x=comparison_df['Strategy'],
                y=comparison_df['Validation Fitness'],
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Strategy Performance Comparison",
                xaxis_title="Strategy",
                yaxis_title="Fitness",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            if st.button("📥 Download Evolution Results"):
                results_summary = {
                    'evolution_parameters': {
                        'population_size': population_size,
                        'max_generations': max_generations,
                        'mutation_rate': mutation_rate,
                        'crossover_rate': crossover_rate,
                        'objective': objective
                    },
                    'best_strategies': [
                        {
                            'rank': i+1,
                            'expression': strategy['human_readable'],
                            'train_fitness': strategy['train_fitness'],
                            'validation_fitness': strategy['validation_fitness'],
                            'overfitting_score': strategy['overfitting_score'],
                            'complexity': strategy['complexity']
                        }
                        for i, strategy in enumerate(results['best_strategies'])
                    ],
                    'convergence_info': results['convergence_info']
                }
                
                st.download_button(
                    "Download Results (JSON)",
                    data=pd.Series(results_summary).to_json(indent=2),
                    file_name=f"evolution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        st.info("👆 Run the evolution process to see results and generate automated trading strategies.")
