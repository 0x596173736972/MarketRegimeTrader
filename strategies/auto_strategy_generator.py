
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import random
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy
from backtesting.backtest_engine import BacktestEngine

class StrategyExpression:
    """
    Represents a symbolic expression for trading strategy generation.
    Uses a tree-based structure to represent mathematical and logical operations.
    """
    
    def __init__(self, expression_str=None, tree=None):
        self.expression_str = expression_str
        self.tree = tree
        self.fitness = None
        self.complexity = 0
        
    def evaluate(self, features):
        """Evaluate the expression given input features."""
        if self.tree is None:
            return np.zeros(len(features))
        return self._evaluate_tree(self.tree, features)
    
    def _evaluate_tree(self, node, features):
        """Recursively evaluate the expression tree."""
        if isinstance(node, dict):
            op = node['op']
            
            if op == 'feature':
                feature_name = node['feature']
                if feature_name in features.columns:
                    return features[feature_name].values
                else:
                    return np.zeros(len(features))
            
            elif op == 'constant':
                return np.full(len(features), node['value'])
            
            elif op in ['add', 'sub', 'mul', 'div', 'max', 'min']:
                left = self._evaluate_tree(node['left'], features)
                right = self._evaluate_tree(node['right'], features)
                
                if op == 'add':
                    return left + right
                elif op == 'sub':
                    return left - right
                elif op == 'mul':
                    return left * right
                elif op == 'div':
                    return np.divide(left, right, out=np.zeros_like(left), where=right!=0)
                elif op == 'max':
                    return np.maximum(left, right)
                elif op == 'min':
                    return np.minimum(left, right)
            
            elif op in ['sin', 'cos', 'exp', 'log', 'abs', 'sqrt', 'tanh']:
                operand = self._evaluate_tree(node['operand'], features)
                
                if op == 'sin':
                    return np.sin(operand)
                elif op == 'cos':
                    return np.cos(operand)
                elif op == 'exp':
                    return np.clip(np.exp(operand), -1e6, 1e6)
                elif op == 'log':
                    return np.log(np.abs(operand) + 1e-10)
                elif op == 'abs':
                    return np.abs(operand)
                elif op == 'sqrt':
                    return np.sqrt(np.abs(operand))
                elif op == 'tanh':
                    return np.tanh(operand)
            
            elif op in ['>', '<', '>=', '<=', '==']:
                left = self._evaluate_tree(node['left'], features)
                right = self._evaluate_tree(node['right'], features)
                
                if op == '>':
                    return (left > right).astype(float)
                elif op == '<':
                    return (left < right).astype(float)
                elif op == '>=':
                    return (left >= right).astype(float)
                elif op == '<=':
                    return (left <= right).astype(float)
                elif op == '==':
                    return (np.abs(left - right) < 1e-10).astype(float)
            
            elif op in ['and', 'or']:
                left = self._evaluate_tree(node['left'], features)
                right = self._evaluate_tree(node['right'], features)
                
                if op == 'and':
                    return np.logical_and(left > 0.5, right > 0.5).astype(float)
                elif op == 'or':
                    return np.logical_or(left > 0.5, right > 0.5).astype(float)
        
        return np.zeros(len(features))
    
    def to_string(self):
        """Convert expression tree to human-readable string."""
        if self.tree is None:
            return "Empty"
        return self._tree_to_string(self.tree)
    
    def _tree_to_string(self, node):
        """Recursively convert tree to string."""
        if isinstance(node, dict):
            op = node['op']
            
            if op == 'feature':
                return node['feature']
            elif op == 'constant':
                return f"{node['value']:.3f}"
            elif op in ['add', 'sub', 'mul', 'div', 'max', 'min', '>', '<', '>=', '<=', '==', 'and', 'or']:
                left_str = self._tree_to_string(node['left'])
                right_str = self._tree_to_string(node['right'])
                return f"({left_str} {op} {right_str})"
            elif op in ['sin', 'cos', 'exp', 'log', 'abs', 'sqrt', 'tanh']:
                operand_str = self._tree_to_string(node['operand'])
                return f"{op}({operand_str})"
        
        return "unknown"

class AutoStrategyGenerator:
    """
    Automated strategy generation using genetic programming.
    
    Evolves a population of trading strategies represented as symbolic expressions
    that combine technical indicators, regime states, and statistical signals.
    """
    
    def __init__(self, 
                 population_size=100,
                 max_generations=50,
                 mutation_rate=0.15,
                 crossover_rate=0.8,
                 max_depth=6,
                 tournament_size=3,
                 early_stopping_patience=10,
                 diversity_threshold=0.95,
                 random_state=42):
        """
        Initialize the AutoStrategy Generator.
        
        Parameters:
        -----------
        population_size : int
            Number of strategies in the population
        max_generations : int
            Maximum number of evolution generations
        mutation_rate : float
            Probability of mutation
        crossover_rate : float
            Probability of crossover
        max_depth : int
            Maximum depth of expression trees
        tournament_size : int
            Size of tournament selection
        early_stopping_patience : int
            Generations without improvement before stopping
        diversity_threshold : float
            Minimum diversity threshold to maintain
        random_state : int
            Random seed for reproducibility
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.early_stopping_patience = early_stopping_patience
        self.diversity_threshold = diversity_threshold
        self.random_state = random_state
        
        self.population = []
        self.best_strategies = []
        self.generation_stats = []
        self.feature_names = []
        
        # Set random seeds
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Available operators
        self.binary_ops = ['add', 'sub', 'mul', 'div', 'max', 'min', '>', '<', '>=', '<=', 'and', 'or']
        self.unary_ops = ['sin', 'cos', 'exp', 'log', 'abs', 'sqrt', 'tanh']
        self.comparison_ops = ['>', '<', '>=', '<=', '==']
        self.logical_ops = ['and', 'or']
    
    def generate_features(self, data, regimes, regime_probs=None, lookback_window=20, include_tda=True):
        """
        Generate comprehensive feature set for strategy evolution including TDA features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV market data
        regimes : pd.Series
            Detected market regimes
        regime_probs : pd.DataFrame, optional
            Regime probabilities
        lookback_window : int
            Lookback window for indicator calculations
        include_tda : bool
            Whether to include topological features
            
        Returns:
        --------
        pd.DataFrame : Feature matrix for strategy evolution
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['price_momentum'] = data['Close'] / data['Close'].shift(lookback_window) - 1
        features['price_mean_reversion'] = (data['Close'] - data['Close'].rolling(lookback_window).mean()) / data['Close'].rolling(lookback_window).std()
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(lookback_window).std()
        features['vol_ratio'] = features['volatility'] / features['volatility'].rolling(lookback_window*2).mean()
        
        # Technical indicators
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_normalized'] = (features['rsi'] - 50) / 50
        
        # Moving averages
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_ratio'] = features['sma_5'] / features['sma_20'] - 1
        features['price_to_sma'] = data['Close'] / features['sma_20'] - 1
        
        # Bollinger Bands
        bb_middle = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_middle
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Volume features (if available)
        if 'Volume' in data.columns:
            features['volume_sma'] = data['Volume'].rolling(lookback_window).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma']
            features['price_volume'] = features['returns'] * np.log1p(features['volume_ratio'])
        
        # Regime-based features
        aligned_regimes = regimes.reindex(data.index, method='ffill')
        for regime_val in aligned_regimes.unique():
            if not pd.isna(regime_val):
                features[f'regime_{int(regime_val)}'] = (aligned_regimes == regime_val).astype(float)
        
        # Regime probabilities
        if regime_probs is not None:
            regime_probs_aligned = regime_probs.reindex(data.index, method='ffill')
            for col in regime_probs_aligned.columns:
                features[f'prob_{col}'] = regime_probs_aligned[col]
        
        # Regime transitions
        regime_changes = (aligned_regimes != aligned_regimes.shift(1)).astype(float)
        features['regime_change'] = regime_changes
        
        # High-Low features
        if all(col in data.columns for col in ['High', 'Low']):
            features['hl_ratio'] = (data['High'] - data['Low']) / data['Close']
            features['price_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # TDA Features
        if include_tda:
            try:
                from tda.topological_features import TopologicalFeatureExtractor
                
                print("Computing TDA features for strategy generation...")
                tda_extractor = TopologicalFeatureExtractor(
                    max_dimension=2,
                    window_size=min(50, len(data) // 4),
                    overlap=0.5
                )
                
                tda_features = tda_extractor.extract_tda_features(data, embedding_dim=3, delay=1)
                
                # Add key TDA features that are useful for strategy generation
                tda_feature_mapping = {
                    'topological_complexity': 'topo_complexity',
                    'persistence_score': 'persistence_score',
                    'market_structure_index': 'market_structure',
                    'betti_0': 'connected_components',
                    'betti_1': 'loops',
                    'persistence_entropy_0': 'entropy_h0',
                    'persistence_entropy_1': 'entropy_h1',
                    'topological_stability': 'topo_stability',
                    'max_persistence_1': 'max_loop_persistence',
                    'topological_complexity_volatility': 'topo_volatility'
                }
                
                for original_name, feature_name in tda_feature_mapping.items():
                    if original_name in tda_features.columns:
                        # Align and normalize TDA features
                        tda_values = tda_features[original_name].reindex(data.index, method='ffill').fillna(0)
                        if tda_values.std() > 1e-6:
                            tda_values = np.tanh(tda_values / tda_values.std())  # Smooth normalization
                        features[feature_name] = tda_values
                
                # Create composite TDA indicators for GP
                if 'persistence_score' in features.columns and 'topo_complexity' in features.columns:
                    features['tda_composite'] = (
                        0.6 * features['persistence_score'] + 
                        0.4 * features['topo_complexity']
                    )
                
                # TDA-regime interaction features
                for regime_col in [col for col in features.columns if col.startswith('regime_')]:
                    if 'persistence_score' in features.columns:
                        features[f'{regime_col}_tda_interaction'] = (
                            features[regime_col] * features['persistence_score']
                        )
                
                print(f"Added {len([f for f in features.columns if any(tda in f for tda in tda_feature_mapping.values())])} TDA features")
                
            except Exception as e:
                print(f"Warning: Could not compute TDA features for strategy generation: {e}")
                print("Continuing without TDA features...")
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
            # Add lagged TDA features if available
            if 'persistence_score' in features.columns:
                features[f'persistence_score_lag_{lag}'] = features['persistence_score'].shift(lag)
        
        # Rolling statistics
        features['returns_skew'] = features['returns'].rolling(lookback_window).skew()
        features['returns_kurt'] = features['returns'].rolling(lookback_window).kurt()
        
        # Cross-features including TDA
        features['vol_momentum'] = features['volatility'] * features['price_momentum']
        features['rsi_momentum'] = features['rsi_normalized'] * features['price_momentum']
        
        # TDA cross-features
        if 'persistence_score' in features.columns:
            features['tda_momentum'] = features['persistence_score'] * features['price_momentum']
            features['tda_volatility'] = features['persistence_score'] * features['volatility']
        
        # Clean features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        return features
    
    def create_random_expression(self, depth=0, max_depth=None):
        """Create a random expression tree."""
        if max_depth is None:
            max_depth = self.max_depth
            
        if depth >= max_depth or (depth > 1 and random.random() < 0.3):
            # Terminal node
            if random.random() < 0.7 and self.feature_names:
                # Feature terminal
                return {
                    'op': 'feature',
                    'feature': random.choice(self.feature_names)
                }
            else:
                # Constant terminal
                return {
                    'op': 'constant',
                    'value': random.uniform(-2, 2)
                }
        else:
            # Non-terminal node
            if random.random() < 0.8:
                # Binary operator
                op = random.choice(self.binary_ops)
                return {
                    'op': op,
                    'left': self.create_random_expression(depth + 1, max_depth),
                    'right': self.create_random_expression(depth + 1, max_depth)
                }
            else:
                # Unary operator
                op = random.choice(self.unary_ops)
                return {
                    'op': op,
                    'operand': self.create_random_expression(depth + 1, max_depth)
                }
    
    def initialize_population(self):
        """Initialize the population with random expressions."""
        self.population = []
        for _ in range(self.population_size):
            tree = self.create_random_expression()
            expr = StrategyExpression(tree=tree)
            expr.complexity = self._calculate_complexity(tree)
            self.population.append(expr)
    
    def _calculate_complexity(self, tree):
        """Calculate the complexity of an expression tree."""
        if isinstance(tree, dict):
            if tree['op'] in ['feature', 'constant']:
                return 1
            elif tree['op'] in self.unary_ops:
                return 1 + self._calculate_complexity(tree['operand'])
            else:
                return 1 + self._calculate_complexity(tree['left']) + self._calculate_complexity(tree['right'])
        return 0
    
    def evaluate_fitness(self, expression, features, data, regimes, objective='sharpe_ratio'):
        """
        Evaluate the fitness of a strategy expression.
        
        Parameters:
        -----------
        expression : StrategyExpression
            The strategy expression to evaluate
        features : pd.DataFrame
            Feature matrix
        data : pd.DataFrame
            Market data
        regimes : pd.Series
            Market regimes
        objective : str
            Objective function ('sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'profit_factor')
            
        Returns:
        --------
        float : Fitness score
        """
        try:
            # Generate signals from expression
            signal_values = expression.evaluate(features)
            
            # Normalize signals to [-1, 1] range
            if np.std(signal_values) > 0:
                signal_values = np.tanh(signal_values)
            else:
                signal_values = np.zeros_like(signal_values)
            
            # Create signals DataFrame
            signals = pd.DataFrame(index=features.index)
            signals['signal'] = signal_values
            signals['position_size'] = signal_values * 0.1  # 10% max position
            
            # Run backtest
            backtest_engine = BacktestEngine(
                initial_capital=100000,
                transaction_cost=0.001,
                slippage=0.0005
            )
            
            results = backtest_engine.run_backtest(data.loc[features.index], signals)
            
            if not results or 'metrics' not in results:
                return -1000  # Penalty for invalid strategies
                
            metrics = results['metrics']
            
            if not metrics or not isinstance(metrics, dict):
                return -1000  # Penalty for invalid strategies
            
            # Calculate fitness based on objective
            if objective == 'sharpe_ratio':
                fitness = metrics.get('sharpe_ratio', 0)
            elif objective == 'max_drawdown':
                fitness = -metrics.get('max_drawdown', -1)  # Minimize drawdown
            elif objective == 'calmar_ratio':
                fitness = metrics.get('calmar_ratio', 0)
            elif objective == 'profit_factor':
                fitness = metrics.get('profit_factor', 0)
            else:
                fitness = metrics.get('sharpe_ratio', 0)
            
            # Penalty for excessive complexity
            complexity_penalty = expression.complexity * 0.01
            fitness -= complexity_penalty
            
            # Penalty for too few trades
            num_trades = metrics.get('total_trades', 0)
            if num_trades < 10:
                fitness -= (10 - num_trades) * 0.1
            
            return fitness
            
        except Exception as e:
            return -1000  # Penalty for errors
    
    def tournament_selection(self, tournament_size=None):
        """Select an individual using tournament selection."""
        if tournament_size is None:
            tournament_size = self.tournament_size
            
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness if x.fitness is not None else -1000)
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        child1_tree = self._copy_tree(parent1.tree)
        child2_tree = self._copy_tree(parent2.tree)
        
        # Find crossover points
        nodes1 = self._get_all_nodes(child1_tree)
        nodes2 = self._get_all_nodes(child2_tree)
        
        if len(nodes1) > 1 and len(nodes2) > 1:
            # Select random nodes for crossover
            node1 = random.choice(nodes1[1:])  # Skip root
            node2 = random.choice(nodes2[1:])  # Skip root
            
            # Swap subtrees
            temp = self._copy_tree(node1)
            self._replace_node(child1_tree, node1, node2)
            self._replace_node(child2_tree, node2, temp)
        
        child1 = StrategyExpression(tree=child1_tree)
        child2 = StrategyExpression(tree=child2_tree)
        child1.complexity = self._calculate_complexity(child1_tree)
        child2.complexity = self._calculate_complexity(child2_tree)
        
        return child1, child2
    
    def mutate(self, individual):
        """Mutate an individual."""
        mutated_tree = self._copy_tree(individual.tree)
        
        # Get all nodes
        nodes = self._get_all_nodes(mutated_tree)
        
        if nodes:
            # Select random node to mutate
            node_to_mutate = random.choice(nodes)
            
            # Mutation types
            mutation_type = random.choice(['value', 'operator', 'subtree'])
            
            if mutation_type == 'value' and node_to_mutate.get('op') == 'constant':
                # Mutate constant value
                node_to_mutate['value'] += random.gauss(0, 0.1)
                node_to_mutate['value'] = np.clip(node_to_mutate['value'], -5, 5)
            
            elif mutation_type == 'operator' and node_to_mutate.get('op') in self.binary_ops + self.unary_ops:
                # Change operator
                if node_to_mutate.get('op') in self.binary_ops:
                    node_to_mutate['op'] = random.choice(self.binary_ops)
                else:
                    node_to_mutate['op'] = random.choice(self.unary_ops)
            
            elif mutation_type == 'subtree':
                # Replace with new random subtree
                new_subtree = self.create_random_expression(max_depth=3)
                self._replace_node(mutated_tree, node_to_mutate, new_subtree)
        
        mutated_individual = StrategyExpression(tree=mutated_tree)
        mutated_individual.complexity = self._calculate_complexity(mutated_tree)
        
        return mutated_individual
    
    def _copy_tree(self, tree):
        """Deep copy a tree structure."""
        if isinstance(tree, dict):
            copied = {'op': tree['op']}
            for key, value in tree.items():
                if key != 'op':
                    if isinstance(value, dict):
                        copied[key] = self._copy_tree(value)
                    else:
                        copied[key] = value
            return copied
        return tree
    
    def _get_all_nodes(self, tree):
        """Get all nodes in a tree."""
        nodes = [tree]
        if isinstance(tree, dict):
            for key, value in tree.items():
                if key != 'op' and isinstance(value, dict):
                    nodes.extend(self._get_all_nodes(value))
        return nodes
    
    def _replace_node(self, tree, old_node, new_node):
        """Replace a node in the tree."""
        if tree == old_node:
            tree.clear()
            tree.update(new_node)
            return True
        
        if isinstance(tree, dict):
            for key, value in tree.items():
                if key != 'op' and isinstance(value, dict):
                    if value == old_node:
                        tree[key] = new_node
                        return True
                    elif self._replace_node(value, old_node, new_node):
                        return True
        return False
    
    def calculate_diversity(self):
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 1.0
        
        # Calculate fitness diversity
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
        if len(fitness_values) < 2:
            return 1.0
        
        fitness_std = np.std(fitness_values)
        fitness_mean = np.mean(fitness_values)
        
        if fitness_mean != 0:
            diversity = fitness_std / abs(fitness_mean)
        else:
            diversity = fitness_std
        
        return min(diversity, 1.0)
    
    def evolve_strategies(self, features, data, regimes, objective='sharpe_ratio', 
                         validation_split=0.3, progress_callback=None):
        """
        Evolve trading strategies using genetic programming.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        data : pd.DataFrame
            Market data
        regimes : pd.Series
            Market regimes
        objective : str
            Objective function
        validation_split : float
            Fraction of data for validation
        progress_callback : callable
            Progress callback function
            
        Returns:
        --------
        dict : Evolution results
        """
        # Split data for validation
        split_idx = int(len(features) * (1 - validation_split))
        train_features = features.iloc[:split_idx]
        train_data = data.iloc[:split_idx]
        train_regimes = regimes.iloc[:split_idx]
        
        val_features = features.iloc[split_idx:]
        val_data = data.iloc[split_idx:]
        val_regimes = regimes.iloc[split_idx:]
        
        # Initialize population
        self.initialize_population()
        
        best_fitness_history = []
        diversity_history = []
        generations_without_improvement = 0
        best_fitness = -np.inf
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all individuals
            for individual in self.population:
                if individual.fitness is None:
                    individual.fitness = self.evaluate_fitness(
                        individual, train_features, train_data, train_regimes, objective
                    )
            
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness if x.fitness is not None else -1000, reverse=True)
            
            # Track best fitness
            current_best_fitness = self.population[0].fitness if self.population[0].fitness is not None else -1000
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            best_fitness_history.append(current_best_fitness)
            
            # Calculate diversity
            diversity = self.calculate_diversity()
            diversity_history.append(diversity)
            
            # Generation statistics
            valid_fitness = [ind.fitness for ind in self.population if ind.fitness is not None]
            avg_fitness = np.mean(valid_fitness) if valid_fitness else 0.0
            
            gen_stats = {
                'generation': generation,
                'best_fitness': current_best_fitness if current_best_fitness is not None else 0.0,
                'avg_fitness': avg_fitness,
                'diversity': diversity,
                'best_complexity': self.population[0].complexity
            }
            self.generation_stats.append(gen_stats)
            
            # Progress callback
            if progress_callback:
                progress_callback(generation, self.max_generations, gen_stats)
            
            # Early stopping
            if generations_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at generation {generation}")
                break
            
            # Diversity enforcement
            if diversity < (1 - self.diversity_threshold):
                # Add random individuals to increase diversity
                num_random = self.population_size // 10
                for _ in range(num_random):
                    random_individual = StrategyExpression(tree=self.create_random_expression())
                    random_individual.complexity = self._calculate_complexity(random_individual.tree)
                    self.population[-1] = random_individual  # Replace worst individual
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = self.population_size // 10
            new_population.extend(self.population[:elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = self.tournament_selection()
                    parent2 = self.tournament_selection()
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    # Reproduction
                    parent = self.tournament_selection()
                    new_population.append(self._copy_tree_to_expression(parent.tree))
            
            # Mutation
            for i in range(elite_size, len(new_population)):
                if random.random() < self.mutation_rate:
                    new_population[i] = self.mutate(new_population[i])
            
            # Truncate to population size
            self.population = new_population[:self.population_size]
        
        # Validate best strategies on out-of-sample data
        top_strategies = self.population[:10]  # Top 10 strategies
        validated_strategies = []
        
        for strategy in top_strategies:
            # Evaluate on validation set
            val_fitness = self.evaluate_fitness(
                strategy, val_features, val_data, val_regimes, objective
            )
            
            # Calculate overfitting measure
            train_fitness = strategy.fitness if strategy.fitness is not None else 0.0
            val_fitness = val_fitness if val_fitness is not None else 0.0
            overfitting_score = abs(train_fitness - val_fitness) / (abs(train_fitness) + 1e-6)
            
            strategy_result = {
                'expression': strategy,
                'train_fitness': train_fitness,
                'validation_fitness': val_fitness,
                'overfitting_score': overfitting_score,
                'complexity': strategy.complexity,
                'human_readable': strategy.to_string()
            }
            validated_strategies.append(strategy_result)
        
        # Sort by validation fitness
        validated_strategies.sort(key=lambda x: x['validation_fitness'], reverse=True)
        
        return {
            'best_strategies': validated_strategies,
            'generation_stats': self.generation_stats,
            'best_fitness_history': best_fitness_history,
            'diversity_history': diversity_history,
            'final_generation': generation,
            'convergence_info': {
                'converged': generations_without_improvement >= self.early_stopping_patience,
                'generations_without_improvement': generations_without_improvement
            }
        }
    
    def _copy_tree_to_expression(self, tree):
        """Convert tree to expression object."""
        copied_tree = self._copy_tree(tree)
        expr = StrategyExpression(tree=copied_tree)
        expr.complexity = self._calculate_complexity(copied_tree)
        return expr

class AutoStrategyAdapter(BaseStrategy):
    """
    Adapter class to use evolved strategies in the backtesting framework.
    """
    
    def __init__(self, strategy_expression, feature_generator_func, **kwargs):
        """
        Initialize adapter with evolved strategy.
        
        Parameters:
        -----------
        strategy_expression : StrategyExpression
            The evolved strategy expression
        feature_generator_func : callable
            Function to generate features from data
        """
        super().__init__(**kwargs)
        self.strategy_expression = strategy_expression
        self.feature_generator_func = feature_generator_func
    
    def generate_signals(self, data, regimes, regime_probs=None):
        """Generate trading signals using the evolved strategy."""
        # Generate features
        features = self.feature_generator_func(data, regimes, regime_probs)
        
        # Get signals from expression
        signal_values = self.strategy_expression.evaluate(features)
        
        # Normalize signals
        if np.std(signal_values) > 0:
            signal_values = np.tanh(signal_values)
        else:
            signal_values = np.zeros_like(signal_values)
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=features.index)
        signals['signal'] = signal_values
        signals['position_size'] = signal_values * self.max_position_size
        
        # Apply money management
        signals = self.apply_money_management(signals, data.loc[features.index])
        
        return signals
