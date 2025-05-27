import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.

    Provides realistic backtesting with transaction costs, slippage,
    and comprehensive performance metrics.
    """

    def __init__(self, initial_capital=100000, transaction_cost=0.001, slippage=0.0005):
        """
        Initialize backtest engine.

        Parameters:
        -----------
        initial_capital : float
            Starting capital amount
        transaction_cost : float
            Transaction cost as decimal (e.g., 0.001 = 0.1%)
        slippage : float
            Market impact/slippage as decimal
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.trades = []
        self.equity_curve = None

    def run_backtest(self, data, signals):
        """
        Run comprehensive backtest.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV market data
        signals : pd.DataFrame
            Trading signals with columns ['signal', 'position_size']

        Returns:
        --------
        dict : Backtest results including metrics, trades, and equity curve
        """
        # Align signals with data
        aligned_signals = signals.reindex(data.index, method='ffill').fillna(0)

        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position = 0.0
        position_value = 0.0

        # Results tracking
        equity_curve = []
        trades = []

        for i, (timestamp, row) in enumerate(data.iterrows()):
            if timestamp not in aligned_signals.index:
                continue

            current_price = row['Close']
            signal = aligned_signals.loc[timestamp, 'signal'] if 'signal' in aligned_signals.columns else 0
            position_size = aligned_signals.loc[timestamp, 'position_size'] if 'position_size' in aligned_signals.columns else signal

            # Calculate target position value
            target_position_value = portfolio_value * position_size
            target_shares = target_position_value / current_price if current_price > 0 else 0

            # Execute trade if position change is significant
            shares_to_trade = target_shares - position

            if abs(shares_to_trade) > 0.001:  # Minimum trade threshold
                # Calculate trade details
                trade_value = shares_to_trade * current_price

                # Apply slippage
                if shares_to_trade > 0:  # Buying
                    execution_price = current_price * (1 + self.slippage)
                else:  # Selling
                    execution_price = current_price * (1 - self.slippage)

                actual_trade_value = shares_to_trade * execution_price

                # Calculate transaction costs
                transaction_fee = abs(actual_trade_value) * self.transaction_cost

                # Update cash and position
                cash -= actual_trade_value + transaction_fee
                position = target_shares

                # Record trade
                trade_record = {
                    'timestamp': timestamp,
                    'type': 'BUY' if shares_to_trade > 0 else 'SELL',
                    'shares': abs(shares_to_trade),
                    'price': execution_price,
                    'value': abs(actual_trade_value),
                    'commission': transaction_fee,
                    'signal': signal,
                    'position_after': position
                }
                trades.append(trade_record)

            # Update portfolio value
            position_value = position * current_price
            portfolio_value = cash + position_value

            # Record equity curve point
            equity_point = {
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'position_value': position_value,
                'position_shares': position,
                'price': current_price
            }
            equity_curve.append(equity_point)

        # Convert to DataFrames
        self.equity_curve = pd.DataFrame(equity_curve).set_index('timestamp')
        trades_df = pd.DataFrame(trades).set_index('timestamp') if trades else pd.DataFrame()

        # Calculate performance metrics
        metrics = self._calculate_metrics(self.equity_curve, trades_df)

        return {
            'metrics': metrics,
            'equity_curve': self.equity_curve,
            'trades': trades_df,
            'returns': self._calculate_returns(self.equity_curve)
        }

    def _calculate_returns(self, equity_curve):
        """Calculate portfolio returns."""
        returns = equity_curve['portfolio_value'].pct_change().dropna()
        return returns

    def _calculate_metrics(self, equity_curve, trades_df):
        """
        Calculate comprehensive performance metrics.

        Parameters:
        -----------
        equity_curve : pd.DataFrame
            Portfolio equity curve
        trades_df : pd.DataFrame
            Individual trades

        Returns:
        --------
        dict : Performance metrics
        """
        if equity_curve.empty:
            return {}

        # Basic returns
        portfolio_returns = self._calculate_returns(equity_curve)

        # Total return
        total_return = (equity_curve['portfolio_value'].iloc[-1] / self.initial_capital) - 1

        # Annualized metrics
        trading_days = len(portfolio_returns)
        years = trading_days / 252  # Assuming 252 trading days per year

        if years > 0:
            annualized_return = (1 + total_return) ** (1/years) - 1
            annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        else:
            annualized_return = 0
            annualized_volatility = 0

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

        # Maximum drawdown
        running_max = equity_curve['portfolio_value'].expanding().max()
        drawdown = (equity_curve['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate and average win/loss
        if not trades_df.empty and len(trades_df) > 1:
            # Calculate trade P&L
            trade_pnl = []
            for i in range(1, len(trades_df)):
                if trades_df.iloc[i]['type'] == 'SELL':
                    # Find corresponding buy
                    prev_trades = trades_df.iloc[:i]
                    last_buy = prev_trades[prev_trades['type'] == 'BUY'].iloc[-1] if not prev_trades[prev_trades['type'] == 'BUY'].empty else None

                    if last_buy is not None:
                        pnl = (trades_df.iloc[i]['price'] - last_buy['price']) * trades_df.iloc[i]['shares']
                        trade_pnl.append(pnl)

            if trade_pnl:
                winning_trades = [pnl for pnl in trade_pnl if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnl if pnl <= 0]

                win_rate = len(winning_trades) / len(trade_pnl) if trade_pnl else 0
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0



        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades_df)
        }

    def calculate_rolling_metrics(self, window_days=252):
        """
        Calculate rolling performance metrics.

        Parameters:
        -----------
        window_days : int
            Rolling window in days

        Returns:
        --------
        pd.DataFrame : Rolling metrics
        """
        if self.equity_curve is None or self.equity_curve.empty:
            return pd.DataFrame()

        returns = self._calculate_returns(self.equity_curve)

        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling Sharpe ratio
        rolling_return = returns.rolling(window=window_days).mean() * 252
        rolling_vol = returns.rolling(window=window_days).std() * np.sqrt(252)
        rolling_metrics['sharpe_ratio'] = rolling_return / rolling_vol

        # Rolling max drawdown
        rolling_max = self.equity_curve['portfolio_value'].rolling(window=window_days).max()
        rolling_drawdown = (self.equity_curve['portfolio_value'] - rolling_max) / rolling_max
        rolling_metrics['max_drawdown'] = rolling_drawdown.rolling(window=window_days).min()

        # Rolling volatility
        rolling_metrics['volatility'] = rolling_vol

        return rolling_metrics

    def generate_trade_analysis(self):
        """
        Generate detailed trade analysis.

        Returns:
        --------
        dict : Trade analysis results
        """
        if self.equity_curve is None:
            return {}

        returns = self._calculate_returns(self.equity_curve)

        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Yearly returns
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)

        # Drawdown analysis
        running_max = self.equity_curve['portfolio_value'].expanding().max()
        drawdown = (self.equity_curve['portfolio_value'] - running_max) / running_max

        # Drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None

        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (> 1%)
                in_drawdown = True
                start_date = date
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_date:
                    duration = (date - start_date).days
                    max_dd_in_period = drawdown[start_date:date].min()
                    drawdown_periods.append({
                        'start': start_date,
                        'end': date,
                        'duration_days': duration,
                        'max_drawdown': max_dd_in_period
                    })

        return {
            'monthly_returns': monthly_returns,
            'yearly_returns': yearly_returns,
            'drawdown_periods': drawdown_periods,
            'longest_drawdown': max(drawdown_periods, key=lambda x: x['duration_days']) if drawdown_periods else None,
            'deepest_drawdown': min(drawdown_periods, key=lambda x: x['max_drawdown']) if drawdown_periods else None
        }

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
                    'mean': float(durations.mean()),
                    'median': float(durations.median()),
                    'max': int(durations.max()),
                    'min': int(durations.min()),
                    'count': len(durations)
                }
            return None
        except Exception as e:
            print(f"Error calculating regime durations: {e}")
            return None