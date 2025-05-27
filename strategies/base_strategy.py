import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Provides common functionality for position sizing, risk management,
    and signal generation that can be inherited by specific strategies.
    """
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001, max_position_size=1.0):
        """
        Initialize base strategy.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital amount
        transaction_cost : float
            Transaction cost as decimal (e.g., 0.001 = 0.1%)
        max_position_size : float
            Maximum position size as fraction of capital (0.0 to 1.0)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.current_position = 0.0
        self.current_capital = initial_capital
        
    @abstractmethod
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
        pd.DataFrame : Trading signals with columns ['signal', 'position_size']
        """
        pass
    
    def calculate_position_size(self, signal_strength, volatility=None, current_price=None):
        """
        Calculate position size based on signal strength and risk management.
        
        Parameters:
        -----------
        signal_strength : float
            Signal strength (-1 to 1, where -1 is strong sell, 1 is strong buy)
        volatility : float, optional
            Current volatility for volatility-based sizing
        current_price : float, optional
            Current price for fixed dollar sizing
            
        Returns:
        --------
        float : Position size as fraction of capital
        """
        # Base position size from signal strength
        base_size = abs(signal_strength) * self.max_position_size
        
        # Volatility adjustment (if provided)
        if volatility is not None and volatility > 0:
            # Reduce position size for high volatility
            vol_adjustment = min(1.0, 0.02 / volatility)  # Target 2% volatility
            base_size *= vol_adjustment
        
        # Ensure position size is within limits
        position_size = np.clip(base_size, 0.0, self.max_position_size)
        
        # Apply sign from signal
        if signal_strength < 0:
            position_size *= -1
        
        return position_size
    
    def apply_money_management(self, signals, data, lookback_window=20):
        """
        Apply money management rules to trading signals.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Raw trading signals
        data : pd.DataFrame
            OHLCV data
        lookback_window : int
            Window for volatility calculation
            
        Returns:
        --------
        pd.DataFrame : Signals with money management applied
        """
        # Calculate rolling volatility
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=lookback_window).std()
        
        # Apply money management to each signal
        managed_signals = signals.copy()
        
        for idx in signals.index:
            if idx in volatility.index and not pd.isna(volatility[idx]):
                # Adjust position size based on volatility
                current_vol = volatility[idx]
                signal_strength = signals.loc[idx, 'signal']
                
                # Calculate volatility-adjusted position size
                adj_position_size = self.calculate_position_size(
                    signal_strength, 
                    current_vol, 
                    data.loc[idx, 'Close'] if idx in data.index else None
                )
                
                managed_signals.loc[idx, 'position_size'] = adj_position_size
        
        return managed_signals
    
    def calculate_stop_loss(self, entry_price, position_type, atr=None, stop_loss_pct=0.02):
        """
        Calculate stop loss level.
        
        Parameters:
        -----------
        entry_price : float
            Entry price for the position
        position_type : str
            'long' or 'short'
        atr : float, optional
            Average True Range for ATR-based stops
        stop_loss_pct : float
            Stop loss percentage (default: 2%)
            
        Returns:
        --------
        float : Stop loss price level
        """
        if atr is not None:
            # ATR-based stop loss
            stop_distance = atr * 2  # 2x ATR
        else:
            # Percentage-based stop loss
            stop_distance = entry_price * stop_loss_pct
        
        if position_type == 'long':
            return entry_price - stop_distance
        else:  # short position
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price, position_type, atr=None, risk_reward_ratio=2.0):
        """
        Calculate take profit level.
        
        Parameters:
        -----------
        entry_price : float
            Entry price for the position
        position_type : str
            'long' or 'short'
        atr : float, optional
            Average True Range for ATR-based targets
        risk_reward_ratio : float
            Risk-reward ratio (default: 2.0)
            
        Returns:
        --------
        float : Take profit price level
        """
        if atr is not None:
            # ATR-based take profit
            profit_distance = atr * 2 * risk_reward_ratio
        else:
            # Use default percentage
            profit_distance = entry_price * 0.02 * risk_reward_ratio
        
        if position_type == 'long':
            return entry_price + profit_distance
        else:  # short position
            return entry_price - profit_distance
    
    def validate_signal(self, signal, current_position, max_position_change=0.5):
        """
        Validate trading signal to prevent excessive position changes.
        
        Parameters:
        -----------
        signal : float
            Proposed position size
        current_position : float
            Current position size
        max_position_change : float
            Maximum allowed position change per signal
            
        Returns:
        --------
        float : Validated signal
        """
        position_change = signal - current_position
        
        # Limit position change
        if abs(position_change) > max_position_change:
            if position_change > 0:
                signal = current_position + max_position_change
            else:
                signal = current_position - max_position_change
        
        return signal
    
    def calculate_atr(self, data, window=14):
        """
        Calculate Average True Range.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        window : int
            ATR calculation window
            
        Returns:
        --------
        pd.Series : ATR values
        """
        high_low = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift(1))
        low_close_prev = abs(data['Low'] - data['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def filter_signals_by_market_hours(self, signals, market_open_hour=9, market_close_hour=16):
        """
        Filter signals to only trade during market hours.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Trading signals with datetime index
        market_open_hour : int
            Market opening hour (24-hour format)
        market_close_hour : int
            Market closing hour (24-hour format)
            
        Returns:
        --------
        pd.DataFrame : Filtered signals
        """
        if not isinstance(signals.index, pd.DatetimeIndex):
            return signals
        
        # Create market hours mask
        market_hours_mask = (
            (signals.index.hour >= market_open_hour) & 
            (signals.index.hour < market_close_hour) &
            (signals.index.weekday < 5)  # Monday=0, Friday=4
        )
        
        # Zero out signals outside market hours
        filtered_signals = signals.copy()
        filtered_signals.loc[~market_hours_mask, 'signal'] = 0
        filtered_signals.loc[~market_hours_mask, 'position_size'] = 0
        
        return filtered_signals
