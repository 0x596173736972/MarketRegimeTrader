import pandas as pd
import numpy as np
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Utility class for loading and preprocessing OHLCV data.
    
    Supports CSV and Parquet formats with automatic data validation
    and preprocessing for financial time series analysis.
    """
    
    def __init__(self):
        """Initialize data loader."""
        self.required_columns = ['Open', 'High', 'Low', 'Close']
        self.optional_columns = ['Volume', 'Adj Close']
        
    def load_data(self, file_path_or_buffer, file_type=None):
        """
        Load OHLCV data from file.
        
        Parameters:
        -----------
        file_path_or_buffer : str or file-like object
            Path to file or file buffer
        file_type : str, optional
            File type ('csv' or 'parquet'). Auto-detected if None
            
        Returns:
        --------
        pd.DataFrame : Processed OHLCV data with datetime index
        """
        # Determine file type
        if file_type is None:
            if hasattr(file_path_or_buffer, 'name'):
                filename = file_path_or_buffer.name
                if filename.endswith('.parquet'):
                    file_type = 'parquet'
                else:
                    file_type = 'csv'
            else:
                file_type = 'csv'  # Default to CSV
        
        # Load data based on file type
        if file_type == 'parquet':
            data = pd.read_parquet(file_path_or_buffer)
        else:  # CSV
            data = self._load_csv(file_path_or_buffer)
        
        # Process and validate data
        processed_data = self._process_data(data)
        
        return processed_data
    
    def _load_csv(self, file_path_or_buffer):
        """
        Load CSV file with flexible parsing.
        
        Parameters:
        -----------
        file_path_or_buffer : str or file-like object
            CSV file path or buffer
            
        Returns:
        --------
        pd.DataFrame : Raw loaded data
        """
        # Try different parsing strategies including semicolon separators
        parsing_strategies = [
            # Strategy 1: Standard CSV with header
            {'header': 0, 'index_col': 0, 'parse_dates': True},
            # Strategy 2: CSV with Date column
            {'header': 0, 'parse_dates': ['Date'], 'index_col': 'Date'},
            # Strategy 3: CSV with Datetime column
            {'header': 0, 'parse_dates': ['Datetime'], 'index_col': 'Datetime'},
            # Strategy 4: CSV with timestamp column
            {'header': 0, 'parse_dates': ['Timestamp'], 'index_col': 'Timestamp'},
            # Strategy 5: Semicolon separated with Date column
            {'header': 0, 'sep': ';', 'parse_dates': ['Date'], 'index_col': 'Date'},
            # Strategy 6: Semicolon separated with no index parsing
            {'header': 0, 'sep': ';'},
            # Strategy 7: No index parsing
            {'header': 0}
        ]
        
        data = None
        for i, strategy in enumerate(parsing_strategies):
            try:
                # Reset file pointer if it's a file object
                if hasattr(file_path_or_buffer, 'seek'):
                    file_path_or_buffer.seek(0)
                
                data = pd.read_csv(file_path_or_buffer, **strategy)
                
                # Check if we got valid data
                if not data.empty and len(data.columns) >= 4:
                    break
                    
            except Exception as e:
                if i == len(parsing_strategies) - 1:  # Last strategy failed
                    raise ValueError(f"Failed to parse CSV file: {str(e)}")
                continue
        
        if data is None or data.empty:
            raise ValueError("No valid data found in file")
        
        return data
    
    def _process_data(self, data):
        """
        Process and validate loaded data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw loaded data
            
        Returns:
        --------
        pd.DataFrame : Processed and validated data
        """
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Standardize column names
        processed_data = self._standardize_columns(processed_data)
        
        # Handle date index
        processed_data = self._process_datetime_index(processed_data)
        
        # Validate required columns
        self._validate_columns(processed_data)
        
        # Clean and validate data
        processed_data = self._clean_data(processed_data)
        
        # Sort by date
        processed_data = processed_data.sort_index()
        
        # Add derived columns if missing
        processed_data = self._add_derived_columns(processed_data)
        
        return processed_data
    
    def _standardize_columns(self, data):
        """
        Standardize column names to expected format.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame : Data with standardized column names
        """
        # Common column name mappings
        column_mappings = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adjusted close': 'Adj Close',
            'adj_close': 'Adj Close',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }
        
        # Rename columns
        new_columns = {}
        for col in data.columns:
            col_lower = col.lower().strip()
            if col_lower in column_mappings:
                new_columns[col] = column_mappings[col_lower]
        
        if new_columns:
            data = data.rename(columns=new_columns)
        
        return data
    
    def _process_datetime_index(self, data):
        """
        Process datetime index from various formats.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame : Data with proper datetime index
        """
        # If index is already datetime, we're good
        if isinstance(data.index, pd.DatetimeIndex):
            return data
        
        # Look for date columns
        date_columns = ['Date', 'Datetime', 'Timestamp', 'Time']
        for col in date_columns:
            if col in data.columns:
                try:
                    data[col] = pd.to_datetime(data[col])
                    data = data.set_index(col)
                    return data
                except:
                    continue
        
        # Try to convert index to datetime
        try:
            data.index = pd.to_datetime(data.index)
            return data
        except:
            pass
        
        # If we have a numeric index, create a date range
        if len(data) > 0:
            # Create a simple date range starting from 2020
            date_range = pd.date_range(
                start='2020-01-01',
                periods=len(data),
                freq='D'
            )
            data.index = date_range
        
        return data
    
    def _validate_columns(self, data):
        """
        Validate that required columns are present.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to validate
            
        Raises:
        -------
        ValueError : If required columns are missing
        """
        missing_columns = []
        for col in self.required_columns:
            if col not in data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            available_columns = list(data.columns)
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_columns}"
            )
    
    def _clean_data(self, data):
        """
        Clean and validate financial data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame : Cleaned data
        """
        # Remove rows with missing OHLC data
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        data = data.dropna(subset=ohlc_columns)
        
        # Validate OHLC relationships
        data = self._validate_ohlc_data(data)
        
        # Remove rows with non-positive prices
        for col in ohlc_columns:
            data = data[data[col] > 0]
        
        # Handle volume data
        if 'Volume' in data.columns:
            # Replace negative volumes with 0
            data.loc[data['Volume'] < 0, 'Volume'] = 0
            # Fill missing volumes with 0
            data['Volume'] = data['Volume'].fillna(0)
        
        # Remove duplicate dates
        data = data[~data.index.duplicated(keep='first')]
        
        return data
    
    def _validate_ohlc_data(self, data):
        """
        Validate OHLC relationships and fix inconsistencies.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame : Validated data
        """
        # Check that High >= Low
        invalid_hl = data['High'] < data['Low']
        if invalid_hl.any():
            # Swap High and Low for invalid rows
            data.loc[invalid_hl, ['High', 'Low']] = data.loc[invalid_hl, ['Low', 'High']].values
        
        # Check that High >= Open, Close and Low <= Open, Close
        data['High'] = np.maximum.reduce([data['High'], data['Open'], data['Close']])
        data['Low'] = np.minimum.reduce([data['Low'], data['Open'], data['Close']])
        
        return data
    
    def _add_derived_columns(self, data):
        """
        Add derived columns if missing.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame : Data with derived columns
        """
        # Add Volume if missing
        if 'Volume' not in data.columns:
            data['Volume'] = 1000000  # Default volume
        
        # Add Adjusted Close if missing
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        
        return data
    
    def validate_data_quality(self, data, min_periods=100):
        """
        Validate data quality and provide quality report.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to validate
        min_periods : int
            Minimum number of periods required
            
        Returns:
        --------
        dict : Data quality report
        """
        quality_report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check minimum periods
        if len(data) < min_periods:
            quality_report['warnings'].append(
                f"Data has only {len(data)} periods, minimum recommended is {min_periods}"
            )
        
        # Check for gaps in data
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for large gaps
            time_diffs = data.index.to_series().diff()
            median_diff = time_diffs.median()
            large_gaps = time_diffs > median_diff * 3
            
            if large_gaps.any():
                quality_report['warnings'].append(
                    f"Found {large_gaps.sum()} large time gaps in data"
                )
        
        # Check for unusual price movements
        returns = data['Close'].pct_change()
        extreme_returns = abs(returns) > 0.50  # 50% single-day moves
        
        if extreme_returns.any():
            quality_report['warnings'].append(
                f"Found {extreme_returns.sum()} extreme price movements (>50%)"
            )
        
        # Statistical summary
        quality_report['statistics'] = {
            'total_periods': len(data),
            'date_range': (data.index.min(), data.index.max()) if len(data) > 0 else (None, None),
            'price_range': (data['Close'].min(), data['Close'].max()),
            'average_volume': data['Volume'].mean() if 'Volume' in data.columns else None,
            'missing_data_pct': data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        }
        
        return quality_report
    
    def resample_data(self, data, frequency='1D', aggregation_rules=None):
        """
        Resample data to different frequency.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        frequency : str
            Target frequency (e.g., '1D', '1W', '1M')
        aggregation_rules : dict, optional
            Custom aggregation rules
            
        Returns:
        --------
        pd.DataFrame : Resampled data
        """
        if aggregation_rules is None:
            aggregation_rules = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
        
        # Only aggregate columns that exist
        valid_rules = {col: rule for col, rule in aggregation_rules.items() 
                      if col in data.columns}
        
        resampled_data = data.resample(frequency).agg(valid_rules)
        
        # Remove periods with no data
        resampled_data = resampled_data.dropna()
        
        return resampled_data
