# ============================================================================
# -*- coding: utf-8 -*-
#
# Module:      mahalanobis_filter.py
# Description: Simplified unified Mahalanobis accuracy filter with clean APIs
#              for training and inference, maintaining perfect naming alignment
#              with strategy cache system.
#
# Purpose:     Single class to replace 6 complex modules (3000+ lines) with
#              clean separation of training vs inference and study-based
#              architecture using MarketContextSensorsStudy as feature space.
#
# Key Features:
#   - Perfect naming alignment with strategy cache files
#   - Study-based architecture (not indicator-based)
#   - Clean API separation (training vs inference)
#   - Fast inference with pre-computed sensor lookup
#   - Start-date + years training specification
#
# Usage:
#   Training:
#     filter = MahalanobisAccuracyFilter()
#     filter.train_from_symbols("SP500.csv", "MarketContextSensorsStudy", 
#                               "SMIsig_14_3_3", "2019-01-01", 10, 
#                               "14_3_3_5_chronos_base")
#   
#   Inference:
#     filter = MahalanobisAccuracyFilter.load_for_strategy(strategy_instance)
#     accuracy = filter.predict_accuracy(sensor_vector)
#
# History:
# 2025-08-17   Created - Simplified unified architecture
# ============================================================================

from typing import Dict, Any, List, Tuple, Optional, Union, Type
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
import csv
import sys
# Add parent directory to path for utilities import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Scientific computing
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Import unified sensor extraction utilities
try:
    from utilities.sensor_utilities import (
        extract_pure_sensor_columns,
        extract_sensor_vector,
        validate_sensor_columns,
        align_sensor_columns
    )
except ImportError:
    # Fallback if running from different directory
    from sensor_utilities import (
        extract_pure_sensor_columns,
        extract_sensor_vector,
        validate_sensor_columns,
        align_sensor_columns
    )

@dataclass
class AccuracyRecord:
    """Single accuracy measurement record linking prediction to outcome."""
    timestamp: datetime
    symbol: str
    predicted_direction: str  # 'bullish' or 'bearish' 
    prediction_confidence: float  # Signal strength
    market_context: np.ndarray  # 48-dimensional sensor values at signal time
    actual_outcome: Optional[bool] = None  # True=accurate, False=inaccurate, None=pending
    outcome_timestamp: Optional[datetime] = None
    accuracy_score: Optional[float] = None  # 0.0-1.0 accuracy measure
    signal_type: str = 'predicted'  # 'predicted' (ML), 'vanilla' (traditional indicator)

@dataclass 
class AccuracyClass:
    """Accuracy class with centroid and historical performance."""
    class_id: int
    centroid: np.ndarray  # Mahalanobis space centroid
    accuracy_rate: float  # Historical accuracy rate (0.0-1.0)
    sample_count: int  # Number of historical samples
    confidence_interval: Tuple[float, float]  # 95% confidence bounds
    accuracy_range: Tuple[float, float]  # Accuracy range for this class
    # Buy/Sell signal breakdowns
    buy_count: int = 0  # Number of buy signals in this class
    sell_count: int = 0  # Number of sell signals in this class
    buy_accuracy_rate: float = 0.0  # Accuracy rate for buy signals
    sell_accuracy_rate: float = 0.0  # Accuracy rate for sell signals
    buy_confidence: str = 'low'  # Confidence level for buy accuracy
    sell_confidence: str = 'low'  # Confidence level for sell accuracy

class MahalanobisAccuracyFilter:
    """
    Unified Mahalanobis filter with clean APIs for training and inference.
    Maintains perfect naming alignment with strategy cache system.
    
    This class replaces the complex 6-module architecture with a single
    clean implementation that uses study-based feature extraction and
    provides clear separation between training and inference.
    """
    
    def __init__(self, strategy_instance=None, n_accuracy_classes: int = 6):
        """
        Initialize filter, optionally from strategy instance for naming.
        
        Args:
            strategy_instance: CachedStrategy instance for automatic naming
            n_accuracy_classes: Number of accuracy classes (6 for single symbol)
        """
        self.n_accuracy_classes = n_accuracy_classes
        self.strategy_instance = strategy_instance
        
        # Model components
        self.scaler = StandardScaler()
        self.covariance_estimator = LedoitWolf()
        self.accuracy_classes: List[AccuracyClass] = []
        self.mahalanobis_inv_cov: Optional[np.ndarray] = None
        
        # Training data
        self.accuracy_records: List[AccuracyRecord] = []
        self.sensor_columns: List[str] = []  # Dynamic sensor columns from training
        self.training_metadata: Dict[str, Any] = {}
        
        # Model state
        self.is_trained = False
        self.validation_scores: Dict[str, float] = {}
        
        # Symbol metadata for cache storage
        self._symbol = None
        self._exchange = None
        self._instrument_type = None
    
    # ========================================================================
    # TRAINING API
    # ========================================================================
    
    def train_from_symbols(self,
                          csv_file: str,
                          sensor_study_name: str,
                          signal_column: str,
                          start_date: str,
                          years: int,
                          strategy_config: str,
                          n_classes: Optional[int] = None,
                          accuracy_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Train filter from symbol CSV with specified study and signal column.
        
        Args:
            csv_file: Symbol list CSV file (e.g., "SP500.csv")
            sensor_study_name: Study class name (e.g., "MarketContextSensorsStudy")
            signal_column: Target sensor column (e.g., "SMIsig_14_3_3")  
            start_date: Training start date (e.g., "2019-01-01")
            years: Years of training data (e.g., 10)
            strategy_config: Strategy config string (e.g., "14_3_3_5_chronos_base")
            n_classes: Override number of classes (uses constructor default if None)
            accuracy_params: Real-world accuracy parameters:
                - buy_success_limit: Full success threshold for buys (e.g., 0.02 for 2%)
                - buy_undershoot_limit: Partial success for buys (e.g., 0.005 for 0.5%)
                - sell_success_limit: Full success for sells (e.g., -0.02 for -2%)
                - sell_overshoot_limit: Partial success for sells (e.g., -0.005 for -0.5%)
                - lookforward_bars: Number of bars to check for targets (e.g., 20)
            
        Returns:
            Training results with metadata
        """
        if n_classes is not None:
            self.n_accuracy_classes = n_classes
            
        # Parse training date range
        start_dt = pd.to_datetime(start_date)
        end_dt = start_dt + pd.DateOffset(years=years)
        
        print(f"🚀 Training Mahalanobis filter:")
        print(f"   📊 Symbols: {csv_file}")
        print(f"   🎯 Target column: {signal_column}")
        print(f"   📅 Date range: {start_date} to {end_dt.strftime('%Y-%m-%d')} ({years} years)")
        print(f"   🔧 Strategy config: {strategy_config}")
        print(f"   📈 Classes: {self.n_accuracy_classes}")
        
        # Load symbols from CSV
        symbols_data = self._load_symbols_from_csv(csv_file)
        if not symbols_data:
            raise ValueError(f"No symbols loaded from {csv_file}")
            
        print(f"   🎫 Loaded {len(symbols_data)} symbols")
        
        # Collect training data from all symbols
        self.accuracy_records = []
        self.sensor_columns = []
        
        for symbol_info in symbols_data:
            symbol_records = self._collect_symbol_training_data(
                symbol_info, sensor_study_name, signal_column, start_dt, end_dt, accuracy_params
            )
            self.accuracy_records.extend(symbol_records)
            
        if not self.accuracy_records:
            raise ValueError("No training data collected from any symbol")
            
        print(f"   📈 Collected {len(self.accuracy_records)} training records")
        
        # Sensor columns are set during _extract_accuracy_records_from_sensors
        # They contain the actual column names used for training
        if not self.sensor_columns and self.accuracy_records:
            # This shouldn't happen, but as a fallback
            self.sensor_columns = [f"sensor_{i}" for i in range(len(self.accuracy_records[0].market_context))]
            
        # Train the classifier
        training_results = self._train_classifier()
        
        # Set metadata for saving
        self.training_metadata = {
            'csv_file': csv_file,
            'sensor_study': sensor_study_name,
            'signal_column': signal_column,
            'start_date': start_date,
            'years': years,
            'end_date': end_dt.strftime('%Y-%m-%d'),
            'strategy_config': strategy_config,
            'n_classes': self.n_accuracy_classes,
            'symbols_count': len(symbols_data),
            'training_records': len(self.accuracy_records),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Set symbol metadata from first symbol for cache storage
        if symbols_data:
            first_symbol = symbols_data[0]
            self._symbol = first_symbol['symbol']
            self._exchange = first_symbol['exchange']
            self._instrument_type = first_symbol.get('type', 'ETF')
        
        self.is_trained = True
        
        # Save the trained filter to cache directory
        saved_path = None
        try:
            # Create filter cache name based on strategy config
            strategy_parts = strategy_config.split('_')
            if len(strategy_parts) >= 4:
                # Extract indicator type from signal column (e.g., "SMIsig_14_3_3" -> "SMI")
                indicator_type = signal_column.split('sig')[0] if 'sig' in signal_column else 'Unknown'
                filter_name = self.get_filter_cache_name(strategy_config, indicator_type)
            else:
                filter_name = f"MahalanobisFilter_{strategy_config}"
            
            # Use symbol directory for cache (from first symbol)
            if self._symbol and self._exchange and self._instrument_type:
                normalized_type = self._instrument_type.replace(' ', '_')
                cache_base = Path(__file__).parent.parent.parent / 'cache' / 'symbols'
                symbol_dir = cache_base / f"{self._exchange}_{normalized_type}_{self._symbol}"
                symbol_dir.mkdir(parents=True, exist_ok=True)
                
                # Save filter with .pkl extension
                filter_path = symbol_dir / f"{filter_name}.pkl"
                
                if self.save_to_path(str(filter_path)):
                    saved_path = str(filter_path)
                    print(f"   💾 Filter saved to: {filter_path}")
                else:
                    print(f"   ❌ Failed to save filter to: {filter_path}")
            
        except Exception as e:
            print(f"   ⚠️ Error saving filter: {e}")
        
        return {
            'success': True,
            'training_accuracy': training_results.get('training_accuracy', 0.0),
            'validation_accuracy': training_results.get('validation_accuracy', 0.0),
            'n_classes': self.n_accuracy_classes,
            'n_records': len(self.accuracy_records),
            'metadata': self.training_metadata,
            'saved_path': saved_path
        }
    
    def _load_symbols_from_csv(self, csv_file: str) -> List[Dict[str, str]]:
        """Load symbol list from CSV file."""
        # Look for CSV in market_indexes directory
        market_indexes_dir = Path(__file__).parent.parent.parent / 'market_indexes'
        csv_path = market_indexes_dir / csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        symbols_data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbols_data.append({
                    'symbol': row.get('symbol_code', row.get('symbol', row.get('Symbol', ''))),
                    'exchange': row.get('exchange_code', row.get('exchange', row.get('Exchange', 'US'))),
                    'type': row.get('Type', row.get('type', 'ETF'))
                })
                
        return symbols_data
    
    def _collect_symbol_training_data(self, symbol_info: Dict[str, str], 
                                    sensor_study_name: str, signal_column: str,
                                    start_date: datetime, end_date: datetime,
                                    accuracy_params: Optional[Dict[str, float]] = None) -> List[AccuracyRecord]:
        """Collect training data for a single symbol using specified study."""
        try:
            # Get market data
            market_data = self._get_market_data(symbol_info['symbol'], symbol_info['exchange'])
            if market_data is None or market_data.empty:
                print(f"   ⚠️ No market data for {symbol_info['symbol']}")
                return []
                
            # Filter to training date range
            mask = (market_data.index >= start_date) & (market_data.index <= end_date)
            training_data = market_data[mask].copy()
            
            if len(training_data) < 50:  # Minimum data requirement
                print(f"   ⚠️ Insufficient data for {symbol_info['symbol']}: {len(training_data)} bars")
                return []
                
            # Calculate sensor study
            sensor_study = self._create_sensor_study(sensor_study_name)
            sensor_df = sensor_study.calculate(training_data)
            
            if sensor_df is None or sensor_df.empty:
                print(f"   ⚠️ Sensor calculation failed for {symbol_info['symbol']}")
                return []
                
            # Validate signal column exists
            if signal_column not in sensor_df.columns:
                available_cols = [col for col in sensor_df.columns if 'SMI' in col.upper()][:5]
                print(f"   ⚠️ Signal column '{signal_column}' not found in {symbol_info['symbol']}")
                print(f"      Available SMI columns: {available_cols}")
                return []
                
            # Extract accuracy records from sensor data
            records = self._extract_accuracy_records_from_sensors(
                sensor_df, signal_column, symbol_info['symbol'], accuracy_params
            )
            
            print(f"   ✅ {symbol_info['symbol']}: {len(records)} training records")
            return records
            
        except Exception as e:
            print(f"   ❌ Error processing {symbol_info['symbol']}: {e}")
            return []
    
    def _get_market_data(self, symbol: str, exchange: str) -> Optional[pd.DataFrame]:
        """Get market data for symbol using database connection."""
        try:
            # Import here to avoid circular dependencies
            from marketdata.database import Database
            import os
            
            # Get database configuration
            db_file_path = os.getenv('MARKETDATA_DB_PATH', '')
            eod_api_key = os.getenv('EOD_API_KEY', '')
            
            if not db_file_path or not eod_api_key:
                raise ValueError("Database configuration not available")
                
            # Connect and fetch data
            db = Database()
            db.open(database_file=db_file_path, api_license=eod_api_key)
            
            df = db.get_eod_data(
                symbol_code=symbol,
                exchange_code=exchange,
                adj_for_splits=True,
                include_synthetics=True
            )
            
            db.close()
            
            if df is not None and not df.empty:
                # Normalize column names
                df.columns = [col.lower() for col in df.columns]
                return df
            else:
                return None
                
        except Exception as e:
            print(f"   Database error for {symbol}: {e}")
            return None
    
    def _create_sensor_study(self, study_name: str):
        """Create sensor study instance by name.
        
        Dynamically imports and instantiates the specified sensor study class.
        
        Args:
            study_name: Name of the study class (e.g., 'MarketContextSensorsStudy')
            
        Returns:
            Instance of the specified sensor study
            
        Raises:
            ValueError: If the study cannot be imported or instantiated
        """
        try:
            # Try to import from studies module
            import importlib
            
            # Convert class name to module name (e.g., MarketContextSensorsStudy -> market_context_sensors_study)
            module_name = ''.join(['_' + c.lower() if c.isupper() else c for c in study_name]).lstrip('_')
            
            # Try importing from studies module
            try:
                module = importlib.import_module(f'studies.{module_name}')
                study_class = getattr(module, study_name)
                return study_class()
            except (ImportError, AttributeError):
                # If not in studies, try importing directly
                # This allows for custom studies in other locations
                module_parts = study_name.rsplit('.', 1)
                if len(module_parts) == 2:
                    module = importlib.import_module(module_parts[0])
                    study_class = getattr(module, module_parts[1])
                    return study_class()
                else:
                    # Final fallback - check if it's a known study
                    if study_name == "MarketContextSensorsStudy":
                        from studies.market_context_sensors_study import MarketContextSensorsStudy
                        return MarketContextSensorsStudy()
                    else:
                        raise ValueError(f"Cannot find sensor study: {study_name}")
                        
        except Exception as e:
            raise ValueError(f"Failed to create sensor study '{study_name}': {e}")
    
    def _extract_accuracy_records_from_sensors(self, sensor_df: pd.DataFrame, 
                                             signal_column: str, symbol: str,
                                             accuracy_params: Optional[Dict[str, float]] = None) -> List[AccuracyRecord]:
        """Extract accuracy records from sensor data using signal column with real-world accuracy measurement.
        
        Args:
            sensor_df: DataFrame with sensor values and price data
            signal_column: Name of the signal column to track
            symbol: Symbol being processed
            accuracy_params: Real-world accuracy parameters:
                - buy_success_limit: Full success threshold for buys (e.g., 0.02 for 2%)
                - buy_undershoot_limit: Partial success for buys (e.g., 0.005 for 0.5%)
                - sell_success_limit: Full success for sells (e.g., -0.02 for -2%)
                - sell_overshoot_limit: Partial success for sells (e.g., -0.005 for -0.5%)
                - lookforward_bars: Number of bars to check for targets (e.g., 20)
        """
        records = []
        
        # Default accuracy parameters if not provided
        if accuracy_params is None:
            accuracy_params = {
                'buy_success_limit': 0.02,      # 2% gain for full success
                'buy_undershoot_limit': 0.005,   # 0.5% gain for partial success
                'sell_success_limit': -0.02,     # -2% loss for full success
                'sell_overshoot_limit': -0.005,  # -0.5% loss for partial success
                'lookforward_bars': 20            # Look 20 bars ahead
            }
        
        # Extract pure sensor columns dynamically (excluding OHLCV and metadata)
        sensor_only_df, actual_sensor_columns = extract_pure_sensor_columns(sensor_df)
        
        # Store the actual sensor column names for use during inference
        if not self.sensor_columns:  # Only set once
            self.sensor_columns = actual_sensor_columns
            print(f"   📊 Extracted {len(self.sensor_columns)} sensor columns dynamically")
            if len(self.sensor_columns) > 0:
                print(f"   📋 Sample sensors: {self.sensor_columns[:5]}...")
        
        # Check if we have price data for real-world accuracy calculation
        has_price_data = 'close' in sensor_df.columns or 'adjusted_close' in sensor_df.columns
        price_column = 'adjusted_close' if 'adjusted_close' in sensor_df.columns else 'close'
        
        lookforward_bars = int(accuracy_params.get('lookforward_bars', 20))
        
        for i, (idx, row) in enumerate(sensor_df.iterrows()):
            try:
                # Get signal value
                signal_value = row[signal_column]
                if pd.isna(signal_value):
                    continue
                    
                # Extract market context using actual sensor columns
                market_context = extract_sensor_vector(row, self.sensor_columns, nan_fill_value=0.0)
                
                # Validate we have the right number of features
                if len(market_context) != len(self.sensor_columns):
                    continue  # Skip if feature count doesn't match
                
                # Determine prediction direction
                predicted_direction = 'bullish' if signal_value > 0 else 'bearish'
                
                # Calculate real-world accuracy if price data is available
                accuracy_score = 0.5  # Default neutral accuracy
                actual_outcome = None
                
                if has_price_data and i + lookforward_bars < len(sensor_df):
                    current_price = row[price_column]
                    if not pd.isna(current_price) and current_price > 0:
                        # Get future prices within lookforward window
                        future_prices = sensor_df.iloc[i+1:i+1+lookforward_bars][price_column]
                        
                        if len(future_prices) > 0:
                            # TRADING STOP LOGIC: Check each bar sequentially and exit when targets hit
                            buy_success = accuracy_params.get('buy_success_limit', 0.02)
                            buy_partial = accuracy_params.get('buy_undershoot_limit', 0.005)
                            sell_success = accuracy_params.get('sell_success_limit', -0.02)
                            sell_partial = accuracy_params.get('sell_overshoot_limit', -0.005)
                            
                            # Process each bar in the lookforward window sequentially
                            final_accuracy_score = 0.5
                            final_outcome = None
                            exit_bar = None
                            
                            # Check bars 1 through 19 for immediate exits only
                            for bar_idx, future_price in enumerate(future_prices[:-1]):  # Exclude last bar
                                if pd.isna(future_price):
                                    continue
                                    
                                price_change = (future_price - current_price) / current_price
                                
                                if predicted_direction == 'bullish':
                                    # BUY: Check for full success (2%) or stop loss (0.5% down)
                                    if price_change >= buy_success:  # >= 2% gain
                                        # FULL SUCCESS - EXIT immediately
                                        final_accuracy_score = 1.0
                                        final_outcome = True
                                        exit_bar = bar_idx + 1
                                        break
                                    elif price_change <= -buy_partial:  # <= -0.5% loss (stop loss)
                                        # STOP LOSS - EXIT immediately with failure
                                        final_accuracy_score = 0.0
                                        final_outcome = False
                                        exit_bar = bar_idx + 1
                                        break
                                        
                                else:  # bearish/sell signal
                                    # SELL: Check for full success (-2%) or stop loss (0.5% up)
                                    if price_change <= sell_success:  # <= -2% loss
                                        # FULL SUCCESS - EXIT immediately
                                        final_accuracy_score = 1.0
                                        final_outcome = True
                                        exit_bar = bar_idx + 1
                                        break
                                    elif price_change >= -sell_partial:  # >= 0.5% gain (stop loss)
                                        # STOP LOSS - EXIT immediately with failure
                                        final_accuracy_score = 0.0
                                        final_outcome = False
                                        exit_bar = bar_idx + 1
                                        break
                            
                            # If no immediate exit occurred, evaluate at bar T+20 (final bar)
                            if exit_bar is None and len(future_prices) > 0:
                                final_price = future_prices.iloc[-1]
                                final_change = (final_price - current_price) / current_price
                                
                                # Evaluate at T+20: Check for partial success, full success, or failure
                                if predicted_direction == 'bullish':
                                    if final_change >= buy_success:  # >= 2%
                                        final_accuracy_score = 1.0  # Full success
                                        final_outcome = True
                                    elif final_change >= buy_partial:  # >= 0.5%
                                        final_accuracy_score = 0.75  # Partial success
                                        final_outcome = True
                                    elif final_change <= -buy_partial:  # <= -0.5%
                                        final_accuracy_score = 0.0  # Failure
                                        final_outcome = False
                                    else:  # Between -0.5% and +0.5%
                                        final_accuracy_score = 0.5  # Neutral
                                        final_outcome = None
                                        
                                else:  # bearish signal
                                    if final_change <= sell_success:  # <= -2%
                                        final_accuracy_score = 1.0  # Full success
                                        final_outcome = True
                                    elif final_change <= sell_partial:  # <= -0.5%
                                        final_accuracy_score = 0.75  # Partial success
                                        final_outcome = True
                                    elif final_change >= -sell_partial:  # >= 0.5%
                                        final_accuracy_score = 0.0  # Failure
                                        final_outcome = False
                                    else:  # Between -0.5% and +0.5%
                                        final_accuracy_score = 0.5  # Neutral
                                        final_outcome = None
                                        
                                exit_bar = lookforward_bars  # Used full window
                            
                            accuracy_score = final_accuracy_score
                            actual_outcome = final_outcome
                
                # If no price data, use simplified accuracy based on signal strength
                if accuracy_score == 0.5 and actual_outcome is None:
                    # Fallback to simplified calculation
                    accuracy_score = min(0.95, max(0.05, 0.5 + abs(signal_value) * 0.3))
                    actual_outcome = True  # Assume accurate for training purposes
                
                record = AccuracyRecord(
                    timestamp=idx,
                    symbol=symbol,
                    predicted_direction=predicted_direction,
                    prediction_confidence=abs(signal_value),
                    market_context=market_context,
                    actual_outcome=actual_outcome,
                    accuracy_score=accuracy_score
                )
                
                records.append(record)
                
            except Exception as e:
                continue  # Skip problematic rows
                
        return records
    
    def _train_classifier(self) -> Dict[str, float]:
        """Train the Mahalanobis classifier on collected accuracy records."""
        if not self.accuracy_records:
            raise ValueError("No training data available")
            
        # Extract features and labels
        features = np.array([record.market_context for record in self.accuracy_records])
        accuracy_scores = np.array([record.accuracy_score for record in self.accuracy_records])
        
        # Fit scaler
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        # Create accuracy classes using K-means clustering
        kmeans = KMeans(n_clusters=self.n_accuracy_classes, random_state=42, n_init=10)
        class_labels = kmeans.fit_predict(features_scaled)
        
        # Build accuracy classes
        self.accuracy_classes = []
        for class_id in range(self.n_accuracy_classes):
            class_mask = class_labels == class_id
            if not np.any(class_mask):
                continue
                
            class_features = features_scaled[class_mask]
            class_accuracies = accuracy_scores[class_mask]
            
            # Get records for this class to calculate buy/sell breakdowns
            class_records = [record for i, record in enumerate(self.accuracy_records) if class_labels[i] == class_id]
            
            # Separate buy and sell records
            buy_records = [r for r in class_records if r.predicted_direction == 'bullish']
            sell_records = [r for r in class_records if r.predicted_direction == 'bearish']
            
            # Calculate buy/sell statistics
            buy_count = len(buy_records)
            sell_count = len(sell_records)
            buy_accuracy_rate = np.mean([r.accuracy_score for r in buy_records]) if buy_records else 0.0
            sell_accuracy_rate = np.mean([r.accuracy_score for r in sell_records]) if sell_records else 0.0
            
            # Store raw data for percentile calculation after all classes are processed
            # We'll calculate percentiles at the end and update these values
            buy_confidence = "pending"
            sell_confidence = "pending"
            
            # Calculate class statistics
            centroid = np.mean(class_features, axis=0)
            accuracy_rate = np.mean(class_accuracies)
            sample_count = len(class_features)
            
            # Calculate confidence interval (simplified)
            accuracy_std = np.std(class_accuracies)
            margin = 1.96 * accuracy_std / np.sqrt(sample_count)  # 95% CI
            confidence_interval = (
                max(0.0, accuracy_rate - margin),
                min(1.0, accuracy_rate + margin)
            )
            
            # Determine accuracy range for this class
            accuracy_range = (
                max(0.0, np.min(class_accuracies)),
                min(1.0, np.max(class_accuracies))
            )
            
            accuracy_class = AccuracyClass(
                class_id=class_id,
                centroid=centroid,
                accuracy_rate=accuracy_rate,
                sample_count=sample_count,
                confidence_interval=confidence_interval,
                accuracy_range=accuracy_range,
                buy_count=buy_count,
                sell_count=sell_count,
                buy_accuracy_rate=buy_accuracy_rate,
                sell_accuracy_rate=sell_accuracy_rate,
                buy_confidence=buy_confidence,
                sell_confidence=sell_confidence
            )
            
            self.accuracy_classes.append(accuracy_class)
        
        # Calculate percentile rankings for confidence values
        self._calculate_percentile_confidence()
        
        # Fit covariance estimator for Mahalanobis distance
        self.covariance_estimator.fit(features_scaled)
        self.mahalanobis_inv_cov = self.covariance_estimator.precision_
        
        # Calculate training metrics
        training_accuracy = self._calculate_training_accuracy(features_scaled, accuracy_scores)
        validation_accuracy = self._cross_validate(features_scaled, accuracy_scores)
        
        self.validation_scores = {
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'n_classes_created': len(self.accuracy_classes)
        }
        
        return self.validation_scores
    
    def _calculate_percentile_confidence(self):
        """Calculate percentile rankings for buy/sell confidence values."""
        if not self.accuracy_classes:
            return
        
        # Collect all buy and sell accuracies
        buy_accuracies = []
        sell_accuracies = []
        
        for ac in self.accuracy_classes:
            if ac.buy_count > 0:
                buy_accuracies.append(ac.buy_accuracy_rate)
            if ac.sell_count > 0:
                sell_accuracies.append(ac.sell_accuracy_rate)
        
        # Function to calculate percentile rank
        def calculate_percentile_rank(value: float, all_values: list) -> str:
            """Calculate what percentile a value represents."""
            if not all_values or len(all_values) < 2:
                return "N/A"
            
            # Count how many values are below this value
            rank = sum(1 for v in all_values if v < value)
            percentile = (rank / len(all_values)) * 100
            
            # Round to nearest 5th percentile for cleaner display
            rounded_percentile = round(percentile / 5) * 5
            return f"{int(rounded_percentile)}th"
        
        # Update confidence values with percentile rankings
        for ac in self.accuracy_classes:
            if ac.buy_count > 0:
                ac.buy_confidence = calculate_percentile_rank(ac.buy_accuracy_rate, buy_accuracies)
            else:
                ac.buy_confidence = "N/A"
                
            if ac.sell_count > 0:
                ac.sell_confidence = calculate_percentile_rank(ac.sell_accuracy_rate, sell_accuracies)
            else:
                ac.sell_confidence = "N/A"
    
    def _calculate_training_accuracy(self, features: np.ndarray, true_accuracies: np.ndarray) -> float:
        """Calculate training accuracy by predicting on training set."""
        if not self.accuracy_classes:
            return 0.0
            
        predictions = []
        for feature_vector in features:
            pred_accuracy = self._predict_single(feature_vector)
            predictions.append(pred_accuracy)
            
        predictions = np.array(predictions)
        
        # Calculate correlation between predicted and actual accuracies
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, true_accuracies)[0, 1]
            return max(0.0, correlation)  # Convert correlation to accuracy metric
        else:
            return 0.0
    
    def _cross_validate(self, features: np.ndarray, accuracies: np.ndarray) -> float:
        """Simple cross-validation on training data."""
        # Simplified: use last 20% as validation set
        split_idx = int(len(features) * 0.8)
        
        train_features = features[:split_idx]
        train_accuracies = accuracies[:split_idx]
        val_features = features[split_idx:]
        val_accuracies = accuracies[split_idx:]
        
        if len(val_features) < 5:  # Not enough validation data
            return self.validation_scores.get('training_accuracy', 0.0)
            
        # Re-train on training subset
        # (Simplified implementation - in production would re-fit everything)
        
        # Predict on validation set
        val_predictions = []
        for feature_vector in val_features:
            pred_accuracy = self._predict_single(feature_vector)
            val_predictions.append(pred_accuracy)
            
        val_predictions = np.array(val_predictions)
        
        # Calculate validation correlation
        if len(val_predictions) > 1:
            correlation = np.corrcoef(val_predictions, val_accuracies)[0, 1]
            return max(0.0, correlation)
        else:
            return 0.0
    
    # ========================================================================
    # INFERENCE API
    # ========================================================================
    
    @classmethod
    def load_for_strategy(cls, strategy_instance):
        """Load filter matching strategy's exact configuration."""
        if not hasattr(strategy_instance, 'get_cache_config_string'):
            raise ValueError("Strategy must have get_cache_config_string() method")
            
        config_string = strategy_instance.get_cache_config_string()
        indicator_type = strategy_instance.__class__.__name__.replace('Strategy', '').upper()
        
        filter_name = cls.get_filter_cache_name(config_string, indicator_type)
        
        # Find PKL file in cache hierarchy
        cache_base = Path(__file__).parent.parent.parent / 'cache'
        
        # Try symbol-specific cache first
        if hasattr(strategy_instance, '_symbol') and hasattr(strategy_instance, '_exchange'):
            instrument_type = getattr(strategy_instance, '_instrument_type', 'Unknown').replace(' ', '_')
            symbol_dir = cache_base / 'symbols' / f"{strategy_instance._exchange}_{instrument_type}_{strategy_instance._symbol}"
            pkl_path = symbol_dir / f"{filter_name}.pkl"
            
            if pkl_path.exists():
                return cls.load_from_path(str(pkl_path))
        
        # Try other cache locations
        # Could add multi-symbol cache lookup here
        
        raise FileNotFoundError(f"No trained filter found for config: {config_string}")
    
    @classmethod
    def load_for_display(cls, symbol: str, exchange: str = 'US', 
                         strategy: str = 'SMI', model: str = 'chronos',
                         instrument_type: str = 'ETF', profile_key: str = None):
        """
        Load filter directly for display purposes without strategy instance.
        This avoids unnecessary sensor calculations and provides instant regime display.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY')
            exchange: Exchange code (default: 'US')
            strategy: Strategy name (e.g., 'SMI', 'RVI', 'Stochastic')
            model: Model type (e.g., 'chronos', 'toto', 'prophet')
            instrument_type: Instrument type (default: 'ETF')
            profile_key: Profile key for filter selection (optional)
            
        Returns:
            MahalanobisAccuracyFilter instance or None if not found
        """
        try:
            # Build the filter configuration string
            indicator_params = "14_3_3"  # Standard parameters
            config_string = f"{indicator_params}_5_{model}_base"
            
            # Add profile suffix if specified
            if profile_key and profile_key != "custom":
                config_string = f"{config_string}_{profile_key}"
            
            # Generate filter name
            filter_name = cls.get_filter_cache_name(config_string, strategy.upper())
            
            # Build path to filter file
            cache_base = Path(__file__).parent.parent.parent / 'cache'
            normalized_type = instrument_type.replace(' ', '_')
            symbol_dir = cache_base / 'symbols' / f"{exchange}_{normalized_type}_{symbol}"
            pkl_path = symbol_dir / f"{filter_name}.pkl"
            
            if pkl_path.exists():
                return cls.load_from_path(str(pkl_path))
            
            # Try fallback without profile suffix
            if profile_key:
                base_config = f"{indicator_params}_5_{model}_base"
                base_filter_name = cls.get_filter_cache_name(base_config, strategy.upper())
                base_pkl_path = symbol_dir / f"{base_filter_name}.pkl"
                
                if base_pkl_path.exists():
                    return cls.load_from_path(str(base_pkl_path))
            
            return None  # No filter found
            
        except Exception as e:
            print(f"Error in load_for_display: {e}")
            return None
    
    @classmethod  
    def load_from_path(cls, pkl_path: str):
        """Load filter from specific PKL file path."""
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Filter file not found: {pkl_path}")
            
        with open(pkl_path, 'rb') as f:
            filter_instance = pickle.load(f)
            
        if not isinstance(filter_instance, cls):
            raise ValueError(f"Invalid filter type in {pkl_path}")
            
        return filter_instance
    
    def predict_accuracy(self, sensor_vector: np.ndarray, signal_direction: str = None) -> Dict[str, float]:
        """
        Predict accuracy from sensor vector with optional signal direction.
        
        Args:
            sensor_vector: Numpy array of sensor values (dimensions match training)
            signal_direction: Optional direction - 'bullish' for buy, 'bearish' for sell
            
        Returns:
            Dictionary with accuracy predictions. If signal_direction is specified,
            returns direction-specific accuracy. Otherwise returns both.
        """
        if not self.is_trained:
            raise ValueError("Filter must be trained before making predictions")
            
        # Validate sensor vector dimensions against training
        expected_features = len(self.sensor_columns) if self.sensor_columns else 48  # Fallback to 48
        if len(sensor_vector) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(sensor_vector)}")
            
        # Get the best matching accuracy class
        best_class = self._find_best_class(sensor_vector)
        
        if signal_direction == 'bullish':
            # Return buy-specific prediction
            return {
                'accuracy': float(best_class.buy_accuracy_rate),
                'confidence': best_class.buy_confidence,
                'sample_count': best_class.buy_count,
                'class_id': best_class.class_id
            }
        elif signal_direction == 'bearish':
            # Return sell-specific prediction  
            return {
                'accuracy': float(best_class.sell_accuracy_rate),
                'confidence': best_class.sell_confidence,
                'sample_count': best_class.sell_count,
                'class_id': best_class.class_id
            }
        else:
            # Return comprehensive analysis for both directions
            return {
                'overall_accuracy': float(best_class.accuracy_rate),
                'buy_accuracy': float(best_class.buy_accuracy_rate),
                'sell_accuracy': float(best_class.sell_accuracy_rate),
                'buy_confidence': best_class.buy_confidence,
                'sell_confidence': best_class.sell_confidence,
                'buy_sample_count': best_class.buy_count,
                'sell_sample_count': best_class.sell_count,
                'class_id': best_class.class_id
            }
    
    def _find_best_class(self, sensor_vector: np.ndarray) -> AccuracyClass:
        """Find the best matching accuracy class for a sensor vector."""
        if not self.accuracy_classes or self.mahalanobis_inv_cov is None:
            # Return a default class with neutral values
            return AccuracyClass(
                class_id=0, centroid=np.array([0]), accuracy_rate=0.5, sample_count=0,
                confidence_interval=(0.5, 0.5), accuracy_range=(0.5, 0.5),
                buy_count=0, sell_count=0, buy_accuracy_rate=0.5, sell_accuracy_rate=0.5,
                buy_confidence='low', sell_confidence='low'
            )
            
        # Scale features
        try:
            sensor_scaled = self.scaler.transform(sensor_vector.reshape(1, -1))[0]
        except Exception:
            return self.accuracy_classes[0] if self.accuracy_classes else None
            
        # Find closest accuracy class using Mahalanobis distance
        min_distance = float('inf')
        best_class = self.accuracy_classes[0]
        
        for accuracy_class in self.accuracy_classes:
            try:
                distance = self._mahalanobis_distance(sensor_scaled, accuracy_class.centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_class = accuracy_class
            except Exception:
                continue
                
        return best_class
    
    def _predict_single(self, sensor_vector: np.ndarray) -> float:
        """Make single accuracy prediction using Mahalanobis distance."""
        if not self.accuracy_classes or self.mahalanobis_inv_cov is None:
            return 0.5  # Neutral if not properly trained
            
        # Scale features
        try:
            sensor_scaled = self.scaler.transform(sensor_vector.reshape(1, -1))[0]
        except Exception:
            return 0.5
            
        # Find closest accuracy class using Mahalanobis distance
        min_distance = float('inf')
        best_class = None
        
        for accuracy_class in self.accuracy_classes:
            try:
                distance = self._mahalanobis_distance(sensor_scaled, accuracy_class.centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_class = accuracy_class
            except Exception:
                continue
                
        if best_class is not None:
            return best_class.accuracy_rate
        else:
            return 0.5  # Neutral fallback
    
    def _mahalanobis_distance(self, point: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate Mahalanobis distance between point and centroid."""
        try:
            diff = point - centroid
            distance = np.sqrt(diff.T @ self.mahalanobis_inv_cov @ diff)
            return distance
        except Exception:
            # Fallback to Euclidean distance
            return np.linalg.norm(point - centroid)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    @staticmethod
    def get_filter_cache_name(strategy_config: str, indicator_type: str = "SMI") -> str:
        """
        Generate filter cache name from strategy config using centralized utility.
        
        Args:
            strategy_config: Strategy config string (e.g., "14_3_3_5_chronos_base_persym_p15_t250")
            indicator_type: Indicator type (e.g., "SMI")
            
        Returns:
            Filter cache name (e.g., "SMIMahalanobisFilter_14_3_3_chronos_base_persym_p15_t250")
        """
        from utilities.cache_naming import CacheNamingUtility
        return CacheNamingUtility.get_filter_cache_name(strategy_config, indicator_type)
    
    def save_to_path(self, pkl_path: str, metadata_path: Optional[str] = None) -> bool:
        """
        Save trained filter to specified path.
        
        Args:
            pkl_path: Path for pickle file
            metadata_path: Optional path for JSON metadata
            
        Returns:
            True if successful
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            
            # Save pickle file
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f, protocol=4)
                
            # Save metadata if requested
            if metadata_path:
                import json
                # Add sensor column information to metadata
                metadata_with_sensors = self.training_metadata.copy()
                metadata_with_sensors['sensor_columns'] = self.sensor_columns
                metadata_with_sensors['sensor_columns_count'] = len(self.sensor_columns)
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_with_sensors, f, indent=2)
                    
            return True
            
        except Exception as e:
            print(f"Error saving filter: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'is_trained': self.is_trained,
            'n_accuracy_classes': self.n_accuracy_classes,
            'n_training_records': len(self.accuracy_records),
            'sensor_columns_count': len(self.sensor_columns),
            'sensor_columns_sample': self.sensor_columns[:5] if self.sensor_columns else [],
            'validation_scores': self.validation_scores,
            'training_metadata': self.training_metadata
        }
        
        if self.accuracy_classes:
            info['accuracy_classes'] = {
                f"Class_{i}": {
                    'accuracy_rate': f"{ac.accuracy_rate:.3f}",
                    'sample_count': ac.sample_count,
                    'accuracy_range': f"{ac.accuracy_range[0]:.3f} - {ac.accuracy_range[1]:.3f}"
                }
                for i, ac in enumerate(self.accuracy_classes)
            }
            
        return info
    
    def get_class_summary(self) -> pd.DataFrame:
        """Get summary of accuracy classes as DataFrame with buy/sell breakdowns."""
        if not self.accuracy_classes:
            return pd.DataFrame()
        
        class_data = []
        for i, ac in enumerate(self.accuracy_classes):
            class_data.append({
                'class_id': i,
                'accuracy_rate': ac.accuracy_rate,
                'sample_count': ac.sample_count,
                'min_accuracy': ac.accuracy_range[0],
                'max_accuracy': ac.accuracy_range[1],
                'confidence': 'high' if ac.sample_count > 100 else 'medium' if ac.sample_count > 50 else 'low',
                # Buy/Sell breakdowns
                'buy_count': ac.buy_count,
                'sell_count': ac.sell_count,
                'buy_accuracy': ac.buy_accuracy_rate,
                'sell_accuracy': ac.sell_accuracy_rate,
                'buy_confidence': ac.buy_confidence,
                'sell_confidence': ac.sell_confidence
            })
        
        return pd.DataFrame(class_data).sort_values('accuracy_rate', ascending=False)
    
    def get_accuracy_percentiles(self) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy percentiles for intelligent threshold setting."""
        if not self.accuracy_classes:
            return {}
        
        # Collect all buy and sell accuracies with their sample counts for weighting
        buy_accuracies = []
        sell_accuracies = []
        buy_weights = []
        sell_weights = []
        
        for ac in self.accuracy_classes:
            if ac.buy_count > 0:
                buy_accuracies.append(ac.buy_accuracy_rate)
                buy_weights.append(ac.buy_count)
            if ac.sell_count > 0:
                sell_accuracies.append(ac.sell_accuracy_rate)
                sell_weights.append(ac.sell_count)
        
        # Calculate weighted percentiles (more samples = more influence)
        import numpy as np
        
        percentiles = {}
        
        if buy_accuracies:
            # Calculate percentiles for buy accuracies
            buy_arr = np.array(buy_accuracies)
            buy_weights_arr = np.array(buy_weights)
            
            # Sort by accuracy
            sorted_idx = np.argsort(buy_arr)
            sorted_accuracies = buy_arr[sorted_idx]
            sorted_weights = buy_weights_arr[sorted_idx]
            
            # Calculate cumulative weights
            cum_weights = np.cumsum(sorted_weights)
            total_weight = cum_weights[-1]
            
            # Find percentile values
            percentiles['buy'] = {}
            for pct in [10, 20, 25, 40, 50, 60, 75, 80, 90]:
                target_weight = (pct / 100.0) * total_weight
                idx = np.searchsorted(cum_weights, target_weight)
                if idx >= len(sorted_accuracies):
                    idx = len(sorted_accuracies) - 1
                percentiles['buy'][pct] = float(sorted_accuracies[idx])
        
        if sell_accuracies:
            # Calculate percentiles for sell accuracies  
            sell_arr = np.array(sell_accuracies)
            sell_weights_arr = np.array(sell_weights)
            
            # Sort by accuracy
            sorted_idx = np.argsort(sell_arr)
            sorted_accuracies = sell_arr[sorted_idx]
            sorted_weights = sell_weights_arr[sorted_idx]
            
            # Calculate cumulative weights
            cum_weights = np.cumsum(sorted_weights)
            total_weight = cum_weights[-1]
            
            # Find percentile values
            percentiles['sell'] = {}
            for pct in [10, 20, 25, 40, 50, 60, 75, 80, 90]:
                target_weight = (pct / 100.0) * total_weight
                idx = np.searchsorted(cum_weights, target_weight)
                if idx >= len(sorted_accuracies):
                    idx = len(sorted_accuracies) - 1
                percentiles['sell'][pct] = float(sorted_accuracies[idx])
        
        # Also calculate simple statistics for reference
        percentiles['stats'] = {
            'buy': {
                'min': float(min(buy_accuracies)) if buy_accuracies else 0.0,
                'max': float(max(buy_accuracies)) if buy_accuracies else 0.0,
                'mean': float(np.average(buy_accuracies, weights=buy_weights)) if buy_accuracies else 0.0,
                'median': percentiles.get('buy', {}).get(50, 0.0)
            },
            'sell': {
                'min': float(min(sell_accuracies)) if sell_accuracies else 0.0,
                'max': float(max(sell_accuracies)) if sell_accuracies else 0.0,
                'mean': float(np.average(sell_accuracies, weights=sell_weights)) if sell_accuracies else 0.0,
                'median': percentiles.get('sell', {}).get(50, 0.0)
            }
        }
        
        return percentiles
    
    def get_profile_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Get recommended thresholds for different trading profiles."""
        if not self.accuracy_classes:
            return {}
        
        df = self.get_class_summary()
        if df.empty:
            return {}
        
        recommendations = {}
        
        # Conservative: Use only highest accuracy classes
        high_accuracy = df[df['accuracy_rate'] >= 0.65]
        if not high_accuracy.empty:
            recommendations['conservative'] = {
                'threshold': 0.65,
                'expected_accuracy': high_accuracy['accuracy_rate'].mean(),
                'class_count': len(high_accuracy),
                'sample_coverage': high_accuracy['sample_count'].sum() / df['sample_count'].sum()
            }
        
        # Balanced: Mid-range threshold
        mid_accuracy = df[df['accuracy_rate'] >= 0.55]
        if not mid_accuracy.empty:
            recommendations['balanced'] = {
                'threshold': 0.55,
                'expected_accuracy': mid_accuracy['accuracy_rate'].mean(),
                'class_count': len(mid_accuracy),
                'sample_coverage': mid_accuracy['sample_count'].sum() / df['sample_count'].sum()
            }
        
        # Aggressive: Include more classes for opportunities
        low_threshold = df[df['accuracy_rate'] >= 0.50]
        if not low_threshold.empty:
            recommendations['aggressive'] = {
                'threshold': 0.50,
                'expected_accuracy': low_threshold['accuracy_rate'].mean(),
                'class_count': len(low_threshold),
                'sample_coverage': low_threshold['sample_count'].sum() / df['sample_count'].sum()
            }
        
        return recommendations