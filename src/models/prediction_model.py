# ============================================================================
# -*- coding: utf-8 -*-
#
# Module:      prediction_model.py
# Description: Abstract base class for all prediction models used in trading
#              indicators. Provides a unified interface for different prediction
#              algorithms including Mock, Burg AR, and Chronos-Bolt models.
#
# History:
# 2025-01-11   Claude Created - Phase 1 of unified prediction architecture
# ============================================================================

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

class PredictionModel(ABC):
    """
    Abstract base class for all prediction models.
    
    This class defines the standard interface that all prediction models must implement
    to work with the BaseIndicator prediction framework. It ensures consistency across
    different prediction algorithms while allowing for model-specific implementations.
    
    The design follows the Strategy pattern, where different prediction algorithms
    can be swapped in and out without changing the client code (BaseIndicator).
    """
    
    def __init__(self, **kwargs):
        """
        Initialize prediction model with configuration parameters.
        
        Args:
            **kwargs: Model-specific configuration parameters
        """
        self.config = kwargs
        self.model_type = self.__class__.__name__.lower().replace('predictor', '')
        self._initialized = True
    
    @abstractmethod
    def predict(self, 
                series: pd.Series, 
                prediction_length: int,
                **kwargs) -> np.ndarray:
        """
        Generate predictions for a time series.
        
        This is the core method that all prediction models must implement.
        It should return future values for the given time series.
        
        Args:
            series: Input time series data (should be clean, no NaN values)
            prediction_length: Number of future steps to predict
            **kwargs: Additional model-specific parameters
            
        Returns:
            Array of predicted values with length equal to prediction_length
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If prediction fails
        """
        pass
    
    @abstractmethod
    def validate_data(self, series: pd.Series) -> bool:
        """
        Validate input data for prediction.
        
        This method should check if the input data is suitable for the specific
        prediction model. For example, AR models need sufficient history,
        while mock models have minimal requirements.
        
        Args:
            series: Input time series data to validate
            
        Returns:
            True if data is valid for prediction, False otherwise
        """
        pass
    
    @abstractmethod
    def get_required_history_length(self, prediction_length: int) -> int:
        """
        Get minimum historical data required for prediction.
        
        Different models have different requirements for historical data.
        This method helps the framework determine if enough data is available.
        
        Args:
            prediction_length: Number of future steps to predict
            
        Returns:
            Minimum number of historical data points required
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, 
                               series: pd.Series, 
                               prediction_length: int,
                               quantile_levels: Optional[List[float]] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Generate predictions with uncertainty quantification.
        
        This method provides predictions along with uncertainty estimates where supported.
        Models that don't support uncertainty should return the same predictions as predict()
        with appropriate metadata indicating no uncertainty information is available.
        
        Args:
            series: Input time series data (should be clean, no NaN values)
            prediction_length: Number of future steps to predict
            quantile_levels: List of quantile levels for uncertainty estimation (e.g., [0.1, 0.5, 0.9])
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing:
            - 'predictions': Array of mean/median predicted values
            - 'quantiles': Dict[str, Array] of quantile predictions (if supported)
            - 'confidence_intervals': Dict[str, Dict[str, Array]] confidence intervals (if supported)  
            - 'uncertainty_metrics': Dict[str, Any] uncertainty measures (if supported)
            - 'supports_uncertainty': bool indicating if uncertainty is available
            - 'metadata': Dict[str, Any] with prediction details
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If prediction fails
        """
        pass
    
    @abstractmethod
    def supports_uncertainty(self) -> bool:
        """
        Check if this model supports uncertainty quantification.
        
        Returns:
            True if model can provide uncertainty estimates, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get model capabilities and limitations.
        
        This method provides metadata about what the model can and cannot do,
        helping with model selection and parameter validation.
        
        Returns:
            Dictionary containing model capabilities, limitations, and characteristics.
            Should include 'supports_uncertainty' key indicating uncertainty support.
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information and configuration.
        
        Returns:
            Dictionary containing model type, configuration, and capabilities
        """
        return {
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "config": self.config,
            "capabilities": self.get_capabilities(),
            "initialized": self._initialized
        }
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: New configuration parameters
        """
        self.config.update(kwargs)
    
    def prepare_series_for_prediction(self, series: pd.Series) -> pd.Series:
        """
        Prepare time series data for prediction.
        
        This method provides common data preprocessing that can be used by
        all prediction models, such as handling NaN values and outliers.
        
        Args:
            series: Input time series data
            
        Returns:
            Cleaned and prepared time series data
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        # Replace infinite values with NaN and forward fill
        clean_series = clean_series.replace([np.inf, -np.inf], np.nan)
        clean_series = clean_series.ffill().bfill()
        
        # Ensure we have some data
        if len(clean_series) == 0:
            raise ValueError("No valid data points after cleaning")
        
        return clean_series
    
    def get_prediction_quality_metrics(self, 
                                     actual: pd.Series, 
                                     predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction quality metrics.
        
        This method provides standard metrics for evaluating prediction quality
        that can be used across all prediction models.
        
        Args:
            actual: Actual observed values
            predicted: Predicted values
            
        Returns:
            Dictionary containing various quality metrics
        """
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have the same length")
        
        # Convert to numpy arrays
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_arr - predicted_arr))
        mse = np.mean((actual_arr - predicted_arr) ** 2)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (handle division by zero)
        mape = np.mean(np.abs((actual_arr - predicted_arr) / np.where(actual_arr != 0, actual_arr, 1))) * 100
        
        # R-squared (coefficient of determination)
        ss_res = np.sum((actual_arr - predicted_arr) ** 2)
        ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "r_squared": r_squared,
            "prediction_length": len(predicted)
        }
    
    def generate_time_period_aware_dates(self, 
                                        df: pd.DataFrame, 
                                        last_date: pd.Timestamp, 
                                        prediction_length: int) -> List[pd.Timestamp]:
        """
        Generate future dates matching the DataFrame's time period pattern.
        
        This method reads the 'interval' column from the DataFrame to determine
        the correct date generation pattern and ensures all prediction models
        generate consistent dates that match the database's time period patterns.
        
        Args:
            df: DataFrame containing the interval column
            last_date: The last date in the historical data
            prediction_length: Number of prediction periods to generate
            
        Returns:
            List of future dates matching the time period pattern
        """
        try:
            # Read the interval from the database to determine date generation pattern
            interval = df['interval'].iloc[0] if 'interval' in df.columns else '1d'
            
            # Ensure last_date is a pandas Timestamp
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            elif not isinstance(last_date, pd.Timestamp):
                last_date = pd.Timestamp(last_date)
            
            # Generate future dates matching the database pattern
            if interval == '1w':
                # Weekly: For weekly aggregation, prediction_length represents trading days, not weeks
                # Calculate how many weekly periods are needed to cover prediction_length trading days
                # Since each week covers ~5 trading days, we need ceil(prediction_length / 5) periods
                import math
                weekly_periods_needed = math.ceil(prediction_length / 5)
                future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, weekly_periods_needed + 1)]
            elif interval == '1m':
                # Monthly: For monthly aggregation, prediction_length represents trading days, not months
                # Calculate how many monthly periods are needed to cover prediction_length trading days
                # Since each month covers ~22 trading days, we need ceil(prediction_length / 22) periods
                import math
                monthly_periods_needed = math.ceil(prediction_length / 22)
                future_dates = pd.date_range(start=last_date, periods=monthly_periods_needed+1, freq='ME')[1:]
            elif interval == '1y':
                # Yearly: For yearly aggregation, prediction_length represents trading days, not years
                # Calculate how many yearly periods are needed to cover prediction_length trading days
                # Since each year covers ~252 trading days, we need ceil(prediction_length / 252) periods
                import math
                yearly_periods_needed = math.ceil(prediction_length / 252)
                future_dates = pd.date_range(start=last_date, periods=yearly_periods_needed+1, freq='YE')[1:]
            else:
                # Daily: Generate business days (matches database daily pattern)
                from pandas.tseries.offsets import BDay
                future_dates = pd.bdate_range(start=last_date + BDay(1), periods=prediction_length)
            
            # Convert to consistent datetime format and return as list
            return pd.to_datetime(future_dates).tolist()
            
        except Exception as e:
            # Fallback to daily business day generation if interval detection fails
            from pandas.tseries.offsets import BDay
            future_dates = pd.bdate_range(start=last_date + BDay(1), periods=prediction_length)
            return pd.to_datetime(future_dates).tolist()
    
    def __str__(self) -> str:
        """String representation of the prediction model."""
        return f"{self.__class__.__name__}(type={self.model_type}, config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the prediction model."""
        return f"{self.__class__.__name__}({self.config})"