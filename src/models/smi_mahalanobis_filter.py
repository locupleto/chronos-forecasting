# ============================================================================
# -*- coding: utf-8 -*-
#
# Module:      smi_mahalanobis_filter.py
# Description: SMI-specific implementation of Mahalanobis distance-based signal 
#              accuracy classification for SMI (Stochastic Momentum Index) signals.
#
# Purpose:     Concrete implementation for SMI indicator accuracy filtering.
#              Provides the same functionality as the original MahalanobisSignalClassifier
#              but through the new generic architecture.
#
# Key Features:
#   - SMI 14,3,3 parameter specification
#   - Both ML predicted and vanilla signal accuracy tracking
#   - Enhanced directional accuracy breakdown
#   - Backward compatibility with existing tests
#
# Usage:
#   filter = SMIMahalanobisFilter()
#   filter.collect_accuracy_data(market_data, sensor_data, predictions, outcomes, symbol)
#   filter.train_classifier()
#   accuracy = filter.predict_accuracy(market_context)
#
# History:
# 2025-01-16   Created - SMI-specific Mahalanobis filter implementation
# ============================================================================

from typing import Dict, Any, Tuple, Type, Optional
import pandas as pd
from models.mahalanobis_filter import MahalanobisAccuracyFilter
from indicators.base_indicator import BaseIndicator

class SMIMahalanobisFilter(MahalanobisAccuracyFilter):
    """
    SMI-specific Mahalanobis signal accuracy classifier.
    
    This class provides signal accuracy classification specifically for SMI
    (Stochastic Momentum Index) indicators with 14,3,3 parameters.
    
    The SMI indicator is prediction-capable, so this filter provides both:
    - ML predicted signal accuracy (early crossover predictions)
    - Vanilla indicator accuracy (traditional crossover signals)
    
    Features:
    - SMI %K and Signal line crossover detection
    - 5-bar prediction horizon for ML signals
    - Price movement validation for vanilla signals
    - Enhanced directional accuracy breakdown
    """
    
    def specify_target_indicator(self) -> Tuple[Type[BaseIndicator], Dict[str, Any]]:
        """
        Specify SMI indicator with 14,3,3 parameters for accuracy evaluation.
        
        Returns:
            Tuple of (SMI_pred_Indicator, parameters_dict)
        """
        # Import here to avoid circular dependencies
        from indicators.smi_pred_indicator import SMI_pred_Indicator
        
        # SMI indicator parameters (matching original implementation)
        smi_params = {
            'k_period': 14,        # %K period (main SMI calculation)
            'k_smooth': 3,         # %K smoothing periods  
            'd_smooth': 3,         # Signal line smoothing periods
            'enable_predictions': True,  # Enable ML predictions
            'prediction_zone_length': 5,      # 5-bar prediction horizon (FIXED: was prediction_length)
            'prediction_model_type': 'chronos',  # Use Chronos model
            'chronos_model_size': 'base',        # Base model size
            'evaluation_prediction_offset': 0    # No evaluation offset
        }
        
        return SMI_pred_Indicator, smi_params
    
    @classmethod
    def get_display_name(cls) -> str:
        """Get human-readable display name for this filter."""
        return "SMI (14,3,3) Accuracy Filter"
    
    @classmethod
    def get_indicator_type(cls) -> str:
        """Get indicator type identifier for this filter."""
        return "smi"
    
    def get_signal_column_patterns(self) -> Dict[str, str]:
        """
        Get SMI-specific signal column naming patterns.
        
        Returns:
            Dictionary mapping signal types to column patterns
        """
        return {
            'predicted_bullish': 'SMI_PREDICTED_BULLISH',
            'predicted_bearish': 'SMI_PREDICTED_BEARISH',
            'actual_bullish': 'SMI_CROSSOVER_BULLISH_14_3',
            'actual_bearish': 'SMI_CROSSOVER_BEARISH_14_3'
        }
        
    def get_ground_truth_sensors(self) -> Dict[str, str]:
        """
        Get SMI-specific ground truth sensor column names.
        
        Returns:
            Dictionary mapping sensor types to column names
        """
        return {
            'crossover_bullish': 'SMI_CROSSOVER_BULLISH_14_3',
            'crossover_bearish': 'SMI_CROSSOVER_BEARISH_14_3', 
            'price_pivot_bullish': 'ENHANCED_PRICE_PIVOT_BULLISH',
            'price_pivot_bearish': 'ENHANCED_PRICE_PIVOT_BEARISH'
        }

    def get_class_summary(self) -> pd.DataFrame:
        """Get summary of accuracy classes as DataFrame for UI display."""
        # Delegate to parent class if it has this method
        if hasattr(super(), 'get_class_summary'):
            return super().get_class_summary()
        
        # Otherwise provide basic implementation
        import pandas as pd
        if not hasattr(self, 'accuracy_classes') or not self.accuracy_classes:
            return pd.DataFrame()
        
        class_data = []
        for i, ac in enumerate(self.accuracy_classes):
            class_data.append({
                'class_id': i,
                'accuracy_rate': ac.accuracy_rate,
                'sample_count': ac.sample_count,
                'min_accuracy': ac.accuracy_range[0] if hasattr(ac, 'accuracy_range') else 0,
                'max_accuracy': ac.accuracy_range[1] if hasattr(ac, 'accuracy_range') else 1,
                'confidence': 'high' if ac.sample_count > 100 else 'medium' if ac.sample_count > 50 else 'low'
            })
        
        return pd.DataFrame(class_data).sort_values('accuracy_rate', ascending=False)

# Backward compatibility - alias to the original class name
# This allows existing code to continue working without changes
MahalanobisSignalClassifier = SMIMahalanobisFilter

# Example usage and testing
if __name__ == "__main__":
    # Quick test to verify the implementation works
    print(f"SMI Filter Display Name: {SMIMahalanobisFilter.get_display_name()}")
    print(f"SMI Filter Indicator Type: {SMIMahalanobisFilter.get_indicator_type()}")
    
    # Create instance and verify configuration
    smi_filter = SMIMahalanobisFilter()
    indicator_class, params = smi_filter.specify_target_indicator()
    
    print(f"Target Indicator: {indicator_class.__name__}")
    print(f"Prediction Capability: {smi_filter.target_indicator_supports_predictions}")
    print(f"Parameters: {params}")