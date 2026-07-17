# ============================================================================
# -*- coding: utf-8 -*-
#
# Module:      filter_profiles.py
# Description: Preset filter profiles for simplified filter configuration.
#              Maps trading objectives to optimized filter configurations.
#
# Purpose:     Provide users with simple, goal-oriented filter choices instead
#              of complex manual configuration. Each profile represents a
#              thoroughly tested combination of training parameters and
#              runtime thresholds.
#
# Architecture:
#   - FilterProfile: Individual profile configuration
#   - FilterProfileManager: Manages profiles and filter discovery
#   - Preset profiles for common trading objectives
#   - Automatic filter discovery and validation
#
# Usage:
#   manager = FilterProfileManager()
#   profile = manager.get_profile("Balanced")
#   filter_model = profile.load_filter(symbol="SPY")
#   metrics = profile.get_expected_metrics()
#
# History:
# 2025-01-21   Created - Filter profile system for simplified UI
# ============================================================================

import os
import json
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterProfile:
    """
    Represents a preset filter configuration optimized for specific trading objectives.
    """
    name: str
    display_name: str
    description: str
    objective: str
    
    # Filter configuration
    training_params: Dict[str, float]
    buy_threshold: float  # Minimum accuracy for buy signals (used for fixed mode)
    sell_threshold: float  # Minimum accuracy for sell signals (used for fixed mode)
    min_accuracy: float
    
    # Expected performance metrics
    expected_win_rate: float
    expected_profit_factor: float
    expected_trades_per_year: int
    expected_expectancy: float
    
    # Characteristics
    risk_level: str  # "low", "medium", "high"
    trade_frequency: str  # "low", "medium", "high"
    
    # Percentile-based thresholds (new intelligent system) - with defaults
    buy_threshold_percentile: Optional[int] = None  # Percentile for buy signals (0-100)
    sell_threshold_percentile: Optional[int] = None  # Percentile for sell signals (0-100)
    threshold_mode: str = "adaptive"  # "fixed", "percentile", "adaptive"
    
    # Optional fields with defaults
    class_selection_mode: str = "threshold"
    suitable_for: List[str] = field(default_factory=list)
    filter_path_override: Optional[str] = None
    
    def get_filter_path(self, symbol: str, exchange: str = "US", 
                       strategy: str = "SMI", model: str = "chronos") -> Optional[str]:
        """
        Get the appropriate filter path for this profile.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange code
            strategy: Strategy type (SMI, RVI, etc.)
            model: Prediction model type
            
        Returns:
            Path to filter file or None if not found
        """
        if self.filter_path_override:
            if os.path.exists(self.filter_path_override):
                return self.filter_path_override
        
        # Use centralized naming for consistent filter paths
        from utilities.cache_naming import CacheNamingUtility
        
        # Generate correct filter name using centralized utility
        strategy_config = CacheNamingUtility.get_standard_config(strategy, model)
        base_filter_name = CacheNamingUtility.get_filter_cache_name(strategy_config, strategy)
        
        # Build filter name based on training parameters
        filter_suffix = self._get_filter_suffix()
        
        # Try different cache locations in priority order
        search_paths = [
            # Symbol-specific with profile suffix
            f"cache/symbols/{exchange}_ETF_{symbol}/{base_filter_name}{filter_suffix}.pkl",
            # Symbol-specific standard
            f"cache/symbols/{exchange}_ETF_{symbol}/{base_filter_name}.pkl",
            # Multi-symbol with profile suffix
            f"cache/test_small/{base_filter_name}{filter_suffix}.pkl",
            # Multi-symbol standard
            f"cache/test_small/{base_filter_name}.pkl",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found filter for profile '{self.name}': {path}")
                return path
        
        logger.warning(f"No filter found for profile '{self.name}' with symbol {symbol}")
        return None
    
    def _get_filter_suffix(self) -> str:
        """Get filter file suffix based on training parameters."""
        # Map training parameters to known suffixes
        if self.training_params.get("success") == 1.5 and self.training_params.get("stop") == 1.0:
            return "_Conservative"
        elif self.training_params.get("success") == 2.5 and self.training_params.get("stop") == 2.5:
            return "_Balanced"
        elif self.training_params.get("success") == 4.0 and self.training_params.get("stop") == 2.5:
            return "_Aggressive"
        return ""
    
    def load_filter(self, symbol: str, exchange: str = "US", 
                   strategy: str = "SMI", model: str = "toto"):
        """
        Load the actual Mahalanobis filter for this profile.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange code  
            strategy: Strategy type
            model: Prediction model type
            
        Returns:
            Loaded MahalanobisAccuracyFilter instance or None if not found
        """
        filter_path = self.get_filter_path(symbol, exchange, strategy, model)
        if not filter_path:
            logger.warning(f"No filter path found for profile '{self.name}' with {symbol}/{exchange}")
            return None
            
        try:
            # Import here to avoid circular imports
            from models.mahalanobis_filter import MahalanobisAccuracyFilter
            
            # Load the filter from the path
            filter_instance = MahalanobisAccuracyFilter.load(filter_path)
            logger.info(f"Successfully loaded filter for profile '{self.name}' from {filter_path}")
            return filter_instance
            
        except Exception as e:
            logger.error(f"Failed to load filter for profile '{self.name}' from {filter_path}: {e}")
            return None

    def get_expected_metrics(self) -> Dict[str, Any]:
        """Get expected performance metrics for display."""
        return {
            "win_rate": f"{self.expected_win_rate:.1%}",
            "profit_factor": f"{self.expected_profit_factor:.2f}",
            "trades_per_year": self.expected_trades_per_year,
            "expectancy": f"{self.expected_expectancy:.2%}",
            "risk_level": self.risk_level.capitalize(),
            "trade_frequency": self.trade_frequency.capitalize()
        }
    
    def get_effective_thresholds(self, accuracy_percentiles: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate effective thresholds based on mode and available percentile data.
        
        Args:
            accuracy_percentiles: Percentile data from MahalanobisAccuracyFilter.get_accuracy_percentiles()
            
        Returns:
            Dictionary with 'buy_threshold' and 'sell_threshold' keys
        """
        if self.threshold_mode == "fixed":
            # Use fixed thresholds regardless of data
            return {
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'mode': 'fixed'
            }
        
        elif self.threshold_mode == "percentile" and accuracy_percentiles:
            # Use percentile-based thresholds
            buy_pct = self.buy_threshold_percentile or 50
            sell_pct = self.sell_threshold_percentile or 50
            
            buy_thresh = accuracy_percentiles.get('buy', {}).get(buy_pct, self.buy_threshold)
            sell_thresh = accuracy_percentiles.get('sell', {}).get(sell_pct, self.sell_threshold)
            
            return {
                'buy_threshold': buy_thresh,
                'sell_threshold': sell_thresh,
                'buy_percentile': buy_pct,
                'sell_percentile': sell_pct,
                'mode': 'percentile'
            }
        
        elif self.threshold_mode == "adaptive" and accuracy_percentiles:
            # Adaptive: Use percentiles if available, fall back to adjusted fixed thresholds
            buy_pct = self.buy_threshold_percentile or 50
            sell_pct = self.sell_threshold_percentile or 50
            
            # Get percentile thresholds
            buy_percentile_thresh = accuracy_percentiles.get('buy', {}).get(buy_pct)
            sell_percentile_thresh = accuracy_percentiles.get('sell', {}).get(sell_pct)
            
            # Get stats for intelligent fallback
            buy_stats = accuracy_percentiles.get('stats', {}).get('buy', {})
            sell_stats = accuracy_percentiles.get('stats', {}).get('sell', {})
            
            # Adaptive logic: Use percentile if reasonable, otherwise use adjusted fixed
            buy_thresh = buy_percentile_thresh if buy_percentile_thresh and buy_percentile_thresh > 0.05 else min(self.buy_threshold, buy_stats.get('max', self.buy_threshold))
            sell_thresh = sell_percentile_thresh if sell_percentile_thresh and sell_percentile_thresh > 0.05 else min(self.sell_threshold, sell_stats.get('max', self.sell_threshold))
            
            return {
                'buy_threshold': buy_thresh,
                'sell_threshold': sell_thresh,
                'buy_percentile': buy_pct,
                'sell_percentile': sell_pct,
                'buy_max_available': buy_stats.get('max', 0.0),
                'sell_max_available': sell_stats.get('max', 0.0),
                'mode': 'adaptive'
            }
        
        else:
            # Fallback to fixed thresholds
            return {
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'mode': 'fixed_fallback'
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "objective": self.objective,
            "training_params": self.training_params,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "buy_threshold_percentile": self.buy_threshold_percentile,
            "sell_threshold_percentile": self.sell_threshold_percentile,
            "threshold_mode": self.threshold_mode,
            "min_accuracy": self.min_accuracy,
            "class_selection_mode": self.class_selection_mode,
            "expected_metrics": self.get_expected_metrics(),
            "risk_level": self.risk_level,
            "trade_frequency": self.trade_frequency,
            "suitable_for": self.suitable_for
        }


class FilterProfileManager:
    """
    Manages filter profiles and provides discovery/loading functionality.
    """
    
    def __init__(self):
        """Initialize filter profile manager with preset profiles."""
        self.profiles = self._load_preset_profiles()
        self.optimization_cache = {}
        self._load_optimization_results()
    
    def _load_preset_profiles(self) -> Dict[str, FilterProfile]:
        """Load preset filter profiles optimized for different objectives."""
        profiles = {}
        
        # Conservative Profile - Adjusted for realistic trading frequency (5 trades/year target)
        # Lowered percentiles significantly to allow trading opportunities
        profiles["conservative"] = FilterProfile(
            name="conservative",
            display_name="Conservative (High Win Rate)",
            description="Minimize losses with focus on winning trades (5 trades/year target)",
            objective="high_win_rate",
            training_params={"success": 1.5, "stop": 1.0, "partial": 0.75, "overshoot": -0.75},
            buy_threshold=0.288,  # OPTIMIZED: Updated from 0.60 based on filter optimization results
            sell_threshold=0.55,  # Lowered from 0.60
            buy_threshold_percentile=22,  # OPTIMIZED: Updated from 50th to 22nd percentile
            sell_threshold_percentile=50,  # Lowered from 80 (top 50% instead of top 20%)
            threshold_mode="adaptive",
            min_accuracy=0.60,  # Lowered from 0.70
            expected_win_rate=0.67,
            expected_profit_factor=2.34,
            expected_trades_per_year=5,  # Updated target
            expected_expectancy=0.018,
            risk_level="low",
            trade_frequency="low",
            suitable_for=["risk_averse", "capital_preservation", "beginners"]
        )
        
        # Balanced Profile - Adjusted for realistic trading frequency (5-10 trades/year target)
        # Moderate percentiles for reasonable trading opportunities
        profiles["balanced"] = FilterProfile(
            name="balanced",
            display_name="Balanced (Optimal Expectancy)",
            description="Best risk-adjusted returns with good trade frequency (5-10 trades/year)",
            objective="optimal_expectancy",
            training_params={"success": 2.5, "stop": 2.5, "partial": 2.0, "overshoot": -2.0},
            buy_threshold=0.288,  # OPTIMIZED: Updated from 0.52 based on filter optimization results
            sell_threshold=0.48,  # Raised from 0.45
            buy_threshold_percentile=22,  # OPTIMIZED: Updated from 40th to 22nd percentile
            sell_threshold_percentile=40,  # Lowered from 60 (top 60% instead of top 40%)
            threshold_mode="adaptive",
            min_accuracy=0.55,  # Lowered from 0.60
            expected_win_rate=0.65,
            expected_profit_factor=2.8,
            expected_trades_per_year=8,  # Updated target (5-10 range)
            expected_expectancy=0.025,
            risk_level="medium",
            trade_frequency="medium",
            suitable_for=["general_trading", "consistent_returns", "intermediate"]
        )
        
        # Aggressive Profile - Adjusted for higher trading frequency (10-20 trades/year target)
        # Lower percentiles to capture more trading opportunities
        profiles["aggressive"] = FilterProfile(
            name="aggressive",
            display_name="Aggressive (Maximum Profit)",
            description="Maximize gains with higher risk tolerance (10-20 trades/year)",
            objective="maximum_profit",
            training_params={"success": 4.0, "stop": 2.5, "partial": 2.5, "overshoot": -2.0},
            buy_threshold=0.288,  # OPTIMIZED: Updated from 0.48 based on filter optimization results
            sell_threshold=0.45,  # Raised from 0.40
            buy_threshold_percentile=22,  # OPTIMIZED: Updated from 30th to 22nd percentile
            sell_threshold_percentile=30,  # Lowered from 40 (top 70% instead of top 60%)
            threshold_mode="adaptive",
            min_accuracy=0.50,  # Lowered from 0.55
            expected_win_rate=0.58,
            expected_profit_factor=3.2,
            expected_trades_per_year=15,  # Updated target (10-20 range)
            expected_expectancy=0.035,
            risk_level="high",
            trade_frequency="high",  # Changed from medium to high
            suitable_for=["growth_focus", "risk_tolerant", "experienced"]
        )
        
        # Volume Trader Profile - Focused on BUY signal optimization for higher frequency
        # Very low buy percentile to capture maximum opportunities
        profiles["volume"] = FilterProfile(
            name="volume",
            display_name="Volume Trader (High Frequency)",
            description="Maximum trading opportunities focused on buy signal quality (20+ trades/year)",
            objective="high_frequency",
            training_params={"success": 2.0, "stop": 2.0, "partial": 1.5, "overshoot": -1.5},
            buy_threshold=0.288,  # OPTIMIZED: Updated from 0.45 based on filter optimization results
            sell_threshold=0.35,  # Less important - keeping as secondary
            buy_threshold_percentile=22,  # OPTIMIZED: Updated from 20th to 22nd percentile
            sell_threshold_percentile=20,  # Secondary concern
            threshold_mode="adaptive",
            min_accuracy=0.50,  # Lowered from 0.52
            expected_win_rate=0.55,
            expected_profit_factor=2.1,
            expected_trades_per_year=25,  # Updated target (20+ trades)
            expected_expectancy=0.015,
            risk_level="medium",
            trade_frequency="high",
            suitable_for=["active_trading", "frequent_signals", "experienced_traders"]
        )
        
        return profiles
    
    def _load_optimization_results(self):
        """Load optimization results from symbol-specific directories and consolidated index."""
        # Load from new consolidated index
        consolidated_file = Path("cache/profile_recommendations_index.json")
        if consolidated_file.exists():
            try:
                with open(consolidated_file, 'r') as f:
                    consolidated = json.load(f)
                
                for symbol, symbol_data in consolidated.get("symbols", {}).items():
                    # Load symbol-specific recommendations
                    symbol_file = Path(symbol_data["file_path"])
                    if symbol_file.exists():
                        with open(symbol_file, 'r') as f:
                            recommendations = json.load(f)
                        
                        # Convert to DataFrame format for compatibility
                        profiles_data = []
                        for profile_name, profile_data in recommendations.get("profiles", {}).items():
                            profiles_data.append({
                                'scenario_name': profile_data.get('scenario', profile_name),
                                'threshold': profile_data.get('threshold', 0.5),
                                'win_rate': profile_data.get('win_rate', 0.6),
                                'profit_factor': profile_data.get('profit_factor', 2.0),
                                'expectancy': profile_data.get('expectancy', 0.02),
                                'total_trades': profile_data.get('total_trades', 30),
                                'training_success_target': profile_data.get('training_params', {}).get('success', 2.5),
                                'training_stop_target': profile_data.get('training_params', {}).get('stop', 2.5)
                            })
                        
                        if profiles_data:
                            self.optimization_cache[symbol] = pd.DataFrame(profiles_data)
                            logger.info(f"Loaded optimization recommendations for {symbol}")
                
            except Exception as e:
                logger.warning(f"Could not load consolidated recommendations: {e}")
        
        # Fallback: Look for legacy CSV files
        result_files = glob.glob("complete_pipeline_optimization_*.csv")
        for file_path in result_files:
            try:
                df = pd.read_csv(file_path)
                if 'symbol' in df.columns:
                    symbol = df['symbol'].iloc[0] if len(df) > 0 else 'DEFAULT'
                else:
                    symbol = 'DEFAULT'
                
                if symbol not in self.optimization_cache:
                    self.optimization_cache[symbol] = df
                    logger.info(f"Loaded legacy optimization results from {file_path}")
                
            except Exception as e:
                logger.warning(f"Could not load optimization results from {file_path}: {e}")
    
    def get_profile(self, name: str, model: str = 'chronos', symbol: str = 'SPY') -> Optional[FilterProfile]:
        """
        Get a filter profile by name, with model-specific optimization results if available.
        
        Args:
            name: Profile name (conservative, balanced, aggressive, volume)
            model: Model name (chronos, toto) for loading model-specific optimization results
            symbol: Symbol to check for model-specific optimization results
            
        Returns:
            FilterProfile object or None if not found
        """
        profile = self.profiles.get(name.lower())
        if not profile:
            return None
            
        # Try to load model-specific optimization results if available
        model_specific_profile = self._get_model_specific_profile(name.lower(), model, symbol)
        if model_specific_profile:
            return model_specific_profile
            
        return profile
    
    def _get_model_specific_profile(self, profile_name: str, model: str, symbol: str) -> Optional[FilterProfile]:
        """
        Load model-specific profile metrics from optimization results if available.
        
        Args:
            profile_name: Profile name (conservative, balanced, etc.)
            model: Model name (chronos, toto)
            symbol: Symbol to load results for
            
        Returns:
            FilterProfile with model-specific metrics or None if not available
        """
        # Path to optimization results - updated to match actual optimization script output
        # The optimization script saves to cache/symbols/US_ETF_{SYMBOL}/recommendations.json
        symbols_results_path = Path(f"cache/symbols/US_ETF_{symbol}/recommendations.json")
        
        # Check if the symbol-specific results exist and match the model
        if symbols_results_path.exists():
            try:
                with open(symbols_results_path, 'r') as f:
                    data = json.load(f)
                    result_model = data.get('model', 'chronos')
                    if result_model == model:
                        model_results_path = symbols_results_path
                    else:
                        # Results don't match requested model
                        return None
            except Exception:
                return None
        else:
            # Legacy path fallback for backward compatibility
            model_results_path = Path(f"cache/profile_recommendations/{symbol}_{model}/recommendations.json")
            if not model_results_path.exists():
                results_path = Path(f"cache/profile_recommendations/{symbol}/recommendations.json")
                if results_path.exists():
                    try:
                        with open(results_path, 'r') as f:
                            data = json.load(f)
                            result_model = data.get('model', 'chronos')
                            if result_model != model:
                                return None
                            model_results_path = results_path
                    except Exception:
                        return None
                else:
                    return None
        
        try:
            with open(model_results_path, 'r') as f:
                data = json.load(f)
                
            profile_data = data.get('profiles', {}).get(profile_name)
            if not profile_data:
                return None
                
            # Get base profile for structure
            base_profile = self.profiles.get(profile_name)
            if not base_profile:
                return None
                
            # Create new profile with updated metrics from optimization results
            return FilterProfile(
                name=base_profile.name,
                display_name=base_profile.display_name,
                description=base_profile.description,
                objective=base_profile.objective,
                training_params=profile_data.get('training_params', base_profile.training_params),
                buy_threshold=profile_data.get('threshold', base_profile.buy_threshold),
                sell_threshold=base_profile.sell_threshold,
                min_accuracy=base_profile.min_accuracy,
                expected_win_rate=profile_data.get('win_rate', base_profile.expected_win_rate),
                expected_profit_factor=profile_data.get('profit_factor', base_profile.expected_profit_factor),
                expected_trades_per_year=profile_data.get('total_trades', base_profile.expected_trades_per_year),
                expected_expectancy=profile_data.get('expectancy', base_profile.expected_expectancy),
                risk_level=base_profile.risk_level,
                trade_frequency=base_profile.trade_frequency,
                suitable_for=base_profile.suitable_for
            )
            
        except Exception as e:
            logger.warning(f"Could not load model-specific profile {profile_name} for {model}/{symbol}: {e}")
            return None
    
    def get_all_profiles(self) -> List[FilterProfile]:
        """Get all available profiles."""
        return list(self.profiles.values())
    
    def get_profile_names(self) -> List[str]:
        """Get list of available profile names for UI."""
        return [p.display_name for p in self.profiles.values()]
    
    def discover_available_filters(self, symbol: str, exchange: str = "US", 
                                  strategy: str = "SMI") -> Dict[str, List[str]]:
        """
        Discover which filter profiles are available for a given symbol/strategy.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange code
            strategy: Strategy type
            
        Returns:
            Dictionary mapping profile names to available filter paths
        """
        available = {}
        
        for profile_name, profile in self.profiles.items():
            filter_path = profile.get_filter_path(symbol, exchange, strategy)
            if filter_path:
                available[profile_name] = filter_path
        
        return available
    
    def get_best_profile_for_objective(self, objective: str, 
                                      symbol: Optional[str] = None) -> Optional[FilterProfile]:
        """
        Get the best profile for a specific trading objective.
        
        Args:
            objective: Trading objective (e.g., "high_win_rate", "maximum_profit")
            symbol: Optional symbol to check optimization results
            
        Returns:
            Best FilterProfile for the objective
        """
        # If we have optimization results for this symbol, use them
        if symbol and symbol in self.optimization_cache:
            return self._get_optimized_profile(symbol, objective)
        
        # Otherwise return preset profile matching objective
        for profile in self.profiles.values():
            if profile.objective == objective:
                return profile
        
        return None
    
    def _get_optimized_profile(self, symbol: str, objective: str) -> Optional[FilterProfile]:
        """Get profile based on actual optimization results for symbol."""
        df = self.optimization_cache.get(symbol)
        if df is None or df.empty:
            return None
        
        # Select best configuration based on objective
        if objective == "high_win_rate":
            best_row = df.loc[df['win_rate'].idxmax()]
        elif objective == "maximum_profit":
            best_row = df.loc[df['expectancy'].idxmax()]
        elif objective == "optimal_expectancy":
            # Use composite score if available
            if 'composite_score' in df.columns:
                best_row = df.loc[df['composite_score'].idxmax()]
            else:
                best_row = df.loc[df['expectancy'].idxmax()]
        else:
            return None
        
        # Create custom profile from optimization results
        return FilterProfile(
            name=f"optimized_{objective}",
            display_name=f"Optimized ({objective.replace('_', ' ').title()})",
            description=f"Custom optimized for {symbol}",
            objective=objective,
            training_params={
                "success": best_row.get('training_success_target', 2.5),
                "stop": best_row.get('training_stop_target', 2.5),
                "partial": best_row.get('training_partial_target', 2.0),
                "overshoot": -best_row.get('training_stop_target', 2.5) * 0.75
            },
            buy_threshold=best_row.get('threshold', 0.5),
            sell_threshold=best_row.get('threshold', 0.5),
            min_accuracy=best_row.get('threshold', 0.5),
            expected_win_rate=best_row.get('win_rate', 0.6),
            expected_profit_factor=best_row.get('profit_factor', 2.0),
            expected_trades_per_year=int(best_row.get('total_trades', 30)),
            expected_expectancy=best_row.get('expectancy', 0.02),
            risk_level="medium",
            trade_frequency="medium",
            suitable_for=["optimized", symbol]
        )
    
    def export_summary(self, output_path: str = "cache/filter_optimization_summary.json"):
        """
        Export consolidated filter profile summary (legacy format for compatibility).
        
        Args:
            output_path: Path to save summary JSON
        """
        summary = {
            "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
            "optimization_results": {},
            "metadata": {
                "version": "1.0",
                "last_updated": pd.Timestamp.now().isoformat()
            }
        }
        
        # Add optimization results if available
        for symbol, df in self.optimization_cache.items():
            # Get top configurations for each objective
            summary["optimization_results"][symbol] = {
                "best_win_rate": self._get_best_config(df, "win_rate"),
                "best_expectancy": self._get_best_config(df, "expectancy"),
                "best_profit_factor": self._get_best_config(df, "profit_factor"),
                "total_configurations_tested": len(df)
            }
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported filter profile summary to {output_path}")
    
    def _get_best_config(self, df: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Get best configuration for a specific metric."""
        if metric not in df.columns:
            return {}
        
        best_row = df.loc[df[metric].idxmax()]
        return {
            "scenario": best_row.get('scenario_name', 'unknown'),
            "threshold": best_row.get('threshold', 0.5),
            metric: float(best_row[metric]),
            "trades": int(best_row.get('total_trades', 0)),
            "win_rate": float(best_row.get('win_rate', 0))
        }


# Example usage and testing
if __name__ == "__main__":
    # Quick test of the profile system
    manager = FilterProfileManager()
    
    print("Available Profiles:")
    print("-" * 50)
    for profile in manager.get_all_profiles():
        print(f"\n{profile.display_name}")
        print(f"  Description: {profile.description}")
        print(f"  Expected Metrics:")
        for key, value in profile.get_expected_metrics().items():
            print(f"    - {key}: {value}")
    
    # Test filter discovery
    print("\n\nFilter Discovery for SPY:")
    print("-" * 50)
    available = manager.discover_available_filters("SPY", "US", "SMI")
    for profile_name, path in available.items():
        print(f"  {profile_name}: {path}")
    
    # Export summary
    manager.export_summary()
    print("\n✅ Filter profile summary exported to cache/filter_optimization_summary.json")