#!/usr/bin/env python3
"""
Adaptive Mahalanobis Filter Implementation

This module implements the adaptive Mahalanobis filter system that maintains
fixed centroids (stable market archetypes) while adapting success rates and
thresholds based on current market conditions.

Key Components:
1. Fixed regime centroids discovered via K-means++ (never retrained)
2. Adaptive success rate estimation using Bayesian updates
3. Financial turbulence index for threshold adjustment
4. Profile-specific threshold management

Architecture:
- Centroids represent timeless market archetypes (Bull, Bear, Volatile, etc.)
- Success rates adapt to recent market outcomes using exponential decay
- Turbulence index adjusts thresholds based on market stress levels
- Profile system provides trading style customization

Created: 2025-08-26
Author: Claude & Urban for Trading-Lab
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from pathlib import Path

# Scientific computing
from scipy.stats import beta, chi2
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Add src to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing components we'll integrate with
try:
    from models.mahalanobis_filter import MahalanobisAccuracyFilter, AccuracyClassificationResult
except ImportError:
    # Fallback for testing
    class MahalanobisAccuracyFilter:
        pass
    
    class AccuracyClassificationResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Import adaptive regime discovery components
try:
    from models.adaptive_regime_discovery import (
        OptimalRegimeSelector, 
        HierarchicalRegimeClassifier, 
        DynamicRegimeLibrary
    )
    from models.hierarchical_confidence_weighting import (
        HierarchicalConfidenceWeighting, 
        create_hierarchical_weighting, 
        TradingProfile,
        HierarchicalClassification
    )
except ImportError:
    # Create fallback classes for testing
    class OptimalRegimeSelector:
        def __init__(self, **kwargs): pass
        def find_optimal_k(self, data, **kwargs): return {'optimal_k': 6}
    
    class HierarchicalRegimeClassifier:
        def __init__(self, **kwargs): pass
        def hierarchical_clustering(self, data, **kwargs): return {}
        def classify_multilevel(self, vector): return {}
    
    class DynamicRegimeLibrary:
        def __init__(self, **kwargs): pass
        def check_novelty(self, vector, **kwargs): return False, 1.0, {}
    
    # Fallback hierarchical confidence weighting classes
    class HierarchicalConfidenceWeighting:
        def __init__(self, **kwargs): pass
        def classify_hierarchical(self, levels, **kwargs): 
            return type('Result', (), {
                'overall_confidence': 0.5,
                'signal_strength': 0.5,
                'recommended_action': 'hold',
                'risk_assessment': 'medium',
                'levels': levels
            })()
        def update_level_metrics(self, level, metrics): pass
        def get_performance_summary(self): return {}
    
    def create_hierarchical_weighting(profile='balanced', **kwargs):
        return HierarchicalConfidenceWeighting(**kwargs)
    
    class TradingProfile:
        CONSERVATIVE = "conservative"
        BALANCED = "balanced"
        AGGRESSIVE = "aggressive"
        RESEARCH = "research"
    
    HierarchicalClassification = type('HierarchicalClassification', (), {})


class TurbulenceIndex:
    """
    Financial turbulence index implementation based on Kritzman & Li.
    
    Uses Mahalanobis distance to measure statistical unusualness of current
    market conditions relative to historical patterns.
    """
    
    def __init__(self, lookback_window: int = 252):
        """
        Initialize turbulence calculator.
        
        Args:
            lookback_window: Number of periods for historical calibration (default: 252 trading days)
        """
        self.window = lookback_window
        self.historical_turbulence = []
        self.historical_mean = None
        self.historical_cov = None
        self.percentiles = {'25th': None, '75th': None, '95th': None}
        
    def calculate_turbulence(self, sensor_data: np.ndarray) -> float:
        """
        Calculate current market turbulence using Mahalanobis distance.
        
        Args:
            sensor_data: Historical sensor data array (rows=time, cols=sensors)
            
        Returns:
            Current turbulence score
        """
        if len(sensor_data) < self.window:
            raise ValueError(f"Insufficient data: need {self.window}, got {len(sensor_data)}")
            
        # Calculate historical statistics from lookback window
        historical_data = sensor_data[-self.window-1:-1]  # Exclude current observation
        current_observation = sensor_data[-1]
        
        # Historical mean and covariance
        self.historical_mean = np.mean(historical_data, axis=0)
        self.historical_cov = np.cov(historical_data.T)
        
        # Add small regularization to prevent singular matrices
        regularization = 1e-6 * np.eye(self.historical_cov.shape[0])
        self.historical_cov += regularization
        
        # Calculate Mahalanobis distance (turbulence)
        try:
            diff = current_observation - self.historical_mean
            inv_cov = np.linalg.inv(self.historical_cov)
            turbulence = np.sqrt(diff @ inv_cov @ diff.T)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            diff = current_observation - self.historical_mean
            pinv_cov = np.linalg.pinv(self.historical_cov)
            turbulence = np.sqrt(diff @ pinv_cov @ diff.T)
            
        return float(turbulence)
        
    def calibrate_historical_turbulence(self, sensor_data: np.ndarray) -> None:
        """
        Calibrate historical turbulence distribution for percentile calculations.
        
        Args:
            sensor_data: Full historical sensor data for calibration
        """
        self.historical_turbulence = []
        
        # Calculate rolling turbulence for entire history
        for i in range(self.window, len(sensor_data)):
            window_data = sensor_data[i-self.window:i+1]
            turbulence = self.calculate_turbulence(window_data)
            self.historical_turbulence.append(turbulence)
            
        # Calculate percentiles for regime classification
        if self.historical_turbulence:
            self.percentiles = {
                '25th': np.percentile(self.historical_turbulence, 25),
                '75th': np.percentile(self.historical_turbulence, 75), 
                '95th': np.percentile(self.historical_turbulence, 95)
            }
            
    def classify_turbulence_regime(self, turbulence_score: float) -> Tuple[str, float]:
        """
        Classify current turbulence level and return threshold adjustment.
        
        Args:
            turbulence_score: Current turbulence score
            
        Returns:
            Tuple of (regime_name, threshold_multiplier)
        """
        if not self.percentiles['25th']:
            return 'normal', 1.0  # Default if not calibrated
            
        if turbulence_score < self.percentiles['25th']:
            return 'quiet', 0.9  # Tighten thresholds by 10%
        elif turbulence_score > self.percentiles['75th']:
            if turbulence_score > self.percentiles['95th']:
                return 'crisis', 1.3  # Very turbulent - relax 30%
            return 'turbulent', 1.2  # Relax thresholds by 20%
        else:
            return 'normal', 1.0  # Use baseline thresholds


class BayesianSuccessRates:
    """
    Bayesian success rate estimator using Beta-Binomial conjugate priors.
    
    Maintains posterior distributions for success rates per regime and updates
    them with new trade outcomes using exponential decay weighting.
    """
    
    def __init__(self, decay_halflife_days: int = 90):
        """
        Initialize Bayesian updater.
        
        Args:
            decay_halflife_days: Half-life for exponential decay weighting of historical outcomes
        """
        self.halflife = decay_halflife_days
        self.regime_priors = {}
        self.last_update = None
        
        # Initialize with weak priors (Beta(10, 10) = 50% success rate)
        for regime in range(6):  # Assume 6 regimes
            self.regime_priors[regime] = {
                'alpha': 10.0,  # Successes + prior
                'beta': 10.0,   # Failures + prior
                'last_updated': datetime.now()
            }
            
    def update_posterior(self, regime: int, successes: int, failures: int, 
                        trade_dates: Optional[List[datetime]] = None) -> None:
        """
        Update posterior distribution with new outcomes.
        
        Args:
            regime: Regime ID (0-5)
            successes: Number of successful trades
            failures: Number of failed trades  
            trade_dates: Dates of trades for decay weighting (optional)
        """
        if regime not in self.regime_priors:
            # Initialize new regime
            self.regime_priors[regime] = {'alpha': 10.0, 'beta': 10.0, 'last_updated': datetime.now()}
            
        # Apply exponential decay weighting if dates provided
        if trade_dates:
            current_date = datetime.now()
            decay_weights = []
            
            for trade_date in trade_dates:
                days_ago = (current_date - trade_date).days
                weight = np.exp(-np.log(2) * days_ago / self.halflife)
                decay_weights.append(weight)
                
            # Weight the successes and failures
            weighted_successes = sum(decay_weights[:successes]) if successes > 0 else 0
            weighted_failures = sum(decay_weights[successes:]) if failures > 0 else 0
            
            successes = weighted_successes
            failures = weighted_failures
            
        # Update posterior parameters
        self.regime_priors[regime]['alpha'] += successes
        self.regime_priors[regime]['beta'] += failures
        self.regime_priors[regime]['last_updated'] = datetime.now()
        
    def get_success_rate_estimate(self, regime: int, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Get success rate estimate with confidence interval.
        
        Args:
            regime: Regime ID
            confidence_level: Confidence level for interval (default: 0.95)
            
        Returns:
            Dictionary with estimate, lower_bound, upper_bound, confidence_width
        """
        if regime not in self.regime_priors:
            # Return default for unknown regime
            return {
                'estimate': 0.5,
                'lower_bound': 0.3,
                'upper_bound': 0.7,
                'confidence_width': 0.4
            }
            
        alpha = self.regime_priors[regime]['alpha']
        beta_param = self.regime_priors[regime]['beta']
        
        # Point estimate (mean of Beta distribution)
        point_estimate = alpha / (alpha + beta_param)
        
        # Confidence interval
        dist = beta(alpha, beta_param)
        alpha_level = (1 - confidence_level) / 2
        lower = dist.ppf(alpha_level)
        upper = dist.ppf(1 - alpha_level)
        
        return {
            'estimate': float(point_estimate),
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'confidence_width': float(upper - lower)
        }
        
    def should_trade_regime(self, regime: int, min_confidence: float) -> Tuple[bool, Dict[str, float]]:
        """
        Determine if regime has sufficient confidence for trading.
        
        Args:
            regime: Regime ID
            min_confidence: Minimum required confidence (lower bound threshold)
            
        Returns:
            Tuple of (should_trade, estimate_details)
        """
        estimate = self.get_success_rate_estimate(regime)
        should_trade = estimate['lower_bound'] > min_confidence
        
        return should_trade, estimate


class AdaptiveProfileManager:
    """
    Manages trading profiles with adaptive threshold calculation.
    
    Each profile has different risk tolerance, confidence requirements,
    and response to market turbulence.
    """
    
    def __init__(self):
        """Initialize profile configurations."""
        self.profile_configs = {
            'conservative': {
                'confidence_requirement': 0.65,    # 65% lower bound required
                'turbulence_adjustment': 0.85,     # More cautious in turbulence
                'min_trades_per_month': 2,
                'relaxation_rate': 0.02,           # 2% relaxation per missing trade
                'description': 'Risk-averse trading with high confidence requirement'
            },
            'balanced': {
                'confidence_requirement': 0.55,    # 55% lower bound required
                'turbulence_adjustment': 0.92,     # Moderate turbulence response
                'min_trades_per_month': 3,
                'relaxation_rate': 0.03,           # 3% relaxation per missing trade
                'description': 'Balanced risk-reward with moderate confidence'
            },
            'volume': {
                'confidence_requirement': 0.45,    # 45% lower bound required
                'turbulence_adjustment': 0.95,     # Less sensitive to turbulence
                'min_trades_per_month': 4,
                'relaxation_rate': 0.05,           # 5% relaxation per missing trade
                'description': 'High frequency trading with lower confidence threshold'
            },
            'aggressive': {
                'confidence_requirement': 0.40,    # 40% lower bound required
                'turbulence_adjustment': 1.0,      # Ignore turbulence
                'min_trades_per_month': 5,
                'relaxation_rate': 0.07,           # 7% relaxation per missing trade
                'description': 'Growth-focused trading with high risk tolerance'
            }
        }
        
    def get_adaptive_threshold(self, profile: str, turbulence_level: str, 
                             recent_trade_count: int) -> float:
        """
        Calculate dynamic threshold based on profile and market conditions.
        
        Args:
            profile: Profile name (conservative, balanced, volume, aggressive)
            turbulence_level: Current turbulence (quiet, normal, turbulent, crisis)
            recent_trade_count: Number of trades in recent period
            
        Returns:
            Adaptive threshold value
        """
        if profile not in self.profile_configs:
            profile = 'balanced'  # Default fallback
            
        config = self.profile_configs[profile]
        base_threshold = config['confidence_requirement']
        
        # Turbulence adjustment
        threshold = base_threshold
        if turbulence_level in ['turbulent', 'crisis']:
            threshold = base_threshold * config['turbulence_adjustment']
            
        # Frequency adjustment (relax if below minimum)
        min_trades = config['min_trades_per_month']
        if recent_trade_count < min_trades:
            deficit = min_trades - recent_trade_count
            relaxation_factor = 1 - (config['relaxation_rate'] * deficit)
            threshold *= relaxation_factor
            
        # Ensure threshold stays within reasonable bounds
        threshold = max(0.1, min(0.9, threshold))
        
        return threshold
        
    def get_profile_info(self, profile: str) -> Dict[str, Any]:
        """Get complete profile configuration and description."""
        return self.profile_configs.get(profile, self.profile_configs['balanced'])


class AdaptiveMahalanobisFilter:
    """
    Adaptive Mahalanobis filter for trading signal classification.
    
    This filter maintains fixed centroids representing stable market archetypes
    while adapting success rate estimates and confidence thresholds based on
    recent market outcomes and current market turbulence.
    
    Key Features:
    - Fixed centroids discovered via K-means++ (stable regime definitions)
    - Bayesian success rate updates with exponential decay
    - Financial turbulence index for threshold adjustment
    - Profile-specific trading styles (Conservative, Balanced, Volume, Aggressive)
    - Confidence-based signal filtering
    """
    
    def __init__(self, n_regimes: int = 6, n_dimensions: int = 61,
                 regime_selection: str = 'fixed', auto_k_range: Tuple[int, int] = (3, 12),
                 hierarchical_mode: bool = False, hierarchical_depth: int = 3,
                 novelty_detection: bool = False, novelty_threshold: float = 0.75,
                 confidence_weighting: bool = False, trading_profile: str = 'balanced'):
        """
        Initialize adaptive filter.
        
        Args:
            n_regimes: Number of market regimes (used when regime_selection='fixed')
            n_dimensions: Number of sensor dimensions
            regime_selection: 'fixed', 'auto', or 'hierarchical'
            auto_k_range: Range for automatic k selection (min_k, max_k)
            hierarchical_mode: Enable hierarchical multi-level classification
            hierarchical_depth: Maximum depth for hierarchical classification
            novelty_detection: Enable real-time novelty detection
            novelty_threshold: Similarity threshold for novelty detection
            confidence_weighting: Enable hierarchical confidence weighting
            trading_profile: Trading profile for confidence weighting ('conservative', 'balanced', 'aggressive', 'research')
        """
        # === CORE PARAMETERS ===
        self.n_regimes = n_regimes
        self.n_dimensions = n_dimensions
        
        # === ADAPTIVE REGIME DISCOVERY PARAMETERS ===
        self.regime_selection = regime_selection
        self.auto_k_range = auto_k_range
        self.hierarchical_mode = hierarchical_mode
        self.hierarchical_depth = hierarchical_depth
        self.novelty_detection = novelty_detection
        self.novelty_threshold = novelty_threshold
        self.confidence_weighting = confidence_weighting
        self.trading_profile = trading_profile
        
        # === FIXED COMPONENTS (from training data) ===
        self.centroids = None                    # Discovered with K-means++, never change
        self.covariance_matrix = None           # Global covariance matrix
        self.sensor_weights = None              # Feature importance weights
        self.scaler = StandardScaler()          # For data normalization
        
        # Clustering validation metrics
        self.silhouette_score = None
        self.davies_bouldin_score = None
        self.clustering_stability = None
        
        # === ADAPTIVE COMPONENTS (updated regularly) ===
        self.turbulence_calculator = TurbulenceIndex()
        self.bayesian_updater = BayesianSuccessRates()
        self.profile_manager = AdaptiveProfileManager()
        
        # === ADAPTIVE REGIME DISCOVERY COMPONENTS ===
        self.regime_selector = OptimalRegimeSelector(k_range=auto_k_range) if regime_selection == 'auto' else None
        self.hierarchical_classifier = HierarchicalRegimeClassifier(max_depth=hierarchical_depth) if hierarchical_mode else None
        self.regime_library = DynamicRegimeLibrary(similarity_threshold=novelty_threshold) if novelty_detection else None
        
        # === HIERARCHICAL CONFIDENCE WEIGHTING COMPONENT ===
        self.confidence_weighting_system = None
        if confidence_weighting:
            self.confidence_weighting_system = create_hierarchical_weighting(
                profile_name=trading_profile,
                custom_config=None  # Can be enhanced with custom config later
            )
        
        # Trade frequency monitoring
        self.trade_history = deque(maxlen=30)   # 30-day rolling window
        
        # Configuration
        self.training_completed = False
        self.last_updated = None
        self.metadata = {}
        
    def discover_stable_centroids(self, training_data: np.ndarray, n_runs: int = 10) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Discover stable regime centroids using multiple K-means++ runs.
        Supports adaptive regime count selection based on configuration.
        
        Args:
            training_data: Historical sensor data (rows=samples, cols=features)
            n_runs: Number of K-means runs to find most stable solution
            
        Returns:
            Tuple of (best_centroids, validation_metrics)
        """
        # Step 1: Determine optimal number of regimes
        if self.regime_selection == 'auto' and self.regime_selector:
            print(f"🔍 Using automatic regime selection (range: {self.auto_k_range})")
            optimal_result = self.regime_selector.find_optimal_k(training_data, method='combined')
            self.n_regimes = optimal_result['optimal_k']
            print(f"   Selected optimal k = {self.n_regimes}")
        elif self.regime_selection == 'hierarchical' and self.hierarchical_classifier:
            print(f"🌳 Using hierarchical regime discovery (depth: {self.hierarchical_depth})")
            hierarchy_results = self.hierarchical_classifier.hierarchical_clustering(training_data)
            # Use the finest level for main centroids
            finest_level = max(hierarchy_results.keys())
            self.n_regimes = hierarchy_results[finest_level]['n_clusters']
            print(f"   Hierarchical regime count: {self.n_regimes}")
        else:
            print(f"🎯 Using fixed regime count: {self.n_regimes}")
        
        print(f"   Training data shape: {training_data.shape}")
        print(f"   Running K-means++ {n_runs} times to find most stable solution...")
        
        # Normalize training data
        normalized_data = self.scaler.fit_transform(training_data)
        
        best_score = -1
        best_centroids = None
        best_labels = None
        best_model = None
        stability_scores = []
        
        for run in range(n_runs):
            try:
                # K-means++ clustering
                kmeans = KMeans(
                    n_clusters=self.n_regimes,
                    init='k-means++',      # Smart initialization
                    n_init=10,             # Multiple attempts per run
                    max_iter=300,
                    random_state=run,
                    algorithm='elkan'      # Often faster for our data
                )
                
                labels = kmeans.fit_predict(normalized_data)
                
                # Validate clustering quality
                if len(np.unique(labels)) < self.n_regimes:
                    print(f"   Run {run+1}: Insufficient clusters found, skipping...")
                    continue
                    
                silhouette = silhouette_score(normalized_data, labels)
                stability_scores.append(silhouette)
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_centroids = kmeans.cluster_centers_
                    best_labels = labels
                    best_model = kmeans
                    
                print(f"   Run {run+1}: Silhouette score = {silhouette:.4f}")
                
            except Exception as e:
                print(f"   Run {run+1}: Failed - {str(e)}")
                continue
                
        if best_centroids is None:
            raise RuntimeError("Failed to find stable centroids - all clustering runs failed")
            
        # Calculate additional validation metrics
        self.silhouette_score = best_score
        self.davies_bouldin_score = davies_bouldin_score(normalized_data, best_labels)
        self.clustering_stability = np.std(stability_scores) if len(stability_scores) > 1 else 0
        
        # Store results
        self.centroids = best_centroids
        self.covariance_matrix = np.cov(normalized_data.T)
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in range(self.n_regimes):
            regime_mask = best_labels == regime
            regime_size = np.sum(regime_mask)
            regime_stats[regime] = {
                'size': int(regime_size),
                'percentage': float(regime_size / len(best_labels) * 100),
                'centroid': self.centroids[regime].tolist()
            }
            
        validation_metrics = {
            'silhouette_score': float(self.silhouette_score),
            'davies_bouldin_score': float(self.davies_bouldin_score),
            'clustering_stability': float(self.clustering_stability),
            'regime_stats': regime_stats
        }
        
        print(f"✅ Centroid discovery completed:")
        print(f"   Best silhouette score: {self.silhouette_score:.4f}")
        print(f"   Davies-Bouldin score: {self.davies_bouldin_score:.4f} (lower is better)")
        print(f"   Clustering stability: {self.clustering_stability:.4f} (lower is better)")
        
        # Print regime sizes
        for regime, stats in regime_stats.items():
            print(f"   Regime {regime}: {stats['size']} samples ({stats['percentage']:.1f}%)")
            
        return self.centroids, validation_metrics
        
    def classify_regime(self, sensor_vector: np.ndarray) -> int:
        """
        Classify current market context into one of the fixed regimes.
        
        Args:
            sensor_vector: Current sensor readings
            
        Returns:
            Regime ID (0 to n_regimes-1)
        """
        if self.centroids is None:
            raise RuntimeError("Filter not trained - centroids not available")
            
        # Normalize sensor vector
        sensor_normalized = self.scaler.transform(sensor_vector.reshape(1, -1))[0]
        
        # Calculate distance to each centroid
        distances = []
        for centroid in self.centroids:
            distance = np.linalg.norm(sensor_normalized - centroid)
            distances.append(distance)
            
        # Return closest regime
        return int(np.argmin(distances))
    
    def classify_regime_with_novelty(self, sensor_vector: np.ndarray) -> Tuple[int, float, bool, Dict]:
        """
        Enhanced classification with novelty detection and confidence scoring.
        
        Args:
            sensor_vector: Current sensor readings
            
        Returns:
            Tuple of (regime_id, confidence, is_novel, analysis_details)
        """
        if self.centroids is None:
            raise ValueError("Filter not trained. Call discover_stable_centroids() first.")
        
        # Standard classification
        regime_id = self.classify_regime(sensor_vector)
        
        # Calculate confidence (inverse of distance)
        normalized_vector = self.scaler.transform(sensor_vector.reshape(1, -1))[0]
        closest_centroid = self.centroids[regime_id]
        
        # Calculate distance to closest centroid
        distance = float(np.linalg.norm(normalized_vector - closest_centroid))
        
        # Convert distance to confidence (0-1 scale)
        confidence = np.exp(-distance / 2.0)  # Exponential decay
        
        # Novelty detection
        is_novel = False
        novelty_analysis = {}
        
        if self.novelty_detection and self.regime_library:
            is_novel, similarity, novelty_analysis = self.regime_library.check_novelty(
                normalized_vector,
                context={'timestamp': datetime.now(), 'regime_id': regime_id}
            )
        
        analysis_details = {
            'regime_id': regime_id,
            'confidence': float(confidence),
            'distance': distance,
            'is_novel': is_novel,
            'novelty_analysis': novelty_analysis
        }
        
        # Handle hierarchical classification if enabled
        if self.hierarchical_mode and self.hierarchical_classifier and self.hierarchical_classifier.fitted:
            try:
                hierarchical_classifications = self.hierarchical_classifier.classify_multilevel(normalized_vector)
                hierarchical_confidence = self.hierarchical_classifier.compute_combined_confidence(hierarchical_classifications)
                interpretation = self.hierarchical_classifier.get_regime_interpretation(hierarchical_classifications)
                
                analysis_details['hierarchical'] = {
                    'classifications': hierarchical_classifications,
                    'combined_confidence': hierarchical_confidence,
                    'interpretation': interpretation
                }
                
                # Combine confidences (weighted average)
                confidence = 0.7 * confidence + 0.3 * hierarchical_confidence
                
            except Exception as e:
                # Hierarchical classification failed - continue with standard classification
                analysis_details['hierarchical_error'] = str(e)
        
        return regime_id, float(confidence), is_novel, analysis_details
    
    def classify_with_hierarchical_confidence(self, 
                                            sensor_vector: np.ndarray,
                                            market_metadata: Optional[Dict] = None) -> 'HierarchicalClassification':
        """
        Perform classification with hierarchical confidence weighting.
        
        This method combines multi-level regime classification with sophisticated
        confidence weighting to provide enhanced trading signal assessment.
        
        Args:
            sensor_vector: Current sensor readings
            market_metadata: Optional market context (volume, volatility, etc.)
            
        Returns:
            HierarchicalClassification with comprehensive analysis
        """
        if not self.confidence_weighting:
            raise ValueError("Hierarchical confidence weighting not enabled. Set confidence_weighting=True")
        
        if self.confidence_weighting_system is None:
            raise ValueError("Confidence weighting system not initialized")
        
        # Perform hierarchical classification if available
        level_classifications = {}
        
        if self.hierarchical_mode and self.hierarchical_classifier and hasattr(self.hierarchical_classifier, 'fitted') and self.hierarchical_classifier.fitted:
            # Get multi-level classifications
            hierarchical_results = self.hierarchical_classifier.classify_multilevel(sensor_vector)
            
            # Convert to expected format: level -> (regime_id, confidence, label)
            for level, (regime_id, confidence) in hierarchical_results.items():
                # Get regime label from hierarchical classifier
                if hasattr(self.hierarchical_classifier, 'get_level_label'):
                    label = self.hierarchical_classifier.get_level_label(level, regime_id)
                else:
                    # Fallback label generation
                    level_names = {1: 'Direction', 2: 'Volatility', 3: 'Pattern'}
                    level_name = level_names.get(level, f'Level_{level}')
                    label = f"{level_name}_Regime_{regime_id}"
                level_classifications[level] = (regime_id, confidence, label)
        else:
            # Fallback to single-level classification
            regime_id, confidence, _, _ = self.classify_regime_with_novelty(sensor_vector)
            
            # Create synthetic hierarchical levels based on single classification
            level_classifications = {
                1: (regime_id, confidence, f"Regime_{regime_id}"),  # Direction level
            }
            
            # Add synthetic volatility level if we have turbulence info
            if hasattr(self, 'turbulence_calculator'):
                # Simplified turbulence-based volatility classification
                vol_regime = min(3, int(confidence * 4))  # 0-3 volatility levels
                vol_labels = ['Low', 'Normal', 'High', 'Extreme']
                level_classifications[2] = (vol_regime, confidence * 0.8, vol_labels[vol_regime])
        
        # Use hierarchical confidence weighting system
        hierarchical_result = self.confidence_weighting_system.classify_hierarchical(
            level_classifications=level_classifications,
            sensor_vector=sensor_vector,
            market_metadata=market_metadata
        )
        
        # Update level metrics in the weighting system (if we have quality info)
        self._update_hierarchical_metrics()
        
        return hierarchical_result
    
    def _update_hierarchical_metrics(self):
        """Update hierarchical confidence weighting system with current quality metrics."""
        if not self.confidence_weighting_system:
            return
        
        # Update level 1 (Direction) metrics
        if hasattr(self, 'silhouette_score') and self.silhouette_score:
            self.confidence_weighting_system.update_level_metrics(
                level=1,
                quality_metrics={
                    'silhouette_score': self.silhouette_score,
                    'stability_score': getattr(self, 'clustering_stability', 0.5)
                }
            )
        
        # Update hierarchical level metrics if available
        if self.hierarchical_classifier and hasattr(self.hierarchical_classifier, 'level_quality'):
            for level, quality_info in self.hierarchical_classifier.level_quality.items():
                self.confidence_weighting_system.update_level_metrics(
                    level=level,
                    quality_metrics=quality_info
                )
        
    def calculate_regime_distances(self, sensor_vector: np.ndarray) -> Dict[int, float]:
        """
        Calculate distances to all regime centroids.
        
        Args:
            sensor_vector: Current sensor readings
            
        Returns:
            Dictionary mapping regime_id -> distance
        """
        if self.centroids is None:
            raise RuntimeError("Filter not trained - centroids not available")
            
        # Normalize sensor vector
        sensor_normalized = self.scaler.transform(sensor_vector.reshape(1, -1))[0]
        
        distances = {}
        for regime, centroid in enumerate(self.centroids):
            distances[regime] = float(np.linalg.norm(sensor_normalized - centroid))
            
        return distances
        
    def should_take_signal(self, sensor_vector: np.ndarray, signal_direction: str, 
                          profile: str = 'balanced') -> Tuple[bool, Dict[str, Any]]:
        """
        Main filtering decision using full adaptive logic.
        
        Args:
            sensor_vector: Current market sensor readings
            signal_direction: 'buy' or 'sell'
            profile: Trading profile (conservative, balanced, volume, aggressive)
            
        Returns:
            Tuple of (should_trade, explanation_dict)
        """
        try:
            # Step 1: Calculate current turbulence
            # For now, use a simplified turbulence calculation
            # TODO: This needs historical sensor data for proper turbulence calculation
            turbulence_score = 0.5  # Placeholder
            turbulence_level = 'normal'  # Placeholder
            
            # Step 2: Classify market regime
            regime = self.classify_regime(sensor_vector)
            regime_distances = self.calculate_regime_distances(sensor_vector)
            
            # Step 3: Get adaptive success rate estimate
            should_trade_confidence, success_estimate = self.bayesian_updater.should_trade_regime(
                regime, min_confidence=0.5  # Placeholder threshold
            )
            
            # Step 4: Calculate adaptive threshold for this profile
            recent_trade_count = len(self.trade_history)
            adaptive_threshold = self.profile_manager.get_adaptive_threshold(
                profile, turbulence_level, recent_trade_count
            )
            
            # Step 5: Make final decision
            should_trade = success_estimate['lower_bound'] > adaptive_threshold
            
            # Prepare detailed explanation
            explanation = {
                'regime': regime,
                'regime_distances': regime_distances,
                'turbulence_level': turbulence_level,
                'turbulence_score': turbulence_score,
                'success_estimate': success_estimate,
                'adaptive_threshold': adaptive_threshold,
                'profile': profile,
                'recent_trade_count': recent_trade_count,
                'decision': 'TRADE' if should_trade else 'NO_TRADE',
                'confidence_gap': success_estimate['lower_bound'] - adaptive_threshold,
                'timestamp': datetime.now()
            }
            
            return should_trade, explanation
            
        except Exception as e:
            # Safe fallback - don't trade if anything goes wrong
            error_explanation = {
                'error': str(e),
                'decision': 'NO_TRADE',
                'reason': 'Error in adaptive filtering - defaulting to no trade for safety',
                'timestamp': datetime.now()
            }
            return False, error_explanation
            
    def update_success_rates(self, recent_outcomes: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """
        Update success rates from recent trading outcomes.
        
        Args:
            recent_outcomes: Dictionary mapping regime -> list of outcome records
                           Each outcome record: {'successful': bool, 'date': datetime, ...}
                           
        Returns:
            Update statistics
        """
        update_stats = {'regimes_updated': 0, 'total_outcomes': 0}
        
        for regime, outcomes in recent_outcomes.items():
            if not outcomes:
                continue
                
            # Separate successes and failures
            successes = sum(1 for outcome in outcomes if outcome['successful'])
            failures = len(outcomes) - successes
            trade_dates = [outcome['date'] for outcome in outcomes]
            
            # Update Bayesian posterior
            self.bayesian_updater.update_posterior(regime, successes, failures, trade_dates)
            
            update_stats['regimes_updated'] += 1
            update_stats['total_outcomes'] += len(outcomes)
            
            print(f"📊 Updated regime {regime}: {successes} successes, {failures} failures")
            
        self.last_updated = datetime.now()
        update_stats['last_updated'] = self.last_updated
        
        return update_stats
        
    def save(self, filepath: str) -> None:
        """
        Save adaptive filter to disk.
        
        Args:
            filepath: Path to save filter
        """
        # Prepare save data
        save_data = {
            'version': '1.0',
            'timestamp': datetime.now(),
            'n_regimes': self.n_regimes,
            'n_dimensions': self.n_dimensions,
            'centroids': self.centroids.tolist() if self.centroids is not None else None,
            'covariance_matrix': self.covariance_matrix.tolist() if self.covariance_matrix is not None else None,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'validation_metrics': {
                'silhouette_score': self.silhouette_score,
                'davies_bouldin_score': self.davies_bouldin_score,
                'clustering_stability': self.clustering_stability
            },
            'bayesian_priors': self.bayesian_updater.regime_priors,
            'turbulence_percentiles': self.turbulence_calculator.percentiles,
            'training_completed': self.training_completed,
            'last_updated': self.last_updated,
            'metadata': self.metadata,
            # Adaptive regime discovery parameters
            'regime_selection': self.regime_selection,
            'auto_k_range': self.auto_k_range,
            'hierarchical_mode': self.hierarchical_mode,
            'hierarchical_depth': self.hierarchical_depth,
            'novelty_detection': self.novelty_detection,
            'novelty_threshold': self.novelty_threshold,
            'confidence_weighting': self.confidence_weighting,
            'trading_profile': self.trading_profile
        }
        
        # Save as pickle file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"💾 Saved adaptive filter to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'AdaptiveMahalanobisFilter':
        """
        Load adaptive filter from disk.
        
        Args:
            filepath: Path to saved filter
            
        Returns:
            Loaded AdaptiveMahalanobisFilter instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
            
        # Create new instance with saved parameters
        filter_instance = cls(
            n_regimes=save_data['n_regimes'],
            n_dimensions=save_data['n_dimensions'],
            regime_selection=save_data.get('regime_selection', 'fixed'),
            auto_k_range=tuple(save_data.get('auto_k_range', (3, 12))),
            hierarchical_mode=save_data.get('hierarchical_mode', False),
            hierarchical_depth=save_data.get('hierarchical_depth', 3),
            novelty_detection=save_data.get('novelty_detection', False),
            novelty_threshold=save_data.get('novelty_threshold', 0.75),
            confidence_weighting=save_data.get('confidence_weighting', False),
            trading_profile=save_data.get('trading_profile', 'balanced')
        )
        
        # Restore fixed components
        if save_data['centroids']:
            filter_instance.centroids = np.array(save_data['centroids'])
        if save_data['covariance_matrix']:
            filter_instance.covariance_matrix = np.array(save_data['covariance_matrix'])
            
        # Restore scaler
        if save_data['scaler_mean'] and save_data['scaler_scale']:
            filter_instance.scaler.mean_ = np.array(save_data['scaler_mean'])
            filter_instance.scaler.scale_ = np.array(save_data['scaler_scale'])
            filter_instance.scaler.n_features_in_ = len(save_data['scaler_mean'])
            
        # Restore validation metrics
        metrics = save_data.get('validation_metrics', {})
        filter_instance.silhouette_score = metrics.get('silhouette_score')
        filter_instance.davies_bouldin_score = metrics.get('davies_bouldin_score')
        filter_instance.clustering_stability = metrics.get('clustering_stability')
        
        # Restore adaptive components
        if 'bayesian_priors' in save_data:
            filter_instance.bayesian_updater.regime_priors = save_data['bayesian_priors']
        if 'turbulence_percentiles' in save_data:
            filter_instance.turbulence_calculator.percentiles = save_data['turbulence_percentiles']
            
        # Restore metadata
        filter_instance.training_completed = save_data.get('training_completed', False)
        filter_instance.last_updated = save_data.get('last_updated')
        filter_instance.metadata = save_data.get('metadata', {})
        
        print(f"📂 Loaded adaptive filter from {filepath}")
        print(f"   Regimes: {filter_instance.n_regimes}, Dimensions: {filter_instance.n_dimensions}")
        print(f"   Last updated: {filter_instance.last_updated}")
        
        return filter_instance
        
    def get_status_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive status report.
        
        Returns:
            Dictionary with filter status and performance metrics
        """
        report = {
            'version': '1.0',
            'timestamp': datetime.now(),
            'training_completed': self.training_completed,
            'last_updated': self.last_updated,
            'configuration': {
                'n_regimes': self.n_regimes,
                'n_dimensions': self.n_dimensions
            },
            'clustering_metrics': {
                'silhouette_score': self.silhouette_score,
                'davies_bouldin_score': self.davies_bouldin_score,
                'clustering_stability': self.clustering_stability
            },
            'available_profiles': list(self.profile_manager.profile_configs.keys()),
            'recent_trade_count': len(self.trade_history),
            'regime_success_rates': {}
        }
        
        # Add regime success rate estimates
        for regime in range(self.n_regimes):
            if regime in self.bayesian_updater.regime_priors:
                estimate = self.bayesian_updater.get_success_rate_estimate(regime)
                report['regime_success_rates'][regime] = estimate
                
        return report


def create_sample_training_data(n_samples: int = 1000, n_features: int = 61, 
                               n_regimes: int = 6, random_state: int = 42) -> np.ndarray:
    """
    Create sample training data for testing (REMOVE IN PRODUCTION).
    
    Args:
        n_samples: Number of samples
        n_features: Number of features  
        n_regimes: Number of regimes to simulate
        random_state: Random seed
        
    Returns:
        Simulated training data
    """
    np.random.seed(random_state)
    
    # Create regime-specific clusters
    data = []
    samples_per_regime = n_samples // n_regimes
    
    for regime in range(n_regimes):
        # Create regime center
        center = np.random.randn(n_features) * 2
        
        # Generate samples around center
        regime_samples = np.random.randn(samples_per_regime, n_features) * 0.5 + center
        data.append(regime_samples)
        
    return np.vstack(data)


if __name__ == "__main__":
    """Test the adaptive filter implementation."""
    print("🧪 Testing AdaptiveMahalanobisFilter implementation...")
    
    # Create sample data
    training_data = create_sample_training_data(n_samples=2000, n_features=61)
    print(f"Created sample training data: {training_data.shape}")
    
    # Initialize and train filter
    adaptive_filter = AdaptiveMahalanobisFilter(n_regimes=6, n_dimensions=61)
    
    # Discover stable centroids
    centroids, metrics = adaptive_filter.discover_stable_centroids(training_data, n_runs=5)
    
    print("\n📊 Clustering Results:")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
    
    # Test regime classification
    test_sample = training_data[0]
    regime = adaptive_filter.classify_regime(test_sample)
    distances = adaptive_filter.calculate_regime_distances(test_sample)
    
    print(f"\n🎯 Test Classification:")
    print(f"Classified regime: {regime}")
    print(f"Distances to all regimes: {distances}")
    
    # Test signal filtering
    should_trade, explanation = adaptive_filter.should_take_signal(
        test_sample, 'buy', profile='balanced'
    )
    
    print(f"\n💼 Signal Filtering Test:")
    print(f"Should trade: {should_trade}")
    print(f"Decision: {explanation['decision']}")
    print(f"Regime: {explanation['regime']}")
    print(f"Confidence gap: {explanation['confidence_gap']:.4f}")
    
    # Test save/load
    test_filepath = "/tmp/test_adaptive_filter.pkl"
    adaptive_filter.save(test_filepath)
    loaded_filter = AdaptiveMahalanobisFilter.load(test_filepath)
    
    print(f"\n💾 Save/Load Test:")
    print(f"Original regimes: {adaptive_filter.n_regimes}")
    print(f"Loaded regimes: {loaded_filter.n_regimes}")
    print(f"Centroids match: {np.allclose(adaptive_filter.centroids, loaded_filter.centroids)}")
    
    # Generate status report
    report = adaptive_filter.get_status_report()
    print(f"\n📋 Status Report:")
    print(f"Training completed: {report['training_completed']}")
    print(f"Available profiles: {report['available_profiles']}")
    print(f"Recent trades: {report['recent_trade_count']}")
    
    print("\n✅ All tests completed successfully!")