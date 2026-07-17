"""
Hierarchical Confidence Weighting System for Multi-Resolution Market Regime Analysis

This module implements a sophisticated hierarchical confidence weighting system that combines
multi-level market regime classifications (Direction → Volatility → Pattern) into unified 
trading signal confidence scores.

Key Components:
- HierarchicalConfidenceWeighting: Main weighting system
- LevelMetrics: Per-level classification quality tracking
- SignalAggregator: Multi-level signal combination
- ProfileManager: Trading profile-specific confidence thresholds

Author: Trading Lab Adaptive Regime Discovery System
Created: 2025-08-27
"""

from typing import Dict, List, Tuple, Optional, NamedTuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import json
import warnings

logger = logging.getLogger(__name__)


class TradingProfile(Enum):
    """Trading profile definitions with different risk/confidence characteristics."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced" 
    AGGRESSIVE = "aggressive"
    RESEARCH = "research"


class ConfidenceLevel(Enum):
    """Classification confidence levels."""
    VERY_HIGH = "very_high"    # >0.9
    HIGH = "high"              # 0.7-0.9
    MEDIUM = "medium"          # 0.5-0.7
    LOW = "low"               # 0.3-0.5
    VERY_LOW = "very_low"     # <0.3


@dataclass
class LevelClassification:
    """Single level classification result."""
    level: int
    regime_id: int
    confidence: float
    label: str
    raw_similarity: float
    metadata: Dict = field(default_factory=dict)


@dataclass 
class HierarchicalClassification:
    """Complete hierarchical classification across all levels."""
    levels: Dict[int, LevelClassification]
    overall_confidence: float
    signal_strength: float
    recommended_action: str
    risk_assessment: str
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class LevelMetrics:
    """Quality metrics for a specific hierarchical level."""
    level: int
    average_confidence: float
    stability_score: float
    separation_quality: float  # Silhouette-like score
    prediction_accuracy: float  # Historical accuracy if available
    sample_count: int
    confidence_distribution: Dict[str, float] = field(default_factory=dict)


class HierarchicalConfidenceWeighting:
    """
    Advanced hierarchical confidence weighting system for multi-resolution market analysis.
    
    Combines classifications from multiple hierarchical levels (Direction, Volatility, Pattern)
    into unified confidence scores with profile-specific thresholds and risk management.
    """
    
    # Default hierarchical weights based on reliability and market importance
    DEFAULT_WEIGHTS = {
        1: 0.50,  # Direction (Bull/Bear) - Most reliable, highest impact
        2: 0.30,  # Volatility (Low/Normal/High/Extreme) - Important for risk
        3: 0.20   # Pattern (Trending/Choppy/Ranging/etc.) - Tactical, less reliable
    }
    
    # Profile-specific confidence thresholds and characteristics
    PROFILE_CONFIGS = {
        TradingProfile.CONSERVATIVE: {
            'min_confidence': 0.75,
            'level_weights': {1: 0.60, 2: 0.30, 3: 0.10},  # Emphasize direction
            'risk_tolerance': 0.2,
            'require_consensus': True,  # All levels must agree
            'novelty_handling': 'skip'
        },
        TradingProfile.BALANCED: {
            'min_confidence': 0.60,
            'level_weights': {1: 0.50, 2: 0.30, 3: 0.20},  # Default weights
            'risk_tolerance': 0.4,
            'require_consensus': False,
            'novelty_handling': 'cautious'
        },
        TradingProfile.AGGRESSIVE: {
            'min_confidence': 0.45,
            'level_weights': {1: 0.40, 2: 0.25, 3: 0.35},  # Higher pattern weight
            'risk_tolerance': 0.7,
            'require_consensus': False,
            'novelty_handling': 'accept'
        },
        TradingProfile.RESEARCH: {
            'min_confidence': 0.30,  # Accept low confidence for analysis
            'level_weights': {1: 0.33, 2: 0.33, 3: 0.34},  # Equal weights
            'risk_tolerance': 1.0,
            'require_consensus': False,
            'novelty_handling': 'track'
        }
    }
    
    def __init__(self, 
                 profile: TradingProfile = TradingProfile.BALANCED,
                 custom_weights: Optional[Dict[int, float]] = None,
                 adaptive_weighting: bool = True,
                 confidence_memory: int = 1000):
        """
        Initialize hierarchical confidence weighting system.
        
        Args:
            profile: Trading profile determining risk tolerance and thresholds
            custom_weights: Optional custom level weights (overrides profile defaults)
            adaptive_weighting: Enable dynamic weight adjustment based on performance
            confidence_memory: Number of recent classifications to remember for adaptation
        """
        self.profile = profile
        self.config = self.PROFILE_CONFIGS[profile].copy()
        
        # Set weights (custom overrides profile defaults)
        if custom_weights:
            self.config['level_weights'] = custom_weights.copy()
        
        self.adaptive_weighting = adaptive_weighting
        self.confidence_history = deque(maxlen=confidence_memory)
        
        # Performance tracking for adaptive weighting
        self.level_metrics = {}
        self.performance_history = defaultdict(list)
        
        # Signal aggregation components
        self.signal_aggregator = SignalAggregator(self.config)
        
        # Initialize level quality tracking
        self._initialize_level_tracking()
        
        logger.info(f"🎯 HierarchicalConfidenceWeighting initialized")
        logger.info(f"   Profile: {profile.value}")
        logger.info(f"   Level weights: {self.config['level_weights']}")
        logger.info(f"   Min confidence: {self.config['min_confidence']}")
    
    def _initialize_level_tracking(self):
        """Initialize tracking structures for each hierarchical level."""
        for level in [1, 2, 3]:
            self.level_metrics[level] = LevelMetrics(
                level=level,
                average_confidence=0.0,
                stability_score=0.0,
                separation_quality=0.0,
                prediction_accuracy=0.0,
                sample_count=0
            )
    
    def classify_hierarchical(self, 
                             level_classifications: Dict[int, Tuple[int, float, str]],
                             sensor_vector: Optional[np.ndarray] = None,
                             market_metadata: Optional[Dict] = None) -> HierarchicalClassification:
        """
        Perform complete hierarchical classification with confidence weighting.
        
        Args:
            level_classifications: Dict mapping level -> (regime_id, confidence, label)
            sensor_vector: Optional sensor data for additional analysis
            market_metadata: Optional market context (volatility, volume, etc.)
            
        Returns:
            HierarchicalClassification with weighted confidence and recommendations
        """
        # Convert input to LevelClassification objects
        levels = {}
        for level, (regime_id, confidence, label) in level_classifications.items():
            levels[level] = LevelClassification(
                level=level,
                regime_id=regime_id,
                confidence=confidence,
                label=label,
                raw_similarity=confidence,  # Assume confidence == similarity for now
                metadata={'sensor_available': sensor_vector is not None}
            )
        
        # Compute overall confidence using weighted combination
        overall_confidence = self._compute_weighted_confidence(levels)
        
        # Compute signal strength (direction-adjusted confidence)
        signal_strength = self._compute_signal_strength(levels, market_metadata)
        
        # Generate trading recommendation
        recommended_action = self._generate_recommendation(overall_confidence, signal_strength, levels)
        
        # Assess risk level
        risk_assessment = self._assess_risk_level(levels, overall_confidence, market_metadata)
        
        # Create hierarchical classification result
        result = HierarchicalClassification(
            levels=levels,
            overall_confidence=overall_confidence,
            signal_strength=signal_strength,
            recommended_action=recommended_action,
            risk_assessment=risk_assessment,
            timestamp=pd.Timestamp.now()
        )
        
        # Update performance tracking
        self._update_performance_tracking(result)
        
        # Adaptive weight adjustment if enabled
        if self.adaptive_weighting:
            self._update_adaptive_weights(result)
        
        return result
    
    def _compute_weighted_confidence(self, levels: Dict[int, LevelClassification]) -> float:
        """
        Compute overall confidence using weighted combination of level confidences.
        
        Args:
            levels: Classification results for each level
            
        Returns:
            Weighted overall confidence score [0, 1]
        """
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        weights = self.config['level_weights']
        
        for level, classification in levels.items():
            if level in weights:
                weight = weights[level]
                confidence = classification.confidence
                
                # Apply quality adjustment based on historical performance
                quality_multiplier = self._get_level_quality_multiplier(level)
                adjusted_weight = weight * quality_multiplier
                
                total_weighted_confidence += adjusted_weight * confidence
                total_weight += adjusted_weight
        
        if total_weight == 0:
            logger.warning("⚠️ No valid level weights found, using equal weighting")
            return np.mean([c.confidence for c in levels.values()])
        
        overall_confidence = total_weighted_confidence / total_weight
        
        # Apply profile-specific confidence adjustments
        overall_confidence = self._apply_profile_adjustments(overall_confidence, levels)
        
        return np.clip(overall_confidence, 0.0, 1.0)
    
    def _get_level_quality_multiplier(self, level: int) -> float:
        """
        Get quality multiplier for a level based on historical performance.
        
        Args:
            level: Hierarchical level (1, 2, or 3)
            
        Returns:
            Quality multiplier [0.5, 1.5] - adjusts weight based on performance
        """
        if level not in self.level_metrics:
            return 1.0
        
        metrics = self.level_metrics[level]
        
        # Base multiplier on separation quality and stability
        separation_factor = np.clip(metrics.separation_quality, 0.0, 1.0)  
        stability_factor = np.clip(metrics.stability_score, 0.0, 1.0)
        
        # Combine factors (emphasize separation over stability)
        quality_score = 0.7 * separation_factor + 0.3 * stability_factor
        
        # Map to multiplier range [0.5, 1.5]
        multiplier = 0.5 + quality_score
        
        return multiplier
    
    def _compute_signal_strength(self, 
                                levels: Dict[int, LevelClassification],
                                market_metadata: Optional[Dict] = None) -> float:
        """
        Compute signal strength considering directional alignment and market context.
        
        Args:
            levels: Classification results for each level
            market_metadata: Optional market context information
            
        Returns:
            Signal strength [0, 1] - higher means stronger directional signal
        """
        # Base signal strength on level 1 (direction) confidence
        base_strength = levels.get(1, LevelClassification(1, 0, 0.0, "unknown", 0.0)).confidence
        
        # Enhance with volatility context (level 2)
        if 2 in levels:
            vol_classification = levels[2]
            # Higher volatility can amplify or dampen signals based on profile
            vol_adjustment = self._get_volatility_adjustment(vol_classification.label)
            base_strength *= vol_adjustment
        
        # Pattern confirmation (level 3)
        if 3 in levels:
            pattern_classification = levels[3] 
            pattern_boost = self._get_pattern_confirmation_boost(
                pattern_classification.label, pattern_classification.confidence
            )
            base_strength += pattern_boost
        
        # Market context adjustments
        if market_metadata:
            context_multiplier = self._get_market_context_multiplier(market_metadata)
            base_strength *= context_multiplier
        
        return np.clip(base_strength, 0.0, 1.0)
    
    def _get_volatility_adjustment(self, volatility_label: str) -> float:
        """Get signal adjustment factor based on volatility regime."""
        vol_adjustments = {
            TradingProfile.CONSERVATIVE: {
                'Low': 1.1, 'Normal': 1.0, 'High': 0.8, 'Extreme': 0.6
            },
            TradingProfile.BALANCED: {
                'Low': 1.05, 'Normal': 1.0, 'High': 0.9, 'Extreme': 0.7
            },
            TradingProfile.AGGRESSIVE: {
                'Low': 0.9, 'Normal': 1.0, 'High': 1.1, 'Extreme': 1.2  # Embrace volatility
            },
            TradingProfile.RESEARCH: {
                'Low': 1.0, 'Normal': 1.0, 'High': 1.0, 'Extreme': 1.0  # No adjustment
            }
        }
        
        profile_adjustments = vol_adjustments.get(self.profile, vol_adjustments[TradingProfile.BALANCED])
        return profile_adjustments.get(volatility_label, 1.0)
    
    def _get_pattern_confirmation_boost(self, pattern_label: str, pattern_confidence: float) -> float:
        """Get signal boost from pattern confirmation."""
        # Strong pattern confirmation provides small boost
        pattern_boosts = {
            'Trending': 0.05,
            'Momentum': 0.03,
            'Breakout': 0.04,
            'Choppy': -0.02,
            'Ranging': -0.01,
            'Reversal': 0.02
        }
        
        base_boost = pattern_boosts.get(pattern_label, 0.0)
        # Scale by pattern confidence
        return base_boost * pattern_confidence
    
    def _get_market_context_multiplier(self, market_metadata: Dict) -> float:
        """Get signal multiplier based on broader market context."""
        multiplier = 1.0
        
        # Volume confirmation
        if 'volume_ratio' in market_metadata:
            vol_ratio = market_metadata['volume_ratio']
            if vol_ratio > 1.5:  # High volume
                multiplier *= 1.1
            elif vol_ratio < 0.7:  # Low volume
                multiplier *= 0.9
        
        # Market stress (VIX-like indicators)
        if 'market_stress' in market_metadata:
            stress = market_metadata['market_stress']
            if self.profile == TradingProfile.CONSERVATIVE:
                multiplier *= max(0.5, 1.0 - stress * 0.5)  # Reduce in high stress
            elif self.profile == TradingProfile.AGGRESSIVE:
                multiplier *= min(1.5, 1.0 + stress * 0.3)  # Increase in high stress
        
        return multiplier
    
    def _apply_profile_adjustments(self, 
                                  confidence: float,
                                  levels: Dict[int, LevelClassification]) -> float:
        """Apply profile-specific confidence adjustments."""
        adjusted_confidence = confidence
        
        # Conservative profile: Require consensus across levels
        if self.profile == TradingProfile.CONSERVATIVE and self.config['require_consensus']:
            min_level_confidence = min(c.confidence for c in levels.values())
            # If any level is very uncertain, reduce overall confidence
            if min_level_confidence < 0.4:
                adjusted_confidence *= 0.7
        
        # Aggressive profile: Boost confidence when pattern level is strong
        elif self.profile == TradingProfile.AGGRESSIVE:
            if 3 in levels and levels[3].confidence > 0.8:
                adjusted_confidence *= 1.1
        
        return adjusted_confidence
    
    def _generate_recommendation(self, 
                                overall_confidence: float,
                                signal_strength: float,
                                levels: Dict[int, LevelClassification]) -> str:
        """
        Generate trading recommendation based on confidence and signal strength.
        
        Args:
            overall_confidence: Weighted confidence score
            signal_strength: Directional signal strength
            levels: Individual level classifications
            
        Returns:
            Recommendation string: 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell', 'skip'
        """
        min_confidence = self.config['min_confidence']
        
        # Skip if below minimum confidence threshold
        if overall_confidence < min_confidence:
            return 'skip'
        
        # Get direction from level 1 (if available)
        direction_bullish = True  # Default assumption
        if 1 in levels:
            direction_label = levels[1].label.lower()
            direction_bullish = 'bull' in direction_label or 'up' in direction_label
        
        # Determine signal intensity based on confidence and strength
        if overall_confidence >= 0.8 and signal_strength >= 0.8:
            intensity = 'strong'
        elif overall_confidence >= 0.6 and signal_strength >= 0.6:
            intensity = 'normal'
        else:
            return 'hold'  # Uncertain conditions
        
        # Combine direction and intensity
        if direction_bullish:
            return f"{intensity}_buy" if intensity == 'strong' else 'buy'
        else:
            return f"{intensity}_sell" if intensity == 'strong' else 'sell'
    
    def _assess_risk_level(self,
                          levels: Dict[int, LevelClassification],
                          overall_confidence: float,
                          market_metadata: Optional[Dict] = None) -> str:
        """
        Assess risk level of the current classification.
        
        Returns:
            Risk assessment: 'very_low', 'low', 'medium', 'high', 'very_high'
        """
        # Base risk on confidence (higher confidence = lower risk)
        confidence_risk = 1.0 - overall_confidence
        
        # Adjust for volatility (level 2)
        volatility_risk = 0.0
        if 2 in levels:
            vol_label = levels[2].label.lower()
            if 'extreme' in vol_label:
                volatility_risk = 0.4
            elif 'high' in vol_label:
                volatility_risk = 0.2
            elif 'low' in vol_label:
                volatility_risk = -0.1
        
        # Market context risk
        context_risk = 0.0
        if market_metadata:
            if 'market_stress' in market_metadata:
                context_risk = market_metadata['market_stress'] * 0.3
        
        # Combine risk factors
        total_risk = confidence_risk + volatility_risk + context_risk
        total_risk = np.clip(total_risk, 0.0, 1.0)
        
        # Map to risk levels
        if total_risk < 0.2:
            return 'very_low'
        elif total_risk < 0.4:
            return 'low'
        elif total_risk < 0.6:
            return 'medium'
        elif total_risk < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _update_performance_tracking(self, classification: HierarchicalClassification):
        """Update performance tracking with new classification."""
        self.confidence_history.append(classification)
        
        # Update level metrics
        for level, level_result in classification.levels.items():
            if level in self.level_metrics:
                metrics = self.level_metrics[level]
                
                # Update running averages
                alpha = 0.1  # Exponential moving average factor
                metrics.average_confidence = (
                    alpha * level_result.confidence + 
                    (1 - alpha) * metrics.average_confidence
                )
                metrics.sample_count += 1
    
    def _update_adaptive_weights(self, classification: HierarchicalClassification):
        """Update level weights based on performance (if adaptive weighting enabled)."""
        if len(self.confidence_history) < 50:  # Wait for sufficient history
            return
        
        # Simple adaptive adjustment - boost weights of consistently high-performing levels
        for level in [1, 2, 3]:
            if level in classification.levels:
                recent_confidences = [
                    c.levels[level].confidence for c in list(self.confidence_history)[-20:]
                    if level in c.levels
                ]
                
                if recent_confidences:
                    avg_recent_confidence = np.mean(recent_confidences)
                    
                    # Small adjustment based on performance
                    if avg_recent_confidence > 0.8:
                        self.config['level_weights'][level] *= 1.01  # Slight boost
                    elif avg_recent_confidence < 0.4:
                        self.config['level_weights'][level] *= 0.99  # Slight reduction
        
        # Renormalize weights
        total_weight = sum(self.config['level_weights'].values())
        if total_weight > 0:
            for level in self.config['level_weights']:
                self.config['level_weights'][level] /= total_weight
    
    def update_level_metrics(self, level: int, quality_metrics: Dict):
        """
        Update quality metrics for a specific level (called from external training).
        
        Args:
            level: Hierarchical level (1, 2, or 3)
            quality_metrics: Dict with 'silhouette_score', 'stability_score', etc.
        """
        if level in self.level_metrics:
            metrics = self.level_metrics[level]
            
            # Update quality metrics from external source (e.g., clustering analysis)
            if 'silhouette_score' in quality_metrics:
                metrics.separation_quality = quality_metrics['silhouette_score']
            
            if 'stability_score' in quality_metrics:
                metrics.stability_score = quality_metrics['stability_score']
            
            if 'prediction_accuracy' in quality_metrics:
                metrics.prediction_accuracy = quality_metrics['prediction_accuracy']
            
            # Commented out to reduce log noise during evaluation
            # logger.info(f"📊 Updated level {level} metrics: "
            #            f"separation={metrics.separation_quality:.3f}, "
            #            f"stability={metrics.stability_score:.3f}")
    
    def get_confidence_distribution(self, lookback_days: int = 30) -> Dict[str, float]:
        """
        Get distribution of confidence levels over recent period.
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            Dict mapping confidence levels to percentages
        """
        if not self.confidence_history:
            return {}
        
        # Filter recent classifications
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        recent_classifications = [
            c for c in self.confidence_history 
            if c.timestamp and c.timestamp >= cutoff_time
        ]
        
        if not recent_classifications:
            return {}
        
        # Categorize confidence levels
        distribution = {level.value: 0 for level in ConfidenceLevel}
        
        for classification in recent_classifications:
            confidence = classification.overall_confidence
            
            if confidence >= 0.9:
                distribution[ConfidenceLevel.VERY_HIGH.value] += 1
            elif confidence >= 0.7:
                distribution[ConfidenceLevel.HIGH.value] += 1
            elif confidence >= 0.5:
                distribution[ConfidenceLevel.MEDIUM.value] += 1
            elif confidence >= 0.3:
                distribution[ConfidenceLevel.LOW.value] += 1
            else:
                distribution[ConfidenceLevel.VERY_LOW.value] += 1
        
        # Convert to percentages
        total = len(recent_classifications)
        return {level: count / total * 100 for level, count in distribution.items()}
    
    def export_configuration(self) -> Dict:
        """Export current configuration for persistence."""
        return {
            'profile': self.profile.value,
            'level_weights': self.config['level_weights'].copy(),
            'min_confidence': self.config['min_confidence'],
            'adaptive_weighting': self.adaptive_weighting,
            'level_metrics': {
                level: {
                    'average_confidence': metrics.average_confidence,
                    'stability_score': metrics.stability_score,
                    'separation_quality': metrics.separation_quality,
                    'sample_count': metrics.sample_count
                }
                for level, metrics in self.level_metrics.items()
            }
        }
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.confidence_history:
            return {'status': 'No classification history available'}
        
        recent_classifications = list(self.confidence_history)[-100:]  # Last 100 classifications
        
        confidences = [c.overall_confidence for c in recent_classifications]
        signal_strengths = [c.signal_strength for c in recent_classifications]
        
        return {
            'total_classifications': len(self.confidence_history),
            'recent_window': len(recent_classifications),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'signal_strength_stats': {
                'mean': float(np.mean(signal_strengths)),
                'std': float(np.std(signal_strengths)),
                'min': float(np.min(signal_strengths)),
                'max': float(np.max(signal_strengths))
            },
            'recommendation_distribution': self._get_recommendation_distribution(recent_classifications),
            'confidence_distribution': self.get_confidence_distribution(),
            'level_performance': {
                level: {
                    'avg_confidence': metrics.average_confidence,
                    'quality_multiplier': self._get_level_quality_multiplier(level),
                    'current_weight': self.config['level_weights'].get(level, 0.0)
                }
                for level, metrics in self.level_metrics.items()
            }
        }
    
    def _get_recommendation_distribution(self, classifications: List[HierarchicalClassification]) -> Dict[str, float]:
        """Get distribution of recommendations."""
        if not classifications:
            return {}
        
        recommendations = [c.recommended_action for c in classifications]
        unique_recs, counts = np.unique(recommendations, return_counts=True)
        
        total = len(recommendations)
        return {rec: count / total * 100 for rec, count in zip(unique_recs, counts)}


class SignalAggregator:
    """
    Helper class for aggregating signals across multiple hierarchical levels.
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def aggregate_signals(self, 
                         level_signals: Dict[int, Tuple[str, float]]) -> Tuple[str, float]:
        """
        Aggregate multiple level signals into unified signal.
        
        Args:
            level_signals: Dict mapping level -> (signal_type, confidence)
            
        Returns:
            (aggregated_signal, aggregated_confidence)
        """
        # Implementation for signal aggregation
        # This is a placeholder - full implementation would depend on specific signal types
        
        if not level_signals:
            return ('hold', 0.0)
        
        # Simple majority voting with confidence weighting
        weighted_signals = []
        total_weight = 0.0
        
        weights = self.config.get('level_weights', {1: 0.5, 2: 0.3, 3: 0.2})
        
        for level, (signal, confidence) in level_signals.items():
            if level in weights:
                weight = weights[level] * confidence
                weighted_signals.append((signal, weight))
                total_weight += weight
        
        if total_weight == 0:
            return ('hold', 0.0)
        
        # Aggregate by signal type
        signal_weights = defaultdict(float)
        for signal, weight in weighted_signals:
            signal_weights[signal] += weight
        
        # Get strongest signal
        best_signal = max(signal_weights.items(), key=lambda x: x[1])
        aggregated_confidence = best_signal[1] / total_weight
        
        return best_signal[0], aggregated_confidence


# Utility functions for external integration

def create_hierarchical_weighting(profile_name: str = "balanced", 
                                custom_config: Optional[Dict] = None) -> HierarchicalConfidenceWeighting:
    """
    Factory function to create hierarchical weighting system with profile.
    
    Args:
        profile_name: Profile name ('conservative', 'balanced', 'aggressive', 'research')
        custom_config: Optional custom configuration overrides
        
    Returns:
        Configured HierarchicalConfidenceWeighting instance
    """
    try:
        profile = TradingProfile(profile_name.lower())
    except ValueError:
        logger.warning(f"Unknown profile '{profile_name}', using 'balanced'")
        profile = TradingProfile.BALANCED
    
    weighting_system = HierarchicalConfidenceWeighting(profile=profile)
    
    if custom_config:
        # Apply custom configuration overrides
        if 'level_weights' in custom_config:
            weighting_system.config['level_weights'].update(custom_config['level_weights'])
        
        if 'min_confidence' in custom_config:
            weighting_system.config['min_confidence'] = custom_config['min_confidence']
        
        # Renormalize level weights if modified
        total_weight = sum(weighting_system.config['level_weights'].values())
        if total_weight > 0:
            for level in weighting_system.config['level_weights']:
                weighting_system.config['level_weights'][level] /= total_weight
    
    return weighting_system


def analyze_hierarchical_performance(weighting_system: HierarchicalConfidenceWeighting,
                                   save_report: bool = True) -> Dict:
    """
    Analyze performance of hierarchical confidence weighting system.
    
    Args:
        weighting_system: Configured weighting system with classification history
        save_report: Whether to save detailed report to file
        
    Returns:
        Comprehensive performance analysis
    """
    performance = weighting_system.get_performance_summary()
    
    if save_report and performance.get('total_classifications', 0) > 0:
        # Save detailed performance report
        report_path = f"hierarchical_confidence_report_{weighting_system.profile.value}.json"
        
        with open(report_path, 'w') as f:
            json.dump(performance, f, indent=2)
        
        logger.info(f"📊 Performance report saved to: {report_path}")
    
    return performance


if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing HierarchicalConfidenceWeighting system...")
    
    # Create weighting system with balanced profile
    weighting = create_hierarchical_weighting("balanced")
    
    # Example hierarchical classifications
    test_classifications = {
        1: (1, 0.85, "Bull Market"),      # Level 1: Direction
        2: (2, 0.72, "High Volatility"),  # Level 2: Volatility  
        3: (5, 0.61, "Trending Pattern")  # Level 3: Pattern
    }
    
    # Perform hierarchical classification
    result = weighting.classify_hierarchical(test_classifications)
    
    print(f"\n🎯 Hierarchical Classification Result:")
    print(f"Overall Confidence: {result.overall_confidence:.3f}")
    print(f"Signal Strength: {result.signal_strength:.3f}")
    print(f"Recommendation: {result.recommended_action}")
    print(f"Risk Assessment: {result.risk_assessment}")
    
    # Test different profiles
    for profile_name in ["conservative", "aggressive", "research"]:
        profile_weighting = create_hierarchical_weighting(profile_name)
        profile_result = profile_weighting.classify_hierarchical(test_classifications)
        
        print(f"\n{profile_name.upper()} Profile:")
        print(f"  Confidence: {profile_result.overall_confidence:.3f}")
        print(f"  Recommendation: {profile_result.recommended_action}")
    
    print("\n✅ HierarchicalConfidenceWeighting system test completed!")