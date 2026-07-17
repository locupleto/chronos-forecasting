#!/usr/bin/env python3
"""
Adaptive Regime Discovery Module

This module implements the core components for adaptive market regime discovery:
- OptimalRegimeSelector: Automatically determine optimal number of regimes
- HierarchicalRegimeClassifier: Multi-level regime classification 
- DynamicRegimeLibrary: Growing library with novelty detection

Key Design Principles:
1. Data-driven regime count selection (not fixed k=6)
2. Hierarchical structure for multi-resolution analysis
3. Online novelty detection for evolving market conditions
4. Stability analysis across multiple timeframes

Created: 2025-08-27
Author: Claude & Urban for Trading-Lab
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from pathlib import Path
import warnings
import logging

# Scientific computing
from scipy.stats import beta, chi2
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add src to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logger = logging.getLogger(__name__)


class OptimalRegimeSelector:
    """
    Automatically determine optimal number of regimes from market data.
    
    Uses multiple clustering validation metrics:
    - Elbow method (within-cluster sum of squares)
    - Silhouette analysis 
    - Davies-Bouldin index
    - Calinski-Harabasz index
    - Stability analysis across multiple runs
    """
    
    def __init__(self, k_range: Tuple[int, int] = (3, 12), n_stability_runs: int = 10):
        """
        Initialize regime selector.
        
        Args:
            k_range: Range of cluster counts to test (min_k, max_k)
            n_stability_runs: Number of runs for stability analysis
        """
        self.k_range = k_range
        self.n_stability_runs = n_stability_runs
        self.validation_results = {}
        
    def find_optimal_k(self, data: np.ndarray, method: str = 'combined') -> Dict[str, Any]:
        """
        Find optimal number of clusters using specified method.
        
        Args:
            data: Normalized sensor data (n_samples, n_features)
            method: 'elbow', 'silhouette', 'davies_bouldin', 'combined'
            
        Returns:
            Dictionary with optimal k and validation metrics
        """
        logger.info(f"🔍 Finding optimal regime count in range {self.k_range}")
        
        if len(data) < self.k_range[1] * 2:
            logger.warning(f"Insufficient data points ({len(data)}) for max k={self.k_range[1]}")
            # Reduce k_range based on available data
            max_k = min(self.k_range[1], len(data) // 2)
            k_range = (self.k_range[0], max_k)
        else:
            k_range = self.k_range
            
        results = {}
        
        # Test each k value
        for k in range(k_range[0], k_range[1] + 1):
            logger.debug(f"   Testing k={k}...")
            
            try:
                # Multiple runs for stability
                k_results = self._evaluate_k(data, k)
                results[k] = k_results
                
            except Exception as e:
                logger.warning(f"   Failed to evaluate k={k}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No valid clustering solutions found")
            
        # Select optimal k based on method
        optimal_result = self._select_optimal_k(results, method)
        
        self.validation_results = results
        return optimal_result
    
    def _evaluate_k(self, data: np.ndarray, k: int) -> Dict[str, Any]:
        """Evaluate clustering quality for specific k."""
        
        scores = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'inertia': [],
            'stability': []
        }
        
        centroids_list = []
        
        # Multiple runs for stability analysis
        for run in range(self.n_stability_runs):
            try:
                # K-means clustering
                kmeans = KMeans(
                    n_clusters=k,
                    init='k-means++',
                    n_init=1,  # We're doing multiple runs ourselves
                    random_state=42 + run,  # Different seed each run
                    max_iter=300
                )
                
                labels = kmeans.fit_predict(data)
                
                # Skip if all points assigned to single cluster
                if len(np.unique(labels)) < k:
                    continue
                    
                # Calculate metrics
                sil_score = silhouette_score(data, labels)
                db_score = davies_bouldin_score(data, labels)
                ch_score = calinski_harabasz_score(data, labels)
                
                scores['silhouette'].append(sil_score)
                scores['davies_bouldin'].append(db_score)
                scores['calinski_harabasz'].append(ch_score)
                scores['inertia'].append(kmeans.inertia_)
                
                centroids_list.append(kmeans.cluster_centers_)
                
            except Exception as e:
                logger.debug(f"      Run {run+1} failed: {str(e)}")
                continue
        
        # Calculate stability (consistency of centroids across runs)
        if len(centroids_list) > 1:
            stability_score = self._calculate_centroid_stability(centroids_list)
            scores['stability'] = stability_score
        else:
            scores['stability'] = 0.0
            
        # Aggregate scores
        result = {
            'k': k,
            'mean_silhouette': np.mean(scores['silhouette']) if scores['silhouette'] else 0,
            'std_silhouette': np.std(scores['silhouette']) if scores['silhouette'] else 0,
            'mean_davies_bouldin': np.mean(scores['davies_bouldin']) if scores['davies_bouldin'] else float('inf'),
            'std_davies_bouldin': np.std(scores['davies_bouldin']) if scores['davies_bouldin'] else 0,
            'mean_calinski_harabasz': np.mean(scores['calinski_harabasz']) if scores['calinski_harabasz'] else 0,
            'std_calinski_harabasz': np.std(scores['calinski_harabasz']) if scores['calinski_harabasz'] else 0,
            'mean_inertia': np.mean(scores['inertia']) if scores['inertia'] else 0,
            'stability_score': scores['stability'],
            'successful_runs': len(scores['silhouette'])
        }
        
        return result
    
    def _calculate_centroid_stability(self, centroids_list: List[np.ndarray]) -> float:
        """
        Calculate stability of centroids across multiple runs.
        
        Uses average pairwise distance between corresponding centroids.
        Lower values indicate more stable clustering.
        """
        if len(centroids_list) < 2:
            return 0.0
            
        # For each pair of centroid sets, find best matching and calculate distance
        pairwise_distances = []
        
        for i in range(len(centroids_list)):
            for j in range(i + 1, len(centroids_list)):
                centroids1 = centroids_list[i]
                centroids2 = centroids_list[j]
                
                # Find best matching between centroids (Hungarian algorithm approximation)
                distances = []
                
                for c1 in centroids1:
                    min_dist = min([np.linalg.norm(c1 - c2) for c2 in centroids2])
                    distances.append(min_dist)
                
                avg_distance = np.mean(distances)
                pairwise_distances.append(avg_distance)
        
        # Return average instability (lower is better)
        return np.mean(pairwise_distances) if pairwise_distances else 0.0
    
    def _select_optimal_k(self, results: Dict[int, Dict], method: str) -> Dict[str, Any]:
        """Select optimal k based on specified method."""
        
        if method == 'silhouette':
            # Highest silhouette score
            best_k = max(results.keys(), key=lambda k: results[k]['mean_silhouette'])
            
        elif method == 'davies_bouldin':
            # Lowest Davies-Bouldin score
            best_k = min(results.keys(), key=lambda k: results[k]['mean_davies_bouldin'])
            
        elif method == 'elbow':
            # Elbow method on inertia
            best_k = self._find_elbow_point(results)
            
        elif method == 'combined':
            # Weighted combination of metrics
            best_k = self._combined_selection(results)
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        optimal_result = {
            'optimal_k': best_k,
            'method': method,
            'validation_metrics': results[best_k],
            'all_results': results,
            'confidence': self._calculate_selection_confidence(results, best_k)
        }
        
        logger.info(f"✅ Optimal regime count: k={best_k} (method: {method})")
        logger.info(f"   Silhouette: {results[best_k]['mean_silhouette']:.4f}")
        logger.info(f"   Davies-Bouldin: {results[best_k]['mean_davies_bouldin']:.4f}")
        logger.info(f"   Selection confidence: {optimal_result['confidence']:.3f}")
        
        return optimal_result
    
    def _find_elbow_point(self, results: Dict[int, Dict]) -> int:
        """Find elbow point in inertia curve."""
        
        k_values = sorted(results.keys())
        inertias = [results[k]['mean_inertia'] for k in k_values]
        
        if len(inertias) < 3:
            return k_values[0]  # Not enough points for elbow detection
        
        # Calculate rate of change
        rate_changes = []
        for i in range(1, len(inertias) - 1):
            rate_change = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            rate_changes.append(rate_change)
        
        # Find maximum rate of change (elbow point)
        if rate_changes:
            elbow_idx = np.argmax(rate_changes) + 1  # +1 because we started from index 1
            return k_values[elbow_idx]
        
        return k_values[0]
    
    def _combined_selection(self, results: Dict[int, Dict]) -> int:
        """Select optimal k using weighted combination of metrics."""
        
        # Weights for different metrics (higher weight = more important)
        weights = {
            'silhouette': 0.4,      # Higher is better
            'davies_bouldin': 0.3,  # Lower is better  
            'stability': 0.2,       # Lower is better
            'calinski_harabasz': 0.1 # Higher is better
        }
        
        k_scores = {}
        
        # Normalize each metric to [0, 1] range
        for metric in weights.keys():
            if metric == 'silhouette':
                values = [results[k]['mean_silhouette'] for k in results.keys()]
                min_val, max_val = min(values), max(values)
                
            elif metric == 'davies_bouldin':
                values = [results[k]['mean_davies_bouldin'] for k in results.keys()]
                min_val, max_val = min(values), max(values)
                
            elif metric == 'stability':
                values = [results[k]['stability_score'] for k in results.keys()]
                min_val, max_val = min(values), max(values)
                
            elif metric == 'calinski_harabasz':
                values = [results[k]['mean_calinski_harabasz'] for k in results.keys()]
                min_val, max_val = min(values), max(values)
            
            # Normalize and weight
            for k in results.keys():
                if k not in k_scores:
                    k_scores[k] = 0.0
                
                if metric == 'silhouette':
                    raw_score = results[k]['mean_silhouette']
                    normalized = (raw_score - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    
                elif metric == 'davies_bouldin':
                    raw_score = results[k]['mean_davies_bouldin']
                    normalized = 1.0 - (raw_score - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    
                elif metric == 'stability':
                    raw_score = results[k]['stability_score']
                    normalized = 1.0 - (raw_score - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    
                elif metric == 'calinski_harabasz':
                    raw_score = results[k]['mean_calinski_harabasz']
                    normalized = (raw_score - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                
                k_scores[k] += weights[metric] * normalized
        
        # Select k with highest combined score
        best_k = max(k_scores.keys(), key=lambda k: k_scores[k])
        return best_k
    
    def _calculate_selection_confidence(self, results: Dict[int, Dict], selected_k: int) -> float:
        """
        Calculate confidence in k selection based on metric stability and separation.
        
        Returns value between 0 and 1 (higher is more confident).
        """
        if selected_k not in results:
            return 0.0
            
        selected_result = results[selected_k]
        
        # Factors that increase confidence:
        # 1. High silhouette score
        # 2. Low standard deviation in metrics
        # 3. High number of successful runs
        # 4. Clear separation from other k values
        
        confidence_factors = []
        
        # Silhouette score confidence
        sil_score = selected_result['mean_silhouette']
        if sil_score > 0.3:
            confidence_factors.append(0.9)
        elif sil_score > 0.2:
            confidence_factors.append(0.7)
        elif sil_score > 0.1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Stability confidence
        stability_conf = 1.0 / (1.0 + selected_result['stability_score'])
        confidence_factors.append(min(stability_conf, 1.0))
        
        # Run success rate
        success_rate = selected_result['successful_runs'] / self.n_stability_runs
        confidence_factors.append(success_rate)
        
        # Standard deviation penalty
        std_penalty = 1.0 / (1.0 + selected_result['std_silhouette'])
        confidence_factors.append(min(std_penalty, 1.0))
        
        # Overall confidence is geometric mean
        overall_confidence = np.power(np.prod(confidence_factors), 1.0/len(confidence_factors))
        return float(overall_confidence)
    
    def analyze_stability(self, data: np.ndarray, k: int, n_runs: int = 20) -> Dict[str, Any]:
        """
        Detailed stability analysis for specific k.
        
        Args:
            data: Sensor data
            k: Number of clusters to analyze
            n_runs: Number of clustering runs
            
        Returns:
            Detailed stability metrics
        """
        logger.info(f"🔄 Analyzing stability for k={k} over {n_runs} runs...")
        
        results = []
        centroids_history = []
        
        for run in range(n_runs):
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    init='k-means++',
                    n_init=1,
                    random_state=42 + run,
                    max_iter=300
                )
                
                labels = kmeans.fit_predict(data)
                
                if len(np.unique(labels)) == k:
                    sil_score = silhouette_score(data, labels)
                    db_score = davies_bouldin_score(data, labels)
                    
                    results.append({
                        'run': run,
                        'silhouette': sil_score,
                        'davies_bouldin': db_score,
                        'inertia': kmeans.inertia_
                    })
                    
                    centroids_history.append(kmeans.cluster_centers_)
                    
            except Exception as e:
                logger.debug(f"   Run {run} failed: {str(e)}")
                continue
        
        if not results:
            raise ValueError(f"No successful runs for k={k}")
        
        # Calculate stability metrics
        stability_analysis = {
            'k': k,
            'n_successful_runs': len(results),
            'success_rate': len(results) / n_runs,
            'silhouette_mean': np.mean([r['silhouette'] for r in results]),
            'silhouette_std': np.std([r['silhouette'] for r in results]),
            'silhouette_range': np.ptp([r['silhouette'] for r in results]),
            'davies_bouldin_mean': np.mean([r['davies_bouldin'] for r in results]),
            'davies_bouldin_std': np.std([r['davies_bouldin'] for r in results]),
            'centroid_stability': self._calculate_centroid_stability(centroids_history),
            'coefficient_of_variation': np.std([r['silhouette'] for r in results]) / np.mean([r['silhouette'] for r in results])
        }
        
        # Stability rating
        if stability_analysis['coefficient_of_variation'] < 0.1 and stability_analysis['success_rate'] > 0.8:
            stability_rating = 'High'
        elif stability_analysis['coefficient_of_variation'] < 0.2 and stability_analysis['success_rate'] > 0.6:
            stability_rating = 'Medium'
        else:
            stability_rating = 'Low'
            
        stability_analysis['stability_rating'] = stability_rating
        
        logger.info(f"✅ Stability analysis complete:")
        logger.info(f"   Success rate: {stability_analysis['success_rate']:.1%}")
        logger.info(f"   Silhouette CV: {stability_analysis['coefficient_of_variation']:.3f}")
        logger.info(f"   Stability rating: {stability_rating}")
        
        return stability_analysis


class HierarchicalRegimeClassifier:
    """
    Multi-level regime classification for different analysis granularities.
    
    Implements a hierarchical approach:
    Level 1: Market Direction (Bull/Bear/Sideways) - 2-3 regimes
    Level 2: Volatility Context (Low/Normal/High/Extreme) - 4-6 regimes  
    Level 3: Fine Patterns (Trending/Choppy/Ranging/etc.) - 8-12 regimes
    
    Each level provides different confidence and granularity trade-offs.
    """
    
    def __init__(self, max_depth: int = 3):
        """
        Initialize hierarchical classifier.
        
        Args:
            max_depth: Maximum depth of hierarchy (1-4)
        """
        self.max_depth = max_depth
        self.levels = {}
        self.fitted = False
        
        # Default level configuration
        self.level_config = {
            1: {
                "name": "Direction", 
                "target_k": 3, 
                "description": "Market direction and trend",
                "confidence_weight": 0.5
            },
            2: {
                "name": "Volatility", 
                "target_k": 6, 
                "description": "Volatility regime within direction",
                "confidence_weight": 0.3
            },
            3: {
                "name": "Pattern", 
                "target_k": 12, 
                "description": "Fine-grained pattern recognition",
                "confidence_weight": 0.2
            }
        }
    
    def hierarchical_clustering(self, data: np.ndarray, method: str = 'kmeans') -> Dict[int, Dict]:
        """
        Perform hierarchical clustering with increasing granularity.
        
        Args:
            data: Normalized sensor data
            method: 'kmeans' or 'agglomerative'
            
        Returns:
            Dictionary mapping level -> clustering results
        """
        logger.info(f"🌳 Performing hierarchical clustering (max depth: {self.max_depth})")
        
        results = {}
        
        for level in range(1, self.max_depth + 1):
            logger.info(f"   Level {level}: {self.level_config[level]['name']}")
            
            target_k = min(self.level_config[level]['target_k'], len(data) // 2)
            
            if method == 'kmeans':
                clustering_result = self._kmeans_clustering(data, target_k, level)
            elif method == 'agglomerative':
                clustering_result = self._agglomerative_clustering(data, target_k, level)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            results[level] = clustering_result
            
            logger.info(f"      k={clustering_result['n_clusters']}, "
                       f"silhouette={clustering_result['silhouette_score']:.4f}")
        
        self.levels = results
        self.fitted = True
        
        return results
    
    def _kmeans_clustering(self, data: np.ndarray, k: int, level: int) -> Dict[str, Any]:
        """Perform K-means clustering for specific level."""
        
        # Use stability-focused approach
        best_score = -1
        best_result = None
        
        for run in range(10):  # Multiple runs for stability
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=1,
                random_state=42 + run,
                max_iter=300
            )
            
            labels = kmeans.fit_predict(data)
            
            if len(np.unique(labels)) == k:
                try:
                    sil_score = silhouette_score(data, labels)
                    
                    if sil_score > best_score:
                        best_score = sil_score
                        best_result = {
                            'level': level,
                            'method': 'kmeans',
                            'n_clusters': k,
                            'labels': labels,
                            'centroids': kmeans.cluster_centers_,
                            'silhouette_score': sil_score,
                            'davies_bouldin_score': davies_bouldin_score(data, labels),
                            'inertia': kmeans.inertia_,
                            'model': kmeans
                        }
                except Exception as e:
                    continue
        
        if best_result is None:
            raise ValueError(f"Failed to fit clustering for level {level}")
            
        return best_result
    
    def _agglomerative_clustering(self, data: np.ndarray, k: int, level: int) -> Dict[str, Any]:
        """Perform agglomerative clustering for specific level."""
        
        clustering = AgglomerativeClustering(
            n_clusters=k,
            linkage='ward'
        )
        
        labels = clustering.fit_predict(data)
        
        # Calculate metrics
        sil_score = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else 0
        db_score = davies_bouldin_score(data, labels) if len(np.unique(labels)) > 1 else float('inf')
        
        # Calculate centroids manually
        centroids = []
        for cluster_id in range(k):
            cluster_points = data[labels == cluster_id]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(centroid)
        
        result = {
            'level': level,
            'method': 'agglomerative',
            'n_clusters': k,
            'labels': labels,
            'centroids': np.array(centroids) if centroids else None,
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score,
            'inertia': None,  # Not applicable for agglomerative
            'model': clustering
        }
        
        return result
    
    def classify_multilevel(self, sensor_vector: np.ndarray) -> Dict[int, Tuple[int, float]]:
        """
        Classify sensor vector at multiple hierarchical levels.
        
        Args:
            sensor_vector: Single sensor observation
            
        Returns:
            Dictionary mapping level -> (regime_id, confidence)
        """
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call hierarchical_clustering() first.")
        
        classifications = {}
        
        for level, level_data in self.levels.items():
            regime_id, confidence = self._classify_single_level(sensor_vector, level_data)
            classifications[level] = (regime_id, confidence)
        
        return classifications
    
    def _classify_single_level(self, sensor_vector: np.ndarray, level_data: Dict) -> Tuple[int, float]:
        """Classify at single hierarchical level."""
        
        centroids = level_data['centroids']
        
        # Calculate distances to all centroids
        distances = []
        for centroid in centroids:
            distance = np.linalg.norm(sensor_vector - centroid)
            distances.append(distance)
        
        # Find closest regime
        regime_id = np.argmin(distances)
        min_distance = distances[regime_id]
        
        # Convert distance to confidence (higher distance = lower confidence)
        # Use exponential decay: confidence = exp(-distance / scale)
        scale = np.std(distances) if len(distances) > 1 else 1.0
        confidence = np.exp(-min_distance / (scale + 1e-6))
        
        return regime_id, float(confidence)
    
    def compute_combined_confidence(self, classifications: Dict[int, Tuple[int, float]]) -> float:
        """
        Combine multi-level classifications into single confidence score.
        
        Args:
            classifications: Dict mapping level -> (regime_id, confidence)
            
        Returns:
            Combined confidence score (0-1)
        """
        if not classifications:
            return 0.0
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for level, (regime_id, confidence) in classifications.items():
            if level in self.level_config:
                weight = self.level_config[level]['confidence_weight']
                total_weighted_confidence += weight * confidence
                total_weight += weight
        
        if total_weight == 0:
            return np.mean([conf for _, conf in classifications.values()])
        
        return total_weighted_confidence / total_weight
    
    def get_regime_interpretation(self, classifications: Dict[int, Tuple[int, float]]) -> Dict[str, Any]:
        """
        Provide human-readable interpretation of regime classification.
        
        Args:
            classifications: Multi-level classification results
            
        Returns:
            Interpretation dictionary with regime descriptions
        """
        interpretation = {
            'combined_confidence': self.compute_combined_confidence(classifications),
            'levels': {}
        }
        
        for level, (regime_id, confidence) in classifications.items():
            if level in self.level_config:
                level_info = {
                    'regime_id': regime_id,
                    'confidence': confidence,
                    'name': self.level_config[level]['name'],
                    'description': self.level_config[level]['description']
                }
                
                # Add specific regime interpretation based on level
                if level == 1:  # Direction
                    if regime_id == 0:
                        level_info['interpretation'] = 'Bull Market'
                    elif regime_id == 1:
                        level_info['interpretation'] = 'Bear Market'
                    else:
                        level_info['interpretation'] = 'Sideways Market'
                        
                elif level == 2:  # Volatility
                    vol_labels = ['Low Vol', 'Normal Vol', 'High Vol', 'Extreme Vol', 'Crisis Vol', 'Panic Vol']
                    level_info['interpretation'] = vol_labels[min(regime_id, len(vol_labels)-1)]
                    
                elif level == 3:  # Patterns
                    pattern_labels = ['Strong Trend', 'Weak Trend', 'Range-Bound', 'Choppy', 
                                    'Breakout', 'Reversal', 'Consolidation', 'Momentum', 
                                    'Mean Reversion', 'Volatile', 'Calm', 'Transition']
                    level_info['interpretation'] = pattern_labels[min(regime_id, len(pattern_labels)-1)]
                
                interpretation['levels'][level] = level_info
        
        return interpretation


class DynamicRegimeLibrary:
    """
    Growing library of discovered regimes with online novelty detection.
    
    Maintains a comprehensive library of market regimes encountered over time,
    with the ability to detect novel regimes and adapt to changing conditions.
    """
    
    def __init__(self, similarity_threshold: float = 0.75, max_library_size: int = 50):
        """
        Initialize dynamic regime library.
        
        Args:
            similarity_threshold: Minimum similarity to existing regime (0-1)
            max_library_size: Maximum number of regimes to maintain
        """
        self.similarity_threshold = similarity_threshold
        self.max_library_size = max_library_size
        
        # Core library storage
        self.regime_library = {}  # regime_id -> regime_data
        self.active_regimes = set()  # Currently relevant regimes
        self.emergence_history = []  # Historical regime emergence
        
        # Tracking and statistics
        self.regime_counter = 0
        self.similarity_history = deque(maxlen=1000)
        self.novelty_alerts = []
        
        # Caching for performance
        self._centroid_cache = {}
        self._scaler = StandardScaler()
        self._scaler_fitted = False
        
    def add_regime(self, regime_data: Dict[str, Any], metadata: Dict[str, Any]) -> int:
        """
        Add new regime to library.
        
        Args:
            regime_data: Regime information (centroids, labels, etc.)
            metadata: Additional metadata (discovery date, source, etc.)
            
        Returns:
            Assigned regime ID
        """
        regime_id = self.regime_counter
        self.regime_counter += 1
        
        # Store regime with timestamp
        regime_entry = {
            'regime_id': regime_id,
            'data': regime_data,
            'metadata': metadata,
            'discovery_date': datetime.now(),
            'usage_count': 0,
            'last_seen': datetime.now(),
            'confidence_history': [],
            'is_active': True
        }
        
        self.regime_library[regime_id] = regime_entry
        self.active_regimes.add(regime_id)
        
        # Update emergence history
        self.emergence_history.append({
            'regime_id': regime_id,
            'timestamp': datetime.now(),
            'source': metadata.get('source', 'unknown'),
            'trigger': metadata.get('trigger', 'manual')
        })
        
        logger.info(f"📚 Added new regime {regime_id} to library")
        logger.info(f"   Source: {metadata.get('source', 'unknown')}")
        logger.info(f"   Total regimes: {len(self.regime_library)}")
        
        # Cleanup if library too large
        if len(self.regime_library) > self.max_library_size:
            self._cleanup_old_regimes()
            
        return regime_id
    
    def check_novelty(self, sensor_vector: np.ndarray, 
                     context: Optional[Dict] = None) -> Tuple[bool, float, Dict]:
        """
        Check if sensor vector represents a novel regime.
        
        Args:
            sensor_vector: Current market sensor observation
            context: Optional context information (date, symbol, etc.)
            
        Returns:
            Tuple of (is_novel, similarity_to_closest, analysis_details)
        """
        if not self.regime_library:
            # Empty library - everything is novel
            return True, 0.0, {'reason': 'empty_library', 'closest_regime': None}
        
        # Find most similar regime
        closest_regime_id, max_similarity = self._find_most_similar_regime(sensor_vector)
        
        # Record similarity for tracking
        self.similarity_history.append({
            'timestamp': datetime.now(),
            'similarity': max_similarity,
            'closest_regime': closest_regime_id
        })
        
        # Update regime usage
        if closest_regime_id in self.regime_library:
            self.regime_library[closest_regime_id]['usage_count'] += 1
            self.regime_library[closest_regime_id]['last_seen'] = datetime.now()
            self.regime_library[closest_regime_id]['confidence_history'].append(max_similarity)
        
        # Determine if novel
        is_novel = max_similarity < self.similarity_threshold
        
        analysis_details = {
            'closest_regime': closest_regime_id,
            'max_similarity': max_similarity,
            'threshold': self.similarity_threshold,
            'library_size': len(self.regime_library),
            'active_regimes': len(self.active_regimes)
        }
        
        # Add context-specific analysis
        if context:
            analysis_details['context'] = context
            
        if is_novel:
            logger.info(f"🔍 Novel regime detected (similarity: {max_similarity:.3f} < {self.similarity_threshold:.3f})")
            
            # Generate novelty alert
            self._generate_novelty_alert(sensor_vector, max_similarity, analysis_details)
        
        return is_novel, max_similarity, analysis_details
    
    def _find_most_similar_regime(self, sensor_vector: np.ndarray) -> Tuple[Optional[int], float]:
        """Find most similar regime in library."""
        
        if not self.regime_library:
            return None, 0.0
        
        max_similarity = 0.0
        closest_regime_id = None
        
        for regime_id, regime_entry in self.regime_library.items():
            # Skip inactive regimes
            if not regime_entry['is_active']:
                continue
                
            # Get regime centroids
            centroids = self._get_regime_centroids(regime_entry)
            
            if centroids is not None:
                # Calculate similarity to each centroid
                for centroid in centroids:
                    similarity = self._calculate_similarity(sensor_vector, centroid)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        closest_regime_id = regime_id
        
        return closest_regime_id, max_similarity
    
    def _get_regime_centroids(self, regime_entry: Dict) -> Optional[np.ndarray]:
        """Extract centroids from regime entry."""
        
        regime_data = regime_entry['data']
        
        # Handle different data formats
        if 'centroids' in regime_data:
            return regime_data['centroids']
        elif 'cluster_centers_' in regime_data:
            return regime_data['cluster_centers_']
        elif 'centers' in regime_data:
            return regime_data['centers']
        
        return None
    
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate similarity between two vectors.
        
        Uses cosine similarity for scale-invariant comparison.
        """
        # Handle edge cases
        if len(vector1) != len(vector2):
            return 0.0
        
        # Cosine similarity
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vector1, vector2) / (norm1 * norm2)
        
        # Convert to 0-1 range
        similarity = (cosine_sim + 1) / 2
        
        return float(similarity)
    
    def _generate_novelty_alert(self, sensor_vector: np.ndarray, similarity: float, 
                               analysis: Dict):
        """Generate alert for novel regime detection."""
        
        alert = {
            'timestamp': datetime.now(),
            'type': 'novel_regime',
            'similarity': similarity,
            'threshold': self.similarity_threshold,
            'analysis': analysis,
            'sensor_vector': sensor_vector.copy(),
            'severity': self._calculate_novelty_severity(similarity)
        }
        
        self.novelty_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.novelty_alerts) > 100:
            self.novelty_alerts = self.novelty_alerts[-100:]
        
        logger.warning(f"🚨 Novel regime alert: severity={alert['severity']}, "
                      f"similarity={similarity:.3f}")
    
    def _calculate_novelty_severity(self, similarity: float) -> str:
        """Calculate severity of novelty based on similarity score."""
        
        if similarity < 0.3:
            return 'EXTREME'  # Completely unprecedented
        elif similarity < 0.5:
            return 'HIGH'     # Very different from known regimes
        elif similarity < 0.7:
            return 'MEDIUM'   # Moderately different
        else:
            return 'LOW'      # Slight variation
    
    def get_regime_evolution(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Track how regimes evolved over time.
        
        Args:
            lookback_days: Days to look back for analysis
            
        Returns:
            DataFrame with regime evolution statistics
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        evolution_data = []
        
        for regime_id, regime_entry in self.regime_library.items():
            if regime_entry['discovery_date'] >= cutoff_date:
                
                # Calculate regime statistics
                confidence_history = regime_entry['confidence_history']
                
                stats = {
                    'regime_id': regime_id,
                    'discovery_date': regime_entry['discovery_date'],
                    'usage_count': regime_entry['usage_count'],
                    'last_seen': regime_entry['last_seen'],
                    'is_active': regime_entry['is_active'],
                    'days_since_discovery': (datetime.now() - regime_entry['discovery_date']).days,
                    'days_since_last_seen': (datetime.now() - regime_entry['last_seen']).days,
                    'avg_confidence': np.mean(confidence_history) if confidence_history else 0,
                    'confidence_trend': self._calculate_confidence_trend(confidence_history),
                    'source': regime_entry['metadata'].get('source', 'unknown')
                }
                
                evolution_data.append(stats)
        
        return pd.DataFrame(evolution_data)
    
    def _calculate_confidence_trend(self, confidence_history: List[float]) -> str:
        """Calculate trend in confidence over time."""
        
        if len(confidence_history) < 5:
            return 'insufficient_data'
        
        # Linear trend over recent history
        recent = confidence_history[-10:]  # Last 10 observations
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _cleanup_old_regimes(self):
        """Remove old, unused regimes to maintain library size."""
        
        # Sort regimes by usage and recency
        regime_scores = []
        
        for regime_id, regime_entry in self.regime_library.items():
            days_old = (datetime.now() - regime_entry['discovery_date']).days
            days_since_seen = (datetime.now() - regime_entry['last_seen']).days
            
            # Score based on usage, recency, and discovery date
            score = (regime_entry['usage_count'] * 10 +  # Usage weight
                    max(0, 365 - days_old) +             # Discovery recency
                    max(0, 30 - days_since_seen))        # Last seen recency
            
            regime_scores.append((regime_id, score))
        
        # Sort by score (highest first)
        regime_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top regimes
        keep_regimes = set([r[0] for r in regime_scores[:self.max_library_size]])
        
        # Remove others
        removed_count = 0
        for regime_id in list(self.regime_library.keys()):
            if regime_id not in keep_regimes:
                del self.regime_library[regime_id]
                self.active_regimes.discard(regime_id)
                removed_count += 1
        
        logger.info(f"🧹 Cleaned up {removed_count} old regimes from library")
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""
        
        if not self.regime_library:
            return {'status': 'empty'}
        
        recent_similarities = [s['similarity'] for s in list(self.similarity_history)[-100:]]
        
        stats = {
            'total_regimes': len(self.regime_library),
            'active_regimes': len(self.active_regimes),
            'recent_avg_similarity': np.mean(recent_similarities) if recent_similarities else 0,
            'recent_novel_rate': sum(1 for s in recent_similarities if s < self.similarity_threshold) / len(recent_similarities) if recent_similarities else 0,
            'similarity_threshold': self.similarity_threshold,
            'novelty_alerts_count': len(self.novelty_alerts),
            'oldest_regime': min([r['discovery_date'] for r in self.regime_library.values()]),
            'newest_regime': max([r['discovery_date'] for r in self.regime_library.values()]),
            'most_used_regime': max(self.regime_library.items(), key=lambda x: x[1]['usage_count'])[0],
            'library_health': self._assess_library_health()
        }
        
        return stats
    
    def _assess_library_health(self) -> str:
        """Assess overall health of regime library."""
        
        if not self.regime_library:
            return 'empty'
        
        # Check various health indicators
        active_ratio = len(self.active_regimes) / len(self.regime_library)
        recent_usage = sum([r['usage_count'] for r in self.regime_library.values()]) / len(self.regime_library)
        
        if active_ratio > 0.8 and recent_usage > 5:
            return 'excellent'
        elif active_ratio > 0.6 and recent_usage > 2:
            return 'good'
        elif active_ratio > 0.4:
            return 'fair'
        else:
            return 'needs_attention'


# Module-level convenience functions
def find_optimal_regimes(data: np.ndarray, k_range: Tuple[int, int] = (3, 12)) -> Dict[str, Any]:
    """Convenience function for optimal regime selection."""
    selector = OptimalRegimeSelector(k_range=k_range)
    return selector.find_optimal_k(data, method='combined')


def create_hierarchical_classifier(data: np.ndarray, max_depth: int = 3) -> HierarchicalRegimeClassifier:
    """Convenience function for hierarchical classification."""
    classifier = HierarchicalRegimeClassifier(max_depth=max_depth)
    classifier.hierarchical_clustering(data)
    return classifier


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Adaptive Regime Discovery Module...")
    
    # Generate synthetic market data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic regimes
    regime1 = np.random.normal(0, 1, (300, n_features))      # Bull market
    regime2 = np.random.normal(-1, 1.5, (300, n_features))   # Bear market  
    regime3 = np.random.normal(0.5, 0.5, (200, n_features))  # Low vol
    regime4 = np.random.normal(-0.5, 2, (200, n_features))   # High vol
    
    test_data = np.vstack([regime1, regime2, regime3, regime4])
    
    # Test optimal regime selection
    print("\n1. Testing Optimal Regime Selection:")
    selector = OptimalRegimeSelector(k_range=(2, 8))
    optimal_result = selector.find_optimal_k(test_data)
    print(f"   Optimal k: {optimal_result['optimal_k']}")
    
    # Test hierarchical classification
    print("\n2. Testing Hierarchical Classification:")
    hierarchical = HierarchicalRegimeClassifier(max_depth=3)
    hierarchy_results = hierarchical.hierarchical_clustering(test_data)
    
    # Test single classification
    test_vector = test_data[0]
    classifications = hierarchical.classify_multilevel(test_vector)
    interpretation = hierarchical.get_regime_interpretation(classifications)
    print(f"   Combined confidence: {interpretation['combined_confidence']:.3f}")
    
    # Test dynamic library
    print("\n3. Testing Dynamic Regime Library:")
    library = DynamicRegimeLibrary()
    
    # Add some regimes
    for i, regime_data in enumerate([regime1, regime2]):
        regime_info = {
            'centroids': np.mean(regime_data, axis=0).reshape(1, -1),
            'n_samples': len(regime_data)
        }
        metadata = {'source': f'test_regime_{i}', 'trigger': 'synthetic'}
        library.add_regime(regime_info, metadata)
    
    # Test novelty detection
    novel_vector = np.random.normal(5, 1, n_features)  # Very different
    is_novel, similarity, analysis = library.check_novelty(novel_vector)
    print(f"   Novel regime detected: {is_novel} (similarity: {similarity:.3f})")
    
    # Library stats
    stats = library.get_library_stats()
    print(f"   Library health: {stats['library_health']}")
    
    print("\n✅ All tests completed successfully!")