"""
Drift Detection and Monitoring for Uncertainty Quantification

Monitors model performance, detects distribution shifts, and tracks
calibration degradation over time.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
import json


@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics."""
    timestamp: datetime
    calibration_error: float
    coverage: float
    mean_uncertainty: float
    prediction_accuracy: float
    drift_score: float
    ood_score: float
    metadata: Dict = field(default_factory=dict)


class DriftDetector:
    """
    Detects distribution drift in inputs and outputs.
    
    Monitors various statistical measures to detect when the data
    distribution has shifted from training/calibration.
    """
    
    def __init__(
        self,
        reference_data: np.ndarray,
        method: str = 'ks',
        window_size: int = 100,
        threshold: float = 0.05
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference data (training/validation)
            method: Detection method ('ks', 'mmd', 'psi', 'kl')
            window_size: Size of sliding window for monitoring
            threshold: Significance threshold for drift detection
        """
        self.reference_data = reference_data
        self.method = method
        self.window_size = window_size
        self.threshold = threshold
        
        # Compute reference statistics
        self._compute_reference_stats()
        
        # Initialize monitoring buffer
        self.buffer = []
        self.drift_scores = []
    
    def _compute_reference_stats(self):
        """Compute statistics on reference data."""
        self.ref_mean = np.mean(self.reference_data, axis=0)
        self.ref_std = np.std(self.reference_data, axis=0)
        self.ref_median = np.median(self.reference_data, axis=0)
        
        # For PSI, create bins
        if self.method == 'psi':
            self.bins = []
            n_features = self.reference_data.shape[1] if len(self.reference_data.shape) > 1 else 1
            
            if n_features == 1:
                self.bins = np.percentile(self.reference_data, np.linspace(0, 100, 11))
            else:
                for i in range(n_features):
                    self.bins.append(np.percentile(self.reference_data[:, i], np.linspace(0, 100, 11)))
    
    def detect_drift(self, new_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect drift in new data.
        
        Args:
            new_data: New data to test for drift
            
        Returns:
            Tuple of (drift_detected, drift_score)
        """
        if self.method == 'ks':
            drift_score = self._kolmogorov_smirnov_test(new_data)
        elif self.method == 'mmd':
            drift_score = self._maximum_mean_discrepancy(new_data)
        elif self.method == 'psi':
            drift_score = self._population_stability_index(new_data)
        elif self.method == 'kl':
            drift_score = self._kl_divergence(new_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        drift_detected = drift_score > self.threshold
        
        # Update buffer
        self.buffer.extend(new_data.tolist())
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
        
        self.drift_scores.append(drift_score)
        
        return drift_detected, drift_score
    
    def _kolmogorov_smirnov_test(self, new_data: np.ndarray) -> float:
        """Kolmogorov-Smirnov test for drift detection."""
        if len(new_data.shape) == 1:
            _, p_value = stats.ks_2samp(self.reference_data, new_data)
            return 1 - p_value
        else:
            # Multivariate: average p-values across features
            p_values = []
            for i in range(new_data.shape[1]):
                _, p = stats.ks_2samp(self.reference_data[:, i], new_data[:, i])
                p_values.append(p)
            return 1 - np.mean(p_values)
    
    def _maximum_mean_discrepancy(self, new_data: np.ndarray) -> float:
        """Maximum Mean Discrepancy for drift detection."""
        # Gaussian kernel
        def gaussian_kernel(X, Y, sigma=1.0):
            """Compute Gaussian kernel matrix."""
            XX = np.sum(X**2, axis=1, keepdims=True)
            YY = np.sum(Y**2, axis=1, keepdims=True)
            XY = X @ Y.T
            
            distances = XX + YY.T - 2 * XY
            return np.exp(-distances / (2 * sigma**2))
        
        n_ref = len(self.reference_data)
        n_new = len(new_data)
        
        # Ensure 2D arrays
        if len(self.reference_data.shape) == 1:
            ref_data = self.reference_data.reshape(-1, 1)
            new_data = new_data.reshape(-1, 1)
        else:
            ref_data = self.reference_data
        
        # Compute kernel matrices
        K_xx = gaussian_kernel(ref_data, ref_data)
        K_yy = gaussian_kernel(new_data, new_data)
        K_xy = gaussian_kernel(ref_data, new_data)
        
        # MMD statistic
        mmd = (np.sum(K_xx) / (n_ref * n_ref) +
               np.sum(K_yy) / (n_new * n_new) -
               2 * np.sum(K_xy) / (n_ref * n_new))
        
        return max(0, mmd)
    
    def _population_stability_index(self, new_data: np.ndarray) -> float:
        """Population Stability Index for drift detection."""
        psi_values = []
        
        if len(new_data.shape) == 1:
            # Single feature
            ref_hist, _ = np.histogram(self.reference_data, bins=self.bins)
            new_hist, _ = np.histogram(new_data, bins=self.bins)
            
            # Normalize
            ref_hist = ref_hist / np.sum(ref_hist) + 1e-10
            new_hist = new_hist / np.sum(new_hist) + 1e-10
            
            # PSI calculation
            psi = np.sum((new_hist - ref_hist) * np.log(new_hist / ref_hist))
            psi_values.append(psi)
        else:
            # Multiple features
            for i in range(new_data.shape[1]):
                ref_hist, _ = np.histogram(self.reference_data[:, i], bins=self.bins[i])
                new_hist, _ = np.histogram(new_data[:, i], bins=self.bins[i])
                
                ref_hist = ref_hist / np.sum(ref_hist) + 1e-10
                new_hist = new_hist / np.sum(new_hist) + 1e-10
                
                psi = np.sum((new_hist - ref_hist) * np.log(new_hist / ref_hist))
                psi_values.append(psi)
        
        return np.mean(psi_values)
    
    def _kl_divergence(self, new_data: np.ndarray) -> float:
        """KL divergence for drift detection."""
        # Create histograms
        n_bins = 20
        
        if len(new_data.shape) == 1:
            ref_hist, bins = np.histogram(self.reference_data, bins=n_bins)
            new_hist, _ = np.histogram(new_data, bins=bins)
            
            # Normalize
            ref_hist = ref_hist / np.sum(ref_hist) + 1e-10
            new_hist = new_hist / np.sum(new_hist) + 1e-10
            
            # KL divergence
            kl = np.sum(ref_hist * np.log(ref_hist / new_hist))
            return kl
        else:
            kl_values = []
            for i in range(new_data.shape[1]):
                ref_hist, bins = np.histogram(self.reference_data[:, i], bins=n_bins)
                new_hist, _ = np.histogram(new_data[:, i], bins=bins)
                
                ref_hist = ref_hist / np.sum(ref_hist) + 1e-10
                new_hist = new_hist / np.sum(new_hist) + 1e-10
                
                kl = np.sum(ref_hist * np.log(ref_hist / new_hist))
                kl_values.append(kl)
            
            return np.mean(kl_values)


class CalibrationMonitor:
    """
    Monitors calibration quality over time.
    
    Tracks calibration metrics and detects calibration degradation.
    """
    
    def __init__(
        self,
        target_coverage: float = 0.9,
        ece_threshold: float = 0.05,
        window_size: int = 100
    ):
        """
        Initialize calibration monitor.
        
        Args:
            target_coverage: Target coverage for prediction intervals
            ece_threshold: Threshold for acceptable ECE
            window_size: Size of sliding window
        """
        self.target_coverage = target_coverage
        self.ece_threshold = ece_threshold
        self.window_size = window_size
        
        self.predictions_buffer = []
        self.labels_buffer = []
        self.calibration_history = []
    
    def update(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Update calibration monitoring with new data.
        
        Args:
            predictions: Model predictions
            labels: True labels
            uncertainties: Uncertainty estimates
            
        Returns:
            Dictionary of calibration metrics
        """
        # Update buffers
        self.predictions_buffer.extend(predictions.tolist())
        self.labels_buffer.extend(labels.tolist())
        
        if len(self.predictions_buffer) > self.window_size:
            self.predictions_buffer = self.predictions_buffer[-self.window_size:]
            self.labels_buffer = self.labels_buffer[-self.window_size:]
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            np.array(self.predictions_buffer),
            np.array(self.labels_buffer),
            uncertainties
        )
        
        self.calibration_history.append(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainties: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate calibration metrics."""
        from ..metrics.calibration_metrics import (
            expected_calibration_error,
            brier_score,
            negative_log_likelihood
        )
        
        metrics = {}
        
        # ECE
        if len(predictions.shape) == 2:
            # Classification with probabilities
            metrics['ece'] = expected_calibration_error(labels, predictions)
            metrics['brier'] = np.mean([brier_score(labels == i, predictions[:, i]) 
                                        for i in range(predictions.shape[1])])
            metrics['nll'] = negative_log_likelihood(labels, predictions)
        else:
            # Regression or binary classification
            if uncertainties is not None:
                # Check coverage
                lower = predictions - 2 * uncertainties  # Assuming Gaussian
                upper = predictions + 2 * uncertainties
                coverage = np.mean((labels >= lower) & (labels <= upper))
                metrics['coverage'] = coverage
                metrics['coverage_gap'] = abs(coverage - self.target_coverage)
        
        # Accuracy
        if len(predictions.shape) == 2:
            pred_classes = np.argmax(predictions, axis=1)
            metrics['accuracy'] = np.mean(pred_classes == labels)
        
        return metrics
    
    def check_calibration_drift(self) -> bool:
        """
        Check if calibration has drifted.
        
        Returns:
            True if calibration has degraded
        """
        if len(self.calibration_history) < 2:
            return False
        
        recent_metrics = self.calibration_history[-1]
        
        # Check ECE threshold
        if 'ece' in recent_metrics and recent_metrics['ece'] > self.ece_threshold:
            return True
        
        # Check coverage gap
        if 'coverage_gap' in recent_metrics and recent_metrics['coverage_gap'] > 0.05:
            return True
        
        # Check trend
        if len(self.calibration_history) >= 10:
            recent_eces = [m.get('ece', 0) for m in self.calibration_history[-10:]]
            if len(recent_eces) > 0:
                trend = np.polyfit(range(len(recent_eces)), recent_eces, 1)[0]
                if trend > 0.01:  # Increasing ECE trend
                    return True
        
        return False


class OODDetector:
    """
    Out-of-Distribution detection for monitoring.
    
    Detects when inputs are outside the training distribution.
    """
    
    def __init__(
        self,
        reference_features: np.ndarray,
        method: str = 'mahalanobis',
        threshold: Optional[float] = None
    ):
        """
        Initialize OOD detector.
        
        Args:
            reference_features: Training data features
            method: Detection method ('mahalanobis', 'energy', 'density')
            threshold: Detection threshold (auto-computed if None)
        """
        self.reference_features = reference_features
        self.method = method
        
        # Compute reference statistics
        self._fit_reference()
        
        # Set threshold
        if threshold is None:
            self.threshold = self._compute_threshold()
        else:
            self.threshold = threshold
    
    def _fit_reference(self):
        """Fit reference distribution."""
        self.ref_mean = np.mean(self.reference_features, axis=0)
        self.ref_cov = np.cov(self.reference_features.T)
        
        # Add regularization for numerical stability
        self.ref_cov += 1e-6 * np.eye(self.ref_cov.shape[0])
        
        # Compute inverse covariance
        self.ref_cov_inv = np.linalg.inv(self.ref_cov)
    
    def _compute_threshold(self) -> float:
        """Compute automatic threshold at 95th percentile."""
        scores = self.score(self.reference_features)
        return np.percentile(scores, 95)
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute OOD scores.
        
        Args:
            features: Input features
            
        Returns:
            OOD scores (higher = more OOD)
        """
        if self.method == 'mahalanobis':
            return self._mahalanobis_distance(features)
        elif self.method == 'energy':
            return self._energy_score(features)
        elif self.method == 'density':
            return self._density_score(features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mahalanobis_distance(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance."""
        diff = features - self.ref_mean
        
        if len(features.shape) == 1:
            diff = diff.reshape(1, -1)
        
        distances = np.sqrt(np.sum((diff @ self.ref_cov_inv) * diff, axis=1))
        return distances
    
    def _energy_score(self, features: np.ndarray) -> np.ndarray:
        """Compute energy-based OOD score."""
        # Simplified energy score using nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(self.reference_features)
        
        distances, _ = nn.kneighbors(features)
        energy_scores = -np.log(np.mean(np.exp(-distances), axis=1))
        
        return energy_scores
    
    def _density_score(self, features: np.ndarray) -> np.ndarray:
        """Compute density-based OOD score."""
        from sklearn.neighbors import KernelDensity
        
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
        kde.fit(self.reference_features)
        
        log_density = kde.score_samples(features)
        return -log_density  # Negative log-likelihood
    
    def is_ood(self, features: np.ndarray) -> np.ndarray:
        """
        Detect OOD samples.
        
        Args:
            features: Input features
            
        Returns:
            Boolean array indicating OOD samples
        """
        scores = self.score(features)
        return scores > self.threshold


class UncertaintyMonitor:
    """
    Comprehensive monitoring system for uncertainty quantification.
    
    Combines drift detection, calibration monitoring, and OOD detection.
    """
    
    def __init__(
        self,
        reference_features: np.ndarray,
        reference_predictions: np.ndarray,
        reference_labels: np.ndarray,
        config: Optional[Dict] = None
    ):
        """
        Initialize uncertainty monitor.
        
        Args:
            reference_features: Reference input features
            reference_predictions: Reference predictions
            reference_labels: Reference labels
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.drift_detector = DriftDetector(
            reference_features,
            method=self.config.get('drift_method', 'ks'),
            threshold=self.config.get('drift_threshold', 0.05)
        )
        
        self.calibration_monitor = CalibrationMonitor(
            target_coverage=self.config.get('target_coverage', 0.9),
            ece_threshold=self.config.get('ece_threshold', 0.05)
        )
        
        self.ood_detector = OODDetector(
            reference_features,
            method=self.config.get('ood_method', 'mahalanobis')
        )
        
        # Initialize history
        self.monitoring_history = []
    
    def update(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> MonitoringMetrics:
        """
        Update monitoring with new data.
        
        Args:
            features: Input features
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            labels: True labels (if available)
            
        Returns:
            Monitoring metrics
        """
        # Drift detection
        drift_detected, drift_score = self.drift_detector.detect_drift(features)
        
        # OOD detection
        ood_scores = self.ood_detector.score(features)
        ood_rate = np.mean(self.ood_detector.is_ood(features))
        
        # Calibration monitoring (if labels available)
        calibration_metrics = {}
        if labels is not None:
            calibration_metrics = self.calibration_monitor.update(
                predictions, labels, uncertainties
            )
        
        # Create metrics object
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            calibration_error=calibration_metrics.get('ece', -1),
            coverage=calibration_metrics.get('coverage', -1),
            mean_uncertainty=np.mean(uncertainties),
            prediction_accuracy=calibration_metrics.get('accuracy', -1),
            drift_score=drift_score,
            ood_score=np.mean(ood_scores),
            metadata={
                'drift_detected': drift_detected,
                'ood_rate': ood_rate,
                'n_samples': len(features),
                **calibration_metrics
            }
        )
        
        self.monitoring_history.append(metrics)
        
        return metrics
    
    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        if not self.monitoring_history:
            return {}
        
        recent = self.monitoring_history[-1]
        
        summary = {
            'latest_metrics': {
                'timestamp': recent.timestamp.isoformat(),
                'calibration_error': recent.calibration_error,
                'coverage': recent.coverage,
                'mean_uncertainty': recent.mean_uncertainty,
                'drift_score': recent.drift_score,
                'ood_score': recent.ood_score
            },
            'alerts': []
        }
        
        # Check for alerts
        if recent.drift_score > self.drift_detector.threshold:
            summary['alerts'].append('Distribution drift detected')
        
        if recent.calibration_error > self.calibration_monitor.ece_threshold:
            summary['alerts'].append('Calibration degradation detected')
        
        if recent.metadata.get('ood_rate', 0) > 0.1:
            summary['alerts'].append('High OOD rate detected')
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export monitoring metrics to file."""
        metrics_data = []
        
        for metric in self.monitoring_history:
            metrics_data.append({
                'timestamp': metric.timestamp.isoformat(),
                'calibration_error': metric.calibration_error,
                'coverage': metric.coverage,
                'mean_uncertainty': metric.mean_uncertainty,
                'prediction_accuracy': metric.prediction_accuracy,
                'drift_score': metric.drift_score,
                'ood_score': metric.ood_score,
                'metadata': metric.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)