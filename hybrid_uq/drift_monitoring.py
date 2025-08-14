"""
Non-Canonically Invasive Drift Monitoring System

This module provides advanced drift detection capabilities specifically designed for 
monitoring distribution shifts that are subtle, non-standard, and potentially 
adversarial in nature. It goes beyond traditional drift detection to identify 
complex, multi-dimensional shifts that might evade standard statistical tests.

Key Features:
- Multi-dimensional drift detection using advanced statistical tests
- Non-parametric methods for unknown distribution families
- Adversarial drift detection for security-aware monitoring
- Real-time streaming drift detection with adaptive thresholds
- Integration with conformal prediction for coverage-aware monitoring
- Hierarchical drift analysis (feature-level, model-level, system-level)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

@dataclass
class DriftAlert:
    """Container for drift detection alerts."""
    timestamp: float
    drift_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_features: List[str]
    test_statistic: float
    p_value: Optional[float]
    threshold: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

class BaseDriftDetector(ABC):
    """Abstract base class for drift detectors."""
    
    def __init__(self, name: str, threshold: float = 0.05):
        self.name = name
        self.threshold = threshold
        self.is_fitted = False
        self.reference_data = None
        
    @abstractmethod
    def fit(self, X_ref: np.ndarray) -> None:
        """Fit detector on reference data."""
        pass
        
    @abstractmethod
    def detect(self, X_test: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift in test data."""
        pass

class KolmogorovSmirnovDriftDetector(BaseDriftDetector):
    """
    Kolmogorov-Smirnov test for univariate drift detection.
    
    Detects changes in the cumulative distribution function of individual features.
    """
    
    def __init__(self, threshold: float = 0.05, bonferroni_correction: bool = True):
        super().__init__("KS_Drift", threshold)
        self.bonferroni_correction = bonferroni_correction
        self.feature_distributions = None
        
    def fit(self, X_ref: np.ndarray) -> None:
        """Fit on reference data."""
        self.reference_data = X_ref.copy()
        self.n_features = X_ref.shape[1]
        self.is_fitted = True
        
    def detect(self, X_test: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift using KS test."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
            
        n_features = X_test.shape[1]
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")
        
        p_values = []
        statistics = []
        drift_features = []
        
        # Adjusted threshold for multiple comparisons
        alpha = self.threshold / n_features if self.bonferroni_correction else self.threshold
        
        for i in range(n_features):
            ref_feature = self.reference_data[:, i]
            test_feature = X_test[:, i]
            
            # KS test
            ks_stat, p_val = stats.ks_2samp(ref_feature, test_feature)
            
            p_values.append(p_val)
            statistics.append(ks_stat)
            
            if p_val < alpha:
                drift_features.append(i)
        
        # Overall drift decision
        drift_detected = len(drift_features) > 0
        min_p_value = min(p_values) if p_values else 1.0
        max_statistic = max(statistics) if statistics else 0.0
        
        metadata = {
            'p_values': p_values,
            'statistics': statistics,
            'drift_features': drift_features,
            'n_drift_features': len(drift_features),
            'bonferroni_corrected': self.bonferroni_correction
        }
        
        return drift_detected, max_statistic, metadata

class MaximumMeanDiscrepancyDetector(BaseDriftDetector):
    """
    Maximum Mean Discrepancy (MMD) for multivariate drift detection.
    
    Uses kernel methods to detect changes in distribution without 
    assumptions about the underlying distribution family.
    """
    
    def __init__(self, threshold: float = 0.05, kernel: str = 'rbf', 
                 gamma: float = 1.0, n_permutations: int = 1000):
        super().__init__("MMD_Drift", threshold)
        self.kernel = kernel
        self.gamma = gamma
        self.n_permutations = n_permutations
        
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """RBF kernel computation."""
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        K = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * K)
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Linear kernel computation."""
        return np.dot(X, Y.T)
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute MMD statistic."""
        m, n = len(X), len(Y)
        
        if self.kernel == 'rbf':
            Kxx = self._rbf_kernel(X, X)
            Kyy = self._rbf_kernel(Y, Y)
            Kxy = self._rbf_kernel(X, Y)
        elif self.kernel == 'linear':
            Kxx = self._linear_kernel(X, X)
            Kyy = self._linear_kernel(Y, Y)
            Kxy = self._linear_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # MMD^2 = 1/m^2 * sum(Kxx) + 1/n^2 * sum(Kyy) - 2/(m*n) * sum(Kxy)
        mmd_squared = (np.sum(Kxx) / (m * m) + 
                      np.sum(Kyy) / (n * n) - 
                      2 * np.sum(Kxy) / (m * n))
        
        return np.sqrt(max(0, mmd_squared))
    
    def fit(self, X_ref: np.ndarray) -> None:
        """Fit on reference data."""
        self.reference_data = X_ref.copy()
        self.is_fitted = True
        
    def detect(self, X_test: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift using MMD test with permutation."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        # Compute observed MMD
        observed_mmd = self._compute_mmd(self.reference_data, X_test)
        
        # Permutation test
        combined_data = np.vstack([self.reference_data, X_test])
        m = len(self.reference_data)
        n = len(X_test)
        
        permutation_mmds = []
        for _ in range(self.n_permutations):
            # Random permutation
            perm_indices = np.random.permutation(len(combined_data))
            perm_X = combined_data[perm_indices[:m]]
            perm_Y = combined_data[perm_indices[m:]]
            
            perm_mmd = self._compute_mmd(perm_X, perm_Y)
            permutation_mmds.append(perm_mmd)
        
        # Compute p-value
        p_value = np.mean(np.array(permutation_mmds) >= observed_mmd)
        drift_detected = p_value < self.threshold
        
        metadata = {
            'observed_mmd': observed_mmd,
            'permutation_mmds': permutation_mmds,
            'p_value': p_value,
            'kernel': self.kernel,
            'n_permutations': self.n_permutations
        }
        
        return drift_detected, observed_mmd, metadata

class AdversarialDriftDetector(BaseDriftDetector):
    """
    Adversarial drift detection using domain classification.
    
    Trains a classifier to distinguish between reference and test data.
    High classification accuracy indicates distribution shift.
    """
    
    def __init__(self, threshold: float = 0.6, model_type: str = 'neural_net',
                 n_epochs: int = 100, patience: int = 10):
        super().__init__("Adversarial_Drift", threshold)
        self.model_type = model_type
        self.n_epochs = n_epochs
        self.patience = patience
        self.classifier = None
        
    def _create_classifier(self, input_dim: int) -> nn.Module:
        """Create domain classifier."""
        if self.model_type == 'neural_net':
            return nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_ref: np.ndarray) -> None:
        """Fit on reference data."""
        self.reference_data = X_ref.copy()
        self.input_dim = X_ref.shape[1]
        self.is_fitted = True
        
    def detect(self, X_test: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift using adversarial training."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        # Prepare training data
        X_combined = np.vstack([self.reference_data, X_test])
        y_combined = np.concatenate([
            np.zeros(len(self.reference_data)),  # Reference = 0
            np.ones(len(X_test))                 # Test = 1
        ])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_combined)
        y_tensor = torch.FloatTensor(y_combined).unsqueeze(1)
        
        # Create and train classifier
        self.classifier = self._create_classifier(self.input_dim)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            self.classifier.train()
            optimizer.zero_grad()
            
            outputs = self.classifier(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Evaluate classifier performance
        self.classifier.eval()
        with torch.no_grad():
            predictions = self.classifier(X_tensor).numpy().flatten()
            
        # Compute AUC as drift statistic
        auc_score = roc_auc_score(y_combined, predictions)
        
        # Drift detected if classifier can distinguish well
        drift_detected = auc_score > self.threshold
        
        metadata = {
            'auc_score': auc_score,
            'final_loss': best_loss,
            'n_epochs_trained': epoch + 1,
            'predictions': predictions
        }
        
        return drift_detected, auc_score, metadata

class EnergyDistanceDetector(BaseDriftDetector):
    """
    Energy distance-based drift detection.
    
    Non-parametric test based on energy statistics that can detect
    changes in any aspect of the distribution.
    """
    
    def __init__(self, threshold: float = 0.05, n_bootstrap: int = 1000):
        super().__init__("Energy_Distance", threshold)
        self.n_bootstrap = n_bootstrap
        
    def _energy_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute energy distance between two samples."""
        m, n = len(X), len(Y)
        
        # Compute pairwise distances
        XX_dist = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
        YY_dist = np.sqrt(np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
        XY_dist = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
        
        # Energy distance formula
        term1 = np.mean(XX_dist)
        term2 = np.mean(YY_dist)
        term3 = np.mean(XY_dist)
        
        energy_dist = 2 * term3 - term1 - term2
        return energy_dist
    
    def fit(self, X_ref: np.ndarray) -> None:
        """Fit on reference data."""
        self.reference_data = X_ref.copy()
        self.is_fitted = True
        
    def detect(self, X_test: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift using energy distance."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        # Compute observed energy distance
        observed_energy = self._energy_distance(self.reference_data, X_test)
        
        # Bootstrap null distribution
        combined_data = np.vstack([self.reference_data, X_test])
        m = len(self.reference_data)
        
        bootstrap_energies = []
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(combined_data), size=len(combined_data), replace=True)
            bootstrap_sample = combined_data[indices]
            
            # Split into two groups of same size as original
            boot_X = bootstrap_sample[:m]
            boot_Y = bootstrap_sample[m:m + len(X_test)]
            
            if len(boot_Y) > 0:
                boot_energy = self._energy_distance(boot_X, boot_Y)
                bootstrap_energies.append(boot_energy)
        
        # Compute p-value
        p_value = np.mean(np.array(bootstrap_energies) >= observed_energy)
        drift_detected = p_value < self.threshold
        
        metadata = {
            'observed_energy': observed_energy,
            'bootstrap_energies': bootstrap_energies,
            'p_value': p_value,
            'n_bootstrap': self.n_bootstrap
        }
        
        return drift_detected, observed_energy, metadata

class NonCanonicalDriftMonitor:
    """
    Comprehensive drift monitoring system for non-canonical distributions.
    
    Combines multiple drift detection methods and provides real-time
    monitoring with adaptive thresholds and alert management.
    """
    
    def __init__(
        self,
        detectors: List[BaseDriftDetector] = None,
        window_size: int = 1000,
        alert_cooldown: int = 100,
        severity_thresholds: Dict[str, float] = None
    ):
        """
        Initialize drift monitoring system.
        
        Args:
            detectors: List of drift detectors to use
            window_size: Size of sliding window for streaming detection
            alert_cooldown: Minimum samples between alerts
            severity_thresholds: Thresholds for alert severity levels
        """
        if detectors is None:
            detectors = [
                KolmogorovSmirnovDriftDetector(),
                MaximumMeanDiscrepancyDetector(),
                AdversarialDriftDetector(),
                EnergyDistanceDetector()
            ]
        
        self.detectors = {detector.name: detector for detector in detectors}
        self.window_size = window_size
        self.alert_cooldown = alert_cooldown
        
        if severity_thresholds is None:
            severity_thresholds = {
                'low': 0.7,
                'medium': 0.8,
                'high': 0.9,
                'critical': 0.95
            }
        self.severity_thresholds = severity_thresholds
        
        # Streaming data storage
        self.data_buffer = deque(maxlen=window_size)
        self.alerts_history = []
        self.detection_history = defaultdict(list)
        self.last_alert_time = 0
        
        # Monitoring state
        self.is_fitted = False
        self.reference_data = None
        
    def fit(self, X_ref: np.ndarray) -> None:
        """Fit all detectors on reference data."""
        self.reference_data = X_ref.copy()
        
        for detector in self.detectors.values():
            detector.fit(X_ref)
            
        self.is_fitted = True
        
    def detect_drift(self, X_test: np.ndarray, 
                    feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive drift detection using all detectors.
        
        Args:
            X_test: Test data
            feature_names: Names of features for reporting
            
        Returns:
            Dictionary with detection results
        """
        if not self.is_fitted:
            raise ValueError("Monitor must be fitted before detection")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        detection_results = {}
        drift_scores = []
        drift_detected_any = False
        
        # Run all detectors
        for name, detector in self.detectors.items():
            try:
                drift_detected, score, metadata = detector.detect(X_test)
                
                detection_results[name] = {
                    'drift_detected': drift_detected,
                    'score': score,
                    'metadata': metadata
                }
                
                drift_scores.append(score if not np.isnan(score) else 0.0)
                if drift_detected:
                    drift_detected_any = True
                    
            except Exception as e:
                warnings.warn(f"Detector {name} failed: {str(e)}")
                detection_results[name] = {
                    'drift_detected': False,
                    'score': 0.0,
                    'metadata': {'error': str(e)}
                }
                drift_scores.append(0.0)
        
        # Aggregate results
        consensus_score = np.mean(drift_scores) if drift_scores else 0.0
        n_detectors_positive = sum(
            result['drift_detected'] for result in detection_results.values()
        )
        
        # Determine severity
        severity = self._determine_severity(consensus_score, n_detectors_positive)
        
        # Create alert if needed
        alert = None
        if drift_detected_any and self._should_create_alert():
            affected_features = self._identify_affected_features(
                detection_results, feature_names
            )
            
            alert = DriftAlert(
                timestamp=time.time(),
                drift_type="multi_detector",
                severity=severity,
                affected_features=affected_features,
                test_statistic=consensus_score,
                p_value=None,
                threshold=0.5,  # Consensus threshold
                confidence=n_detectors_positive / len(self.detectors),
                metadata={
                    'detection_results': detection_results,
                    'n_detectors_positive': n_detectors_positive,
                    'consensus_score': consensus_score
                }
            )
            
            self.alerts_history.append(alert)
            self.last_alert_time = len(self.data_buffer)
        
        return {
            'drift_detected': drift_detected_any,
            'consensus_score': consensus_score,
            'n_detectors_positive': n_detectors_positive,
            'severity': severity,
            'detection_results': detection_results,
            'alert': alert
        }
    
    def update_streaming(self, X_new: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Update with streaming data and detect drift.
        
        Args:
            X_new: New data batch
            
        Returns:
            Detection results if drift detected, None otherwise
        """
        # Add to buffer
        for sample in X_new:
            self.data_buffer.append(sample)
        
        # Check if we have enough data for detection
        if len(self.data_buffer) < self.window_size // 2:
            return None
        
        # Convert buffer to array for detection
        current_window = np.array(list(self.data_buffer))
        
        # Detect drift
        results = self.detect_drift(current_window)
        
        # Store detection history
        timestamp = time.time()
        for detector_name, result in results['detection_results'].items():
            self.detection_history[detector_name].append({
                'timestamp': timestamp,
                'drift_detected': result['drift_detected'],
                'score': result['score']
            })
        
        return results if results['drift_detected'] else None
    
    def _determine_severity(self, consensus_score: float, n_positive: int) -> str:
        """Determine alert severity based on consensus."""
        total_detectors = len(self.detectors)
        consensus_ratio = n_positive / total_detectors if total_detectors > 0 else 0
        
        combined_score = 0.7 * consensus_score + 0.3 * consensus_ratio
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if combined_score >= self.severity_thresholds[severity]:
                return severity
        
        return 'low'
    
    def _should_create_alert(self) -> bool:
        """Check if enough time has passed since last alert."""
        return len(self.data_buffer) - self.last_alert_time >= self.alert_cooldown
    
    def _identify_affected_features(self, detection_results: Dict, 
                                  feature_names: List[str]) -> List[str]:
        """Identify which features are affected by drift."""
        affected_features = set()
        
        for detector_name, result in detection_results.items():
            if result['drift_detected'] and 'drift_features' in result['metadata']:
                drift_indices = result['metadata']['drift_features']
                affected_features.update(feature_names[i] for i in drift_indices)
        
        return list(affected_features)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        # Alert statistics
        alert_counts = defaultdict(int)
        for alert in self.alerts_history:
            alert_counts[alert.severity] += 1
        
        # Detection history statistics
        detector_stats = {}
        for detector_name, history in self.detection_history.items():
            if history:
                recent_detections = [h['drift_detected'] for h in history[-100:]]
                detector_stats[detector_name] = {
                    'total_detections': len(history),
                    'recent_drift_rate': np.mean(recent_detections),
                    'avg_score': np.mean([h['score'] for h in history])
                }
        
        return {
            'status': 'monitoring',
            'n_samples_processed': len(self.data_buffer),
            'n_alerts_total': len(self.alerts_history),
            'alert_counts_by_severity': dict(alert_counts),
            'detector_statistics': detector_stats,
            'last_alert': self.alerts_history[-1] if self.alerts_history else None
        }
    
    def reset_monitoring(self) -> None:
        """Reset monitoring state while keeping fitted detectors."""
        self.data_buffer.clear()
        self.alerts_history.clear()
        self.detection_history.clear()
        self.last_alert_time = 0

# Integration with conformal prediction
def create_coverage_aware_drift_monitor(
    conformal_predictor,
    coverage_threshold: float = 0.05,
    **monitor_kwargs
) -> 'CoverageAwareDriftMonitor':
    """
    Create drift monitor that considers conformal prediction coverage.
    
    Args:
        conformal_predictor: Fitted conformal predictor
        coverage_threshold: Threshold for coverage degradation
        **monitor_kwargs: Arguments for NonCanonicalDriftMonitor
        
    Returns:
        CoverageAwareDriftMonitor instance
    """
    
    class CoverageAwareDriftMonitor(NonCanonicalDriftMonitor):
        def __init__(self, conformal_predictor, coverage_threshold, **kwargs):
            super().__init__(**kwargs)
            self.conformal_predictor = conformal_predictor
            self.coverage_threshold = coverage_threshold
            
        def detect_drift_with_coverage(self, X_test: np.ndarray, 
                                     y_test: np.ndarray) -> Dict[str, Any]:
            """Detect drift considering conformal coverage."""
            # Standard drift detection
            drift_results = self.detect_drift(X_test)
            
            # Evaluate conformal coverage
            coverage_results = self.conformal_predictor.evaluate_coverage(
                X_test, y_test
            )
            
            # Check for coverage degradation
            coverage_drift = (
                coverage_results['coverage_error'] > self.coverage_threshold
            )
            
            # Combine results
            combined_drift = drift_results['drift_detected'] or coverage_drift
            
            drift_results.update({
                'coverage_drift': coverage_drift,
                'coverage_results': coverage_results,
                'combined_drift': combined_drift
            })
            
            return drift_results
    
    return CoverageAwareDriftMonitor(conformal_predictor, coverage_threshold, **monitor_kwargs)

# Utility functions for integration
def setup_default_monitoring_pipeline(
    model,
    X_ref: np.ndarray,
    y_ref: Optional[np.ndarray] = None,
    alpha: float = 0.1
) -> Tuple[NonCanonicalDriftMonitor, Optional[Any]]:
    """
    Set up complete monitoring pipeline with conformal prediction.
    
    Args:
        model: Trained model
        X_ref: Reference data
        y_ref: Reference targets (optional, for conformal prediction)
        alpha: Miscoverage level for conformal prediction
        
    Returns:
        Tuple of (drift_monitor, conformal_predictor)
    """
    # Create drift monitor
    drift_monitor = NonCanonicalDriftMonitor()
    drift_monitor.fit(X_ref)
    
    # Create conformal predictor if targets provided
    conformal_predictor = None
    if y_ref is not None:
        from .conformal_node_rk4 import RK4ConformalPredictor
        conformal_predictor = RK4ConformalPredictor(model, alpha=alpha)
        conformal_predictor.fit(X_ref, y_ref)
    
    return drift_monitor, conformal_predictor