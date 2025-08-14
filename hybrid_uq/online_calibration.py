"""
Online Calibration Refresh Mechanisms

This module provides real-time calibration updates for maintaining model 
reliability in production environments. It integrates with conformal prediction
and drift monitoring to automatically adapt to changing data distributions.

Key Features:
- Online conformal prediction updates with adaptive quantiles
- Temperature scaling refresh for neural network calibration
- Isotonic regression updates for non-parametric calibration
- Drift-aware calibration with automatic triggers
- Coverage monitoring and automatic recalibration
- Integration with monitoring systems for seamless operation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import warnings
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import time

@dataclass
class CalibrationMetrics:
    """Container for calibration quality metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    nll: float  # Negative Log-Likelihood
    coverage: Optional[float] = None  # For regression
    interval_width: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

class OnlineTemperatureScaling:
    """
    Online temperature scaling for neural network calibration.
    
    Maintains and updates temperature parameter in real-time as new
    predictions and labels become available.
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        window_size: int = 1000,
        update_frequency: int = 100
    ):
        """
        Initialize online temperature scaling.
        
        Args:
            initial_temperature: Starting temperature value
            learning_rate: Learning rate for temperature updates
            momentum: Momentum for smoothing updates
            window_size: Size of sliding window for calibration data
            update_frequency: How often to update temperature (in samples)
        """
        self.temperature = initial_temperature
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        # Data storage
        self.logits_buffer = deque(maxlen=window_size)
        self.labels_buffer = deque(maxlen=window_size)
        
        # Update tracking
        self.samples_since_update = 0
        self.temperature_history = [initial_temperature]
        self.calibration_history = []
        
        # Momentum state
        self.velocity = 0.0
        
    def add_batch(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Add new batch of logits and labels."""
        for i in range(len(logits)):
            self.logits_buffer.append(logits[i])
            self.labels_buffer.append(labels[i])
        
        self.samples_since_update += len(logits)
        
        # Update temperature if needed
        if self.samples_since_update >= self.update_frequency:
            self._update_temperature()
            self.samples_since_update = 0
    
    def _update_temperature(self) -> None:
        """Update temperature using current buffer data."""
        if len(self.logits_buffer) < 10:  # Need minimum data
            return
            
        # Convert buffer to arrays
        logits_array = np.array(list(self.logits_buffer))
        labels_array = np.array(list(self.labels_buffer))
        
        # Compute gradient of NLL w.r.t. temperature
        scaled_logits = logits_array / self.temperature
        probs = self._softmax(scaled_logits)
        
        # Gradient computation
        # d(NLL)/d(T) = (1/T^2) * sum(logits * (probs - one_hot))
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels_array)), labels_array] = 1
        
        gradient = np.sum(logits_array * (probs - one_hot)) / (self.temperature ** 2)
        gradient /= len(logits_array)  # Average over batch
        
        # Momentum update
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        new_temperature = self.temperature + self.velocity
        
        # Constrain temperature to reasonable range
        self.temperature = np.clip(new_temperature, 0.1, 10.0)
        self.temperature_history.append(self.temperature)
        
        # Compute and store calibration metrics
        metrics = self._compute_calibration_metrics(logits_array, labels_array)
        self.calibration_history.append(metrics)
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Stable softmax computation."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _compute_calibration_metrics(self, logits: np.ndarray, labels: np.ndarray) -> CalibrationMetrics:
        """Compute calibration metrics for current data."""
        scaled_logits = logits / self.temperature
        probs = self._softmax(scaled_logits)
        
        # Expected Calibration Error
        ece = self._compute_ece(probs, labels)
        
        # Maximum Calibration Error
        mce = self._compute_mce(probs, labels)
        
        # Brier Score
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels)), labels] = 1
        brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
        
        # Negative Log-Likelihood
        nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-8))
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            nll=nll
        )
    
    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(accuracies[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                bin_weight = np.sum(bin_mask) / len(confidences)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _compute_mce(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_error = 0
        
        for i in range(n_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(accuracies[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                max_error = max(max_error, abs(bin_accuracy - bin_confidence))
        
        return max_error
    
    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply current temperature scaling to logits."""
        return logits / self.temperature
    
    def get_current_metrics(self) -> Optional[CalibrationMetrics]:
        """Get most recent calibration metrics."""
        return self.calibration_history[-1] if self.calibration_history else None

class OnlineIsotonicRegression:
    """
    Online isotonic regression for non-parametric calibration.
    
    Maintains isotonic regression model that adapts to new data
    while preserving monotonicity constraints.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        update_frequency: int = 100,
        min_samples_for_update: int = 50
    ):
        """
        Initialize online isotonic regression.
        
        Args:
            window_size: Size of sliding window for calibration data
            update_frequency: How often to refit the model
            min_samples_for_update: Minimum samples needed for model update
        """
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.min_samples_for_update = min_samples_for_update
        
        # Data storage
        self.scores_buffer = deque(maxlen=window_size)
        self.labels_buffer = deque(maxlen=window_size)
        
        # Model
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        
        # Update tracking
        self.samples_since_update = 0
        self.calibration_history = []
        
    def add_batch(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Add new batch of prediction scores and labels."""
        for i in range(len(scores)):
            self.scores_buffer.append(scores[i])
            self.labels_buffer.append(labels[i])
        
        self.samples_since_update += len(scores)
        
        # Update model if needed
        if (self.samples_since_update >= self.update_frequency and 
            len(self.scores_buffer) >= self.min_samples_for_update):
            self._update_model()
            self.samples_since_update = 0
    
    def _update_model(self) -> None:
        """Refit isotonic regression model with current buffer data."""
        scores_array = np.array(list(self.scores_buffer))
        labels_array = np.array(list(self.labels_buffer))
        
        # Fit isotonic regression
        self.isotonic_model.fit(scores_array, labels_array)
        self.is_fitted = True
        
        # Compute calibration metrics
        calibrated_probs = self.isotonic_model.predict(scores_array)
        metrics = self._compute_calibration_metrics(calibrated_probs, labels_array)
        self.calibration_history.append(metrics)
    
    def _compute_calibration_metrics(self, probs: np.ndarray, labels: np.ndarray) -> CalibrationMetrics:
        """Compute calibration metrics."""
        # Brier Score
        brier = brier_score_loss(labels, probs)
        
        # NLL for binary classification
        probs_clipped = np.clip(probs, 1e-8, 1 - 1e-8)
        nll = -np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped))
        
        # ECE and MCE
        ece = self._compute_ece_binary(probs, labels)
        mce = self._compute_mce_binary(probs, labels)
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            nll=nll
        )
    
    def _compute_ece_binary(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute ECE for binary classification."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(labels[bin_mask])
                bin_confidence = np.mean(probs[bin_mask])
                bin_weight = np.sum(bin_mask) / len(probs)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _compute_mce_binary(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute MCE for binary classification."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_error = 0
        
        for i in range(n_bins):
            bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(labels[bin_mask])
                bin_confidence = np.mean(probs[bin_mask])
                max_error = max(max_error, abs(bin_accuracy - bin_confidence))
        
        return max_error
    
    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to prediction scores."""
        if not self.is_fitted:
            return scores  # Return uncalibrated if not fitted
        
        return self.isotonic_model.predict(scores)

class OnlineConformalCalibration:
    """
    Online conformal prediction calibration.
    
    Maintains conformal quantiles that adapt to changing distributions
    while preserving coverage guarantees.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 1000,
        update_frequency: int = 50,
        adaptive_alpha: bool = True
    ):
        """
        Initialize online conformal calibration.
        
        Args:
            alpha: Nominal miscoverage level
            window_size: Size of sliding window for conformity scores
            update_frequency: How often to update quantiles
            adaptive_alpha: Whether to adapt alpha based on observed coverage
        """
        self.alpha = alpha
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.adaptive_alpha = adaptive_alpha
        
        # Data storage
        self.conformity_scores = deque(maxlen=window_size)
        self.coverage_buffer = deque(maxlen=window_size)
        
        # Current quantile
        self.current_quantile = 0.0
        self.quantile_history = []
        
        # Adaptive alpha tracking
        self.alpha_history = [alpha]
        self.target_coverage = 1 - alpha
        
        # Update tracking
        self.samples_since_update = 0
        self.coverage_history = []
        
    def add_conformity_scores(self, scores: np.ndarray, 
                            coverage_indicators: Optional[np.ndarray] = None) -> None:
        """
        Add new conformity scores and coverage indicators.
        
        Args:
            scores: New conformity scores
            coverage_indicators: Binary indicators of whether predictions were covered
        """
        for i, score in enumerate(scores):
            self.conformity_scores.append(score)
            
            if coverage_indicators is not None:
                self.coverage_buffer.append(coverage_indicators[i])
        
        self.samples_since_update += len(scores)
        
        # Update quantile if needed
        if self.samples_since_update >= self.update_frequency:
            self._update_quantile()
            self.samples_since_update = 0
    
    def _update_quantile(self) -> None:
        """Update conformal quantile based on current scores."""
        if len(self.conformity_scores) < 10:
            return
        
        # Adapt alpha if enabled and we have coverage data
        if self.adaptive_alpha and len(self.coverage_buffer) >= 50:
            self._adapt_alpha()
        
        # Compute new quantile
        scores_array = np.array(list(self.conformity_scores))
        n = len(scores_array)
        q_level = (n + 1) * (1 - self.alpha) / n
        q_level = min(q_level, 1.0)
        
        self.current_quantile = np.quantile(scores_array, q_level)
        self.quantile_history.append(self.current_quantile)
        
        # Compute coverage metrics if available
        if len(self.coverage_buffer) > 0:
            recent_coverage = np.mean(list(self.coverage_buffer)[-100:])  # Last 100 samples
            coverage_metrics = CalibrationMetrics(
                ece=0.0,  # Not applicable for conformal
                mce=0.0,  # Not applicable for conformal
                brier_score=0.0,  # Not applicable for conformal
                nll=0.0,  # Not applicable for conformal
                coverage=recent_coverage,
                interval_width=self.current_quantile
            )
            self.coverage_history.append(coverage_metrics)
    
    def _adapt_alpha(self) -> None:
        """Adapt alpha based on observed coverage."""
        recent_coverage = np.mean(list(self.coverage_buffer)[-100:])  # Last 100 samples
        coverage_error = recent_coverage - self.target_coverage
        
        # Simple adaptive rule: adjust alpha to correct coverage
        alpha_adjustment = -0.1 * coverage_error  # Negative because we want inverse relationship
        new_alpha = self.alpha + alpha_adjustment
        
        # Constrain alpha to reasonable range
        self.alpha = np.clip(new_alpha, 0.01, 0.5)
        self.alpha_history.append(self.alpha)
        self.target_coverage = 1 - self.alpha
    
    def get_prediction_interval(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction intervals using current quantile."""
        lower = predictions - self.current_quantile
        upper = predictions + self.current_quantile
        return lower, upper
    
    def get_current_coverage_metrics(self) -> Optional[CalibrationMetrics]:
        """Get most recent coverage metrics."""
        return self.coverage_history[-1] if self.coverage_history else None

class DriftAwareCalibrationManager:
    """
    Manages calibration updates based on drift detection.
    
    Integrates with drift monitoring to trigger recalibration
    when distribution shifts are detected.
    """
    
    def __init__(
        self,
        drift_monitor,
        temperature_scaler: Optional[OnlineTemperatureScaling] = None,
        isotonic_calibrator: Optional[OnlineIsotonicRegression] = None,
        conformal_calibrator: Optional[OnlineConformalCalibration] = None,
        drift_threshold: float = 0.5,
        forced_recalibration_interval: int = 10000
    ):
        """
        Initialize drift-aware calibration manager.
        
        Args:
            drift_monitor: Drift monitoring system
            temperature_scaler: Temperature scaling calibrator
            isotonic_calibrator: Isotonic regression calibrator
            conformal_calibrator: Conformal prediction calibrator
            drift_threshold: Drift score threshold for triggering recalibration
            forced_recalibration_interval: Force recalibration after this many samples
        """
        self.drift_monitor = drift_monitor
        self.temperature_scaler = temperature_scaler
        self.isotonic_calibrator = isotonic_calibrator
        self.conformal_calibrator = conformal_calibrator
        self.drift_threshold = drift_threshold
        self.forced_recalibration_interval = forced_recalibration_interval
        
        # Tracking
        self.samples_processed = 0
        self.last_forced_recalibration = 0
        self.recalibration_history = []
        
    def process_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        logits: Optional[np.ndarray] = None,
        conformity_scores: Optional[np.ndarray] = None,
        coverage_indicators: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process new batch with drift detection and calibration updates.
        
        Args:
            X: Input features
            y: True labels
            predictions: Model predictions
            logits: Raw logits (for temperature scaling)
            conformity_scores: Conformity scores (for conformal prediction)
            coverage_indicators: Coverage indicators (for conformal prediction)
            
        Returns:
            Dictionary with processing results and calibration status
        """
        self.samples_processed += len(X)
        
        # Check for drift
        drift_results = self.drift_monitor.update_streaming(X)
        drift_detected = drift_results is not None and drift_results['drift_detected']
        
        # Determine if recalibration is needed
        should_recalibrate = (
            drift_detected or 
            (self.samples_processed - self.last_forced_recalibration >= self.forced_recalibration_interval)
        )
        
        recalibration_triggered = False
        
        if should_recalibrate:
            # Trigger intensive recalibration
            self._trigger_recalibration(X, y, predictions, logits)
            self.last_forced_recalibration = self.samples_processed
            recalibration_triggered = True
            
            # Log recalibration event
            self.recalibration_history.append({
                'timestamp': time.time(),
                'samples_processed': self.samples_processed,
                'trigger_reason': 'drift' if drift_detected else 'forced',
                'drift_score': drift_results['consensus_score'] if drift_results else 0.0
            })
        
        # Always update online calibrators
        if self.temperature_scaler and logits is not None:
            self.temperature_scaler.add_batch(logits, y)
        
        if self.isotonic_calibrator:
            # Use max probability as score for binary case
            scores = np.max(predictions, axis=1) if len(predictions.shape) > 1 else predictions
            binary_labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
            self.isotonic_calibrator.add_batch(scores, binary_labels)
        
        if self.conformal_calibrator and conformity_scores is not None:
            self.conformal_calibrator.add_conformity_scores(conformity_scores, coverage_indicators)
        
        # Get current calibration metrics
        current_metrics = {}
        if self.temperature_scaler:
            temp_metrics = self.temperature_scaler.get_current_metrics()
            if temp_metrics:
                current_metrics['temperature_scaling'] = temp_metrics
        
        if self.isotonic_calibrator:
            iso_metrics = self.isotonic_calibrator.calibration_history
            if iso_metrics:
                current_metrics['isotonic_regression'] = iso_metrics[-1]
        
        if self.conformal_calibrator:
            conf_metrics = self.conformal_calibrator.get_current_coverage_metrics()
            if conf_metrics:
                current_metrics['conformal_prediction'] = conf_metrics
        
        return {
            'drift_detected': drift_detected,
            'drift_results': drift_results,
            'recalibration_triggered': recalibration_triggered,
            'current_metrics': current_metrics,
            'samples_processed': self.samples_processed
        }
    
    def _trigger_recalibration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        logits: Optional[np.ndarray] = None
    ) -> None:
        """Trigger intensive recalibration of all calibrators."""
        warnings.warn("Drift detected - triggering recalibration", UserWarning)
        
        # For temperature scaling, we can force an immediate update
        if self.temperature_scaler and logits is not None:
            # Force update by setting samples_since_update
            self.temperature_scaler.samples_since_update = self.temperature_scaler.update_frequency
            self.temperature_scaler.add_batch(logits, y)
        
        # For isotonic regression, force update
        if self.isotonic_calibrator:
            scores = np.max(predictions, axis=1) if len(predictions.shape) > 1 else predictions
            binary_labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
            self.isotonic_calibrator.samples_since_update = self.isotonic_calibrator.update_frequency
            self.isotonic_calibrator.add_batch(scores, binary_labels)
        
        # For conformal prediction, force quantile update
        if self.conformal_calibrator:
            self.conformal_calibrator.samples_since_update = self.conformal_calibrator.update_frequency
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get comprehensive calibration summary."""
        summary = {
            'samples_processed': self.samples_processed,
            'n_recalibrations': len(self.recalibration_history),
            'last_recalibration': self.recalibration_history[-1] if self.recalibration_history else None
        }
        
        # Add calibrator-specific summaries
        if self.temperature_scaler:
            summary['temperature_scaling'] = {
                'current_temperature': self.temperature_scaler.temperature,
                'n_updates': len(self.temperature_scaler.temperature_history) - 1,
                'current_metrics': self.temperature_scaler.get_current_metrics()
            }
        
        if self.isotonic_calibrator:
            summary['isotonic_regression'] = {
                'is_fitted': self.isotonic_calibrator.is_fitted,
                'n_updates': len(self.isotonic_calibrator.calibration_history),
                'current_metrics': (self.isotonic_calibrator.calibration_history[-1] 
                                  if self.isotonic_calibrator.calibration_history else None)
            }
        
        if self.conformal_calibrator:
            summary['conformal_prediction'] = {
                'current_quantile': self.conformal_calibrator.current_quantile,
                'current_alpha': self.conformal_calibrator.alpha,
                'n_updates': len(self.conformal_calibrator.quantile_history),
                'current_metrics': self.conformal_calibrator.get_current_coverage_metrics()
            }
        
        return summary

# Utility function for easy setup
def create_online_calibration_pipeline(
    drift_monitor,
    calibration_methods: List[str] = None,
    **calibrator_kwargs
) -> DriftAwareCalibrationManager:
    """
    Create complete online calibration pipeline.
    
    Args:
        drift_monitor: Drift monitoring system
        calibration_methods: List of calibration methods to use
        **calibrator_kwargs: Arguments for individual calibrators
        
    Returns:
        DriftAwareCalibrationManager instance
    """
    if calibration_methods is None:
        calibration_methods = ['temperature_scaling', 'conformal_prediction']
    
    # Create calibrators based on requested methods
    temperature_scaler = None
    isotonic_calibrator = None
    conformal_calibrator = None
    
    if 'temperature_scaling' in calibration_methods:
        temp_kwargs = {k: v for k, v in calibrator_kwargs.items() if k.startswith('temp_')}
        temp_kwargs = {k[5:]: v for k, v in temp_kwargs.items()}  # Remove 'temp_' prefix
        temperature_scaler = OnlineTemperatureScaling(**temp_kwargs)
    
    if 'isotonic_regression' in calibration_methods:
        iso_kwargs = {k: v for k, v in calibrator_kwargs.items() if k.startswith('iso_')}
        iso_kwargs = {k[4:]: v for k, v in iso_kwargs.items()}  # Remove 'iso_' prefix
        isotonic_calibrator = OnlineIsotonicRegression(**iso_kwargs)
    
    if 'conformal_prediction' in calibration_methods:
        conf_kwargs = {k: v for k, v in calibrator_kwargs.items() if k.startswith('conf_')}
        conf_kwargs = {k[5:]: v for k, v in conf_kwargs.items()}  # Remove 'conf_' prefix
        conformal_calibrator = OnlineConformalCalibration(**conf_kwargs)
    
    return DriftAwareCalibrationManager(
        drift_monitor=drift_monitor,
        temperature_scaler=temperature_scaler,
        isotonic_calibrator=isotonic_calibrator,
        conformal_calibrator=conformal_calibrator
    )