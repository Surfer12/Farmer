"""
Conformal Prediction for Calibrated Uncertainty Quantification

This module implements conformal prediction methods to provide
distribution-free, finite-sample coverage guarantees for the
hybrid AI-physics system.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import warnings
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt


class ConformalPredictor(ABC):
    """Base class for conformal predictors"""
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage level (1-alpha is target coverage)
        """
        self.alpha = alpha
        self.coverage_level = 1 - alpha
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, cal_scores: torch.Tensor) -> None:
        """Fit conformal predictor on calibration scores"""
        pass
    
    @abstractmethod
    def predict(self, predictions: torch.Tensor, 
               scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate conformal prediction sets"""
        pass


class SplitConformalPredictor(ConformalPredictor):
    """
    Split conformal prediction for regression
    
    Uses absolute residuals as conformity scores and provides
    prediction intervals with finite-sample coverage guarantees.
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha)
        self.quantile = None
        
    def fit(self, cal_scores: torch.Tensor) -> None:
        """
        Fit split conformal predictor
        
        Args:
            cal_scores: Calibration scores (typically absolute residuals)
                       Shape: [n_calibration_samples]
        """
        if len(cal_scores.shape) > 1:
            cal_scores = cal_scores.flatten()
            
        n = len(cal_scores)
        if n == 0:
            raise ValueError("Empty calibration set")
            
        # Compute empirical quantile
        # For finite-sample coverage, use ceiling adjustment
        q_level = np.ceil((n + 1) * self.coverage_level) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0
        
        self.quantile = torch.quantile(cal_scores, q_level)
        self.is_fitted = True
        
    def predict(self, predictions: torch.Tensor, 
               scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate prediction intervals
        
        Args:
            predictions: Point predictions [batch_size, ...]
            scores: Optional conformity scores for adaptive intervals
            
        Returns:
            Dictionary with lower, upper bounds and interval widths
        """
        if not self.is_fitted:
            raise RuntimeError("Conformal predictor not fitted")
            
        if scores is not None:
            # Adaptive intervals based on local conformity scores
            interval_width = scores
        else:
            # Fixed interval width
            interval_width = self.quantile
            
        lower = predictions - interval_width
        upper = predictions + interval_width
        
        return {
            'lower': lower,
            'upper': upper,
            'width': 2 * interval_width,
            'quantile': self.quantile
        }


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive conformal prediction with locally-weighted scores
    
    Adapts prediction intervals based on local difficulty/uncertainty.
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.1):
        super().__init__(alpha)
        self.gamma = gamma  # Learning rate for online updates
        self.quantiles = None
        self.difficulty_scores = None
        
    def fit(self, cal_scores: torch.Tensor, 
           difficulty_features: Optional[torch.Tensor] = None) -> None:
        """
        Fit adaptive conformal predictor
        
        Args:
            cal_scores: Calibration conformity scores
            difficulty_features: Features indicating local difficulty
        """
        if difficulty_features is not None:
            # Locally-weighted quantiles based on difficulty
            self.difficulty_scores = difficulty_features
            # Simple approach: use difficulty as weights for quantile estimation
            weights = torch.softmax(difficulty_features, dim=0)
            
            # Weighted quantile (approximate)
            sorted_indices = torch.argsort(cal_scores)
            sorted_scores = cal_scores[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cumulative_weights = torch.cumsum(sorted_weights, dim=0)
            
            # Find quantile position
            target_weight = self.coverage_level
            quantile_idx = torch.searchsorted(cumulative_weights, target_weight)
            quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
            
            self.quantiles = sorted_scores[quantile_idx]
        else:
            # Standard quantile
            n = len(cal_scores)
            q_level = np.ceil((n + 1) * self.coverage_level) / n
            self.quantiles = torch.quantile(cal_scores, q_level)
            
        self.is_fitted = True
        
    def predict(self, predictions: torch.Tensor, 
               difficulty_features: Optional[torch.Tensor] = None,
               scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate adaptive prediction intervals"""
        if not self.is_fitted:
            raise RuntimeError("Adaptive conformal predictor not fitted")
            
        if difficulty_features is not None and self.difficulty_scores is not None:
            # Adapt interval width based on local difficulty
            similarity = torch.exp(-torch.abs(difficulty_features.unsqueeze(0) - 
                                             self.difficulty_scores.unsqueeze(1)))
            weights = torch.softmax(similarity, dim=0)
            adaptive_quantile = torch.sum(weights * self.quantiles.unsqueeze(1), dim=0)
            interval_width = adaptive_quantile
        else:
            interval_width = self.quantiles
            
        if scores is not None:
            # Further adapt based on conformity scores
            interval_width = interval_width * (1 + scores)
            
        lower = predictions - interval_width
        upper = predictions + interval_width
        
        return {
            'lower': lower,
            'upper': upper, 
            'width': 2 * interval_width,
            'adaptive_quantile': interval_width
        }


class OnlineConformalPredictor(ConformalPredictor):
    """
    Online conformal prediction for streaming data
    
    Updates prediction intervals online as new data arrives,
    maintaining approximate coverage guarantees.
    """
    
    def __init__(self, alpha: float = 0.1, window_size: int = 1000):
        super().__init__(alpha)
        self.window_size = window_size
        self.score_buffer = []
        self.quantile = None
        self.update_count = 0
        
    def fit(self, cal_scores: torch.Tensor) -> None:
        """Initialize with calibration scores"""
        if len(cal_scores.shape) > 1:
            cal_scores = cal_scores.flatten()
            
        self.score_buffer = cal_scores.tolist()[-self.window_size:]
        self._update_quantile()
        self.is_fitted = True
        
    def _update_quantile(self):
        """Update quantile estimate from current buffer"""
        if len(self.score_buffer) == 0:
            return
            
        scores_tensor = torch.tensor(self.score_buffer)
        n = len(scores_tensor)
        q_level = np.ceil((n + 1) * self.coverage_level) / n
        self.quantile = torch.quantile(scores_tensor, q_level)
        
    def update(self, new_score: float):
        """
        Update with new conformity score
        
        Args:
            new_score: New conformity score from latest prediction
        """
        self.score_buffer.append(new_score)
        
        # Maintain sliding window
        if len(self.score_buffer) > self.window_size:
            self.score_buffer.pop(0)
            
        # Update quantile periodically for efficiency
        self.update_count += 1
        if self.update_count % 10 == 0:  # Update every 10 samples
            self._update_quantile()
            
    def predict(self, predictions: torch.Tensor,
               scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate prediction intervals with current quantile"""
        if not self.is_fitted:
            raise RuntimeError("Online conformal predictor not fitted")
            
        if self.quantile is None:
            self._update_quantile()
            
        interval_width = self.quantile
        if scores is not None:
            interval_width = interval_width * (1 + scores)
            
        lower = predictions - interval_width
        upper = predictions + interval_width
        
        return {
            'lower': lower,
            'upper': upper,
            'width': 2 * interval_width,
            'current_quantile': self.quantile,
            'buffer_size': len(self.score_buffer)
        }


class QuantileConformalPredictor(ConformalPredictor):
    """
    Conformal prediction for quantile regression
    
    Provides conformal prediction intervals when the base model
    already outputs quantile predictions.
    """
    
    def __init__(self, alpha: float = 0.1, quantiles: List[float] = None):
        super().__init__(alpha)
        if quantiles is None:
            # Default: predict median and symmetric quantiles
            self.quantiles = [alpha/2, 0.5, 1-alpha/2]
        else:
            self.quantiles = sorted(quantiles)
            
        self.conformity_quantiles = {}
        
    def fit(self, predicted_quantiles: torch.Tensor, 
           true_values: torch.Tensor) -> None:
        """
        Fit conformal quantile predictor
        
        Args:
            predicted_quantiles: Predicted quantiles [n_samples, n_quantiles]
            true_values: True target values [n_samples]
        """
        n_samples, n_quantiles = predicted_quantiles.shape
        
        if len(self.quantiles) != n_quantiles:
            raise ValueError(f"Expected {len(self.quantiles)} quantiles, got {n_quantiles}")
            
        # Compute conformity scores for each quantile
        for i, q in enumerate(self.quantiles):
            predicted_q = predicted_quantiles[:, i]
            
            if q <= 0.5:
                # Lower quantile: score = max(0, y - q_hat)
                scores = torch.clamp(true_values - predicted_q, min=0)
            else:
                # Upper quantile: score = max(0, q_hat - y)  
                scores = torch.clamp(predicted_q - true_values, min=0)
                
            # Compute conformal quantile
            n = len(scores)
            conf_level = np.ceil((n + 1) * self.coverage_level) / n
            self.conformity_quantiles[q] = torch.quantile(scores, conf_level)
            
        self.is_fitted = True
        
    def predict(self, predicted_quantiles: torch.Tensor,
               scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate conformal quantile predictions"""
        if not self.is_fitted:
            raise RuntimeError("Quantile conformal predictor not fitted")
            
        batch_size, n_quantiles = predicted_quantiles.shape
        adjusted_quantiles = torch.zeros_like(predicted_quantiles)
        
        for i, q in enumerate(self.quantiles):
            adjustment = self.conformity_quantiles[q]
            if scores is not None:
                adjustment = adjustment * (1 + scores)
                
            if q <= 0.5:
                # Adjust lower quantiles downward
                adjusted_quantiles[:, i] = predicted_quantiles[:, i] - adjustment
            else:
                # Adjust upper quantiles upward
                adjusted_quantiles[:, i] = predicted_quantiles[:, i] + adjustment
                
        # Extract prediction intervals (assume symmetric around median)
        if len(self.quantiles) >= 3:
            lower_idx = 0
            upper_idx = -1
            median_idx = len(self.quantiles) // 2
            
            lower = adjusted_quantiles[:, lower_idx]
            upper = adjusted_quantiles[:, upper_idx] 
            median = adjusted_quantiles[:, median_idx]
            width = upper - lower
        else:
            # Fallback for non-standard quantile sets
            lower = torch.min(adjusted_quantiles, dim=1)[0]
            upper = torch.max(adjusted_quantiles, dim=1)[0]
            median = torch.median(adjusted_quantiles, dim=1)[0]
            width = upper - lower
            
        return {
            'lower': lower,
            'upper': upper,
            'median': median,
            'width': width,
            'adjusted_quantiles': adjusted_quantiles,
            'quantile_levels': self.quantiles
        }


class ConformalCalibrator:
    """
    Calibration utilities for conformal prediction
    
    Provides methods to evaluate coverage, compute calibration
    metrics, and visualize conformal prediction performance.
    """
    
    @staticmethod
    def compute_coverage(predictions: Dict[str, torch.Tensor], 
                        true_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute empirical coverage of prediction intervals
        
        Args:
            predictions: Dictionary with 'lower' and 'upper' bounds
            true_values: True target values
            
        Returns:
            Dictionary with coverage metrics
        """
        lower = predictions['lower']
        upper = predictions['upper']
        
        # Check if true values fall within intervals
        covered = (true_values >= lower) & (true_values <= upper)
        empirical_coverage = covered.float().mean().item()
        
        # Interval widths
        widths = predictions.get('width', upper - lower)
        mean_width = widths.mean().item()
        
        # Coverage by percentiles (to check uniformity)
        n_quantiles = 10
        percentiles = torch.quantile(widths, torch.linspace(0, 1, n_quantiles+1))
        coverage_by_width = {}
        
        for i in range(n_quantiles):
            mask = (widths >= percentiles[i]) & (widths < percentiles[i+1])
            if mask.sum() > 0:
                coverage_by_width[f'width_p{i*10}-{(i+1)*10}'] = covered[mask].float().mean().item()
        
        return {
            'empirical_coverage': empirical_coverage,
            'mean_width': mean_width,
            'median_width': widths.median().item(),
            'coverage_by_width': coverage_by_width
        }
    
    @staticmethod
    def compute_efficiency_metrics(predictions: Dict[str, torch.Tensor],
                                 true_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute efficiency metrics for prediction intervals
        
        Args:
            predictions: Prediction intervals
            true_values: True values
            
        Returns:
            Efficiency metrics
        """
        widths = predictions.get('width', predictions['upper'] - predictions['lower'])
        
        # Normalized interval widths (by target range)
        target_range = true_values.max() - true_values.min()
        normalized_widths = widths / (target_range + 1e-8)
        
        # Conditional coverage (coverage given width)
        covered = ((true_values >= predictions['lower']) & 
                  (true_values <= predictions['upper']))
        
        return {
            'mean_normalized_width': normalized_widths.mean().item(),
            'width_std': widths.std().item(),
            'efficiency_score': 1.0 / (1.0 + normalized_widths.mean().item()),
            'coverage_width_correlation': torch.corrcoef(
                torch.stack([covered.float(), widths])
            )[0, 1].item() if len(widths) > 1 else 0.0
        }
    
    @staticmethod
    def plot_coverage_analysis(predictions: Dict[str, torch.Tensor],
                             true_values: torch.Tensor,
                             target_coverage: float = 0.9,
                             save_path: Optional[str] = None):
        """
        Plot coverage analysis for conformal predictions
        
        Args:
            predictions: Prediction intervals
            true_values: True values
            target_coverage: Target coverage level
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        lower = predictions['lower'].detach().numpy()
        upper = predictions['upper'].detach().numpy()
        widths = predictions.get('width', upper - lower).detach().numpy()
        true_vals = true_values.detach().numpy()
        
        covered = (true_vals >= lower) & (true_vals <= upper)
        
        # Plot 1: Prediction intervals
        axes[0, 0].fill_between(range(len(true_vals)), lower, upper, 
                               alpha=0.3, label='Prediction Interval')
        axes[0, 0].scatter(range(len(true_vals)), true_vals, 
                          c=['green' if c else 'red' for c in covered],
                          s=20, label='True Values')
        axes[0, 0].set_title('Prediction Intervals vs True Values')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        
        # Plot 2: Coverage vs Width
        axes[0, 1].scatter(widths, covered.astype(float), alpha=0.6)
        axes[0, 1].axhline(y=target_coverage, color='red', linestyle='--', 
                          label=f'Target Coverage ({target_coverage})')
        axes[0, 1].set_xlabel('Interval Width')
        axes[0, 1].set_ylabel('Coverage (0/1)')
        axes[0, 1].set_title('Coverage vs Interval Width')
        axes[0, 1].legend()
        
        # Plot 3: Width distribution
        axes[1, 0].hist(widths, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=widths.mean(), color='red', linestyle='--',
                          label=f'Mean Width: {widths.mean():.3f}')
        axes[1, 0].set_xlabel('Interval Width')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Interval Widths')
        axes[1, 0].legend()
        
        # Plot 4: Cumulative coverage
        sorted_indices = np.argsort(widths)
        cumulative_coverage = np.cumsum(covered[sorted_indices]) / np.arange(1, len(covered) + 1)
        
        axes[1, 1].plot(np.sort(widths), cumulative_coverage, 'b-', linewidth=2)
        axes[1, 1].axhline(y=target_coverage, color='red', linestyle='--',
                          label=f'Target Coverage ({target_coverage})')
        axes[1, 1].set_xlabel('Interval Width (sorted)')
        axes[1, 1].set_ylabel('Cumulative Coverage')
        axes[1, 1].set_title('Cumulative Coverage by Width')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def test_conformal_predictors():
    """Test conformal prediction implementations"""
    print("=== Testing Conformal Predictors ===")
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_cal = 200
    n_test = 50
    
    # Calibration data
    X_cal = torch.randn(n_cal, 2)
    y_cal = X_cal[:, 0] + 0.5 * X_cal[:, 1] + 0.2 * torch.randn(n_cal)
    pred_cal = X_cal[:, 0] + 0.5 * X_cal[:, 1]  # Perfect model for simplicity
    residuals_cal = torch.abs(y_cal - pred_cal)
    
    # Test data
    X_test = torch.randn(n_test, 2)
    y_test = X_test[:, 0] + 0.5 * X_test[:, 1] + 0.2 * torch.randn(n_test)
    pred_test = X_test[:, 0] + 0.5 * X_test[:, 1]
    
    # Test Split Conformal
    print("\n--- Split Conformal ---")
    split_cp = SplitConformalPredictor(alpha=0.1)
    split_cp.fit(residuals_cal)
    split_intervals = split_cp.predict(pred_test)
    
    split_coverage = ConformalCalibrator.compute_coverage(split_intervals, y_test)
    print(f"Split Conformal Coverage: {split_coverage['empirical_coverage']:.3f}")
    print(f"Mean Interval Width: {split_coverage['mean_width']:.3f}")
    
    # Test Adaptive Conformal
    print("\n--- Adaptive Conformal ---")
    difficulty = torch.abs(X_cal[:, 0])  # Use first feature as difficulty measure
    adaptive_cp = AdaptiveConformalPredictor(alpha=0.1)
    adaptive_cp.fit(residuals_cal, difficulty)
    
    test_difficulty = torch.abs(X_test[:, 0])
    adaptive_intervals = adaptive_cp.predict(pred_test, test_difficulty)
    
    adaptive_coverage = ConformalCalibrator.compute_coverage(adaptive_intervals, y_test)
    print(f"Adaptive Conformal Coverage: {adaptive_coverage['empirical_coverage']:.3f}")
    print(f"Mean Interval Width: {adaptive_coverage['mean_width']:.3f}")
    
    # Test Online Conformal
    print("\n--- Online Conformal ---")
    online_cp = OnlineConformalPredictor(alpha=0.1, window_size=100)
    online_cp.fit(residuals_cal[:100])  # Initialize with first 100 samples
    
    # Simulate online updates
    for i in range(100, len(residuals_cal)):
        online_cp.update(residuals_cal[i].item())
    
    online_intervals = online_cp.predict(pred_test)
    online_coverage = ConformalCalibrator.compute_coverage(online_intervals, y_test)
    print(f"Online Conformal Coverage: {online_coverage['empirical_coverage']:.3f}")
    print(f"Buffer Size: {online_intervals['buffer_size']}")
    
    # Test Quantile Conformal
    print("\n--- Quantile Conformal ---")
    quantiles = [0.05, 0.5, 0.95]
    quantile_cp = QuantileConformalPredictor(alpha=0.1, quantiles=quantiles)
    
    # Generate synthetic quantile predictions
    pred_quantiles_cal = torch.stack([
        pred_cal + torch.quantile(residuals_cal, q) * torch.randn_like(pred_cal) 
        for q in quantiles
    ], dim=1)
    
    quantile_cp.fit(pred_quantiles_cal, y_cal)
    
    pred_quantiles_test = torch.stack([
        pred_test + torch.quantile(residuals_cal, q) * torch.randn_like(pred_test)
        for q in quantiles  
    ], dim=1)
    
    quantile_intervals = quantile_cp.predict(pred_quantiles_test)
    quantile_coverage = ConformalCalibrator.compute_coverage(quantile_intervals, y_test)
    print(f"Quantile Conformal Coverage: {quantile_coverage['empirical_coverage']:.3f}")
    
    # Efficiency comparison
    print("\n=== Efficiency Comparison ===")
    methods = [
        ("Split", split_intervals),
        ("Adaptive", adaptive_intervals),
        ("Online", online_intervals),
        ("Quantile", quantile_intervals)
    ]
    
    for name, intervals in methods:
        efficiency = ConformalCalibrator.compute_efficiency_metrics(intervals, y_test)
        print(f"{name}: Efficiency Score = {efficiency['efficiency_score']:.3f}, "
              f"Normalized Width = {efficiency['mean_normalized_width']:.3f}")


if __name__ == "__main__":
    test_conformal_predictors()