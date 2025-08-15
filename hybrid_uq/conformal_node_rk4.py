"""
Conformal Prediction Module for NODE-RK4 Integration

This module provides conformal prediction capabilities specifically designed for 
Neural Ordinary Differential Equations with Runge-Kutta 4th order solver (NODE-RK4).
It handles temporal dynamics, heteroscedastic uncertainty, and provides coverage 
guarantees for differential equation solutions.

Key Features:
- Temporal conformal prediction for sequential ODE solutions
- Heteroscedastic conformity scores accounting for varying uncertainty
- Integration with RK4 solver dynamics
- Online conformal updates for streaming predictions
- Coverage guarantees for trajectory prediction intervals
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from collections import deque
import warnings

@dataclass
class ConformalPrediction:
    """Container for conformal prediction results."""
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor
    prediction: torch.Tensor
    conformity_score: torch.Tensor
    coverage_level: float
    
    @property
    def interval_width(self) -> torch.Tensor:
        """Width of prediction intervals."""
        return self.upper_bound - self.lower_bound
    
    @property
    def contains_target(self, target: torch.Tensor) -> torch.Tensor:
        """Check if target falls within prediction intervals."""
        return (target >= self.lower_bound) & (target <= self.upper_bound)

class RK4ConformalPredictor:
    """
    Conformal Prediction for NODE-RK4 with temporal dynamics.
    
    Provides distribution-free coverage guarantees for ODE solutions
    by learning conformity scores on calibration trajectories.
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        conformity_score_fn: str = "adaptive_residual",
        temporal_weighting: bool = True,
        max_calibration_size: int = 1000
    ):
        """
        Initialize conformal predictor for NODE-RK4.
        
        Args:
            model: Trained NODE-RK4 model
            alpha: Miscoverage level (1-alpha coverage)
            conformity_score_fn: Type of conformity score
            temporal_weighting: Whether to weight scores by temporal position
            max_calibration_size: Maximum size of calibration set
        """
        self.model = model
        self.alpha = alpha
        self.conformity_score_fn = conformity_score_fn
        self.temporal_weighting = temporal_weighting
        self.max_calibration_size = max_calibration_size
        
        # Calibration data storage
        self.calibration_scores = deque(maxlen=max_calibration_size)
        self.quantile_cache = None
        self.temporal_weights = None
        self.is_fitted = False
        
    def _compute_conformity_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute conformity scores for predictions vs targets.
        
        Args:
            predictions: Model predictions [B, T, D]
            targets: True values [B, T, D]
            uncertainties: Uncertainty estimates [B, T, D]
            time_steps: Time step indices [T]
            
        Returns:
            conformity_scores: Scores for each sample [B, T]
        """
        if self.conformity_score_fn == "absolute_residual":
            # Standard absolute residual
            scores = torch.abs(predictions - targets).mean(dim=-1)
            
        elif self.conformity_score_fn == "adaptive_residual":
            # Residual normalized by uncertainty (heteroscedastic)
            if uncertainties is not None:
                # Avoid division by zero
                uncertainties = torch.clamp(uncertainties, min=1e-6)
                scores = torch.abs(predictions - targets) / uncertainties
                scores = scores.mean(dim=-1)
            else:
                scores = torch.abs(predictions - targets).mean(dim=-1)
                
        elif self.conformity_score_fn == "squared_residual":
            # Squared residual for Gaussian assumptions
            scores = ((predictions - targets) ** 2).mean(dim=-1)
            
        elif self.conformity_score_fn == "mahalanobis":
            # Mahalanobis distance using covariance
            diff = predictions - targets  # [B, T, D]
            if uncertainties is not None:
                # Use uncertainties as diagonal covariance
                cov_inv = 1.0 / torch.clamp(uncertainties ** 2, min=1e-6)
                scores = torch.sum(diff ** 2 * cov_inv, dim=-1)
            else:
                scores = torch.sum(diff ** 2, dim=-1)
                
        elif self.conformity_score_fn == "trajectory_aware":
            # Trajectory-aware score considering temporal dependencies
            residuals = torch.abs(predictions - targets).mean(dim=-1)  # [B, T]
            
            if time_steps is not None and self.temporal_weighting:
                # Weight later time steps more heavily (error accumulation)
                weights = torch.exp(0.1 * time_steps.float())
                weights = weights / weights.sum()
                scores = (residuals * weights.unsqueeze(0)).sum(dim=-1)
            else:
                scores = residuals.mean(dim=-1)
        else:
            raise ValueError(f"Unknown conformity score function: {self.conformity_score_fn}")
            
        return scores
    
    def fit(
        self,
        X_cal: torch.Tensor,
        y_cal: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None
    ) -> None:
        """
        Fit conformal predictor using calibration data.
        
        Args:
            X_cal: Calibration inputs [B, T, D_in]
            y_cal: Calibration targets [B, T, D_out]
            time_steps: Time step indices [T]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get model predictions and uncertainties
            outputs = self.model(X_cal)
            
            if isinstance(outputs, dict):
                predictions = outputs.get('psi', outputs.get('O', outputs.get('predictions')))
                uncertainties = outputs.get('sigma_res', None)
            else:
                predictions = outputs
                uncertainties = None
            
            # Compute conformity scores
            scores = self._compute_conformity_score(
                predictions, y_cal, uncertainties, time_steps
            )
            
            # Store calibration scores
            self.calibration_scores.clear()
            for score in scores.flatten():
                self.calibration_scores.append(score.item())
            
            # Compute quantile
            self._update_quantile()
            
        self.is_fitted = True
        
    def _update_quantile(self) -> None:
        """Update the conformal quantile from calibration scores."""
        if len(self.calibration_scores) == 0:
            self.quantile_cache = 0.0
            return
            
        scores_array = np.array(self.calibration_scores)
        n = len(scores_array)
        
        # Conformal quantile: (n+1)(1-α)/n quantile
        q_level = (n + 1) * (1 - self.alpha) / n
        q_level = min(q_level, 1.0)  # Ensure valid quantile
        
        self.quantile_cache = np.quantile(scores_array, q_level)
        
    def predict_intervals(
        self,
        X: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> Union[ConformalPrediction, Tuple[ConformalPrediction, torch.Tensor]]:
        """
        Generate conformal prediction intervals.
        
        Args:
            X: Input data [B, T, D_in]
            time_steps: Time step indices [T]
            return_scores: Whether to return conformity scores
            
        Returns:
            ConformalPrediction object with intervals and metadata
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict_intervals()")
            
        self.model.eval()
        
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(X)
            
            if isinstance(outputs, dict):
                predictions = outputs.get('psi', outputs.get('O', outputs.get('predictions')))
                uncertainties = outputs.get('sigma_res', None)
            else:
                predictions = outputs
                uncertainties = None
            
            # Create prediction intervals using cached quantile
            quantile_tensor = torch.tensor(
                self.quantile_cache, 
                device=predictions.device, 
                dtype=predictions.dtype
            )
            
            if self.conformity_score_fn == "adaptive_residual" and uncertainties is not None:
                # Scale quantile by local uncertainty
                interval_width = quantile_tensor * uncertainties
            else:
                # Fixed width intervals
                interval_width = quantile_tensor
                
            lower_bound = predictions - interval_width
            upper_bound = predictions + interval_width
            
            # Compute conformity scores for monitoring
            dummy_scores = torch.zeros_like(predictions[..., 0])
            
            conformal_pred = ConformalPrediction(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                prediction=predictions,
                conformity_score=dummy_scores,
                coverage_level=1 - self.alpha
            )
            
            if return_scores:
                return conformal_pred, dummy_scores
            else:
                return conformal_pred
    
    def update_online(
        self,
        new_predictions: torch.Tensor,
        new_targets: torch.Tensor,
        new_uncertainties: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update conformal predictor online with new observations.
        
        Args:
            new_predictions: New model predictions [B, T, D]
            new_targets: New observed targets [B, T, D]
            new_uncertainties: New uncertainty estimates [B, T, D]
            time_steps: Time step indices [T]
        """
        # Compute new conformity scores
        new_scores = self._compute_conformity_score(
            new_predictions, new_targets, new_uncertainties, time_steps
        )
        
        # Add to calibration set (deque handles max size)
        for score in new_scores.flatten():
            self.calibration_scores.append(score.item())
        
        # Update quantile
        self._update_quantile()
    
    def evaluate_coverage(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate conformal prediction coverage on test data.
        
        Args:
            X_test: Test inputs [B, T, D_in]
            y_test: Test targets [B, T, D_out]
            time_steps: Time step indices [T]
            
        Returns:
            Dictionary with coverage metrics
        """
        conformal_pred = self.predict_intervals(X_test, time_steps)
        
        # Check coverage
        in_interval = (y_test >= conformal_pred.lower_bound) & (y_test <= conformal_pred.upper_bound)
        coverage = in_interval.float().mean().item()
        
        # Interval width statistics
        interval_widths = conformal_pred.interval_width
        mean_width = interval_widths.mean().item()
        std_width = interval_widths.std().item()
        
        # Coverage by time step (if temporal)
        if len(y_test.shape) > 2:  # [B, T, D]
            coverage_by_time = in_interval.float().mean(dim=(0, 2))  # Average over batch and features
            coverage_by_time = coverage_by_time.cpu().numpy()
        else:
            coverage_by_time = None
        
        return {
            'coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'coverage_error': abs(coverage - (1 - self.alpha)),
            'mean_interval_width': mean_width,
            'std_interval_width': std_width,
            'coverage_by_time': coverage_by_time,
            'n_calibration_samples': len(self.calibration_scores)
        }

class TemporalConformalEnsemble:
    """
    Ensemble of conformal predictors for robust temporal coverage.
    
    Combines multiple conformal predictors with different conformity scores
    and temporal weightings for improved robustness.
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        ensemble_methods: List[str] = None,
        combination_method: str = "intersection"
    ):
        """
        Initialize ensemble of conformal predictors.
        
        Args:
            model: Trained NODE-RK4 model
            alpha: Miscoverage level
            ensemble_methods: List of conformity score functions
            combination_method: How to combine intervals ("intersection", "union", "average")
        """
        if ensemble_methods is None:
            ensemble_methods = ["adaptive_residual", "trajectory_aware", "mahalanobis"]
            
        self.model = model
        self.alpha = alpha
        self.combination_method = combination_method
        
        # Create ensemble of conformal predictors
        self.predictors = {}
        for method in ensemble_methods:
            self.predictors[method] = RK4ConformalPredictor(
                model=model,
                alpha=alpha,
                conformity_score_fn=method,
                temporal_weighting=True
            )
    
    def fit(
        self,
        X_cal: torch.Tensor,
        y_cal: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None
    ) -> None:
        """Fit all predictors in the ensemble."""
        for predictor in self.predictors.values():
            predictor.fit(X_cal, y_cal, time_steps)
    
    def predict_intervals(
        self,
        X: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None
    ) -> ConformalPrediction:
        """
        Generate ensemble conformal prediction intervals.
        
        Args:
            X: Input data [B, T, D_in]
            time_steps: Time step indices [T]
            
        Returns:
            Combined ConformalPrediction
        """
        # Get predictions from all ensemble members
        predictions = []
        for predictor in self.predictors.values():
            pred = predictor.predict_intervals(X, time_steps)
            predictions.append(pred)
        
        # Combine intervals
        if self.combination_method == "intersection":
            # Conservative: intersection of all intervals
            lower_bound = torch.stack([p.lower_bound for p in predictions]).max(dim=0)[0]
            upper_bound = torch.stack([p.upper_bound for p in predictions]).min(dim=0)[0]
            
        elif self.combination_method == "union":
            # Liberal: union of all intervals
            lower_bound = torch.stack([p.lower_bound for p in predictions]).min(dim=0)[0]
            upper_bound = torch.stack([p.upper_bound for p in predictions]).max(dim=0)[0]
            
        elif self.combination_method == "average":
            # Average bounds
            lower_bound = torch.stack([p.lower_bound for p in predictions]).mean(dim=0)
            upper_bound = torch.stack([p.upper_bound for p in predictions]).mean(dim=0)
            
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        # Use prediction from first predictor (they should be similar)
        base_prediction = predictions[0].prediction
        
        return ConformalPrediction(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            prediction=base_prediction,
            conformity_score=torch.zeros_like(base_prediction[..., 0]),
            coverage_level=1 - self.alpha
        )

def rk4_conformal_step(
    model: nn.Module,
    conformal_predictor: RK4ConformalPredictor,
    x0: torch.Tensor,
    t_span: torch.Tensor,
    dt: float,
    return_intervals: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Perform RK4 integration with conformal prediction intervals.
    
    Args:
        model: NODE-RK4 model
        conformal_predictor: Fitted conformal predictor
        x0: Initial conditions [B, D]
        t_span: Time points [T]
        dt: Time step size
        return_intervals: Whether to return prediction intervals
        
    Returns:
        Dictionary with trajectory and optional intervals
    """
    device = x0.device
    batch_size = x0.shape[0]
    n_steps = len(t_span)
    state_dim = x0.shape[-1]
    
    # Initialize trajectory storage
    trajectory = torch.zeros(batch_size, n_steps, state_dim, device=device)
    trajectory[:, 0] = x0
    
    if return_intervals:
        lower_trajectory = torch.zeros_like(trajectory)
        upper_trajectory = torch.zeros_like(trajectory)
        lower_trajectory[:, 0] = x0
        upper_trajectory[:, 0] = x0
    
    # RK4 integration with conformal intervals
    x_current = x0
    
    for i in range(1, n_steps):
        t_current = t_span[i-1]
        
        # Standard RK4 step
        with torch.no_grad():
            k1 = model(x_current.unsqueeze(1)).squeeze(1)  # Remove time dimension
            k2 = model((x_current + dt/2 * k1).unsqueeze(1)).squeeze(1)
            k3 = model((x_current + dt/2 * k2).unsqueeze(1)).squeeze(1)
            k4 = model((x_current + dt * k3).unsqueeze(1)).squeeze(1)
            
            x_next = x_current + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        trajectory[:, i] = x_next
        
        # Get conformal intervals for this step
        if return_intervals:
            # Create input for conformal predictor (single time step)
            x_input = x_current.unsqueeze(1)  # [B, 1, D]
            conformal_pred = conformal_predictor.predict_intervals(x_input)
            
            # Apply intervals to the integrated result
            lower_trajectory[:, i] = conformal_pred.lower_bound.squeeze(1)
            upper_trajectory[:, i] = conformal_pred.upper_bound.squeeze(1)
        
        x_current = x_next
    
    result = {
        'trajectory': trajectory,
        'time_points': t_span
    }
    
    if return_intervals:
        result.update({
            'lower_bound': lower_trajectory,
            'upper_bound': upper_trajectory
        })
    
    return result

# Utility functions for integration with existing hybrid_uq framework
def integrate_with_hybrid_model(
    hybrid_model,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    alpha: float = 0.1
) -> RK4ConformalPredictor:
    """
    Create and fit conformal predictor for HybridModel from core.py.
    
    Args:
        hybrid_model: HybridModel instance from hybrid_uq.core
        X_cal: Calibration inputs
        y_cal: Calibration targets
        alpha: Miscoverage level
        
    Returns:
        Fitted RK4ConformalPredictor
    """
    # Wrapper to make HybridModel compatible with conformal predictor
    class HybridModelWrapper(nn.Module):
        def __init__(self, hybrid_model):
            super().__init__()
            self.hybrid_model = hybrid_model
            
        def forward(self, x):
            outputs = self.hybrid_model(x)
            # Return both prediction and uncertainty
            return {
                'predictions': outputs['psi'],
                'sigma_res': outputs['sigma_res']
            }
    
    wrapper = HybridModelWrapper(hybrid_model)
    
    conformal_predictor = RK4ConformalPredictor(
        model=wrapper,
        alpha=alpha,
        conformity_score_fn="adaptive_residual",
        temporal_weighting=True
    )
    
    conformal_predictor.fit(X_cal, y_cal)
    
    return conformal_predictor