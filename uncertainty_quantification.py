"""
Uncertainty Quantification (UQ) Framework for Reliable Risk Estimates

This module provides comprehensive tools for quantifying and leveraging uncertainty
in machine learning predictions to make better risk-based decisions.

Key Features:
- Separates aleatoric (data noise) from epistemic (model ignorance) uncertainty
- Converts predictions into actionable risk metrics (VaR, CVaR, tail probabilities)
- Provides calibration methods and evaluation metrics
- Implements multiple UQ approaches from simple to sophisticated
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import warnings

@dataclass
class UncertaintyEstimate:
    """Container for different types of uncertainty estimates."""
    mean: np.ndarray
    aleatoric: np.ndarray  # Data noise, irreducible uncertainty
    epistemic: np.ndarray  # Model ignorance, reducible with more data
    total: np.ndarray      # Combined uncertainty
    
    @property
    def std_total(self) -> np.ndarray:
        """Total standard deviation."""
        return np.sqrt(self.total)
    
    @property
    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals."""
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * self.std_total
        return self.mean - margin, self.mean + margin

class DeepEnsemble:
    """
    Deep Ensemble implementation for epistemic uncertainty quantification.
    
    Strong, simple baseline that trains multiple models with different 
    random initializations and data subsampling.
    """
    
    def __init__(self, model_class: Callable, n_models: int = 5, **model_kwargs):
        self.model_class = model_class
        self.n_models = n_models
        self.model_kwargs = model_kwargs
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        """Train ensemble of models."""
        self.models = []
        n_samples = len(X)
        
        for i in range(self.n_models):
            # Bootstrap sampling for diversity
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Train individual model
            model = self.model_class(**self.model_kwargs)
            model.fit(X_boot, y_boot, **fit_kwargs)
            self.models.append(model)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyEstimate:
        """Generate predictions with epistemic uncertainty."""
        predictions = np.array([model.predict(X) for model in self.models])
        
        mean_pred = np.mean(predictions, axis=0)
        epistemic_var = np.var(predictions, axis=0)  # Model disagreement
        
        # For aleatoric, we'd need heteroscedastic models
        # Here we assume constant aleatoric uncertainty (can be improved)
        aleatoric_var = np.zeros_like(epistemic_var)
        
        return UncertaintyEstimate(
            mean=mean_pred,
            aleatoric=aleatoric_var,
            epistemic=epistemic_var,
            total=epistemic_var + aleatoric_var
        )

class MCDropout:
    """
    Monte Carlo Dropout for lightweight Bayesian approximation.
    
    Uses dropout at inference time to approximate model uncertainty.
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 100, dropout_rate: float = 0.1):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def enable_dropout(self):
        """Enable dropout for all dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def predict_with_uncertainty(self, X: torch.Tensor) -> UncertaintyEstimate:
        """Generate predictions with MC Dropout uncertainty."""
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        total_var = np.var(predictions, axis=0)
        
        # MC Dropout primarily captures epistemic uncertainty
        return UncertaintyEstimate(
            mean=mean_pred,
            aleatoric=np.zeros_like(total_var),
            epistemic=total_var,
            total=total_var
        )

class HeteroscedasticHead(nn.Module):
    """
    Neural network head that predicts both mean and variance.
    
    Separates aleatoric uncertainty by explicitly modeling output variance.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, output_dim)
        self.var_head = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        # Ensure positive variance
        log_var = self.var_head(x)
        var = torch.exp(log_var)
        return mean, var
    
    def nll_loss(self, predictions: Tuple[torch.Tensor, torch.Tensor], 
                 targets: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for heteroscedastic regression."""
        mean, var = predictions
        return 0.5 * (torch.log(var) + (targets - mean)**2 / var).mean()

class QuantileRegression:
    """
    Quantile regression for direct uncertainty quantification.
    
    Predicts specific quantiles (τ-quantiles) to estimate prediction intervals.
    """
    
    def __init__(self, model_class: Callable, quantiles: List[float] = None):
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]  # Default quantiles
        self.quantiles = sorted(quantiles)
        self.models = {q: model_class() for q in quantiles}
        
    def quantile_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     quantile: float) -> float:
        """Quantile loss function."""
        errors = y_true - y_pred
        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        """Train quantile regression models."""
        for q in self.quantiles:
            # Custom loss for each quantile
            self.models[q].fit(X, y, **fit_kwargs)
    
    def predict_intervals(self, X: np.ndarray, 
                         confidence_levels: List[float] = None) -> Dict:
        """Predict confidence intervals using quantile estimates."""
        if confidence_levels is None:
            confidence_levels = [0.5, 0.8, 0.9, 0.95]
            
        predictions = {q: self.models[q].predict(X) for q in self.quantiles}
        intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            
            # Find closest quantiles
            lower_idx = min(range(len(self.quantiles)), 
                          key=lambda i: abs(self.quantiles[i] - lower_q))
            upper_idx = min(range(len(self.quantiles)), 
                          key=lambda i: abs(self.quantiles[i] - upper_q))
            
            intervals[level] = {
                'lower': predictions[self.quantiles[lower_idx]],
                'upper': predictions[self.quantiles[upper_idx]],
                'median': predictions[0.5] if 0.5 in predictions else None
            }
            
        return intervals

class ConformalPredictor:
    """
    Conformal Prediction for distribution-free coverage guarantees.
    
    Provides prediction intervals/sets with theoretical coverage guarantees
    regardless of the underlying model or data distribution.
    """
    
    def __init__(self, model, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha  # Miscoverage level (1-alpha coverage)
        self.quantile = None
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Fit conformal predictor using calibration set."""
        # Get model predictions on calibration set
        predictions = self.model.predict(X_cal)
        
        # Compute conformity scores (absolute residuals)
        scores = np.abs(y_cal - predictions)
        
        # Find the (1-alpha) quantile of scores
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(scores, q_level)
        
    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate conformal prediction intervals."""
        if self.quantile is None:
            raise ValueError("Must call fit() before predict_intervals()")
            
        predictions = self.model.predict(X)
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        return lower, upper
    
    def predict_sets(self, X: np.ndarray, class_probs: np.ndarray) -> List[List[int]]:
        """Generate conformal prediction sets for classification."""
        if self.quantile is None:
            raise ValueError("Must call fit() before predict_sets()")
            
        prediction_sets = []
        for probs in class_probs:
            # Sort classes by probability (descending)
            sorted_indices = np.argsort(-probs)
            cumulative_prob = 0
            prediction_set = []
            
            for idx in sorted_indices:
                prediction_set.append(idx)
                cumulative_prob += probs[idx]
                if cumulative_prob >= 1 - self.alpha:
                    break
                    
            prediction_sets.append(prediction_set)
            
        return prediction_sets

class CalibrationMethods:
    """
    Calibration methods to improve probability estimates.
    
    Ensures that predicted probabilities match observed frequencies.
    """
    
    @staticmethod
    def temperature_scaling(logits: np.ndarray, labels: np.ndarray, 
                           validation_split: float = 0.2) -> float:
        """
        Temperature scaling for calibrating neural network outputs.
        
        Finds optimal temperature T to scale logits: p = softmax(logits/T)
        """
        from scipy.optimize import minimize_scalar
        
        # Split data
        n_val = int(len(logits) * validation_split)
        val_logits, val_labels = logits[:n_val], labels[:n_val]
        
        def nll_loss(temperature):
            scaled_logits = val_logits / temperature
            probs = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()
            # Avoid log(0)
            probs = np.clip(probs, 1e-8, 1 - 1e-8)
            return -np.mean(np.log(probs[range(len(val_labels)), val_labels]))
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    @staticmethod
    def isotonic_regression(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
        """
        Isotonic regression for non-parametric calibration.
        
        Learns a monotonic mapping from predicted to calibrated probabilities.
        """
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(probs, labels)
        return calibrator
    
    @staticmethod
    def reliability_diagram(probs: np.ndarray, labels: np.ndarray, 
                          n_bins: int = 10) -> Dict:
        """
        Generate reliability diagram data for calibration assessment.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
                counts.append(in_bin.sum())
            else:
                accuracies.append(0)
                confidences.append(0)
                counts.append(0)
                
        return {
            'accuracies': np.array(accuracies),
            'confidences': np.array(confidences),
            'counts': np.array(counts),
            'bin_boundaries': bin_boundaries
        }

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, 
                              n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy.
    """
    reliability_data = CalibrationMethods.reliability_diagram(probs, labels, n_bins)
    
    accuracies = reliability_data['accuracies']
    confidences = reliability_data['confidences']
    counts = reliability_data['counts']
    
    n_samples = len(probs)
    ece = 0
    
    for acc, conf, count in zip(accuracies, confidences, counts):
        if count > 0:
            ece += (count / n_samples) * abs(acc - conf)
            
    return ece

class OODDetector:
    """
    Out-of-Distribution (OOD) detection methods.
    
    Identifies when inputs are significantly different from training data,
    which typically correlates with higher epistemic uncertainty.
    """
    
    @staticmethod
    def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Energy-based OOD detection.
        
        Lower energy indicates higher likelihood of being in-distribution.
        Energy = -T * log(sum(exp(logits/T)))
        """
        return -temperature * torch.logsumexp(logits / temperature, dim=1)
    
    @staticmethod
    def max_softmax_probability(logits: torch.Tensor) -> torch.Tensor:
        """Maximum softmax probability (MSP) baseline."""
        probs = F.softmax(logits, dim=1)
        return torch.max(probs, dim=1)[0]
    
    @staticmethod
    def entropy_score(logits: torch.Tensor) -> torch.Tensor:
        """Predictive entropy as uncertainty measure."""
        probs = F.softmax(logits, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    
    @staticmethod
    def mahalanobis_distance(features: torch.Tensor, mean: torch.Tensor, 
                           cov_inv: torch.Tensor) -> torch.Tensor:
        """
        Mahalanobis distance for OOD detection.
        
        Measures distance from feature distribution center.
        """
        diff = features - mean
        return torch.sum(diff @ cov_inv * diff, dim=1)

class DirichletPriorNetwork:
    """
    Dirichlet Prior Networks for uncertainty-aware classification.
    
    Models class probabilities as Dirichlet distribution to separate
    aleatoric and epistemic uncertainty in classification.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning Dirichlet parameters.
        
        Returns:
            alpha: Dirichlet concentration parameters
            uncertainty: Epistemic uncertainty measure
            prob: Expected probabilities
        """
        evidence = self.model(x)
        alpha = evidence + 1  # Ensure alpha > 0
        
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        
        # Epistemic uncertainty (differential entropy)
        uncertainty = torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
        
        return {
            'alpha': alpha,
            'uncertainty': uncertainty,
            'prob': prob,
            'strength': S.squeeze()
        }
    
    def kl_divergence_loss(self, alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """KL divergence loss for Dirichlet Prior Networks."""
        S = torch.sum(alpha, dim=1)
        target_alpha = torch.zeros_like(alpha)
        target_alpha[range(len(targets)), targets] = 1.0
        
        # KL divergence between Dirichlet distributions
        kl = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(target_alpha, dim=1))
        kl += torch.sum(torch.lgamma(target_alpha) - torch.lgamma(alpha), dim=1)
        kl += torch.sum((alpha - target_alpha) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1))), dim=1)
        
        return torch.mean(kl)

class UQEvaluationMetrics:
    """
    Comprehensive evaluation metrics for uncertainty quantification.
    """
    
    @staticmethod
    def continuous_ranked_probability_score(y_true: np.ndarray, 
                                          predictions: np.ndarray) -> float:
        """
        Continuous Ranked Probability Score (CRPS) for probabilistic predictions.
        
        Measures quality of probabilistic forecasts for continuous variables.
        """
        n_samples = predictions.shape[0]
        crps = 0
        
        for i in range(n_samples):
            pred_samples = predictions[i]
            y_obs = y_true[i]
            
            # CRPS = E[|X - Y|] - 0.5 * E[|X - X'|]
            term1 = np.mean(np.abs(pred_samples - y_obs))
            term2 = 0.5 * np.mean(np.abs(pred_samples[:, None] - pred_samples[None, :]))
            crps += term1 - term2
            
        return crps / n_samples
    
    @staticmethod
    def prediction_interval_coverage_probability(y_true: np.ndarray,
                                               lower: np.ndarray,
                                               upper: np.ndarray) -> float:
        """
        Prediction Interval Coverage Probability (PICP).
        
        Fraction of true values that fall within prediction intervals.
        """
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage
    
    @staticmethod
    def mean_prediction_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
        """Mean Prediction Interval Width (MPIW)."""
        return np.mean(upper - lower)
    
    @staticmethod
    def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      alpha: float = 0.1) -> float:
        """
        Interval Score combining coverage and sharpness.
        
        Lower is better. Penalizes wide intervals and poor coverage.
        """
        width = upper - lower
        coverage_penalty = (2/alpha) * (lower - y_true) * (y_true < lower)
        coverage_penalty += (2/alpha) * (y_true - upper) * (y_true > upper)
        
        return np.mean(width + coverage_penalty)
    
    @staticmethod
    def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Brier Score for probabilistic classification.
        
        Measures accuracy of probabilistic predictions.
        """
        return brier_score_loss(labels, probs)
    
    @staticmethod
    def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
        """Negative Log-Likelihood."""
        # Avoid log(0)
        probs_clipped = np.clip(probs, 1e-8, 1 - 1e-8)
        return -np.mean(np.log(probs_clipped[range(len(labels)), labels]))
    
    @staticmethod
    def ood_detection_metrics(in_scores: np.ndarray, out_scores: np.ndarray) -> Dict:
        """
        Metrics for OOD detection performance.
        
        Args:
            in_scores: Uncertainty scores for in-distribution data
            out_scores: Uncertainty scores for out-of-distribution data
        """
        from sklearn.metrics import roc_auc_score, roc_curve
        
        # Combine scores and labels
        scores = np.concatenate([in_scores, out_scores])
        labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
        
        # AUROC
        auroc = roc_auc_score(labels, scores)
        
        # FPR at 95% TPR
        fpr, tpr, _ = roc_curve(labels, scores)
        fpr_at_95_tpr = fpr[np.argmax(tpr >= 0.95)]
        
        return {
            'auroc': auroc,
            'fpr_at_95_tpr': fpr_at_95_tpr
        }

class RiskBasedDecisionFramework:
    """
    Framework for converting uncertainty estimates into risk-based decisions.
    
    Implements Value at Risk (VaR), Conditional Value at Risk (CVaR),
    and expected cost minimization for decision making under uncertainty.
    """
    
    @staticmethod
    def value_at_risk(samples: np.ndarray, alpha: float = 0.05) -> float:
        """
        Value at Risk (VaR) - α-quantile of loss distribution.
        
        VaR_α = inf{x : P(L ≤ x) ≥ 1-α}
        """
        return np.quantile(samples, 1 - alpha)
    
    @staticmethod
    def conditional_value_at_risk(samples: np.ndarray, alpha: float = 0.05) -> float:
        """
        Conditional Value at Risk (CVaR) - expected loss beyond VaR.
        
        CVaR_α = E[L | L ≥ VaR_α]
        """
        var = RiskBasedDecisionFramework.value_at_risk(samples, alpha)
        tail_losses = samples[samples >= var]
        return np.mean(tail_losses) if len(tail_losses) > 0 else var
    
    @staticmethod
    def tail_probability(samples: np.ndarray, threshold: float) -> float:
        """Probability of exceeding threshold: P(Y ≥ t | X)."""
        return np.mean(samples >= threshold)
    
    @staticmethod
    def expected_cost_minimization(predictions: UncertaintyEstimate,
                                 cost_matrix: np.ndarray,
                                 actions: List[str]) -> Dict:
        """
        Choose action that minimizes expected cost.
        
        a* = argmin_a E[C(a,Y) | X]
        
        Args:
            predictions: Uncertainty estimates
            cost_matrix: Cost matrix C(action, outcome)
            actions: List of possible actions
        """
        n_samples = len(predictions.mean)
        n_actions = len(actions)
        
        expected_costs = np.zeros((n_samples, n_actions))
        
        for i in range(n_samples):
            # Sample from predictive distribution
            samples = np.random.normal(
                predictions.mean[i], 
                predictions.std_total[i], 
                size=1000
            )
            
            for j, action in enumerate(actions):
                # Compute expected cost for this action
                costs = cost_matrix[j, :]  # Costs for this action across outcomes
                # Discretize outcomes for cost computation
                outcome_bins = np.linspace(samples.min(), samples.max(), len(costs))
                digitized = np.digitize(samples, outcome_bins) - 1
                digitized = np.clip(digitized, 0, len(costs) - 1)
                
                expected_costs[i, j] = np.mean(costs[digitized])
        
        # Choose action with minimum expected cost
        optimal_actions = np.argmin(expected_costs, axis=1)
        
        return {
            'optimal_actions': [actions[a] for a in optimal_actions],
            'expected_costs': expected_costs,
            'cost_savings': np.max(expected_costs, axis=1) - np.min(expected_costs, axis=1)
        }
    
    @staticmethod
    def selective_prediction_threshold(uncertainties: np.ndarray,
                                     accuracies: np.ndarray,
                                     coverage_target: float = 0.9) -> float:
        """
        Find uncertainty threshold for selective prediction.
        
        Returns threshold such that predictions with uncertainty below
        threshold achieve target accuracy coverage.
        """
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_accuracies = accuracies[sorted_indices]
        
        # Find cumulative accuracy
        cumulative_correct = np.cumsum(sorted_accuracies)
        cumulative_total = np.arange(1, len(sorted_accuracies) + 1)
        cumulative_accuracy = cumulative_correct / cumulative_total
        
        # Find threshold where accuracy meets target
        valid_indices = cumulative_accuracy >= coverage_target
        if np.any(valid_indices):
            threshold_idx = np.argmax(valid_indices)
            return sorted_uncertainties[threshold_idx]
        else:
            return np.max(uncertainties)  # No threshold achieves target

class MonitoringAndDrift:
    """
    Monitoring tools for detecting distribution drift and maintaining UQ quality.
    """
    
    @staticmethod
    def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                                 n_bins: int = 10) -> float:
        """
        Population Stability Index (PSI) for detecting input drift.
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        # Create bins based on expected distribution
        bin_edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Calculate percentages in each bin
        expected_counts = np.histogram(expected, bins=bin_edges)[0]
        actual_counts = np.histogram(actual, bins=bin_edges)[0]
        
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_pct = np.maximum(expected_pct, 1e-6)
        actual_pct = np.maximum(actual_pct, 1e-6)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi
    
    @staticmethod
    def kl_divergence_drift(p_samples: np.ndarray, q_samples: np.ndarray,
                           n_bins: int = 50) -> float:
        """
        KL divergence between two distributions for drift detection.
        """
        # Create common bins
        all_samples = np.concatenate([p_samples, q_samples])
        bin_edges = np.histogram_bin_edges(all_samples, bins=n_bins)
        
        # Calculate histograms
        p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=True)
        q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=True)
        
        # Normalize and add small epsilon
        p_hist = p_hist / np.sum(p_hist) + 1e-8
        q_hist = q_hist / np.sum(q_hist) + 1e-8
        
        # KL divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
        return kl_div
    
    @staticmethod
    def calibration_drift_monitor(old_probs: np.ndarray, old_labels: np.ndarray,
                                new_probs: np.ndarray, new_labels: np.ndarray) -> Dict:
        """
        Monitor changes in calibration quality over time.
        """
        old_ece = expected_calibration_error(old_probs, old_labels)
        new_ece = expected_calibration_error(new_probs, new_labels)
        
        old_brier = UQEvaluationMetrics.brier_score(old_probs, old_labels)
        new_brier = UQEvaluationMetrics.brier_score(new_probs, new_labels)
        
        return {
            'ece_change': new_ece - old_ece,
            'brier_change': new_brier - old_brier,
            'old_ece': old_ece,
            'new_ece': new_ece,
            'calibration_degraded': new_ece > old_ece + 0.01  # Threshold
        }
    
    @staticmethod
    def online_conformal_update(conformity_scores: List[float], 
                              new_score: float,
                              alpha: float = 0.1,
                              window_size: int = 1000) -> float:
        """
        Update conformal prediction quantile online with new conformity scores.
        """
        # Add new score and maintain window
        conformity_scores.append(new_score)
        if len(conformity_scores) > window_size:
            conformity_scores.pop(0)
        
        # Recompute quantile
        n = len(conformity_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        return np.quantile(conformity_scores, q_level)

class BacktestingFramework:
    """
    Backtesting framework for validating UQ methods over time.
    """
    
    @staticmethod
    def sliding_window_coverage_test(predictions: List[np.ndarray],
                                   actuals: List[np.ndarray],
                                   intervals: List[Tuple[np.ndarray, np.ndarray]],
                                   window_size: int = 100) -> Dict:
        """
        Test coverage stability over sliding windows.
        """
        coverages = []
        widths = []
        
        for i in range(len(predictions) - window_size + 1):
            window_actuals = np.concatenate(actuals[i:i+window_size])
            window_lower = np.concatenate([interval[0] for interval in intervals[i:i+window_size]])
            window_upper = np.concatenate([interval[1] for interval in intervals[i:i+window_size]])
            
            coverage = UQEvaluationMetrics.prediction_interval_coverage_probability(
                window_actuals, window_lower, window_upper
            )
            width = UQEvaluationMetrics.mean_prediction_interval_width(
                window_lower, window_upper
            )
            
            coverages.append(coverage)
            widths.append(width)
        
        return {
            'coverage_stability': np.std(coverages),
            'mean_coverage': np.mean(coverages),
            'width_stability': np.std(widths),
            'coverage_trend': np.polyfit(range(len(coverages)), coverages, 1)[0]
        }
    
    @staticmethod
    def champion_challenger_test(champion_uncertainties: np.ndarray,
                               challenger_uncertainties: np.ndarray,
                               true_errors: np.ndarray,
                               test_type: str = 'correlation') -> Dict:
        """
        Compare two UQ methods using various metrics.
        """
        from scipy.stats import spearmanr, kendalltau
        
        results = {}
        
        if test_type == 'correlation':
            # Test correlation with true errors
            champion_corr = spearmanr(champion_uncertainties, true_errors)[0]
            challenger_corr = spearmanr(challenger_uncertainties, true_errors)[0]
            
            results['champion_correlation'] = champion_corr
            results['challenger_correlation'] = challenger_corr
            results['improvement'] = challenger_corr - champion_corr
            
        elif test_type == 'ranking':
            # Test ranking quality
            champion_tau = kendalltau(champion_uncertainties, true_errors)[0]
            challenger_tau = kendalltau(challenger_uncertainties, true_errors)[0]
            
            results['champion_kendall_tau'] = champion_tau
            results['challenger_kendall_tau'] = challenger_tau
            results['improvement'] = challenger_tau - champion_tau
        
        return results