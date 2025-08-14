"""
Calibration Metrics for Uncertainty Quantification

Comprehensive metrics to evaluate the quality of probabilistic predictions
and uncertainty estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List, Union
from scipy import stats
import warnings


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy
    across different confidence bins.
    
    Args:
        y_true: True labels (binary or class indices)
        y_prob: Predicted probabilities (for binary) or confidence scores
        n_bins: Number of bins for calibration
        strategy: Binning strategy ('uniform' or 'quantile')
    
    Returns:
        ECE value (lower is better, 0 is perfect)
    """
    if len(y_prob.shape) == 2:
        # Multiclass: use max probability as confidence
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        # Binary
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
    
    ece = 0.0
    for i in range(n_bins):
        # Find samples in bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if np.sum(in_bin) > 0:
            # Calculate accuracy and confidence in bin
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_weight = np.sum(in_bin) / len(confidences)
            
            # Add to ECE
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def adaptive_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Adaptive Calibration Error (ACE).
    
    Similar to ECE but uses adaptive binning to ensure equal number
    of samples per bin.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        ACE value
    """
    return expected_calibration_error(y_true, y_prob, n_bins, strategy='quantile')


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    Returns the maximum calibration error across all bins.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        MCE value
    """
    if len(y_prob.shape) == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    max_error = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            error = np.abs(bin_accuracy - bin_confidence)
            max_error = max(max_error, error)
    
    return max_error


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Calculate Brier Score.
    
    Measures the mean squared difference between predicted probabilities
    and actual outcomes.
    
    Args:
        y_true: True labels (binary)
        y_prob: Predicted probabilities
    
    Returns:
        Brier score (lower is better)
    """
    return np.mean((y_prob - y_true) ** 2)


def negative_log_likelihood(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Calculate Negative Log-Likelihood.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
    
    Returns:
        NLL value (lower is better)
    """
    eps = 1e-8
    
    if len(y_prob.shape) == 2:
        # Multiclass
        n_samples = len(y_true)
        nll = -np.sum(np.log(y_prob[np.arange(n_samples), y_true] + eps))
    else:
        # Binary
        nll = -np.sum(y_true * np.log(y_prob + eps) + 
                     (1 - y_true) * np.log(1 - y_prob + eps))
    
    return nll / len(y_true)


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    ax: Optional[plt.Axes] = None,
    return_stats: bool = False
) -> Union[plt.Figure, Dict]:
    """
    Create reliability diagram (calibration plot).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        ax: Matplotlib axes (if None, creates new figure)
        return_stats: If True, return statistics instead of plotting
    
    Returns:
        Figure or statistics dictionary
    """
    if len(y_prob.shape) == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        accuracies = y_true
    
    # Compute bin statistics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if np.sum(in_bin) > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    if return_stats:
        return {
            'bin_centers': np.array(bin_centers),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts),
            'ece': expected_calibration_error(y_true, y_prob, n_bins)
        }
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Plot reliability diagram
    ax.scatter(bin_confidences, bin_accuracies, s=np.array(bin_counts)*2, 
              alpha=0.7, label='Observed')
    
    # Add bars showing calibration error
    for conf, acc in zip(bin_confidences, bin_accuracies):
        if conf > 0:
            ax.plot([conf, conf], [conf, acc], 'r-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{title} (ECE={expected_calibration_error(y_true, y_prob, n_bins):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig


def classwise_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[int, float]:
    """
    Calculate per-class Expected Calibration Error.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins
    
    Returns:
        Dictionary mapping class index to ECE
    """
    n_classes = y_prob.shape[1]
    class_ece = {}
    
    for class_idx in range(n_classes):
        # Binary problem: class vs rest
        binary_true = (y_true == class_idx).astype(int)
        binary_prob = y_prob[:, class_idx]
        
        class_ece[class_idx] = expected_calibration_error(
            binary_true, binary_prob, n_bins
        )
    
    return class_ece


def confidence_histogram(
    y_prob: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    n_bins: int = 20,
    title: str = "Confidence Histogram",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot histogram of prediction confidences.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels (for accuracy overlay)
        n_bins: Number of bins
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    if len(y_prob.shape) == 2:
        confidences = np.max(y_prob, axis=1)
    else:
        confidences = y_prob
    
    # Plot histogram
    counts, bins, _ = ax.hist(confidences, bins=n_bins, alpha=0.7, 
                              edgecolor='black', label='Confidence')
    
    # Overlay accuracy if labels provided
    if y_true is not None:
        if len(y_prob.shape) == 2:
            predictions = np.argmax(y_prob, axis=1)
            correct = predictions == y_true
        else:
            predictions = (y_prob > 0.5).astype(int)
            correct = predictions == y_true
        
        # Calculate accuracy per bin
        bin_accs = []
        for i in range(len(bins) - 1):
            in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if np.sum(in_bin) > 0:
                bin_accs.append(np.mean(correct[in_bin]))
            else:
                bin_accs.append(0)
        
        # Plot accuracy
        ax2 = ax.twinx()
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax2.plot(bin_centers, bin_accs, 'r-', marker='o', label='Accuracy')
        ax2.set_ylabel('Accuracy', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 1])
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    nominal_coverage: float = 0.95
) -> Dict[str, float]:
    """
    Calculate Prediction Interval Coverage Probability (PICP) and width.
    
    Args:
        y_true: True values
        lower_bounds: Lower prediction bounds
        upper_bounds: Upper prediction bounds
        nominal_coverage: Expected coverage level
    
    Returns:
        Dictionary with coverage and width metrics
    """
    # Coverage
    covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    picp = np.mean(covered)
    
    # Mean width
    mpiw = np.mean(upper_bounds - lower_bounds)
    
    # Normalized width
    y_range = np.max(y_true) - np.min(y_true)
    nmpiw = mpiw / y_range if y_range > 0 else mpiw
    
    # Coverage gap
    coverage_gap = np.abs(picp - nominal_coverage)
    
    return {
        'picp': picp,
        'mpiw': mpiw,
        'nmpiw': nmpiw,
        'coverage_gap': coverage_gap,
        'nominal_coverage': nominal_coverage
    }


def continuous_ranked_probability_score(
    y_true: np.ndarray,
    y_samples: np.ndarray
) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS).
    
    Measures the quality of probabilistic predictions for continuous variables.
    
    Args:
        y_true: True values (n_samples,)
        y_samples: Sampled predictions (n_samples, n_mc_samples)
    
    Returns:
        Mean CRPS (lower is better)
    """
    n_samples = len(y_true)
    crps_values = []
    
    for i in range(n_samples):
        samples = y_samples[i]
        true_val = y_true[i]
        
        # CRPS = E|Y - y| - 0.5 * E|Y - Y'|
        term1 = np.mean(np.abs(samples - true_val))
        term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
        
        crps_values.append(term1 - term2)
    
    return np.mean(crps_values)


def sharpness(
    y_samples: np.ndarray,
    metric: str = 'std'
) -> float:
    """
    Calculate sharpness of predictions.
    
    Sharpness measures how concentrated predictions are.
    Lower values indicate more confident predictions.
    
    Args:
        y_samples: Sampled predictions or uncertainty estimates
        metric: 'std' for standard deviation, 'entropy' for entropy
    
    Returns:
        Mean sharpness
    """
    if metric == 'std':
        return np.mean(np.std(y_samples, axis=-1))
    elif metric == 'entropy':
        # For probability distributions
        return np.mean(-np.sum(y_samples * np.log(y_samples + 1e-8), axis=-1))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def calibration_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    test: str = 'hosmer-lemeshow',
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Statistical test for calibration.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        test: Test type ('hosmer-lemeshow' or 'spiegelhalter')
        n_bins: Number of bins for Hosmer-Lemeshow test
    
    Returns:
        Dictionary with test statistic and p-value
    """
    if test == 'hosmer-lemeshow':
        # Hosmer-Lemeshow test
        # Group by predicted probability
        order = np.argsort(y_prob)
        y_true_sorted = y_true[order]
        y_prob_sorted = y_prob[order]
        
        # Create bins
        n_samples = len(y_true)
        bin_size = n_samples // n_bins
        
        observed = []
        expected = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else n_samples
            
            bin_true = y_true_sorted[start:end]
            bin_prob = y_prob_sorted[start:end]
            
            observed.append(np.sum(bin_true))
            expected.append(np.sum(bin_prob))
        
        # Chi-square test
        observed = np.array(observed)
        expected = np.array(expected)
        
        # Avoid division by zero
        expected = np.maximum(expected, 1e-8)
        
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, n_bins - 2)
        
        return {
            'statistic': chi2_stat,
            'p_value': p_value,
            'calibrated': p_value > 0.05
        }
    
    elif test == 'spiegelhalter':
        # Spiegelhalter's z-test
        n = len(y_true)
        
        # Calculate z-statistic
        numerator = np.sum((y_true - y_prob) * (1 - 2 * y_prob))
        denominator = np.sqrt(np.sum((1 - 2 * y_prob) ** 2 * y_prob * (1 - y_prob)))
        
        z_stat = numerator / denominator if denominator > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
        
        return {
            'statistic': z_stat,
            'p_value': p_value,
            'calibrated': p_value > 0.05
        }
    
    else:
        raise ValueError(f"Unknown test: {test}")