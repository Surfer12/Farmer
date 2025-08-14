# Uncertainty Quantification for Reliable Risk Estimates

## Executive Summary

Better uncertainty quantification (UQ) transforms unreliable point predictions into trustworthy risk assessments by separating what we don't know (epistemic uncertainty) from inherent randomness (aleatoric uncertainty). This enables actionable risk management through calibrated probabilities, tail risk estimates, and principled abstention decisions.

## What UQ Gives You

### 1. Uncertainty Decomposition
- **Epistemic uncertainty**: Model ignorance, reducible with more data
- **Aleatoric uncertainty**: Inherent noise, irreducible randomness
- **Total uncertainty**: Combined measure for decision making

### 2. Actionable Risk Outputs
- Tail probabilities: P(Y ≥ t | X)
- Confidence/credible intervals: [L(X), U(X)]
- Abstention triggers: High uncertainty → human review
- Risk-adjusted decisions: Expected cost minimization

### 3. Calibrated Predictions
- Predicted probabilities match observed frequencies
- "90% confidence" actually means 90% correct
- Enables reliable risk budgeting and resource allocation

## Core Methods

### Deep Ensembles (n≈5)
**Strengths**: Strong epistemic UQ baseline, easy to implement
**Use case**: Most scenarios, especially when computational budget allows

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class DeepEnsemble:
    def __init__(self, models: List[nn.Module]):
        self.models = models
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and uncertainty estimates"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        epistemic_var = predictions.var(dim=0)  # Between-model variance
        
        return mean, epistemic_var
```

### Lightweight Bayesian Methods

#### MC Dropout
**Strengths**: Minimal code changes, computationally efficient
**Use case**: Quick UQ addition to existing models

```python
class MCDropoutModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enable dropout during inference for uncertainty estimation"""
        self.train()  # Enable dropout
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean, uncertainty
```

### Heteroscedastic Regression
**Strengths**: Predicts both mean and variance
**Use case**: Regression with input-dependent noise

```python
class HeteroscedasticModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        mean = self.mean_head(features)
        var = self.var_head(features)
        return mean, var
    
    def nll_loss(self, y_pred: Tuple[torch.Tensor, torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for heteroscedastic regression"""
        mean, var = y_pred
        return 0.5 * (torch.log(var) + (y_true - mean).pow(2) / var).mean()
```

### Conformal Prediction
**Strengths**: Distribution-free coverage guarantees
**Use case**: When you need guaranteed interval coverage

```python
class ConformalPredictor:
    def __init__(self, base_model, alpha: float = 0.1):
        self.base_model = base_model
        self.alpha = alpha  # Miscoverage rate (1-alpha = coverage)
        self.quantile = None
    
    def calibrate(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        """Calibrate on held-out calibration set"""
        with torch.no_grad():
            predictions = self.base_model(X_cal)
            
        # Compute conformity scores (absolute residuals)
        scores = torch.abs(y_cal - predictions.squeeze())
        
        # Find quantile for desired coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = torch.quantile(scores, q_level)
    
    def predict_interval(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return prediction intervals with coverage guarantee"""
        with torch.no_grad():
            predictions = self.base_model(X)
        
        lower = predictions.squeeze() - self.quantile
        upper = predictions.squeeze() + self.quantile
        
        return lower, upper
```

## Evaluation Framework

### Calibration Metrics

```python
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Plot reliability diagram for calibration assessment"""
    import matplotlib.pyplot as plt
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(confidences, accuracies, s=100, alpha=0.7, label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (PICP)"""
    coverage = ((y_true >= lower) & (y_true <= upper)).mean()
    return coverage

def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width (MPIW)"""
    return (upper - lower).mean()
```

### Temperature Scaling for Calibration

```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50):
        """Fit temperature on validation set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
```

## Risk-Optimal Decision Making

### Expected Cost Framework

```python
class RiskOptimalDecision:
    def __init__(self, cost_matrix: np.ndarray):
        """
        cost_matrix[i,j] = cost of taking action i when true state is j
        """
        self.cost_matrix = cost_matrix
    
    def optimal_action(self, p_y_given_x: np.ndarray) -> int:
        """Choose action that minimizes expected cost"""
        expected_costs = self.cost_matrix @ p_y_given_x
        return np.argmin(expected_costs)
    
    def expected_cost(self, p_y_given_x: np.ndarray, action: int) -> float:
        """Compute expected cost for given action"""
        return (self.cost_matrix[action] * p_y_given_x).sum()

def value_at_risk(samples: np.ndarray, alpha: float) -> float:
    """Value at Risk (VaR) - alpha-quantile of loss distribution"""
    return np.quantile(samples, alpha)

def conditional_value_at_risk(samples: np.ndarray, alpha: float) -> float:
    """Conditional Value at Risk (CVaR) - expected loss beyond VaR"""
    var = value_at_risk(samples, alpha)
    return samples[samples >= var].mean()
```

### Selective Prediction with Abstention

```python
class SelectivePredictor:
    def __init__(self, base_model, uncertainty_threshold: float, max_set_size: int = 5):
        self.base_model = base_model
        self.uncertainty_threshold = uncertainty_threshold
        self.max_set_size = max_set_size
    
    def predict_with_abstention(self, x: torch.Tensor, conformal_sets: List = None) -> dict:
        """Make prediction with option to abstain based on uncertainty"""
        # Get base prediction and uncertainty
        if hasattr(self.base_model, 'predict_with_uncertainty'):
            pred, uncertainty = self.base_model.predict_with_uncertainty(x)
        else:
            pred = self.base_model(x)
            uncertainty = torch.zeros_like(pred)  # Placeholder
        
        # Compute entropy for classification
        if pred.dim() > 1 and pred.size(-1) > 1:
            probs = torch.softmax(pred, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        else:
            entropy = uncertainty
        
        # Decision logic
        should_abstain = entropy > self.uncertainty_threshold
        
        if conformal_sets is not None:
            set_sizes = torch.tensor([len(s) for s in conformal_sets])
            should_abstain = should_abstain | (set_sizes > self.max_set_size)
        
        results = {
            'prediction': pred,
            'uncertainty': uncertainty,
            'entropy': entropy,
            'abstain': should_abstain,
            'action': 'abstain' if should_abstain.any() else 'predict'
        }
        
        return results
```

## Monitoring and Drift Detection

```python
class UQMonitor:
    def __init__(self, reference_data: dict):
        """
        reference_data: dict with 'X', 'y', 'predictions', 'uncertainties'
        """
        self.reference_data = reference_data
        self.reference_stats = self._compute_reference_stats()
    
    def _compute_reference_stats(self) -> dict:
        """Compute reference statistics for drift detection"""
        X_ref = self.reference_data['X']
        y_ref = self.reference_data['y']
        pred_ref = self.reference_data['predictions']
        
        return {
            'input_mean': X_ref.mean(axis=0),
            'input_std': X_ref.std(axis=0),
            'prediction_mean': pred_ref.mean(),
            'prediction_std': pred_ref.std(),
            'calibration_error': expected_calibration_error(y_ref, pred_ref)
        }
    
    def detect_input_drift(self, X_new: np.ndarray, threshold: float = 2.0) -> bool:
        """Detect input distribution drift using z-score"""
        new_mean = X_new.mean(axis=0)
        z_scores = np.abs((new_mean - self.reference_stats['input_mean']) / 
                         (self.reference_stats['input_std'] + 1e-8))
        return (z_scores > threshold).any()
    
    def detect_prediction_drift(self, pred_new: np.ndarray, threshold: float = 0.1) -> bool:
        """Detect prediction drift"""
        new_mean = pred_new.mean()
        drift_score = abs(new_mean - self.reference_stats['prediction_mean'])
        return drift_score > threshold
    
    def assess_calibration_drift(self, y_new: np.ndarray, pred_new: np.ndarray, 
                               threshold: float = 0.05) -> bool:
        """Assess if calibration has degraded"""
        new_ece = expected_calibration_error(y_new, pred_new)
        ref_ece = self.reference_stats['calibration_error']
        return (new_ece - ref_ece) > threshold

def population_stability_index(reference: np.ndarray, current: np.ndarray, 
                             n_bins: int = 10) -> float:
    """Population Stability Index (PSI) for drift detection"""
    ref_hist, bin_edges = np.histogram(reference, bins=n_bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)
    
    # Add small constant to avoid division by zero
    ref_pct = (ref_hist + 1e-8) / (ref_hist.sum() + n_bins * 1e-8)
    cur_pct = (cur_hist + 1e-8) / (cur_hist.sum() + n_bins * 1e-8)
    
    psi = ((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum()
    return psi
```

## Complete Implementation Example

### Classification with Full UQ Pipeline

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class FullUQClassifier:
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, 
                 n_ensemble: int = 5, dropout_rate: float = 0.1):
        self.n_ensemble = n_ensemble
        self.models = []
        self.temperature_scaler = TemperatureScaling()
        self.conformal_predictor = None
        
        # Create ensemble of models
        for _ in range(n_ensemble):
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, n_classes)
            )
            self.models.append(model)
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, 
              X_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 100):
        """Train ensemble and calibrate"""
        # Train each model in ensemble
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_ensemble}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
        
        # Calibrate on validation set
        val_logits = self._get_ensemble_logits(X_val)
        self.temperature_scaler.fit(val_logits, y_val)
        
        # Setup conformal prediction
        calibrated_logits = self.temperature_scaler(val_logits)
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
        self.conformal_predictor = self._setup_conformal(X_val, y_val, calibrated_probs)
    
    def _get_ensemble_logits(self, X: torch.Tensor) -> torch.Tensor:
        """Get average logits from ensemble"""
        logits_list = []
        for model in self.models:
            with torch.no_grad():
                logits = model(X)
                logits_list.append(logits)
        
        return torch.stack(logits_list).mean(dim=0)
    
    def _setup_conformal(self, X_val: torch.Tensor, y_val: torch.Tensor, 
                        probs: torch.Tensor, alpha: float = 0.1):
        """Setup conformal prediction for set-valued predictions"""
        # Use 1 - max probability as conformity score
        scores = 1 - probs.max(dim=-1)[0]
        
        # Find quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile = torch.quantile(scores, q_level)
        
        return {'quantile': quantile, 'alpha': alpha}
    
    def predict_full(self, X: torch.Tensor) -> dict:
        """Full UQ prediction with all uncertainty estimates"""
        # Ensemble predictions
        logits_list = []
        for model in self.models:
            with torch.no_grad():
                logits = model(X)
                logits_list.append(logits)
        
        logits_tensor = torch.stack(logits_list)
        mean_logits = logits_tensor.mean(dim=0)
        epistemic_var = logits_tensor.var(dim=0).mean(dim=-1)  # Average across classes
        
        # Temperature scaling
        calibrated_logits = self.temperature_scaler(mean_logits)
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
        
        # Entropy (aleatoric uncertainty proxy)
        entropy = -(calibrated_probs * torch.log(calibrated_probs + 1e-8)).sum(dim=-1)
        
        # Conformal sets
        if self.conformal_predictor is not None:
            scores = 1 - calibrated_probs.max(dim=-1)[0]
            in_set = scores <= self.conformal_predictor['quantile']
            conformal_sets = []
            for i, prob_row in enumerate(calibrated_probs):
                if in_set[i]:
                    # Include top classes until threshold
                    sorted_indices = torch.argsort(prob_row, descending=True)
                    cumsum = torch.cumsum(prob_row[sorted_indices], dim=0)
                    set_size = (cumsum <= (1 - self.conformal_predictor['quantile'])).sum() + 1
                    conformal_set = sorted_indices[:set_size].tolist()
                else:
                    conformal_set = list(range(len(prob_row)))  # Include all classes
                conformal_sets.append(conformal_set)
        else:
            conformal_sets = None
        
        return {
            'probabilities': calibrated_probs,
            'predictions': calibrated_probs.argmax(dim=-1),
            'epistemic_uncertainty': epistemic_var,
            'entropy': entropy,
            'total_uncertainty': epistemic_var + entropy,
            'conformal_sets': conformal_sets,
            'max_prob': calibrated_probs.max(dim=-1)[0]
        }

# Example usage
def demo_classification_uq():
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                             n_informative=15, random_state=42)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # Create and train UQ classifier
    uq_classifier = FullUQClassifier(input_dim=20, hidden_dim=64, n_classes=3)
    uq_classifier.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # Make predictions with full uncertainty quantification
    results = uq_classifier.predict_full(X_test)
    
    # Evaluate
    accuracy = (results['predictions'] == y_test).float().mean()
    ece = expected_calibration_error(y_test.numpy(), results['max_prob'].numpy())
    
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Expected Calibration Error: {ece:.3f}")
    print(f"Average Epistemic Uncertainty: {results['epistemic_uncertainty'].mean():.3f}")
    print(f"Average Entropy: {results['entropy'].mean():.3f}")
    
    # Coverage analysis for conformal sets
    if results['conformal_sets'] is not None:
        coverage = np.mean([y_test[i].item() in conf_set 
                          for i, conf_set in enumerate(results['conformal_sets'])])
        avg_set_size = np.mean([len(conf_set) for conf_set in results['conformal_sets']])
        print(f"Conformal Set Coverage: {coverage:.3f}")
        print(f"Average Set Size: {avg_set_size:.2f}")

if __name__ == "__main__":
    demo_classification_uq()
```

## Quick-Start Implementation Plan

### Phase 1: Baseline UQ (Week 1)
1. **Deep Ensemble Setup**: Train 5 models with different initializations
2. **Temperature Scaling**: Calibrate on validation set
3. **Basic Metrics**: Implement ECE and reliability diagrams
4. **Target**: ECE ≤ 3% on validation data

### Phase 2: Guarantees (Week 2)
1. **Conformal Prediction**: Add distribution-free intervals/sets
2. **Coverage Validation**: Ensure 90-95% nominal coverage
3. **Width Optimization**: Minimize interval width while maintaining coverage
4. **Target**: Coverage within ±2% of nominal

### Phase 3: Decision Integration (Week 3)
1. **Cost Matrix**: Define business-relevant cost structure
2. **Risk Metrics**: Implement VaR/CVaR calculations
3. **Abstention Rules**: Set uncertainty-based thresholds
4. **Target**: Measurable improvement in expected cost

### Phase 4: Production Monitoring (Week 4)
1. **Drift Detection**: Input and prediction distribution monitoring
2. **Online Calibration**: Periodic recalibration procedures
3. **Alerting System**: Automated threshold-based notifications
4. **Target**: Stable performance under distribution shift

## Minimal Decision Pseudocode

```python
def make_risk_aware_decision(x, model, cost_matrix, uncertainty_threshold=0.1, max_set_size=3):
    """
    Minimal decision framework with uncertainty quantification
    
    Args:
        x: Input features
        model: Trained UQ model
        cost_matrix: Cost of actions vs true outcomes
        uncertainty_threshold: Threshold for abstention
        max_set_size: Maximum conformal set size before abstention
    
    Returns:
        dict: Decision with rationale
    """
    # Get predictions with uncertainty
    results = model.predict_full(x)
    
    # Extract key metrics
    probs = results['probabilities']
    uncertainty = results['total_uncertainty']
    conformal_sets = results['conformal_sets']
    
    # Check abstention conditions
    high_uncertainty = uncertainty > uncertainty_threshold
    large_conformal_set = len(conformal_sets[0]) > max_set_size if conformal_sets else False
    
    if high_uncertainty or large_conformal_set:
        action = "abstain_or_escalate"
        rationale = f"High uncertainty ({uncertainty:.3f}) or large conformal set"
    else:
        # Compute expected cost for each action
        expected_costs = cost_matrix @ probs.numpy()
        action = np.argmin(expected_costs)
        rationale = f"Expected cost minimization: {expected_costs[action]:.3f}"
    
    return {
        'action': action,
        'rationale': rationale,
        'uncertainty': uncertainty.item(),
        'probabilities': probs.numpy(),
        'expected_costs': expected_costs if 'expected_costs' in locals() else None
    }
```

## Link to Ψ Framework

### Calibration Enhancement
- **Improved Trust**: Well-calibrated probabilities increase stakeholder confidence
- **Verifiable Accuracy**: Temperature scaling provides auditable calibration process
- **Consistent Performance**: Stable calibration across different data distributions

### Verifiability (R_v) Improvement
- **Reproducible Methods**: Ensemble training with fixed seeds ensures reproducibility
- **Auditable Decisions**: All uncertainty estimates and thresholds are logged
- **Statistical Guarantees**: Conformal prediction provides mathematically rigorous coverage

### Authority (R_a) Enhancement
- **OOD Detection**: Energy scores and density-based methods identify distribution shift
- **Graceful Degradation**: Abstention mechanisms prevent overconfident predictions
- **Robust Performance**: Deep ensembles maintain accuracy under various conditions

### Ψ Score Integration
As calibration improves and verifiability increases through systematic UQ implementation, the overall Ψ score increases predictably without introducing overconfidence. The framework provides measurable improvements in:

- **Reliability**: ECE ≤ 2-3% demonstrates calibrated confidence
- **Coverage**: Conformal intervals maintain statistical guarantees
- **Decision Quality**: Risk-optimal actions minimize expected costs

## Practical Implementation Targets

### Performance Benchmarks
- **Calibration**: ECE ≤ 2-3% on in-domain validation data
- **Coverage**: Conformal prediction within ±1-2% of nominal (90-95%)
- **OOD Detection**: FPR@95%TPR substantially below in-domain baseline
- **Decision Quality**: 10-20% improvement in expected cost vs baseline

### Resource Requirements
- **Computational**: 5x inference cost for deep ensembles (parallelizable)
- **Memory**: Minimal overhead for temperature scaling and conformal prediction
- **Storage**: Model ensemble requires 5x storage vs single model
- **Latency**: Acceptable for most applications (<100ms additional overhead)

## Implementation Checklist

### Training Phase
- [ ] Train ensemble of 5 models with different random seeds
- [ ] Implement heteroscedastic/quantile heads for regression tasks
- [ ] Reserve 10-20% of data for calibration (separate from validation)
- [ ] Train temperature scaling on held-out calibration set

### Calibration Phase
- [ ] Fit temperature parameter using LBFGS optimization
- [ ] Generate reliability diagrams to verify calibration quality
- [ ] Compute ECE and other calibration metrics
- [ ] Set target ECE ≤ 3% for production deployment

### Guarantee Phase
- [ ] Fit conformal prediction quantiles on calibration set
- [ ] Validate coverage on separate test set
- [ ] Optimize prediction interval width while maintaining coverage
- [ ] Document coverage guarantees for stakeholders

### Decision Phase
- [ ] Define cost matrix based on business requirements
- [ ] Implement expected cost minimization logic
- [ ] Set abstention thresholds using validation data
- [ ] Compute VaR/CVaR for tail risk assessment

### Monitoring Phase
- [ ] Implement drift detection for input distributions
- [ ] Monitor calibration quality over time
- [ ] Set up automated recalibration triggers
- [ ] Create dashboards for uncertainty metrics

### Governance Phase
- [ ] Log all uncertainty estimates and decision rationales
- [ ] Implement human review for high-uncertainty cases
- [ ] Create audit trails for regulatory compliance
- [ ] Document threshold selection and update procedures

This comprehensive guide provides both theoretical foundation and practical implementation details for deploying uncertainty quantification in production machine learning systems. The modular design allows for incremental adoption while the monitoring framework ensures long-term reliability.