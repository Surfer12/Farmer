# Uncertainty Quantification for Reliable Risk Estimates

A comprehensive implementation of uncertainty quantification (UQ) methods for machine learning models, enabling reliable risk assessment and decision-making in production environments.

## Overview

This repository provides a complete framework for implementing uncertainty quantification in machine learning systems, transforming unreliable point predictions into trustworthy risk assessments. The implementation separates epistemic uncertainty (model ignorance) from aleatoric uncertainty (inherent randomness) and provides actionable risk management tools.

## Key Features

### 🎯 Core UQ Methods
- **Deep Ensembles**: Strong baseline for epistemic uncertainty estimation
- **MC Dropout**: Lightweight Bayesian inference for existing models
- **Heteroscedastic Regression**: Input-dependent noise modeling
- **Conformal Prediction**: Distribution-free coverage guarantees
- **Temperature Scaling**: Post-hoc calibration for classification

### 📊 Evaluation Framework
- **Calibration Metrics**: ECE, ACE, Brier score, reliability diagrams
- **Interval Quality**: Coverage probability, interval width, CRPS
- **Drift Detection**: PSI, KL divergence, distribution monitoring
- **Decision Quality**: Expected cost curves, VaR/CVaR analysis

### 🚨 Production Monitoring
- **Real-time Drift Detection**: Input and prediction distribution monitoring
- **Calibration Monitoring**: Continuous quality assessment
- **Automated Alerting**: Threshold-based notifications with cooldown
- **Dashboard Visualization**: Comprehensive monitoring interface

### ⚖️ Risk-Aware Decision Making
- **Expected Cost Minimization**: Optimal action selection
- **Tail Risk Assessment**: VaR and CVaR calculations
- **Selective Prediction**: Abstention based on uncertainty thresholds
- **Business Integration**: Cost matrices and risk budgeting

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd uncertainty-quantification

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Regression with Uncertainty Quantification

```python
from uq_examples import regression_uq_example

# Run comprehensive regression example
results = regression_uq_example()

# Results include:
# - Deep ensemble predictions with epistemic uncertainty
# - MC Dropout uncertainty estimates
# - Heteroscedastic variance predictions
# - Conformal prediction intervals with coverage guarantees
```

#### 2. Classification with Calibration

```python
from uq_examples import classification_uq_example

# Run classification with uncertainty quantification
results = classification_uq_example()

# Results include:
# - Temperature-scaled probabilities
# - Reliability diagrams
# - Entropy-based uncertainty measures
# - Conformal prediction sets
```

#### 3. Production Monitoring

```python
from uq_monitoring import simulate_monitoring_scenario

# Simulate production monitoring with drift detection
monitor, report = simulate_monitoring_scenario()

# Features:
# - Real-time drift detection
# - Calibration quality monitoring
# - Automated alerting system
# - Comprehensive dashboard
```

## Implementation Guide

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

## Core Methods

### Deep Ensembles
```python
from uq_examples import DeepEnsemble

# Create ensemble of models
models = [create_model() for _ in range(5)]
ensemble = DeepEnsemble(models)

# Get predictions with epistemic uncertainty
mean, epistemic_var = ensemble.predict(x_test)
```

### MC Dropout
```python
from uq_examples import MCDropoutModel

# Create MC Dropout model
model = MCDropoutModel(input_dim=10, hidden_dim=64, output_dim=1)

# Get uncertainty estimates
mean, uncertainty = model.predict_with_uncertainty(x_test, n_samples=100)
```

### Conformal Prediction
```python
from uq_examples import ConformalPredictor

# Setup conformal prediction
conformal = ConformalPredictor(base_model, alpha=0.1)  # 90% coverage
conformal.calibrate(x_cal, y_cal)

# Get prediction intervals
lower, upper = conformal.predict_interval(x_test)
```

### Temperature Scaling
```python
from uq_examples import TemperatureScaling

# Calibrate model predictions
temp_scaler = TemperatureScaling()
temp_scaler.fit(logits_val, labels_val)

# Apply calibration
calibrated_logits = temp_scaler(logits_test)
```

## Risk-Aware Decision Making

### Expected Cost Framework
```python
# Define cost matrix (action vs true state)
cost_matrix = np.array([
    [0, 5],    # Action 0: correct=0 cost, wrong=5 cost
    [3, 0],    # Action 1: wrong=3 cost, correct=0 cost
    [1, 1]     # Abstain: always 1 cost
])

# Make risk-optimal decisions
def make_decision(probabilities, uncertainty_threshold=0.1):
    if entropy(probabilities) > uncertainty_threshold:
        return "abstain"
    else:
        expected_costs = cost_matrix @ probabilities
        return np.argmin(expected_costs)
```

### Tail Risk Assessment
```python
def value_at_risk(samples, alpha=0.95):
    """Value at Risk - alpha-quantile of loss distribution"""
    return np.quantile(samples, alpha)

def conditional_value_at_risk(samples, alpha=0.95):
    """Expected loss beyond VaR"""
    var = value_at_risk(samples, alpha)
    return samples[samples >= var].mean()
```

## Production Monitoring

### Setting Up Monitoring
```python
from uq_monitoring import UQProductionMonitor

# Initialize monitor
monitor = UQProductionMonitor(
    psi_threshold=0.1,           # Input drift threshold
    calibration_threshold=0.05,  # Calibration degradation threshold
    coverage_threshold=0.05,     # Coverage drift threshold
    window_size=1000            # Rolling window size
)

# Set reference data
monitor.set_reference_data(
    features=ref_features,
    predictions=ref_predictions,
    uncertainties=ref_uncertainties,
    true_labels=ref_labels,
    prediction_intervals=ref_intervals
)
```

### Adding New Predictions
```python
# Add batch of new predictions
monitor.add_batch(
    features=new_features,
    predictions=new_predictions,
    uncertainties=new_uncertainties,
    true_labels=new_labels,
    prediction_intervals=new_intervals
)

# Generate monitoring report
report = monitor.export_monitoring_report(hours_back=24)
```

## Evaluation Metrics

### Calibration Assessment
```python
from uq_examples import expected_calibration_error, reliability_diagram

# Calculate Expected Calibration Error
ece = expected_calibration_error(y_true, y_prob)

# Plot reliability diagram
reliability_diagram(y_true, y_prob)
```

### Interval Quality
```python
from uq_examples import interval_coverage, interval_width

# Assess prediction intervals
coverage = interval_coverage(y_true, lower_bounds, upper_bounds)
width = interval_width(lower_bounds, upper_bounds)
```

## File Structure

```
uncertainty-quantification/
├── uncertainty_quantification_guide.md  # Comprehensive theory and methods guide
├── uq_examples.py                       # Practical implementation examples
├── uq_monitoring.py                     # Production monitoring system
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── outputs/                            # Generated plots and reports
    ├── regression_uq_results.png
    ├── classification_uq_reliability.png
    ├── monitoring_dashboard.png
    └── risk_aware_decisions.png
```

## Key Performance Targets

### Calibration Quality
- **ECE ≤ 2-3%** on in-domain validation data
- **Reliability diagrams** showing diagonal alignment
- **Brier score** improvement over baseline

### Coverage Guarantees
- **Conformal prediction** within ±1-2% of nominal coverage (90-95%)
- **Adaptive intervals** that adjust to input complexity
- **Minimal width** while maintaining coverage

### Drift Detection
- **PSI < 0.1** for stable operation
- **KL divergence** monitoring for distribution shifts
- **FPR@95%TPR** substantially below in-domain baseline for OOD detection

### Decision Quality
- **10-20% improvement** in expected cost vs naive baseline
- **Reduced tail losses** through VaR/CVaR optimization
- **Appropriate abstention rates** (5-15% depending on application)

## Business Impact

### Risk Management
- **Quantified uncertainty** enables proper risk budgeting
- **Tail risk assessment** for regulatory compliance
- **Automated abstention** prevents costly errors

### Model Reliability
- **Calibrated probabilities** improve stakeholder trust
- **Coverage guarantees** provide statistical assurances
- **Drift detection** maintains performance over time

### Operational Excellence
- **Real-time monitoring** prevents silent failures
- **Automated alerting** enables proactive intervention
- **Audit trails** support regulatory requirements

## Advanced Features

### Multi-Model Ensembles
Combine different architectures for improved uncertainty estimation:
- CNN + Transformer ensembles
- Gradient boosting + neural network combinations
- Bayesian model averaging

### Online Learning
Adapt uncertainty estimates in real-time:
- Incremental conformal prediction updates
- Online temperature scaling recalibration
- Streaming drift detection

### Hierarchical Uncertainty
Model uncertainty at multiple levels:
- Feature-level uncertainty propagation
- Layer-wise uncertainty decomposition
- Multi-task uncertainty sharing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{uncertainty-quantification-2024,
  title={Uncertainty Quantification for Reliable Risk Estimates},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/uncertainty-quantification}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on PyTorch for deep learning components
- Uses scikit-learn for traditional ML utilities
- Inspired by recent advances in uncertainty quantification research
- Designed for production deployment and monitoring

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `uncertainty_quantification_guide.md`
- Review the examples in `uq_examples.py`
- Examine the monitoring system in `uq_monitoring.py`
