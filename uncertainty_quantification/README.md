# Uncertainty Quantification for Reliable Risk Estimation

A production-ready framework for uncertainty quantification (UQ) that separates epistemic from aleatoric uncertainty, provides calibrated predictions, and enables risk-aware decision making.

## 🎯 Key Benefits

- **Separates Ignorance from Noise**: Distinguishes epistemic (model) uncertainty from aleatoric (data) uncertainty
- **Actionable Risk Outputs**: Provides tail probabilities, confidence intervals, and abstention triggers
- **Calibrated Predictions**: Ensures predicted probabilities match observed frequencies
- **Risk-Aware Decisions**: Implements expected cost minimization and tail risk management

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic example
python examples/quick_start.py

# Run comprehensive demo
jupyter notebook notebooks/uq_demo.ipynb
```

## 📊 Core Methods

### Deep Ensembles
- Strong baseline for epistemic uncertainty (n≈5 models)
- Simple to implement, excellent performance

### Lightweight Bayesian
- MC Dropout: Uncertainty from dropout at inference
- SWAG: Stochastic Weight Averaging Gaussian
- Laplace Approximation: Last-layer Bayesian treatment

### Heteroscedastic/Quantile Regression
- Predict mean + variance directly
- Quantile regression for distribution-free intervals

### Conformal Prediction
- Distribution-free coverage guarantees
- Works with any base predictor

## 📈 Evaluation Metrics

- **Calibration**: ECE, ACE, reliability diagrams, Brier score
- **Intervals**: Coverage (PICP), width (MPIW), CRPS
- **OOD Detection**: AUROC, FPR@95%TPR
- **Decision Quality**: Expected cost, VaR, CVaR

## 🎛️ Risk Decision Framework

```python
# Minimal decision pseudocode
if uncertainty > threshold or conformal_set_size > max_size:
    action = "abstain_or_escalate"
else:
    action = argmin_a(expected_cost(a, p_y_given_x))
```

## 📁 Project Structure

```
uncertainty_quantification/
├── src/
│   ├── models/          # UQ model implementations
│   ├── calibration/     # Calibration methods
│   ├── conformal/       # Conformal prediction
│   ├── metrics/         # Evaluation metrics
│   ├── decisions/       # Risk-aware decision making
│   └── monitoring/      # Drift detection and monitoring
├── examples/            # Quick examples
├── notebooks/           # Jupyter notebooks
└── tests/              # Unit tests
```

## 🎯 Performance Targets

- ECE ≤ 2-3% on in-domain data
- Conformal coverage within ±1-2% of nominal
- OOD FPR@95%TPR substantially below baseline
- Improved expected cost and lower tail losses

## 📚 References

See `notebooks/uq_demo.ipynb` for detailed examples and explanations.