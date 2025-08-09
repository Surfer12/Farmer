# Uncertainty Quantification (UQ) Framework for Reliable Risk Estimates

A comprehensive Python framework for quantifying and leveraging uncertainty in machine learning predictions to make better risk-based decisions.

## 🎯 What This Framework Gives You

### Separates Ignorance from Noise
- **Aleatoric Uncertainty**: Irreducible data noise that's inherent to the problem
- **Epistemic Uncertainty**: Model ignorance that can be reduced with more data
- Clear separation helps you understand what can be improved vs. what's fundamental

### Converts Predictions into Actionable Risk
- **Tail Probabilities**: P(Y ≥ threshold | X) for risk assessment
- **Confidence Intervals**: Reliable prediction bounds with coverage guarantees
- **Abstention Triggers**: Know when to escalate to human experts
- **VaR/CVaR**: Value at Risk and Conditional Value at Risk for financial applications

### Improves Calibration
- Predicted probabilities match observed frequencies
- Temperature scaling and isotonic regression for better calibration
- Reliability diagrams for visual assessment

## 🔧 Core Methods Implemented

### Deep Ensembles (n≈5)
Strong, simple baseline for epistemic uncertainty quantification
```python
from uncertainty_quantification import DeepEnsemble
ensemble = DeepEnsemble(RandomForestRegressor, n_models=5)
ensemble.fit(X_train, y_train)
uncertainty_est = ensemble.predict_with_uncertainty(X_test)
```

### Bayesian Approximations
- **MC Dropout**: Lightweight uncertainty via dropout at inference
- **SWAG**: Stochastic Weight Averaging Gaussian
- **Laplace Approximation**: Last-layer Bayesian treatment

### Heteroscedastic Heads
Regress both mean and variance to capture aleatoric uncertainty
```python
from uncertainty_quantification import HeteroscedasticHead
head = HeteroscedasticHead(input_dim=128, output_dim=1)
mean, var = head(features)
```

### Conformal Prediction
Distribution-free coverage guarantees regardless of model or data distribution
```python
from uncertainty_quantification import ConformalPredictor
conformal = ConformalPredictor(base_model, alpha=0.1)  # 90% coverage
conformal.fit(X_cal, y_cal)
lower, upper = conformal.predict_intervals(X_test)
```

### Calibration Methods
- **Temperature Scaling**: Scale logits for better calibration
- **Isotonic Regression**: Non-parametric calibration mapping
- **Reliability Diagrams**: Visual calibration assessment

### OOD Detection
- **Energy Scores**: -T * log(sum(exp(logits/T)))
- **Dirichlet Prior Networks**: Separate aleatoric/epistemic in classification
- **Mahalanobis Distance**: Feature-space OOD detection

## 📊 Evaluation Metrics

### Calibration Assessment
- **Expected Calibration Error (ECE)**: Difference between confidence and accuracy
- **Reliability Diagrams**: Visual calibration quality
- **Brier Score**: Probabilistic prediction accuracy
- **Negative Log-Likelihood (NLL)**: Information-theoretic measure

### Interval Quality
- **Coverage (PICP)**: Fraction of true values within intervals
- **Width (MPIW)**: Average interval width (sharpness)
- **CRPS**: Continuous Ranked Probability Score for distributions
- **Interval Score**: Combined coverage and sharpness metric

### OOD Detection
- **AUROC**: Area under ROC curve for OOD detection
- **FPR@95%TPR**: False positive rate at 95% true positive rate

## ⚠️ Risk-Based Decision Making

### Expected Cost Minimization
Choose actions that minimize expected cost under uncertainty
```python
from uncertainty_quantification import RiskBasedDecisionFramework

optimal_actions = RiskBasedDecisionFramework.expected_cost_minimization(
    predictions=uncertainty_estimates,
    cost_matrix=cost_matrix,
    actions=['conservative', 'moderate', 'aggressive']
)
```

### Tail Risk Metrics
- **VaR(α)**: Value at Risk - α-quantile of loss distribution
- **CVaR(α)**: Conditional Value at Risk - expected loss beyond VaR
- **Tail Probabilities**: P(Y ≥ threshold | X)

### Selective Prediction
Abstain or escalate when uncertainty exceeds threshold
```python
threshold = RiskBasedDecisionFramework.selective_prediction_threshold(
    uncertainties, accuracies, coverage_target=0.9
)
```

## 🔍 Monitoring and Guardrails

### Drift Detection
- **Population Stability Index (PSI)**: Input distribution drift
- **KL Divergence**: Distribution change measurement
- **Calibration Drift**: Monitor ECE and Brier score changes

### Online Updates
- **Conformal Online Updates**: Maintain coverage with streaming data
- **Temperature Re-fitting**: Periodic calibration updates
- **Sliding Window Coverage**: Monitor prediction quality over time

### Backtesting
- **Champion-Challenger**: Compare UQ methods
- **Coverage Audits**: Validate interval coverage over time
- **Canary Deployments**: Safe rollout of new UQ methods

## 🚀 Quick-Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from uq_quickstart_example import QuickStartUQPipeline
from sklearn.ensemble import RandomForestRegressor

# Initialize pipeline
pipeline = QuickStartUQPipeline(RandomForestRegressor, n_ensemble=5)

# Train with calibration set
pipeline.fit(X_train, y_train, X_cal, y_cal)

# Generate predictions with risk analysis
results = pipeline.predict_with_risk_analysis(X_test)

# Evaluate quality
metrics = pipeline.evaluate_quality(X_test, y_test)
```

### 3. Run Complete Demo
```bash
python uq_quickstart_example.py
```

## 🔮 Ψ Framework Integration

This UQ framework integrates with the Ψ framework to enhance reliability:

### Calibration Component (Post)
Better UQ → Better calibrated probabilities → Higher Ψ score

### Verifiability Component (R_v)
Reproducible uncertainty estimates → Lower verifiability risk → Higher Ψ score

### Authority Component (R_a)
Stable predictions across distribution shifts → Better authority scoring → Higher Ψ score

```python
from uq_quickstart_example import PsiFrameworkIntegration

psi_integration = PsiFrameworkIntegration(pipeline)
psi_scores = psi_integration.compute_psi_score(X_val, y_val, X_test, X_shifted)
```

## 📈 Production Deployment Checklist

### Baseline Setup
- [x] Deep ensemble (n=5) for epistemic uncertainty
- [x] Temperature scaling for calibration
- [x] Conformal prediction for coverage guarantees

### For Regression
- [x] Heteroscedastic head or quantile regression
- [x] Report P(Y>t) and prediction intervals
- [x] VaR/CVaR risk metrics

### For Classification
- [x] Predicted probabilities + entropy
- [x] Conformal prediction sets
- [x] Abstention/escalation thresholds

### Monitoring
- [x] Input drift detection (PSI/KL divergence)
- [x] Calibration drift monitoring
- [x] Online conformal updates
- [x] Coverage backtesting

### Decision Integration
- [x] Expected cost minimization
- [x] Risk-based action selection
- [x] Selective prediction thresholds

## 📚 Key Benefits

1. **Reliability**: Theoretical coverage guarantees via conformal prediction
2. **Actionability**: Convert uncertainty into concrete risk metrics and decisions
3. **Robustness**: Multiple complementary UQ methods for comprehensive coverage
4. **Monitoring**: Built-in drift detection and quality assessment
5. **Integration**: Seamless integration with Ψ framework for enhanced reliability
6. **Production-Ready**: Complete pipeline with monitoring and backtesting

## 🎯 Summary

This framework transforms uncertainty from a nuisance into a strategic advantage by:
- Providing reliable, well-calibrated uncertainty estimates
- Converting predictions into actionable risk assessments
- Enabling principled decision-making under uncertainty
- Maintaining quality through comprehensive monitoring
- Integrating with reliability frameworks like Ψ

**Result**: More reliable risk estimates that improve decision quality and reduce unexpected failures.