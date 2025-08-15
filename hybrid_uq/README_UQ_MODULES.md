# Hybrid UQ Framework: Conformal Prediction & Drift Monitoring

This document describes the comprehensive uncertainty quantification (UQ) framework that integrates conformal prediction for NODE-RK4 models with advanced drift monitoring and online calibration capabilities.

## 🎯 Overview

The framework provides production-ready uncertainty quantification with the following key features:

- **Conformal Prediction for NODE-RK4**: Distribution-free coverage guarantees for differential equation solutions
- **Non-Canonically Invasive Drift Monitoring**: Advanced detection of subtle distribution shifts
- **Online Calibration Refresh**: Real-time adaptation to changing conditions
- **Production Integration**: Complete system for reliable risk estimation

## 📦 Module Structure

```
hybrid_uq/
├── conformal_node_rk4.py      # Conformal prediction for NODE-RK4
├── drift_monitoring.py        # Advanced drift detection
├── online_calibration.py      # Real-time calibration updates
├── integration_example.py     # Production system integration
├── core.py                    # Original hybrid model (enhanced)
└── __init__.py               # Main exports
```

## 🚀 Quick Start

### Basic Usage

```python
from hybrid_uq import (
    ProductionUQSystem,
    RK4ConformalPredictor,
    NonCanonicalDriftMonitor
)

# 1. Set up production UQ system
uq_system = ProductionUQSystem(
    model=your_trained_model,
    alpha=0.1,  # 90% coverage
    calibration_methods=['conformal_prediction']
)

# 2. Fit on calibration data
uq_system.fit(X_train, y_train, X_cal, y_cal)

# 3. Process new data with uncertainty quantification
results = uq_system.process_batch(X_new, y_new)

# Results include:
# - Predictions with uncertainty bounds
# - Conformal prediction intervals
# - Drift detection alerts
# - Calibration status
```

### Advanced Configuration

```python
# Custom conformal predictor
conformal_predictor = RK4ConformalPredictor(
    model=model,
    alpha=0.1,
    conformity_score_fn="adaptive_residual",  # Handles heteroscedastic uncertainty
    temporal_weighting=True,  # Weight later time steps more heavily
    max_calibration_size=1000
)

# Advanced drift monitoring
drift_monitor = NonCanonicalDriftMonitor(
    detectors=[
        KolmogorovSmirnovDriftDetector(),
        MaximumMeanDiscrepancyDetector(),
        AdversarialDriftDetector(),
        EnergyDistanceDetector()
    ],
    window_size=1000,
    alert_cooldown=100
)

# Online calibration with multiple methods
calibration_manager = create_online_calibration_pipeline(
    drift_monitor=drift_monitor,
    calibration_methods=['temperature_scaling', 'conformal_prediction'],
    temp_learning_rate=0.01,
    conf_alpha=0.1,
    conf_adaptive_alpha=True
)
```

## 🔬 Core Components

### 1. Conformal Prediction for NODE-RK4

**File**: `conformal_node_rk4.py`

Provides distribution-free coverage guarantees for neural ODE solutions:

- **RK4ConformalPredictor**: Main conformal predictor class
- **TemporalConformalEnsemble**: Ensemble of predictors for robustness
- **Multiple conformity scores**: Adaptive residual, trajectory-aware, Mahalanobis
- **Online updates**: Streaming conformal quantile updates
- **RK4 integration**: Direct integration with uncertainty intervals

**Key Features**:
- Handles temporal dependencies in ODE solutions
- Heteroscedastic uncertainty through adaptive conformity scores
- Online quantile updates for streaming data
- Coverage guarantees regardless of underlying model

### 2. Non-Canonically Invasive Drift Monitoring

**File**: `drift_monitoring.py`

Advanced drift detection for subtle, non-standard distribution shifts:

- **Multiple detection methods**: KS test, MMD, adversarial, energy distance
- **Real-time monitoring**: Streaming detection with adaptive thresholds
- **Alert management**: Severity levels, cooldown periods, affected features
- **Coverage-aware monitoring**: Integration with conformal prediction

**Drift Detectors**:
- **Kolmogorov-Smirnov**: Univariate CDF changes
- **Maximum Mean Discrepancy**: Multivariate kernel-based detection
- **Adversarial**: Domain classification approach
- **Energy Distance**: Non-parametric statistical test

### 3. Online Calibration Refresh

**File**: `online_calibration.py`

Real-time calibration maintenance for production reliability:

- **Temperature Scaling**: Online neural network calibration
- **Isotonic Regression**: Non-parametric probability calibration
- **Conformal Calibration**: Adaptive quantile updates
- **Drift-Aware Management**: Automatic recalibration triggers

**Calibration Methods**:
- **OnlineTemperatureScaling**: Gradient-based temperature updates
- **OnlineIsotonicRegression**: Streaming monotonic calibration
- **OnlineConformalCalibration**: Adaptive coverage maintenance
- **DriftAwareCalibrationManager**: Coordinated calibration system

### 4. Production Integration

**File**: `integration_example.py`

Complete production-ready system with comprehensive examples:

- **ProductionUQSystem**: Main production class
- **Synthetic data generation**: Testing and validation utilities
- **Comprehensive demos**: End-to-end system demonstration
- **RK4 integration examples**: Direct ODE solver integration

## 📊 Key Metrics and Evaluation

### Conformal Prediction Metrics
- **Coverage**: Fraction of true values within prediction intervals
- **Interval Width**: Average width of prediction intervals
- **Coverage by Time**: Temporal coverage analysis for ODE solutions

### Drift Detection Metrics
- **Consensus Score**: Aggregated drift score across detectors
- **Alert Severity**: Low/Medium/High/Critical classification
- **Detection Rate**: Frequency of drift alerts over time

### Calibration Metrics
- **ECE/MCE**: Expected/Maximum Calibration Error
- **Brier Score**: Probabilistic prediction accuracy
- **Coverage Drift**: Changes in conformal coverage over time

## 🛠️ Implementation Details

### Conformity Score Functions

1. **Absolute Residual**: `|y - ŷ|`
2. **Adaptive Residual**: `|y - ŷ| / σ` (heteroscedastic)
3. **Squared Residual**: `(y - ŷ)²`
4. **Mahalanobis**: `(y - ŷ)ᵀ Σ⁻¹ (y - ŷ)`
5. **Trajectory-Aware**: Temporal weighting for ODE solutions

### Drift Detection Algorithms

1. **KS Test**: `D = sup_x |F₁(x) - F₂(x)|`
2. **MMD**: `||μ₁ - μ₂||²_H` in RKHS
3. **Adversarial**: Domain classifier AUC
4. **Energy Distance**: `2E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]`

### Online Update Rules

1. **Temperature**: `T ← T + η ∇_T NLL(T)`
2. **Conformal Quantile**: `q ← Q_{(n+1)(1-α)/n}(scores)`
3. **Isotonic**: Refit on sliding window
4. **Adaptive α**: `α ← α - γ(coverage - target)`

## 🎯 Production Deployment

### System Requirements
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Scikit-learn ≥ 1.0.0

### Performance Considerations
- **Memory**: Sliding windows for streaming data
- **Computation**: Parallel drift detection
- **Latency**: Online updates with configurable frequency
- **Storage**: Efficient calibration score storage

### Monitoring and Alerting
- Real-time drift detection alerts
- Calibration quality tracking
- Coverage monitoring
- System health metrics

## 📈 Performance Targets

Based on the UQ framework specification:

- **ECE ≤ 2-3%** in-domain calibration
- **Conformal coverage within ±1-2%** of nominal
- **OOD FPR@95% TPR** well below in-domain baseline
- **Lower expected cost and tail losses** vs. legacy systems

## 🔧 Configuration Examples

### Conservative Configuration (High Reliability)
```python
uq_system = ProductionUQSystem(
    model=model,
    alpha=0.05,  # 95% coverage
    calibration_methods=['temperature_scaling', 'conformal_prediction'],
    drift_window_size=2000,
    temp_learning_rate=0.005,
    conf_adaptive_alpha=False  # Fixed coverage level
)
```

### Adaptive Configuration (Dynamic Environments)
```python
uq_system = ProductionUQSystem(
    model=model,
    alpha=0.1,  # 90% coverage
    calibration_methods=['conformal_prediction'],
    drift_window_size=500,
    conf_adaptive_alpha=True,  # Adapt to observed coverage
    conf_update_frequency=25   # Frequent updates
)
```

### High-Throughput Configuration (Streaming)
```python
uq_system = ProductionUQSystem(
    model=model,
    alpha=0.1,
    calibration_methods=['conformal_prediction'],
    drift_window_size=1000,
    temp_update_frequency=200,  # Less frequent updates
    conf_update_frequency=100,
    forced_recalibration_interval=50000  # Periodic full recalibration
)
```

## 🧪 Testing and Validation

### Unit Tests
Each module includes comprehensive unit tests:
- Conformal prediction coverage validation
- Drift detection sensitivity analysis
- Online calibration convergence tests

### Integration Tests
- End-to-end system validation
- Streaming data processing
- Multi-component interaction testing

### Performance Benchmarks
- Latency measurements for real-time processing
- Memory usage profiling
- Accuracy validation on synthetic and real data

## 📚 References and Theory

### Conformal Prediction
- Distribution-free coverage guarantees
- Finite-sample validity
- Exchangeability assumptions
- Online conformal prediction theory

### Drift Detection
- Non-parametric statistical tests
- Kernel methods for multivariate detection
- Adversarial domain adaptation
- Energy statistics theory

### Online Calibration
- Temperature scaling theory
- Isotonic regression properties
- Adaptive learning rates
- Convergence guarantees

## 🎉 Getting Started

1. **Install dependencies**: Ensure PyTorch and required packages are installed
2. **Import modules**: `from hybrid_uq import ProductionUQSystem`
3. **Prepare data**: Training, calibration, and test datasets
4. **Initialize system**: Configure UQ system for your use case
5. **Fit and deploy**: Train on calibration data and start processing
6. **Monitor**: Track drift detection and calibration quality

For detailed examples, see `integration_example.py` and run:
```python
from hybrid_uq import run_comprehensive_demo
results = run_comprehensive_demo()
```

This will demonstrate the complete system with synthetic ODE data, showing:
- Model training and calibration
- Conformal prediction setup
- Drift monitoring activation
- Online calibration updates
- Streaming data processing
- Performance metrics and alerts

The framework is designed to be production-ready with minimal configuration while providing extensive customization options for advanced use cases.
