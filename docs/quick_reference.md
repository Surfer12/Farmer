# Quick Reference Guide

**SPDX-License-Identifier: LicenseRef-Internal-Use-Only**

Quick reference for common operations and frequently used APIs in the Farmer project.

## Table of Contents

1. [Common Commands](#common-commands)
2. [Quick Examples](#quick-examples)
3. [Configuration Snippets](#configuration-snippets)
4. [Troubleshooting](#troubleshooting)

---

## Common Commands

### Java Core Operations

```bash
# Build
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')

# Run help
java -cp out-qualia qualia.Core help

# HMC sampling (adaptive)
java -cp out-qualia qualia.Core hmc_adapt warmup=500 iters=1000 thin=2 seed=42

# Multi-chain HMC
java -cp out-qualia qualia.Core hmcmulti chains=2 burnIn=100 samples=100 thin=3

# Unified detector
java -cp out-qualia qualia.Core unified h=1e-3 eps=1e-4 triad=true

# Bifurcation analysis
java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.02

# MCDA demo
java -cp out-qualia qualia.Core mcda
```

### Swift Operations

```bash
# Build
swift build

# Test
swift test

# Run CLI
swift run UOIFCLI

# Generate docs
swift package generate-documentation
```

### Python Operations

```bash
# Install dependencies
pip install -r requirements.txt

# Run UQ examples
python scripts/python/uq_quickstart_example.py

# Run tests
python scripts/python/test_uq_simple.py
```

---

## Quick Examples

### Basic Ψ Calculation (Java)

```java
// Create model with default priors
HierarchicalBayesianModel model = new HierarchicalBayesianModel();

// Create claim data
ClaimData claim = new ClaimData("test", true, 0.1, 0.05, 0.8);

// Create parameters
ModelParameters params = new ModelParameters(0.7, 0.8, 0.6, 1.1);

// Calculate Ψ
double psi = model.calculatePsi(claim, params);
System.out.println("Ψ = " + psi);
```

### HMC Sampling (Java)

```java
// Setup
HierarchicalBayesianModel model = new HierarchicalBayesianModel();
List<ClaimData> dataset = createDataset();
HmcSampler hmc = new HmcSampler(model, dataset);

// Initial parameters (unconstrained space)
double[] z0 = {logit(0.7), logit(0.6), logit(0.5), Math.log(1.0)};

// Adaptive sampling
AdaptiveResult result = hmc.sampleAdaptive(
    1000, 2000, 3, 42L, z0, 0.01, 20, 0.75
);

System.out.println("Acceptance rate: " + result.acceptanceRate);
System.out.println("Samples: " + result.samples.size());
```

### Swift Confidence Assessment

```swift
// Quick confidence calculation
let confidence = ConfidenceHeuristics.overall(
    sources: 0.95,
    hybrid: 0.88,
    penalty: 0.82,
    posterior: 0.90
)

// Use preset evaluation
let eval = Presets.eval2025Results(alpha: 0.6)
let psiScore = eval.confidence.psiOverall
```

### Python Uncertainty Quantification

```python
# Deep ensemble for uncertainty
ensemble = DeepEnsemble(
    model_class=RandomForestRegressor,
    n_models=10,
    n_estimators=100
)

ensemble.fit(X_train, y_train)
estimate = ensemble.predict_with_uncertainty(X_test)

print(f"Mean: {estimate.mean}")
print(f"Epistemic uncertainty: {estimate.epistemic}")
print(f"Total uncertainty: {estimate.total}")
```

---

## Configuration Snippets

### Environment Variables

```bash
# HMC tuning
export HMC_STEP_SIZE=0.01
export HMC_LEAP=30
export HMC_ADAPT=true
export HMC_WARMUP=1000
export HMC_ITERS=3000
export HMC_TARGET_ACC=0.75

# Database
export JDBC_URL=jdbc:postgresql://localhost:5432/qualia
export JDBC_USER=username
export JDBC_PASS=password

# Metrics
export METRICS_ENABLE=true
```

### Java Model Configuration

```java
// Custom priors
ModelPriors priors = new ModelPriors(
    0.85,  // lambda1 (authority risk weight)
    0.15   // lambda2 (verifiability risk weight)
);

// Model with parallelization threshold
HierarchicalBayesianModel model = new HierarchicalBayesianModel(priors, 1024);

// Service locator configuration
ServiceLocator sl = ServiceLocator.builder()
    .fromEnvironment()
    .build();
```

### Swift Model Setup

```swift
// Custom priors
let priors = ModelPriors(
    lambda1: 0.85,
    lambda2: 0.15,
    s_alpha: 1.0, s_beta: 1.0,
    n_alpha: 1.0, n_beta: 1.0,
    alpha_alpha: 1.0, alpha_beta: 1.0,
    beta_mu: 0.0, beta_sigma: 1.0
)

// Model initialization
let model = HierarchicalBayesianModel(priors: priors)
```

### Python UQ Configuration

```python
# MCDropout configuration
mc_dropout = MCDropout(
    model=model,
    n_samples=100,
    dropout_rate=0.1
)

# Conformal prediction
cp = ConformalPrediction(
    base_model=model,
    method="naive",
    alpha=0.05
)

# Risk-aware classifier
risk_classifier = RiskAwareClassifier(
    base_model=model,
    uncertainty_threshold=0.15
)
```

---

## Troubleshooting

### Common Issues

#### HMC Divergences

```bash
# Reduce step size
export HMC_STEP_SIZE=0.005

# Increase leapfrog steps
export HMC_LEAP=50

# Check diagnostics
java -cp out-qualia qualia.Core hmc_adapt warmup=2000 iters=1000
```

#### Memory Issues

```java
// Reduce sample count
int sampleCount = 500;  // Instead of 2000

// Increase thinning
int thin = 5;  // Instead of 2

// Use parallel processing for large datasets
if (model.shouldParallelize(dataset.size())) {
    // Parallel computation enabled
}
```

#### Numerical Stability

```java
// Clamp Ψ to [0,1]
double psi = Math.max(0.0, Math.min(1.0, rawPsi));

// Use log-space for small probabilities
double logProb = Math.log(Math.max(1e-9, Math.min(1.0 - 1e-9, prob)));
```

### Performance Tips

```bash
# Enable parallel processing for large datasets
# Model automatically detects when dataset size >= parallelThreshold

# Use appropriate thinning
# thin=2 for exploration, thin=5+ for final analysis

# Monitor acceptance rates
# Target: 0.65-0.85 for HMC
# Below 0.5: reduce step size
# Above 0.9: increase step size
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with smaller datasets for testing
# Use synthetic data with known properties

# Check metrics endpoint
curl http://localhost:8080/metrics
```

---

## File Locations

### Key Directories

```
Corpus/qualia/           # Java core implementation
Sources/UOIFCore/        # Swift framework
Sources/UOIFCLI/         # Swift CLI
scripts/python/          # Python utilities
docs/                    # Documentation
internal/                # Internal notes and status
```

### Important Files

```
Corpus/qualia/Core.java                    # Main entry point
Corpus/qualia/HierarchicalBayesianModel.java # Core Ψ model
Corpus/qualia/HmcSampler.java              # HMC implementation
Corpus/qualia/UnifiedDetector.java         # Triad gating
Sources/UOIFCore/Confidence.swift          # Swift confidence
Sources/UOIFCore/PINN.swift                # Physics-informed NN
scripts/python/uncertainty_quantification.py # Python UQ framework
```

### Output Files

```
out-qualia/              # Compiled Java classes
hmc-out/                 # HMC output files
data/logs/               # JSONL log files
*.jsonl                  # Results and logs
```

---

## Validation Commands

### HMC Diagnostics

```bash
# Check convergence
java -cp out-qualia qualia.Core hmc_adapt warmup=1000 iters=2000 thin=3

# Look for:
# - R̂ < 1.1 (Gelman-Rubin diagnostic)
# - ESS > 100 (Effective sample size)
# - Acceptance rate: 0.65-0.85
# - Divergence count: 0
```

### Model Validation

```java
// Check Ψ bounds
double psi = model.calculatePsi(claim, params);
assert psi >= 0.0 && psi <= 1.0;

// Verify likelihood
double logLik = model.logLikelihood(claim, params);
assert !Double.isNaN(logLik) && !Double.isInfinite(logLik);
```

### Swift Validation

```swift
// Validate confidence bounds
let confidence = ConfidenceHeuristics.overall(...)
assert(confidence >= 0.0 && confidence <= 1.0)

// Check PINN forward pass
let output = pinn.forward(x: 0.5, t: 0.1)
assert(!output.isNaN && !output.isInfinite)
```

### Python Validation

```python
# Check uncertainty estimates
estimate = ensemble.predict_with_uncertainty(X_test)
assert np.all(estimate.total >= 0)  # Non-negative variance
assert np.all(estimate.epistemic >= 0)
assert np.all(estimate.aleatoric >= 0)

# Validate calibration
metrics = calibrator.evaluate_calibration(uncertainties, errors)
assert metrics['calibration_error'] < 0.1
```

---

## Quick Start Workflows

### 1. Basic Ψ Analysis

```bash
# Build
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')

# Run HMC sampling
java -cp out-qualia qualia.Core hmc_adapt warmup=500 iters=1000 thin=2 seed=42

# Check results
tail -n 1 hmc-out/hmc_adapt.jsonl | jq '.'
```

### 2. Swift Confidence Assessment

```bash
# Build Swift package
swift build

# Run tests
swift test

# Use in your code
import UOIFCore
let confidence = ConfidenceHeuristics.overall(...)
```

### 3. Python UQ Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick start
python scripts/python/uq_quickstart_example.py

# Use framework
python -c "
from scripts.python.uncertainty_quantification import DeepEnsemble
ensemble = DeepEnsemble(RandomForestRegressor, n_models=5)
# ... your code here
"
```

### 4. Full Analysis Pipeline

```bash
# 1. Build everything
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')
swift build

# 2. Run HMC sampling
java -cp out-qualia qualia.Core hmc_adapt warmup=1000 iters=2000 thin=3 seed=42

# 3. Run unified detector
java -cp out-qualia qualia.Core unified h=1e-3 eps=1e-4 triad=true

# 4. Analyze results
python scripts/python/analyze_results.py hmc-out/ data/logs/
```

---

## License

This quick reference is licensed under `LicenseRef-Internal-Use-Only`.

For full API documentation, see [docs/api_reference.md](api_reference.md).