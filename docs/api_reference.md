# API Reference Documentation

**SPDX-License-Identifier: LicenseRef-Internal-Use-Only**

This document provides comprehensive documentation for all public APIs, functions, and components in the Farmer project codebase.

## Table of Contents

1. [Java Core APIs](#java-core-apis)
2. [Swift UOIF Framework](#swift-uoif-framework)
3. [Python Utilities](#python-utilities)
4. [Command Line Interface](#command-line-interface)
5. [Data Models](#data-models)
6. [Configuration](#configuration)

---

## Java Core APIs

### Core Entry Point

#### `qualia.Core`

Main entry point for the Java application with multiple operational modes.

**Usage:**
```bash
java -cp <classpath> qualia.Core <mode> [key=value ...]
```

**Available Modes:**

- `console` - Console audit sink demo
- `file` - File-based audit sink demo  
- `jdbc` - JDBC database audit sink demo
- `stein` - Stein estimator demo
- `hmc` - Hamiltonian Monte Carlo sampling
- `hmc_adapt` - Adaptive HMC with warmup
- `hmcmulti` - Multi-chain HMC runner
- `unified` - Unified detector triad gating
- `bifurc` - Bifurcation analysis
- `mcda` - Multi-criteria decision analysis
- `rmala` - Riemannian manifold Langevin sampling

**Examples:**

```bash
# HMC adaptive sampling
java -cp out-qualia qualia.Core hmc_adapt warmup=500 iters=1000 thin=2 seed=42

# Bifurcation analysis
java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.02

# Unified detector
java -cp out-qualia qualia.Core unified h=1e-3 eps=1e-4 triad=true
```

### Core Model Interface

#### `PsiModel`

Interface for Ψ-scoring models with Bayesian inference capabilities.

```java
public interface PsiModel {
    double calculatePsi(ClaimData claim, ModelParameters params);
    double logLikelihood(ClaimData claim, ModelParameters params);
    double totalLogLikelihood(List<ClaimData> dataset, ModelParameters params);
    double logPriors(ModelParameters params);
    double logPosterior(List<ClaimData> dataset, ModelParameters params);
    boolean shouldParallelize(int datasetSize);
    
    // Optional advanced API
    default List<ModelParameters> performInference(List<ClaimData> dataset, int sampleCount);
    default HmcSampler.AdaptiveResult hmcAdaptive(...);
}
```

**Key Methods:**

- `calculatePsi()` - Compute Ψ confidence score for a claim
- `logLikelihood()` - Log-likelihood of single observation
- `logPosterior()` - Log-posterior for Bayesian inference
- `performInference()` - Generate posterior samples
- `hmcAdaptive()` - Adaptive HMC sampling

### Hierarchical Bayesian Model

#### `HierarchicalBayesianModel`

Implementation of the core Ψ framework using hierarchical Bayesian inference.

**Constructor:**
```java
HierarchicalBayesianModel()                    // Default priors
HierarchicalBayesianModel(ModelPriors priors)  // Custom priors
HierarchicalBayesianModel(ModelPriors priors, int parallelThreshold)
```

**Core Methods:**

```java
// Ψ calculation
double calculatePsi(ClaimData claim, ModelParameters params)

// Likelihood evaluation  
double logLikelihood(ClaimData claim, ModelParameters params)
double totalLogLikelihood(List<ClaimData> dataset, ModelParameters params)

// Bayesian inference
List<ModelParameters> performInference(List<ClaimData> dataset, int sampleCount)

// Diagnostics
Diagnostics diagnose(List<List<ModelParameters>> chains)
```

**Model Structure:**
```
Ψ(x) = min{β·exp(-[λ₁Rₐ + λ₂Rᵥ])·[αS + (1-α)N], 1}
```

Where:
- `S` = Internal signal strength ∈ [0,1]
- `N` = Canonical evidence strength ∈ [0,1]  
- `α` = Evidence allocation parameter ∈ [0,1]
- `Rₐ` = Authority risk ∈ [0,∞)
- `Rᵥ` = Verifiability risk ∈ [0,∞)
- `λ₁,λ₂` = Risk penalty weights > 0
- `β` = Uplift factor ≥ 1

### HMC Sampler

#### `HmcSampler`

Hamiltonian Monte Carlo sampler for Bayesian inference over unconstrained parameter space.

**Constructor:**
```java
HmcSampler(HierarchicalBayesianModel model, List<ClaimData> dataset)
```

**Sampling Methods:**

```java
// Basic sampling
Result sample(int totalIters, int burnIn, int thin, long seed, 
              double[] z0, double stepSize, int leapfrogSteps)

// Adaptive sampling with tuning
AdaptiveResult sampleAdaptive(int warmupIters, int samplingIters, int thin,
                              long seed, double[] z0, double initStepSize, 
                              int leapfrogSteps, double targetAccept)
```

**Result Classes:**

```java
public static final class Result {
    public final List<ModelParameters> samples;
    public final double acceptanceRate;
}

public static final class AdaptiveResult {
    public final List<ModelParameters> samples;
    public final double acceptanceRate;
    public final double tunedStepSize;
    public final double[] massDiag;
    public final int divergenceCount;
}
```

**Usage Example:**
```java
HmcSampler hmc = new HmcSampler(model, dataset);
double[] z0 = {logit(0.7), logit(0.6), logit(0.5), Math.log(1.0)};

// Adaptive sampling
AdaptiveResult result = hmc.sampleAdaptive(
    1000, 2000, 3, 42L, z0, 0.01, 20, 0.75
);

System.out.println("Acceptance rate: " + result.acceptanceRate);
System.out.println("Tuned step size: " + result.tunedStepSize);
```

### Multi-Chain HMC Runner

#### `HmcMultiChainRunner`

Parallel multi-chain HMC execution with comprehensive diagnostics.

**Constructor:**
```java
HmcMultiChainRunner(HierarchicalBayesianModel model, List<ClaimData> dataset,
                    int chains, int burnIn, int samples, int thin, long seed,
                    double[] z0, double stepSize, int leapfrogSteps, 
                    double targetAccept, File outputDir)
```

**Execution:**
```java
HmcMultiChainRunner runner = new HmcMultiChainRunner(...);
Summary summary = runner.run();
```

**Summary Output:**
```json
{
  "chains": [
    {"kept": 1000, "acc": 0.75, "tunedStep": 0.012, "divergences": 0},
    {"kept": 1000, "acc": 0.73, "tunedStep": 0.011, "divergences": 0}
  ],
  "diagnostics": {
    "rhat": {"S": 1.02, "N": 1.01, "alpha": 1.03, "beta": 1.01},
    "ess": {"S": 850.2, "N": 920.1, "alpha": 780.5, "beta": 890.3}
  }
}
```

### Unified Detector

#### `UnifiedDetector`

Triad gating system combining RK4, Taylor series, and geometric invariants for adaptive ODE integration.

**Core Methods:**

```java
// Single step with confidence production
Result step(Dynamics f, double t, double[] y, double hInit, 
            double epsilonTotal, long timeBudgetNanos, Invariant[] invariants)

// Triad gating step
Triad triadStep(Dynamics f, double t, double[] y, double hInit, 
                double epsilonTotal, double epsRk4, double epsTaylor, 
                double epsGeom, long timeBudgetNanos, Invariant[] invariants)
```

**Interfaces:**

```java
interface Dynamics {
    void eval(double t, double[] y, double[] dy);
}

interface Invariant {
    double value(double t, double[] y);
    double reference();
    double tolerance();
}
```

**Result Classes:**

```java
static final class Result {
    final double tNext;
    final double[] yNext;
    final double psi;           // Overall confidence [0,1]
    final double errLocal;      // Local error estimate
    final double agreeMetric;   // Agreement score [0,1]
    final double hUsed;
}

static final class Triad {
    final double tNext;
    final double[] yNext;
    final double psi;
    final double hUsed;
    final double epsRk4;        // RK4 step-doubling error
    final double epsTaylor;     // Taylor remainder proxy
    final double epsGeom;       // Geometric invariant violation
    final double geomDrift;     // Raw geometric drift
    final boolean accepted;
}
```

**Usage Example:**
```java
UnifiedDetector detector = new UnifiedDetector();

// Simple harmonic oscillator
Dynamics sho = (t, y, dy) -> {
    dy[0] = y[1];
    dy[1] = -omega * omega * y[0];
};

// Energy invariant
Invariant energy = new Invariant() {
    @Override public double value(double t, double[] y) {
        return 0.5 * (y[1] * y[1] + omega * omega * y[0] * y[0]);
    }
    @Override public double reference() { return E0; }
    @Override public double tolerance() { return E0 * 1e-3; }
};

// Triad step
Triad result = detector.triadStep(sho, t, y, h, eps, epsRk4, epsTaylor, epsGeom, budget, new Invariant[]{energy});
```

### MCDA Framework

#### `Mcda`

Multi-criteria decision analysis with Ψ integration for confidence-aware decision making.

**Core Classes:**

```java
public enum Direction { BENEFIT, COST }

public record CriterionSpec(String name, Direction direction, double weight)

public static final class Alternative {
    public final String id;
    public final Map<String, Double> rawScores;
    public final double psi;  // Ψ confidence score
}

public record Ranked(Alternative alternative, double score)
```

**Key Methods:**

```java
// Ψ-based gating
static List<Alternative> gateByPsi(Collection<Alternative> alts, double tau)

// Normalization
static Map<Alternative, Map<String, Double>> normalize(Collection<Alternative> alts, List<CriterionSpec> specs)

// Ranking methods
static List<Ranked> rankByWSM(Collection<Alternative> alts, List<CriterionSpec> specs)
static List<Ranked> rankByTOPSIS(Collection<Alternative> alts, List<CriterionSpec> specs)
static List<Ranked> rankByWPM(Collection<Alternative> alts, List<CriterionSpec> specs)
```

**Usage Example:**
```java
// Define alternatives with Ψ scores
Alternative a = new Alternative("A", Map.of("cost", 100.0, "value", 0.8), 0.85);
Alternative b = new Alternative("B", Map.of("cost", 80.0, "value", 0.7), 0.80);

// Gate by confidence threshold
double tau = 0.79;
List<Alternative> feasible = Mcda.gateByPsi(List.of(a, b), tau);

// Define criteria including Ψ
List<CriterionSpec> specs = List.of(
    new CriterionSpec("psi", Direction.BENEFIT, 0.4),
    new CriterionSpec("value", Direction.BENEFIT, 0.4),
    new CriterionSpec("cost", Direction.COST, 0.2)
);

// Rank alternatives
List<Ranked> wsm = Mcda.rankByWSM(feasible, specs);
List<Ranked> topsis = Mcda.rankByTOPSIS(feasible, specs);
```

### Bifurcation Analysis

#### `BifurcationSweep`

Parameter space exploration for dynamical systems bifurcation detection.

**Available Systems:**

```java
// Logistic map bifurcation
static void runLogistic(double rMin, double rMax, double rStep, 
                       int horizon, int burnin, long seed, File out)

// Saddle-node bifurcation
static void runSaddleNode(double muMin, double muMax, double muStep, 
                         int steps, double h, File out)

// Hopf bifurcation
static void runHopf(double muMin, double muMax, double muStep, 
                   double omega, int steps, double h, File out)
```

**Usage:**
```bash
java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.5 rMax=3.6 rStep=0.01 horizon=2000 burnin=1000 seed=42 out=bifurc.jsonl
```

---

## Swift UOIF Framework

### Core Confidence Framework

#### `ConfidenceHeuristics`

High-level confidence assessment combining multiple evidence sources.

```swift
public enum ConfidenceHeuristics {
    public static func overall(
        sources: Double,      // Source reliability
        hybrid: Double,       // Hybrid model confidence
        penalty: Double,      // Risk penalty score
        posterior: Double     // Bayesian posterior
    ) -> Double
}
```

**Usage:**
```swift
let confidence = ConfidenceHeuristics.overall(
    sources: 0.95,
    hybrid: 0.88,
    penalty: 0.82,
    posterior: 0.90
)
// Returns weighted blend in [0,1]
```

#### `Presets`

Pre-configured evaluation scenarios for common use cases.

```swift
public enum Presets {
    // 2025 IMO results (canonical, eased)
    public static func eval2025Results(alpha: Double) -> Evaluation
    
    // 2025 IMO problems (pending canonical)
    public static func eval2025Problems(alpha: Double, N: Double) -> Evaluation
    
    // 2024 DeepMind results
    public static func eval2024(alpha: Double) -> Evaluation
    
    // Hybrid PINN example
    public static func evalHybridPINNExample() -> Evaluation
}
```

**Usage:**
```swift
let eval = Presets.eval2025Results(alpha: 0.6)
let confidence = eval.confidence.psiOverall
```

### Hierarchical Bayesian Model

#### `HierarchicalBayesianModel`

Swift implementation of the core Ψ framework.

**Core Methods:**

```swift
// Ψ calculation
func calculatePsi(for claim: ClaimData, with params: ModelParameters) -> Double

// Likelihood evaluation
func logLikelihood(for claim: ClaimData, with params: ModelParameters) -> Double
func totalLogLikelihood(for dataset: [ClaimData], with params: ModelParameters) -> Double

// Bayesian inference
func performInference(on dataset: [ClaimData], sampleCount: Int) -> [ModelParameters]
```

**Data Structures:**

```swift
struct ClaimData {
    let id: String
    let isVerifiedTrue: Bool
    let riskAuthenticity: Double      // R_a
    let riskVirality: Double         // R_v  
    let probabilityHgivenE: Double   // P(H|E)
}

struct ModelParameters {
    let S: Double       // S ~ Beta(a_S, b_S)
    let N: Double       // N ~ Beta(a_N, b_N)
    let alpha: Double   // alpha ~ Beta(a_alpha, b_alpha)
    let beta: Double    // beta ~ LogNormal(mu_beta, sigma_beta)
}

struct ModelPriors {
    // Beta priors for S, N, alpha
    let s_alpha: Double, s_beta: Double
    let n_alpha: Double, n_beta: Double
    let alpha_alpha: Double, alpha_beta: Double
    
    // LogNormal prior for beta
    let beta_mu: Double, beta_sigma: Double
    
    // Gamma priors for risk parameters
    let ra_shape: Double, ra_scale: Double
    let rv_shape: Double, rv_scale: Double
    
    // Penalty function hyperparameters
    let lambda1: Double, lambda2: Double
}
```

### Physics-Informed Neural Networks

#### `PINN`

Neural network architecture for solving partial differential equations.

**Core Classes:**

```swift
public final class DenseLayer {
    public var weights: [[Double]]  // [output][input]
    public var biases: [Double]     // [output]
    
    public init(inputSize: Int, outputSize: Int, activation: ActivationFunction)
    public func forward(_ input: [Double]) -> [Double]
}

public final class PINN {
    public var layers: [DenseLayer]
    
    public init(hiddenWidth: Int = 20)
    public func forward(x: Double, t: Double) -> Double
}
```

**Activation Functions:**

```swift
public enum ActivationFunction {
    case tanh
    case linear
    
    func apply(_ x: Double) -> Double
}
```

**PDE Residuals:**

```swift
public enum PDE {
    // Heat equation: u_t = u_xx
    public static func residual_heatEquation(model: PINN, x: Double, t: Double, 
                                           dx: Double, dt: Double) -> Double
    
    // Burgers equation: u_t + u u_x = ν u_xx
    public static func residual_burgersEquation(model: PINN, x: Double, t: Double,
                                              nu: Double, dx: Double, dt: Double) -> Double
}
```

**Derivative Estimation:**

```swift
// Central difference first derivative
public func finiteDifferenceFirst(_ f: (Double) -> Double, at x0: Double, dx: Double) -> Double

// Second derivative via three-point stencil  
public func finiteDifferenceSecond(_ f: (Double) -> Double, at x0: Double, dx: Double) -> Double
```

**Usage Example:**
```swift
let pinn = PINN(hiddenWidth: 30)

// Train on heat equation
for epoch in 0..<1000 {
    var totalLoss = 0.0
    
    for _ in 0..<100 {
        let x = Double.random(in: -1...1)
        let t = Double.random(in: 0...1)
        
        let residual = PDE.residual_heatEquation(model: pinn, x: x, t: t, dx: 1e-4, dt: 1e-4)
        totalLoss += residual * residual
    }
    
    // Update weights (gradient descent implementation needed)
    print("Epoch \(epoch): Loss = \(totalLoss)")
}
```

---

## Python Utilities

### Uncertainty Quantification Framework

#### `UncertaintyEstimate`

Container for comprehensive uncertainty estimates.

```python
@dataclass
class UncertaintyEstimate:
    mean: np.ndarray          # Point predictions
    aleatoric: np.ndarray     # Data noise (irreducible)
    epistemic: np.ndarray     # Model ignorance (reducible)
    total: np.ndarray         # Combined uncertainty
    
    @property
    def std_total(self) -> np.ndarray:
        """Total standard deviation."""
        
    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals."""
```

**Usage:**
```python
estimate = UncertaintyEstimate(
    mean=preds,
    aleatoric=aleatoric_var,
    epistemic=epistemic_var,
    total=total_var
)

lower, upper = estimate.confidence_interval(alpha=0.05)
print(f"95% CI: [{lower}, {upper}]")
```

#### `DeepEnsemble`

Ensemble-based epistemic uncertainty quantification.

```python
class DeepEnsemble:
    def __init__(self, model_class: Callable, n_models: int = 5, **model_kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs)
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyEstimate
```

**Usage:**
```python
ensemble = DeepEnsemble(
    model_class=RandomForestRegressor,
    n_models=10,
    n_estimators=100
)

ensemble.fit(X_train, y_train)
estimate = ensemble.predict_with_uncertainty(X_test)

print(f"Mean predictions: {estimate.mean}")
print(f"Epistemic uncertainty: {estimate.epistemic}")
```

#### `MCDropout`

Monte Carlo dropout for lightweight Bayesian approximation.

```python
class MCDropout:
    def __init__(self, model: nn.Module, n_samples: int = 100, dropout_rate: float = 0.1)
    
    def predict_with_uncertainty(self, X: torch.Tensor) -> UncertaintyEstimate
    def enable_dropout(self)
    def disable_dropout(self)
```

**Usage:**
```python
# PyTorch model with dropout layers
model = MyModel(dropout_rate=0.1)

mc_dropout = MCDropout(model, n_samples=100)
estimate = mc_dropout.predict_with_uncertainty(X_test)

print(f"Total uncertainty: {estimate.total}")
```

#### `ConformalPrediction`

Conformal prediction for distribution-free uncertainty quantification.

```python
class ConformalPrediction:
    def __init__(self, base_model, method: str = "naive", alpha: float = 0.05)
    
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray)
    def predict_with_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    def predict_with_quantiles(self, X: np.ndarray, quantiles: List[float]) -> np.ndarray
```

**Usage:**
```python
cp = ConformalPrediction(base_model, method="naive", alpha=0.1)
cp.fit(X_cal, y_cal)

lower, upper = cp.predict_with_intervals(X_test)
print(f"90% prediction intervals: [{lower}, {upper}]")
```

### Risk-Aware Decision Making

#### `RiskAwareClassifier`

Classifier that incorporates uncertainty into decision making.

```python
class RiskAwareClassifier:
    def __init__(self, base_model, uncertainty_threshold: float = 0.1)
    
    def fit(self, X: np.ndarray, y: np.ndarray)
    def predict_with_risk(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    def reject_uncertain(self, X: np.ndarray, threshold: float) -> np.ndarray
```

**Usage:**
```python
classifier = RiskAwareClassifier(base_model, uncertainty_threshold=0.15)
classifier.fit(X_train, y_train)

predictions, probabilities, uncertainties = classifier.predict_with_risk(X_test)
rejected = classifier.reject_uncertain(X_test, threshold=0.2)

print(f"Predictions: {predictions}")
print(f"Uncertainties: {uncertainties}")
print(f"Rejected samples: {np.sum(rejected)}")
```

#### `UncertaintyCalibration`

Calibration methods for uncertainty estimates.

```python
class UncertaintyCalibration:
    def __init__(self, method: str = "isotonic")
    
    def fit(self, uncertainties: np.ndarray, errors: np.ndarray)
    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray
    def evaluate_calibration(self, uncertainties: np.ndarray, errors: np.ndarray) -> Dict[str, float]
```

**Usage:**
```python
calibrator = UncertaintyCalibration(method="isotonic")
calibrator.fit(uncertainties_train, errors_train)

calibrated_uncertainties = calibrator.calibrate(uncertainties_test)
metrics = calibrator.evaluate_calibration(calibrated_uncertainties, errors_test)

print(f"Calibration metrics: {metrics}")
```

---

## Command Line Interface

### Java CLI

**Core Commands:**

```bash
# Basic demos
java -cp out-qualia qualia.Core console
java -cp out-qualia qualia.Core file
java -cp out-qualia qualia.Core jdbc

# Inference methods
java -cp out-qualia qualia.Core stein
java -cp out-qualia qualia.Core hmc
java -cp out-qualia qualia.Core rmala

# Advanced sampling
java -cp out-qualia qualia.Core hmc_adapt warmup=1000 iters=2000 thin=3
java -cp out-qualia qualia.Core hmcmulti chains=4 burnIn=500 samples=1000

# Specialized analysis
java -cp out-qualia qualia.Core unified h=1e-3 eps=1e-4 triad=true
java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.5 rMax=3.6
java -cp out-qualia qualia.Core mcda
```

**Environment Variables:**

```bash
# HMC parameters
export HMC_STEP_SIZE=0.01
export HMC_LEAP=30
export HMC_ADAPT=true
export HMC_WARMUP=1000
export HMC_ITERS=3000
export HMC_TARGET_ACC=0.75

# JDBC connection
export JDBC_URL=jdbc:postgresql://localhost:5432/qualia
export JDBC_USER=username
export JDBC_PASS=password

# Metrics server
export METRICS_ENABLE=true
```

### Swift CLI

**UOIF CLI Commands:**

```bash
# Build and run
swift build
swift run UOIFCLI

# Available commands (check help for full options)
swift run UOIFCLI --help
```

### Python Scripts

**Main UQ Scripts:**

```bash
# Quick start example
python scripts/python/uq_quickstart_example.py

# Simple UQ tests
python scripts/python/test_uq_simple.py

# Main UQ framework
python scripts/python/uncertainty_quantification.py

# Agentic workflow
python scripts/python/agentic_workflow.py
```

---

## Data Models

### Core Data Structures

#### `ClaimData`

Represents a single claim with features and verification outcome.

```java
public record ClaimData(
    String id,                    // Unique identifier
    boolean isVerifiedTrue,       // Observation y_i ∈ {0,1}
    double riskAuthenticity,      // R_a authority risk
    double riskVirality,          // R_v verifiability risk  
    double probabilityHgivenE     // P(H|E) base probability
)
```

#### `ModelParameters`

Single sample from the posterior distribution.

```java
public record ModelParameters(
    double S,       // Internal signal strength
    double N,       // Canonical evidence strength
    double alpha,   // Evidence allocation parameter
    double beta     // Uplift factor
)
```

#### `ModelPriors`

Hyperparameters for prior distributions.

```java
public record ModelPriors(
    double lambda1,     // Authority risk penalty weight
    double lambda2,     // Verifiability risk penalty weight
    // Additional hyperparameters for hierarchical structure
)
```

### JSON Output Formats

#### HMC Results

```json
{
  "config": {
    "warmup": 1000,
    "iters": 2000,
    "thin": 3,
    "leap": 20,
    "target": 0.75,
    "eps0": 0.05
  },
  "chains": [
    {
      "kept": 1000,
      "acc": 0.75,
      "tunedStep": 0.012,
      "divergences": 0
    }
  ],
  "summary": {
    "meanPsi1": 0.823456,
    "meanPsi2": 0.789123
  },
  "diagnostics": {
    "rhat": {
      "S": 1.02,
      "N": 1.01,
      "alpha": 1.03,
      "beta": 1.01
    },
    "ess": {
      "S": 850.2,
      "N": 920.1,
      "alpha": 780.5,
      "beta": 890.3
    }
  }
}
```

#### Unified Detector Results

```json
{
  "t": 0.001000,
  "x": 0.99950000,
  "v": -0.00050000,
  "psi": 0.987654,
  "h": 1.000000e-03,
  "eps_rk4": 1.234567e-06,
  "eps_taylor": 2.345678e-06,
  "eps_geom": 3.456789e-09,
  "geom_drift": 1.234567e-08,
  "accepted": true
}
```

---

## Configuration

### Build Configuration

#### Java Build

```bash
# Compile
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')

# Run tests
./scripts/test_qualia.sh

# CI pipeline
./scripts/ci.sh
```

#### Swift Build

```bash
# Build package
swift build

# Run tests
swift test

# Generate documentation
swift package generate-documentation
```

#### Python Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest scripts/python/

# Development setup
pip install -e .
```

### Environment Configuration

#### Metrics Server

```bash
# Enable metrics (default: true)
export METRICS_ENABLE=true

# Access metrics endpoint
curl http://localhost:8080/metrics
```

#### Database Configuration

```bash
# PostgreSQL connection
export JDBC_URL=jdbc:postgresql://localhost:5432/qualia
export JDBC_USER=username
export JDBC_PASS=password
```

#### HMC Tuning

```bash
# Step size and leapfrog steps
export HMC_STEP_SIZE=0.01
export HMC_LEAP=30

# Adaptive parameters
export HMC_ADAPT=true
export HMC_WARMUP=1000
export HMC_ITERS=3000
export HMC_TARGET_ACC=0.75
```

---

## Error Handling

### Common Error Patterns

#### HMC Divergences

```java
// Check for divergences in results
if (result.divergenceCount > 0) {
    System.err.println("Warning: " + result.divergenceCount + " divergences detected");
    // Consider reducing step size or increasing leapfrog steps
}
```

#### Numerical Stability

```java
// Clamp Ψ to [0,1] for numerical stability
double psi = Math.max(0.0, Math.min(1.0, rawPsi));

// Use log-space for small probabilities
double logProb = Math.log(Math.max(1e-9, Math.min(1.0 - 1e-9, prob)));
```

#### Memory Management

```java
// For large datasets, use parallel processing
if (model.shouldParallelize(dataset.size())) {
    // Parallel likelihood evaluation
}

// Close resources explicitly
try (PrintWriter out = new PrintWriter(file)) {
    // File operations
}
```

---

## Performance Considerations

### Optimization Strategies

#### Parallel Processing

```java
// Enable parallel likelihood evaluation for large datasets
HierarchicalBayesianModel model = new HierarchicalBayesianModel(priors, 2048);

// Use parallel streams for likelihood sums
if (model.shouldParallelize(dataset.size())) {
    // Parallel computation
}
```

#### Caching

```java
// Precompute expensive calculations
HierarchicalBayesianModel.Prepared prep = model.precompute(dataset);

// Use prepared data in samplers
HmcSampler sampler = new HmcSampler(model, dataset);
```

#### Memory Management

```java
// Thin samples to reduce memory usage
int thin = 3;
List<ModelParameters> thinned = samples.stream()
    .filter(sample -> samples.indexOf(sample) % thin == 0)
    .collect(Collectors.toList());
```

---

## Testing and Validation

### Test Suites

#### Java Tests

```bash
# Run all tests
./scripts/test_qualia.sh

# Individual test classes
java -cp out-qualia:test-classes qualia.HmcSamplerTest
java -cp out-qualia:test-classes qualia.HierarchicalBayesianModelTest
```

#### Swift Tests

```bash
# Run package tests
swift test

# Specific test target
swift test --filter UOIFCoreTests
```

#### Python Tests

```bash
# Run UQ tests
python scripts/python/test_uq_simple.py

# Run with pytest
python -m pytest scripts/python/ -v
```

### Validation Metrics

#### HMC Diagnostics

```java
// Gelman-Rubin diagnostic (R̂)
Diagnostics diag = model.diagnose(chains);
System.out.println("R̂ S: " + diag.rHatS);
System.out.println("R̂ N: " + diag.rHatN);

// Effective sample size
System.out.println("ESS S: " + diag.essS);
System.out.println("ESS N: " + diag.essN);
```

#### Uncertainty Calibration

```python
# Calibration metrics
metrics = calibrator.evaluate_calibration(uncertainties, errors)
print(f"Calibration error: {metrics['calibration_error']}")
print(f"Reliability: {metrics['reliability']}")
```

---

## Contributing

### Code Style

#### Java

- Follow Java 21+ conventions
- Use descriptive names (avoid 1-2 letter variables)
- Prefer immutable data structures and records
- Include comprehensive Javadoc for public APIs
- Add SPDX license headers to all files

#### Swift

- Follow Swift naming conventions (camelCase)
- Use structs over classes when possible
- Prefer protocols for abstraction
- Include comprehensive documentation comments
- Add SPDX license headers to all files

#### Python

- Follow PEP 8 style guidelines
- Use type hints for function parameters
- Include docstrings for all public functions
- Use dataclasses for data containers
- Add license headers to all files

### Testing Requirements

- Write unit tests for all public APIs
- Maintain test coverage above 80%
- Include integration tests for complex workflows
- Test edge cases and error conditions
- Keep CI pipeline green

### Documentation Standards

- Update this API reference when adding new public APIs
- Include usage examples for complex functionality
- Document configuration options and environment variables
- Maintain consistency with existing documentation style
- Update internal notation reference as needed

---

## License

This documentation is licensed under `LicenseRef-Internal-Use-Only`.

The codebase includes both public (GPL-3.0-only) and internal components. See [LICENSES/](LICENSES/) for full license details.