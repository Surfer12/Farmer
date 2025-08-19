# Comprehensive API Documentation

This document provides complete documentation for all public APIs, functions, and components in the project.

## Table of Contents

1. [Java APIs (Corpus/qualia)](#java-apis)
2. [Swift APIs (UOIFCore)](#swift-apis)
3. [Python APIs (Uncertainty Quantification)](#python-apis)
4. [Command Line Interfaces](#command-line-interfaces)
5. [Configuration and Setup](#configuration-and-setup)
6. [Examples and Usage Patterns](#examples-and-usage-patterns)

---

## Java APIs

The Java codebase provides the core Ψ framework implementation with Bayesian inference, HMC sampling, and MCDA integration.

### Core Entry Point

#### `qualia.Core`

Main CLI entry point for the Java application.

```java
public final class Core {
    public static void main(String[] args)
}
```

**Usage:**
```bash
# Compile
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')

# Run different modes
java -cp out-qualia qualia.Core help
java -cp out-qualia qualia.Core hmc_adapt chains=2 warmup=500 iters=1000
java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2
```

**Available Commands:**
- `console` - Console audit sink demo
- `file` - File audit sink demo  
- `jdbc` - JDBC audit sink demo
- `stein` - Stein estimation demo
- `hmc` - Basic HMC sampling
- `hmc_adapt` - Adaptive HMC with warmup
- `mcda` - Multi-criteria decision analysis
- `unified` - Unified detector with triad gating
- `bifurc` - Bifurcation analysis

### Ψ Model Interface

#### `PsiModel`

Core interface for Ψ framework implementations.

```java
public interface PsiModel {
    double calculatePsi(ClaimData claim, ModelParameters params);
    double logLikelihood(ClaimData claim, ModelParameters params);
    double totalLogLikelihood(List<ClaimData> dataset, ModelParameters params);
    double logPriors(ModelParameters params);
    double logPosterior(List<ClaimData> dataset, ModelParameters params);
    boolean shouldParallelize(int datasetSize);
}
```

**Key Methods:**

- `calculatePsi()` - Computes Ψ(x) ∈ [0,1] for given claim and parameters
- `logLikelihood()` - Bernoulli likelihood: y | Ψ ~ Bernoulli(Ψ)
- `logPosterior()` - Full log-posterior for Bayesian inference

### Hierarchical Bayesian Model

#### `HierarchicalBayesianModel`

Primary implementation of the Ψ framework with full Bayesian inference.

```java
public final class HierarchicalBayesianModel implements PsiModel {
    public HierarchicalBayesianModel()
    public HierarchicalBayesianModel(ModelPriors priors)
    public HierarchicalBayesianModel(ModelPriors priors, int parallelThreshold)
    
    public double calculatePsi(ClaimData claim, ModelParameters params)
    public boolean shouldParallelize(int datasetSize)
}
```

**Ψ Formula Implementation:**
```
Ψ(x) = min{β·exp(-[λ₁Rₐ + λ₂Rᵥ])·[αS + (1-α)N], 1}
```

**Example:**
```java
// Create model with default priors
HierarchicalBayesianModel model = new HierarchicalBayesianModel();

// Create claim data
ClaimData claim = new ClaimData("claim-1", true, 0.2, 0.1, 0.8);

// Create parameters
ModelParameters params = new ModelParameters(0.6, 0.9, 0.15, 1.2);

// Calculate Ψ
double psi = model.calculatePsi(claim, params);
System.out.println("Ψ = " + psi);
```

### HMC Sampling

#### `HmcSampler`

Hamiltonian Monte Carlo sampler for Bayesian inference.

```java
final class HmcSampler {
    public HmcSampler(HierarchicalBayesianModel model, List<ClaimData> dataset)
    
    public Result sample(int totalIters, int burnIn, int thin, long seed,
                        double[] z0, double stepSize, int leapfrogSteps)
    
    public AdaptiveResult sampleAdaptive(int warmupIters, int samplingIters,
                                       int thin, long seed, double[] z0,
                                       double initStepSize, int leapfrogSteps,
                                       double targetAccept)
}
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

**Example:**
```java
// Setup data and model
List<ClaimData> dataset = loadDataset();
HierarchicalBayesianModel model = new HierarchicalBayesianModel();
HmcSampler sampler = new HmcSampler(model, dataset);

// Run adaptive sampling
double[] initialParams = {0.0, 0.0, 0.0, 0.0}; // z-space
AdaptiveResult result = sampler.sampleAdaptive(
    500,    // warmup iterations
    1000,   // sampling iterations  
    2,      // thinning
    42L,    // seed
    initialParams,
    0.01,   // initial step size
    10,     // leapfrog steps
    0.8     // target acceptance rate
);

System.out.println("Acceptance rate: " + result.acceptanceRate);
System.out.println("Tuned step size: " + result.tunedStepSize);
```

### Multi-Chain HMC

#### `HmcMultiChainRunner`

Parallel multi-chain HMC with convergence diagnostics.

```java
public final class HmcMultiChainRunner {
    public static MultiChainResult runMultiChain(
        HierarchicalBayesianModel model,
        List<ClaimData> dataset,
        int numChains,
        int warmupIters,
        int samplingIters,
        int thin,
        long baseSeed
    )
}
```

**MultiChainResult:**
```java
public static final class MultiChainResult {
    public final List<List<ModelParameters>> chainSamples;
    public final double[] acceptanceRates;
    public final double[] rHat;           // Convergence diagnostic
    public final double[] effectiveSampleSize;
    public final int totalDivergences;
}
```

### Data Types

#### `ClaimData`

Represents a single claim/observation for Ψ evaluation.

```java
public record ClaimData(
    String id,                    // Unique identifier
    boolean isVerifiedTrue,       // Ground truth label
    double riskAuthenticity,      // Authority risk Rₐ
    double riskVirality,         // Verifiability risk Rᵥ  
    double probabilityHgivenE    // Base posterior P(H|E)
)
```

#### `ModelParameters`

Parameter sample from Bayesian inference.

```java
public record ModelParameters(
    double S,      // Internal signal strength
    double N,      // Canonical evidence strength  
    double alpha,  // Evidence allocation parameter
    double beta    // Uplift factor
)
```

#### `ModelPriors`

Hyperparameters for Bayesian priors.

```java
public record ModelPriors(
    double lambda1,    // Authority risk penalty weight
    double lambda2,    // Verifiability risk penalty weight
    // ... other hyperparameters
) {
    public static ModelPriors defaults()
}
```

### MCDA Integration

#### `PsiMcda`

Multi-criteria decision analysis with Ψ integration.

```java
public final class PsiMcda {
    public static Decision evaluate(
        double psiScore,
        double[] otherCriteria,
        double[] weights,
        double threshold
    )
    
    public static double aggregateWithPsi(
        double psiScore,
        double[] criteria,
        double[] weights
    )
}
```

### Audit Trail System

#### `AuditSink`

Interface for audit logging implementations.

```java
public interface AuditSink extends AutoCloseable {
    CompletableFuture<Void> write(AuditRecord record, AuditOptions options);
    CompletableFuture<Void> flush();
    void close();
}
```

**Implementations:**
- `ConsoleAuditSink` - Console output
- `FileAuditSink` - File-based logging with rotation
- `JdbcAuditSink` - Database persistence
- `HttpAuditSink` - HTTP endpoint logging

#### `AuditRecord`

Audit event data structure.

```java
public interface AuditRecord {
    String id();
    Date timestamp();
    String eventType();
    Map<String, Object> metadata();
}
```

---

## Swift APIs

The Swift codebase provides the UOIFCore framework for iOS and macOS applications.

### Ψ Model Computation

#### `PsiModel`

Core Ψ computation functions.

```swift
public enum PsiModel {
    public static func computeHybrid(alpha: Double, S: Double, N: Double) -> Double
    
    public static func computePenalty(
        lambdaAuthority: Double,
        lambdaVerifiability: Double,
        riskAuthority: Double,
        riskVerifiability: Double
    ) -> Double
    
    public static func computePosteriorCapped(basePosterior: Double, beta: Double) -> Double
    
    public static func computePsi(inputs: PsiInputs) -> PsiOutcome
    
    // Convenience method
    public static func computePsi(
        alpha: Double, S: Double, N: Double,
        R_cognitive: Double, R_efficiency: Double,
        lambda1: Double, lambda2: Double,
        basePosterior: Double, beta: Double
    ) -> PsiOutcome
}
```

**Example:**
```swift
import UOIFCore

// Create inputs
let inputs = PsiInputs(
    alpha: 0.15,
    S_symbolic: 0.60,
    N_external: 0.90,
    lambdaAuthority: 0.85,
    lambdaVerifiability: 0.15,
    riskAuthority: 0.12,
    riskVerifiability: 0.04,
    basePosterior: 0.90,
    betaUplift: 1.15
)

// Compute Ψ
let outcome = PsiModel.computePsi(inputs: inputs)
print("Ψ = \(outcome.psi)")
print("Hybrid = \(outcome.hybrid)")
print("Penalty = \(outcome.penalty)")
```

### Data Types

#### `PsiInputs`

Input parameters for Ψ computation.

```swift
public struct PsiInputs {
    public let alpha: Double                 // Evidence allocation
    public let S_symbolic: Double            // Internal signal
    public let N_external: Double            // Canonical evidence
    public let lambdaAuthority: Double       // Authority risk weight
    public let lambdaVerifiability: Double   // Verifiability risk weight
    public let riskAuthority: Double         // Authority risk
    public let riskVerifiability: Double     // Verifiability risk
    public let basePosterior: Double         // Base posterior P(H|E)
    public let betaUplift: Double           // Uplift factor
    
    public init(/* all parameters */)
}
```

#### `PsiOutcome`

Results of Ψ computation.

```swift
public struct PsiOutcome {
    public let hybrid: Double        // O(α) = αS + (1-α)N
    public let penalty: Double       // exp(-[λ₁Rₐ + λ₂Rᵥ])
    public let posterior: Double     // min(βP(H|E), 1)
    public let psi: Double          // Final Ψ score
    public let dPsi_dAlpha: Double  // Sensitivity ∂Ψ/∂α
}
```

#### `Evaluation`

Complete evaluation with metadata.

```swift
public struct Evaluation {
    public let title: String
    public let inputs: PsiInputs
    public let confidence: ConfidenceBundle
    public let label: String
}
```

### Confidence Assessment

#### `ConfidenceHeuristics`

Confidence scoring utilities.

```swift
public enum ConfidenceHeuristics {
    public static func overall(
        sources: Double,
        hybrid: Double, 
        penalty: Double,
        posterior: Double
    ) -> Double
}
```

#### `ConfidenceBundle`

Confidence scores for different components.

```swift
public struct ConfidenceBundle {
    public let sources: Double      // Source reliability
    public let hybrid: Double       // Hybrid computation confidence
    public let penalty: Double      // Risk penalty confidence
    public let posterior: Double    // Posterior confidence
    public let psiOverall: Double  // Overall Ψ confidence
}
```

### Presets and Examples

#### `Presets`

Pre-configured evaluation scenarios.

```swift
public enum Presets {
    // 2025 results with canonical evidence
    public static func eval2025Results(alpha: Double) -> Evaluation
    
    // 2025 problems without canonical evidence
    public static func eval2025Problems(alpha: Double, N: Double) -> Evaluation
    
    // 2024 historical evaluations
    public static func eval2024(alpha: Double) -> Evaluation
}
```

**Example:**
```swift
// Create evaluations with different α values
let eval1 = Presets.eval2025Results(alpha: 0.12)
let eval2 = Presets.eval2025Results(alpha: 0.15)

// Compute outcomes
let outcome1 = PsiModel.computePsi(inputs: eval1.inputs)
let outcome2 = PsiModel.computePsi(inputs: eval2.inputs)

print("α=0.12: Ψ=\(outcome1.psi)")
print("α=0.15: Ψ=\(outcome2.psi)")
```

### Classification Labels

#### `Label`

Evidence classification categories.

```swift
public enum Label: String {
    case interpretive = "Interpretive/Contextual"
    case empiricallyGrounded = "Empirically Grounded" 
    case primitiveEmpiricallyGrounded = "Primitive/Empirically Grounded"
}
```

**Thresholds:**
- **Interpretive/Contextual**: Ψ ≤ 0.70
- **Empirically Grounded**: Ψ > 0.70
- **Primitive/Empirically Grounded**: Ψ > 0.85 with canonical verification

---

## Python APIs

The Python codebase provides uncertainty quantification and risk analysis tools.

### Uncertainty Quantification Framework

#### `UncertaintyEstimate`

Container for uncertainty decomposition.

```python
@dataclass
class UncertaintyEstimate:
    mean: np.ndarray           # Predicted values
    aleatoric: np.ndarray      # Data noise (irreducible)
    epistemic: np.ndarray      # Model ignorance (reducible)
    total: np.ndarray          # Combined uncertainty
    
    @property
    def std_total(self) -> np.ndarray:
        """Total standard deviation."""
        return np.sqrt(self.total)
    
    @property  
    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals."""
```

**Example:**
```python
from uncertainty_quantification import UncertaintyEstimate
import numpy as np

# Create uncertainty estimate
uncertainty = UncertaintyEstimate(
    mean=np.array([2.5, 3.1, 1.8]),
    aleatoric=np.array([0.1, 0.15, 0.08]), 
    epistemic=np.array([0.2, 0.1, 0.3]),
    total=np.array([0.3, 0.25, 0.38])
)

print(f"Predictions: {uncertainty.mean}")
print(f"Total std: {uncertainty.std_total}")

# Get 95% confidence intervals
lower, upper = uncertainty.confidence_interval(alpha=0.05)
print(f"95% CI: [{lower}, {upper}]")
```

### Deep Ensemble

#### `DeepEnsemble`

Epistemic uncertainty via model ensemble.

```python
class DeepEnsemble:
    def __init__(self, model_class: Callable, n_models: int = 5, **model_kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyEstimate
```

**Example:**
```python
from sklearn.ensemble import RandomForestRegressor
from uncertainty_quantification import DeepEnsemble

# Create ensemble
ensemble = DeepEnsemble(RandomForestRegressor, n_models=10, n_estimators=100)

# Fit on training data
ensemble.fit(X_train, y_train)

# Get predictions with uncertainty
uncertainty_est = ensemble.predict_with_uncertainty(X_test)
print(f"Mean predictions: {uncertainty_est.mean}")
print(f"Epistemic uncertainty: {uncertainty_est.epistemic}")
```

### Monte Carlo Dropout

#### `MCDropout`

Lightweight Bayesian approximation via dropout.

```python
class MCDropout:
    def __init__(self, model: nn.Module, n_samples: int = 100, dropout_rate: float = 0.1)
    
    def predict_with_uncertainty(self, X: torch.Tensor) -> UncertaintyEstimate
    
    def enable_dropout_inference(self)
```

**Example:**
```python
import torch
import torch.nn as nn
from uncertainty_quantification import MCDropout

# Define model with dropout
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create MC Dropout wrapper
model = SimpleNet()
mc_dropout = MCDropout(model, n_samples=100)

# Get predictions with uncertainty
X_test = torch.randn(100, 10)
uncertainty_est = mc_dropout.predict_with_uncertainty(X_test)
```

### Conformal Prediction

#### `ConformalPredictor`

Distribution-free prediction intervals.

```python
class ConformalPredictor:
    def __init__(self, base_model, alpha: float = 0.1)
    
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray)
    
    def predict_intervals(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    
    def predict_sets(self, X_test: np.ndarray, y_candidates: np.ndarray) -> List[set]
```

**Example:**
```python
from sklearn.ensemble import RandomForestRegressor
from uncertainty_quantification import ConformalPredictor

# Create base model and conformal wrapper
base_model = RandomForestRegressor()
conformal = ConformalPredictor(base_model, alpha=0.1)  # 90% coverage

# Fit on calibration data
conformal.fit(X_cal, y_cal)

# Get prediction intervals
lower, upper = conformal.predict_intervals(X_test)
print(f"90% prediction intervals: [{lower}, {upper}]")
```

### Risk-Based Decision Framework

#### `RiskBasedDecisionFramework`

Convert predictions to actionable risk metrics.

```python
class RiskBasedDecisionFramework:
    @staticmethod
    def compute_var(samples: np.ndarray, confidence_level: float = 0.05) -> float
        """Value at Risk (VaR)."""
    
    @staticmethod
    def compute_cvar(samples: np.ndarray, confidence_level: float = 0.05) -> float
        """Conditional Value at Risk (CVaR)."""
    
    @staticmethod
    def tail_probability(samples: np.ndarray, threshold: float) -> float
        """P(loss > threshold)."""
    
    def make_risk_based_decision(self, 
                               uncertainty_est: UncertaintyEstimate,
                               risk_thresholds: Dict[str, float],
                               action_costs: Dict[str, float]) -> Dict[str, Any]
```

**Example:**
```python
from uncertainty_quantification import RiskBasedDecisionFramework
import numpy as np

framework = RiskBasedDecisionFramework()

# Sample from predictive distribution
samples = np.random.normal(uncertainty_est.mean, uncertainty_est.std_total, 
                          size=(len(uncertainty_est.mean), 1000))

# Compute risk metrics
for i, sample in enumerate(samples):
    var_95 = framework.compute_var(sample, 0.05)
    cvar_95 = framework.compute_cvar(sample, 0.05) 
    tail_prob = framework.tail_probability(sample, threshold=2.0)
    
    print(f"Sample {i}: VaR_95={var_95:.3f}, CVaR_95={cvar_95:.3f}, P(>2.0)={tail_prob:.3f}")
```

### Calibration Methods

#### `CalibrationMethods`

Model calibration techniques.

```python
class CalibrationMethods:
    @staticmethod
    def temperature_scaling(logits: np.ndarray, 
                          labels: np.ndarray) -> Tuple[float, Callable]
    
    @staticmethod
    def platt_scaling(scores: np.ndarray, 
                     labels: np.ndarray) -> Tuple[Callable, float]
    
    @staticmethod
    def isotonic_regression(scores: np.ndarray,
                          labels: np.ndarray) -> Tuple[Callable, float]
```

### Quick-Start Pipeline

#### `QuickStartUQPipeline`

Complete UQ workflow implementation.

```python
class QuickStartUQPipeline:
    def __init__(self, model_class=RandomForestRegressor, n_ensemble=5)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_cal: np.ndarray, y_cal: np.ndarray)
    
    def predict_with_risk_analysis(self, X_test: np.ndarray,
                                 risk_thresholds: list = None) -> Dict[str, Any]
    
    def generate_report(self, results: Dict[str, Any]) -> str
```

**Complete Example:**
```python
from uncertainty_quantification import QuickStartUQPipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train pipeline
pipeline = QuickStartUQPipeline(n_ensemble=5)
pipeline.fit(X_train, y_train, X_cal, y_cal)

# Get predictions with risk analysis
results = pipeline.predict_with_risk_analysis(X_test, risk_thresholds=[1.0, 2.0, 3.0])

# Generate report
report = pipeline.generate_report(results)
print(report)
```

---

## Command Line Interfaces

### Java CLI

The Java CLI provides access to all core functionality:

```bash
# Basic usage
java -cp out-qualia qualia.Core <command> [options]

# Available commands:
java -cp out-qualia qualia.Core help                    # Show help
java -cp out-qualia qualia.Core console                 # Console demo
java -cp out-qualia qualia.Core hmc                     # Basic HMC
java -cp out-qualia qualia.Core hmc_adapt chains=2 warmup=500 iters=1000 thin=2 seed=42 out=hmc.jsonl
java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.02 horizon=1000 burnin=500 seed=42 out=bifurc.jsonl
java -cp out-qualia qualia.Core unified                 # Unified detector
```

### Swift CLI

The Swift CLI provides Ψ evaluation and PINN integration:

```bash
# Build the CLI
swift build

# Basic usage
.build/debug/uoif-cli                                    # Default evaluations
.build/debug/uoif-cli pinn-demo                         # PINN demo
.build/debug/uoif-cli pinn-demo --alpha=0.3            # Custom alpha
```

### Python Scripts

Direct execution of Python UQ tools:

```bash
# Quick start example
python scripts/python/uq_quickstart_example.py

# Comprehensive uncertainty quantification
python scripts/python/uncertainty_quantification.py

# Simple UQ tests
python scripts/python/test_uq_simple.py
```

---

## Configuration and Setup

### Environment Variables

#### Java Configuration

```bash
# Metrics server (default: enabled)
export METRICS_ENABLE=1

# JDBC configuration for database audit sink
export JDBC_URL="jdbc:postgresql://localhost:5432/qualia"
export JDBC_USER="username"
export JDBC_PASS="password"
```

#### Python Dependencies

Install required packages:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Key dependencies:
pip install torch numpy scikit-learn scipy matplotlib
```

### Build Configuration

#### Java Build

```bash
# Compile all Java sources
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')

# Run tests
bash scripts/test_qualia.sh

# Full CI
bash scripts/ci.sh
```

#### Swift Build

```bash
# Build Swift package
swift build

# Run tests
swift test

# Build for release
swift build -c release
```

### Model Configuration

#### Ψ Framework Parameters

**Recommended Configuration:**
```java
ModelPriors priors = new ModelPriors(
    0.85,  // lambda1 (authority risk weight)
    0.15,  // lambda2 (verifiability risk weight)
    // ... other hyperparameters
);
```

**Prior Distributions:**
- `α ~ Beta(1,1)` - Uniform prior on evidence allocation
- `λ₁,λ₂ ~ Gamma(2,1)` - Weakly informative risk penalties  
- `β ~ Gamma(2,1)` - Weakly informative uplift factor

#### Validation Thresholds

- **Empirically Grounded**: Ψ > 0.70
- **Interpretive/Contextual**: Ψ ≤ 0.70  
- **Primitive**: Ψ > 0.85 with canonical verification

---

## Examples and Usage Patterns

### End-to-End Ψ Evaluation

#### Java Implementation

```java
import java.util.List;
import java.util.ArrayList;

public class PsiEvaluationExample {
    public static void main(String[] args) {
        // 1. Create model with default priors
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        
        // 2. Prepare dataset
        List<ClaimData> dataset = new ArrayList<>();
        dataset.add(new ClaimData("claim-1", true, 0.15, 0.05, 0.85));
        dataset.add(new ClaimData("claim-2", false, 0.25, 0.10, 0.60));
        dataset.add(new ClaimData("claim-3", true, 0.10, 0.03, 0.90));
        
        // 3. Run HMC inference
        HmcSampler sampler = new HmcSampler(model, dataset);
        double[] z0 = {0.0, 0.0, 0.0, 0.0}; // Initial parameters in z-space
        
        HmcSampler.AdaptiveResult result = sampler.sampleAdaptive(
            500,   // warmup
            1000,  // sampling
            2,     // thin
            42L,   // seed
            z0,
            0.01,  // step size
            10,    // leapfrog steps
            0.8    // target acceptance
        );
        
        // 4. Analyze results
        System.out.println("Acceptance rate: " + result.acceptanceRate);
        System.out.println("Final step size: " + result.tunedStepSize);
        System.out.println("Samples collected: " + result.samples.size());
        
        // 5. Compute posterior predictive Ψ scores
        ClaimData newClaim = new ClaimData("new-claim", true, 0.12, 0.04, 0.88);
        List<Double> psiScores = new ArrayList<>();
        
        for (ModelParameters params : result.samples) {
            double psi = model.calculatePsi(newClaim, params);
            psiScores.add(psi);
        }
        
        double meanPsi = psiScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        System.out.println("Posterior mean Ψ: " + meanPsi);
        
        // 6. Classification
        if (meanPsi > 0.85) {
            System.out.println("Classification: Primitive/Empirically Grounded");
        } else if (meanPsi > 0.70) {
            System.out.println("Classification: Empirically Grounded");
        } else {
            System.out.println("Classification: Interpretive/Contextual");
        }
    }
}
```

#### Swift Implementation

```swift
import UOIFCore

// Create evaluation inputs
let inputs = PsiInputs(
    alpha: 0.15,
    S_symbolic: 0.60,
    N_external: 0.90,
    lambdaAuthority: 0.85,
    lambdaVerifiability: 0.15,
    riskAuthority: 0.12,
    riskVerifiability: 0.04,
    basePosterior: 0.90,
    betaUplift: 1.15
)

// Compute Ψ outcome
let outcome = PsiModel.computePsi(inputs: inputs)

// Create confidence bundle
let confidence = ConfidenceBundle(
    sources: 0.95,
    hybrid: 0.92,
    penalty: 0.88,
    posterior: 0.90,
    psiOverall: 0.91
)

// Overall confidence
let overallConf = ConfidenceHeuristics.overall(
    sources: confidence.sources,
    hybrid: confidence.hybrid,
    penalty: confidence.penalty,
    posterior: confidence.posterior
)

// Print results
print("=== Ψ Evaluation Results ===")
print("Input Parameters:")
print("  α = \(inputs.alpha)")
print("  S = \(inputs.S_symbolic)")
print("  N = \(inputs.N_external)")
print("  λ₁ = \(inputs.lambdaAuthority)")
print("  λ₂ = \(inputs.lambdaVerifiability)")
print("  Rₐ = \(inputs.riskAuthority)")
print("  Rᵥ = \(inputs.riskVerifiability)")

print("\nComputation Results:")
print("  Hybrid O(α) = \(String(format: "%.4f", outcome.hybrid))")
print("  Penalty = \(String(format: "%.4f", outcome.penalty))")
print("  Posterior = \(String(format: "%.4f", outcome.posterior))")
print("  Ψ = \(String(format: "%.3f", outcome.psi))")
print("  ∂Ψ/∂α = \(String(format: "%.4f", outcome.dPsi_dAlpha))")

print("\nConfidence Assessment:")
print("  Sources: \(confidence.sources)")
print("  Hybrid: \(confidence.hybrid)")
print("  Penalty: \(confidence.penalty)")
print("  Posterior: \(confidence.posterior)")
print("  Overall: \(String(format: "%.2f", overallConf))")

// Classification
let label: String
if outcome.psi > 0.85 {
    label = Label.primitiveEmpiricallyGrounded.rawValue
} else if outcome.psi > 0.70 {
    label = Label.empiricallyGrounded.rawValue
} else {
    label = Label.interpretive.rawValue
}
print("\nClassification: \(label)")
```

### Uncertainty Quantification Pipeline

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from uncertainty_quantification import QuickStartUQPipeline

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train UQ pipeline
pipeline = QuickStartUQPipeline(
    model_class=RandomForestRegressor,
    n_ensemble=10
)

print("Training UQ Pipeline...")
pipeline.fit(X_train, y_train, X_cal, y_cal)

# Predict with uncertainty and risk analysis
print("Generating predictions with risk analysis...")
results = pipeline.predict_with_risk_analysis(
    X_test, 
    risk_thresholds=[1.0, 2.0, 3.0]
)

# Extract results
predictions = results['predictions']
uncertainty_est = results['uncertainty']
risk_metrics = results['risk_metrics']
conformal_intervals = results['conformal_intervals']

print(f"\nResults Summary:")
print(f"Predictions shape: {predictions.shape}")
print(f"Mean prediction: {np.mean(predictions):.3f}")
print(f"Mean total uncertainty: {np.mean(uncertainty_est.std_total):.3f}")
print(f"Mean epistemic uncertainty: {np.mean(np.sqrt(uncertainty_est.epistemic)):.3f}")

# Risk analysis
print(f"\nRisk Analysis:")
for threshold, metrics in risk_metrics.items():
    print(f"Threshold {threshold}:")
    print(f"  VaR (95%): {metrics['var_95']:.3f}")
    print(f"  CVaR (95%): {metrics['cvar_95']:.3f}")
    print(f"  Tail probability: {metrics['tail_prob']:.3f}")

# Conformal intervals
coverage = np.mean((y_test >= conformal_intervals[0]) & 
                   (y_test <= conformal_intervals[1]))
print(f"\nConformal Prediction:")
print(f"Empirical coverage: {coverage:.3f}")
print(f"Target coverage: 0.90")

# Generate detailed report
report = pipeline.generate_report(results)
print(f"\n{report}")
```

### Multi-Chain HMC Analysis

```java
import java.util.List;
import java.util.ArrayList;

public class MultiChainHMCExample {
    public static void main(String[] args) {
        // Setup
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        List<ClaimData> dataset = generateSyntheticDataset(100);
        
        // Run multi-chain HMC
        HmcMultiChainRunner.MultiChainResult result = 
            HmcMultiChainRunner.runMultiChain(
                model,
                dataset,
                4,      // 4 chains
                1000,   // warmup
                2000,   // sampling
                2,      // thin
                42L     // base seed
            );
        
        // Convergence diagnostics
        System.out.println("=== Multi-Chain HMC Results ===");
        System.out.println("Number of chains: " + result.chainSamples.size());
        
        for (int i = 0; i < result.acceptanceRates.length; i++) {
            System.out.printf("Chain %d: acceptance=%.3f, R-hat=%.3f, ESS=%.1f%n",
                i, result.acceptanceRates[i], result.rHat[i], result.effectiveSampleSize[i]);
        }
        
        System.out.println("Total divergences: " + result.totalDivergences);
        
        // Check convergence (R-hat < 1.1)
        boolean converged = true;
        for (double rhat : result.rHat) {
            if (rhat > 1.1) {
                converged = false;
                break;
            }
        }
        
        if (converged) {
            System.out.println("✅ Chains have converged (R-hat < 1.1)");
        } else {
            System.out.println("⚠️  Chains may not have converged (R-hat > 1.1)");
        }
        
        // Combine samples from all chains
        List<ModelParameters> allSamples = new ArrayList<>();
        for (List<ModelParameters> chainSamples : result.chainSamples) {
            allSamples.addAll(chainSamples);
        }
        
        System.out.println("Total samples: " + allSamples.size());
        
        // Posterior statistics
        double meanAlpha = allSamples.stream()
            .mapToDouble(ModelParameters::alpha)
            .average().orElse(0.0);
        
        double meanBeta = allSamples.stream()
            .mapToDouble(ModelParameters::beta)
            .average().orElse(0.0);
        
        System.out.printf("Posterior means: α=%.3f, β=%.3f%n", meanAlpha, meanBeta);
    }
    
    private static List<ClaimData> generateSyntheticDataset(int n) {
        List<ClaimData> dataset = new ArrayList<>();
        java.util.Random rng = new java.util.Random(42);
        
        for (int i = 0; i < n; i++) {
            String id = "claim-" + i;
            boolean verified = rng.nextBoolean();
            double riskAuth = Math.abs(rng.nextGaussian()) * 0.2;
            double riskViral = Math.abs(rng.nextGaussian()) * 0.1;
            double probHE = 0.5 + 0.3 * rng.nextGaussian();
            probHE = Math.max(0.1, Math.min(0.9, probHE));
            
            dataset.add(new ClaimData(id, verified, riskAuth, riskViral, probHE));
        }
        
        return dataset;
    }
}
```

### Integration with External Systems

#### REST API Integration

```java
// Example HTTP audit sink usage
public class HttpIntegrationExample {
    public static void main(String[] args) {
        // Configure HTTP audit sink
        String endpoint = "https://api.example.com/audit";
        Map<String, String> headers = Map.of(
            "Authorization", "Bearer " + System.getenv("API_TOKEN"),
            "Content-Type", "application/json"
        );
        
        HttpAuditSink sink = new HttpAuditSink(endpoint, headers, 30000);
        
        // Create audit record
        AuditRecord record = new AuditRecordImpl(
            "psi-eval-" + System.currentTimeMillis(),
            new Date()
        );
        
        // Add metadata
        if (record instanceof ExtendedAuditRecord extended) {
            extended.addMetadata("psi_score", 0.75);
            extended.addMetadata("classification", "Empirically Grounded");
            extended.addMetadata("model_version", "v1.2.3");
        }
        
        // Write to HTTP endpoint
        AuditOptions options = AuditOptions.builder()
            .idempotencyKey(record.id())
            .dryRun(false)
            .build();
        
        sink.write(record, options)
            .thenRun(() -> System.out.println("Audit record sent successfully"))
            .exceptionally(throwable -> {
                System.err.println("Failed to send audit record: " + throwable.getMessage());
                return null;
            });
        
        sink.close();
    }
}
```

This comprehensive API documentation covers all public interfaces, provides detailed examples, and includes usage patterns for integrating the Ψ framework, uncertainty quantification, and related tools into production systems.