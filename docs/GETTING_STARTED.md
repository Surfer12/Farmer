# Getting Started Guide

This guide will help you get up and running with the Ψ framework, uncertainty quantification tools, and related APIs.

## Quick Start

### Prerequisites

- **Java 21** or later
- **Swift 5.9** or later (for macOS/iOS development)
- **Python 3.8** or later
- **Git** for cloning the repository

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Compile Java code:**
   ```bash
   javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')
   ```

4. **Build Swift package:**
   ```bash
   swift build
   ```

### Your First Ψ Evaluation

#### Using Java

```java
import java.util.List;

public class FirstPsiExample {
    public static void main(String[] args) {
        // Create model
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        
        // Create a claim
        ClaimData claim = new ClaimData(
            "first-claim",     // ID
            true,              // Verified as true
            0.15,              // Authority risk
            0.05,              // Verifiability risk
            0.85               // Base posterior P(H|E)
        );
        
        // Create parameters
        ModelParameters params = new ModelParameters(
            0.60,  // S (internal signal)
            0.90,  // N (canonical evidence)
            0.15,  // α (evidence allocation)
            1.15   // β (uplift factor)
        );
        
        // Calculate Ψ
        double psi = model.calculatePsi(claim, params);
        
        System.out.println("Ψ score: " + psi);
        
        // Classify result
        if (psi > 0.85) {
            System.out.println("Classification: Primitive/Empirically Grounded");
        } else if (psi > 0.70) {
            System.out.println("Classification: Empirically Grounded");
        } else {
            System.out.println("Classification: Interpretive/Contextual");
        }
    }
}
```

**Run it:**
```bash
javac -cp out-qualia FirstPsiExample.java
java -cp .:out-qualia FirstPsiExample
```

#### Using Swift

```swift
import UOIFCore

// Create inputs
let inputs = PsiInputs(
    alpha: 0.15,
    S_symbolic: 0.60,
    N_external: 0.90,
    lambdaAuthority: 0.85,
    lambdaVerifiability: 0.15,
    riskAuthority: 0.15,
    riskVerifiability: 0.05,
    basePosterior: 0.85,
    betaUplift: 1.15
)

// Compute Ψ
let outcome = PsiModel.computePsi(inputs: inputs)

print("Ψ score: \(outcome.psi)")
print("Components:")
print("  Hybrid: \(outcome.hybrid)")
print("  Penalty: \(outcome.penalty)")
print("  Posterior: \(outcome.posterior)")

// Classification
let label: String
if outcome.psi > 0.85 {
    label = "Primitive/Empirically Grounded"
} else if outcome.psi > 0.70 {
    label = "Empirically Grounded"
} else {
    label = "Interpretive/Contextual"
}
print("Classification: \(label)")
```

**Run it:**
```bash
swift run uoif-cli
```

#### Using Python (Uncertainty Quantification)

```python
from uncertainty_quantification import QuickStartUQPipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

# Generate sample data
X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Create and train UQ pipeline
pipeline = QuickStartUQPipeline()
pipeline.fit(X_train, y_train, X_cal, y_cal)

# Get predictions with uncertainty
results = pipeline.predict_with_risk_analysis(X_test[:10])  # First 10 samples

print("Predictions with uncertainty:")
for i, (pred, std) in enumerate(zip(results['predictions'], results['uncertainty'].std_total)):
    print(f"Sample {i}: {pred:.3f} ± {std:.3f}")

# Risk analysis
risk_metrics = results['risk_metrics']
print(f"\nRisk Analysis (threshold 2.0):")
print(f"VaR (95%): {risk_metrics[2.0]['var_95']:.3f}")
print(f"CVaR (95%): {risk_metrics[2.0]['cvar_95']:.3f}")
print(f"Tail probability: {risk_metrics[2.0]['tail_prob']:.3f}")
```

**Run it:**
```bash
python scripts/python/uq_quickstart_example.py
```

## Core Concepts

### The Ψ Framework

The Ψ (Psi) framework provides a mathematical model for evaluating evidence quality and making reliable decisions under uncertainty.

**Formula:**
```
Ψ(x) = min{β·exp(-[λ₁Rₐ + λ₂Rᵥ])·[αS + (1-α)N], 1}
```

**Components:**
- **S**: Internal signal strength (symbolic reasoning)
- **N**: Canonical evidence strength (external verification)
- **α**: Evidence allocation parameter (0 = trust canonical, 1 = trust internal)
- **Rₐ**: Authority risk (credibility concerns)
- **Rᵥ**: Verifiability risk (difficulty of verification)
- **λ₁, λ₂**: Risk penalty weights
- **β**: Uplift factor for confidence scaling

**Interpretation:**
- **Ψ > 0.85**: Primitive/Empirically Grounded (highest confidence)
- **0.70 < Ψ ≤ 0.85**: Empirically Grounded (moderate confidence)
- **Ψ ≤ 0.70**: Interpretive/Contextual (lower confidence)

### Uncertainty Quantification

UQ separates different types of uncertainty:

- **Aleatoric**: Irreducible data noise
- **Epistemic**: Model uncertainty (reducible with more data)
- **Total**: Combined uncertainty

**Key Benefits:**
- Reliable confidence intervals
- Risk-aware decision making
- Calibrated predictions
- Out-of-distribution detection

## Common Workflows

### 1. Bayesian Inference with HMC

```java
// 1. Prepare data
List<ClaimData> dataset = Arrays.asList(
    new ClaimData("c1", true, 0.1, 0.05, 0.8),
    new ClaimData("c2", false, 0.2, 0.1, 0.6),
    new ClaimData("c3", true, 0.15, 0.08, 0.85)
);

// 2. Create model
HierarchicalBayesianModel model = new HierarchicalBayesianModel();

// 3. Run HMC sampling
HmcSampler sampler = new HmcSampler(model, dataset);
double[] initialParams = {0.0, 0.0, 0.0, 0.0};

HmcSampler.AdaptiveResult result = sampler.sampleAdaptive(
    500,   // warmup
    1000,  // sampling
    2,     // thin
    42L,   // seed
    initialParams,
    0.01,  // step size
    10,    // leapfrog steps
    0.8    // target acceptance
);

// 4. Analyze results
System.out.println("Acceptance rate: " + result.acceptanceRate);
System.out.println("Samples: " + result.samples.size());

// 5. Make predictions
ClaimData newClaim = new ClaimData("new", true, 0.12, 0.06, 0.82);
double meanPsi = result.samples.stream()
    .mapToDouble(params -> model.calculatePsi(newClaim, params))
    .average().orElse(0.0);

System.out.println("Predicted Ψ: " + meanPsi);
```

### 2. Multi-Chain Convergence Analysis

```java
// Run multiple chains
HmcMultiChainRunner.MultiChainResult result = 
    HmcMultiChainRunner.runMultiChain(
        model, dataset, 4, 1000, 2000, 2, 42L
    );

// Check convergence
for (int i = 0; i < result.rHat.length; i++) {
    System.out.printf("Parameter %d: R-hat=%.3f", i, result.rHat[i]);
    if (result.rHat[i] > 1.1) {
        System.out.println(" ⚠️ May not have converged");
    } else {
        System.out.println(" ✅ Converged");
    }
}
```

### 3. Uncertainty-Aware Machine Learning

```python
import numpy as np
from uncertainty_quantification import DeepEnsemble, ConformalPredictor
from sklearn.ensemble import RandomForestRegressor

# 1. Create ensemble for epistemic uncertainty
ensemble = DeepEnsemble(RandomForestRegressor, n_models=10)
ensemble.fit(X_train, y_train)

# 2. Add conformal prediction for guaranteed coverage
base_predictions = ensemble.predict_with_uncertainty(X_cal).mean
conformal = ConformalPredictor(ensemble, alpha=0.1)  # 90% coverage
conformal.fit(X_cal, y_cal)

# 3. Make predictions with uncertainty
uncertainty_est = ensemble.predict_with_uncertainty(X_test)
conf_lower, conf_upper = conformal.predict_intervals(X_test)

# 4. Risk analysis
from uncertainty_quantification import RiskBasedDecisionFramework
risk_framework = RiskBasedDecisionFramework()

for i in range(len(X_test)):
    # Sample from predictive distribution
    samples = np.random.normal(
        uncertainty_est.mean[i],
        uncertainty_est.std_total[i],
        size=1000
    )
    
    # Compute risk metrics
    var_95 = risk_framework.compute_var(samples, 0.05)
    cvar_95 = risk_framework.compute_cvar(samples, 0.05)
    
    print(f"Sample {i}: Pred={uncertainty_est.mean[i]:.2f}, "
          f"CI=[{conf_lower[i]:.2f}, {conf_upper[i]:.2f}], "
          f"VaR={var_95:.2f}, CVaR={cvar_95:.2f}")
```

### 4. Real-time Evaluation Pipeline

```swift
import UOIFCore

class PsiEvaluationService {
    func evaluateClaim(
        internalSignal: Double,
        canonicalEvidence: Double,
        authorityRisk: Double,
        verifiabilityRisk: Double,
        basePosterior: Double
    ) -> (psi: Double, classification: String, confidence: Double) {
        
        // Configure inputs
        let inputs = PsiInputs(
            alpha: 0.15,  // Favor canonical evidence
            S_symbolic: internalSignal,
            N_external: canonicalEvidence,
            lambdaAuthority: 0.85,
            lambdaVerifiability: 0.15,
            riskAuthority: authorityRisk,
            riskVerifiability: verifiabilityRisk,
            basePosterior: basePosterior,
            betaUplift: 1.15
        )
        
        // Compute Ψ
        let outcome = PsiModel.computePsi(inputs: inputs)
        
        // Assess confidence
        let confidence = ConfidenceBundle(
            sources: min(1.0, canonicalEvidence + 0.1),
            hybrid: 0.9,
            penalty: exp(-0.5 * (authorityRisk + verifiabilityRisk)),
            posterior: basePosterior,
            psiOverall: outcome.psi
        )
        
        let overallConfidence = ConfidenceHeuristics.overall(
            sources: confidence.sources,
            hybrid: confidence.hybrid,
            penalty: confidence.penalty,
            posterior: confidence.posterior
        )
        
        // Classification
        let classification: String
        if outcome.psi > 0.85 {
            classification = "Primitive/Empirically Grounded"
        } else if outcome.psi > 0.70 {
            classification = "Empirically Grounded"
        } else {
            classification = "Interpretive/Contextual"
        }
        
        return (outcome.psi, classification, overallConfidence)
    }
}

// Usage
let service = PsiEvaluationService()
let result = service.evaluateClaim(
    internalSignal: 0.75,
    canonicalEvidence: 0.92,
    authorityRisk: 0.08,
    verifiabilityRisk: 0.03,
    basePosterior: 0.88
)

print("Ψ: \(String(format: "%.3f", result.psi))")
print("Classification: \(result.classification)")
print("Confidence: \(String(format: "%.2f", result.confidence))")
```

## Testing Your Setup

### Java Tests

```bash
# Run basic functionality test
java -cp out-qualia qualia.Core console

# Run HMC smoke test
java -cp out-qualia qualia.Core hmc

# Run full test suite
bash scripts/test_qualia.sh
```

### Swift Tests

```bash
# Run Swift tests
swift test

# Run CLI with examples
swift run uoif-cli

# Run PINN demo
swift run uoif-cli pinn-demo --alpha=0.3
```

### Python Tests

```bash
# Run simple UQ tests
python scripts/python/test_uq_simple.py

# Run comprehensive example
python scripts/python/uq_quickstart_example.py
```

## Configuration

### Environment Variables

```bash
# Enable metrics server (Java)
export METRICS_ENABLE=1

# Database configuration (optional)
export JDBC_URL="jdbc:postgresql://localhost:5432/qualia"
export JDBC_USER="username"
export JDBC_PASS="password"
```

### Model Parameters

**Recommended starting values:**
- `α = 0.15` (slight preference for canonical evidence)
- `λ₁ = 0.85` (authority risk weight)
- `λ₂ = 0.15` (verifiability risk weight)
- `β = 1.15` (modest confidence uplift)

**Tuning guidelines:**
- Increase `α` to trust internal signals more
- Increase `λ₁, λ₂` to be more risk-averse
- Adjust `β` based on domain-specific confidence needs

## Next Steps

1. **Read the [API Documentation](API_DOCUMENTATION.md)** for detailed interface specifications
2. **Explore the [Examples](examples/)** directory for more complex scenarios
3. **Check the [Architecture Documentation](architecture.md)** for system design details
4. **Review the [Security Policy](../SECURITY.md)** for production deployment guidelines

## Getting Help

- Check the [FAQ](FAQ.md) for common questions
- Review [troubleshooting guides](TROUBLESHOOTING.md) for common issues
- See [Contributing Guidelines](../CONTRIBUTING.md) for development setup
- Check the [internal documentation](../internal/) for advanced topics

## Common Issues

### Java Compilation Errors

```bash
# Ensure Java 21 is being used
java -version
javac -version

# Clean and recompile
rm -rf out-qualia
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')
```

### Swift Build Issues

```bash
# Clean build
swift package clean
swift build

# Update dependencies
swift package update
```

### Python Import Errors

```bash
# Install requirements
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts/python"
```

This getting started guide should help you quickly become productive with the Ψ framework and uncertainty quantification tools!