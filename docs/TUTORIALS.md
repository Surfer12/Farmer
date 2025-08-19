# Tutorials and Examples

This document provides step-by-step tutorials for common use cases and advanced scenarios.

## Table of Contents

1. [Tutorial 1: Basic Ψ Evaluation](#tutorial-1-basic-ψ-evaluation)
2. [Tutorial 2: Bayesian Inference with HMC](#tutorial-2-bayesian-inference-with-hmc)
3. [Tutorial 3: Multi-Chain Convergence Analysis](#tutorial-3-multi-chain-convergence-analysis)
4. [Tutorial 4: Uncertainty Quantification Pipeline](#tutorial-4-uncertainty-quantification-pipeline)
5. [Tutorial 5: Risk-Based Decision Making](#tutorial-5-risk-based-decision-making)
6. [Tutorial 6: Real-time Evaluation System](#tutorial-6-real-time-evaluation-system)
7. [Tutorial 7: Integration with External APIs](#tutorial-7-integration-with-external-apis)
8. [Tutorial 8: Custom Model Development](#tutorial-8-custom-model-development)

---

## Tutorial 1: Basic Ψ Evaluation

**Goal:** Learn how to compute Ψ scores for individual claims and understand the components.

### Step 1: Understanding the Components

The Ψ framework evaluates evidence quality using this formula:

```
Ψ(x) = min{β·exp(-[λ₁Rₐ + λ₂Rᵥ])·[αS + (1-α)N], 1}
```

Let's break this down:

1. **Hybrid Evidence**: `O(α) = αS + (1-α)N`
   - Combines internal signal (S) and canonical evidence (N)
   - α controls the balance (0 = trust canonical, 1 = trust internal)

2. **Risk Penalty**: `exp(-[λ₁Rₐ + λ₂Rᵥ])`
   - Exponentially penalizes authority and verifiability risks
   - λ₁, λ₂ control penalty strength

3. **Confidence Uplift**: `min(β·P(H|E), 1)`
   - β scales base posterior confidence
   - Capped at 1.0 to prevent overconfidence

### Step 2: Java Implementation

```java
// BasicPsiTutorial.java
import java.util.*;

public class BasicPsiTutorial {
    public static void main(String[] args) {
        System.out.println("=== Ψ Framework Tutorial 1: Basic Evaluation ===\n");
        
        // Create model with default priors
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        
        // Example 1: High-quality claim
        System.out.println("Example 1: High-Quality Claim");
        ClaimData highQuality = new ClaimData(
            "high-quality",
            true,           // Verified as true
            0.05,          // Low authority risk
            0.02,          // Low verifiability risk  
            0.92           // High base posterior
        );
        
        ModelParameters params = new ModelParameters(
            0.65,  // S: Strong internal signal
            0.95,  // N: Strong canonical evidence
            0.15,  // α: Favor canonical evidence
            1.20   // β: Modest confidence uplift
        );
        
        double psi1 = model.calculatePsi(highQuality, params);
        printDetailedAnalysis(highQuality, params, psi1, "High-Quality");
        
        // Example 2: Medium-quality claim
        System.out.println("\nExample 2: Medium-Quality Claim");
        ClaimData mediumQuality = new ClaimData(
            "medium-quality",
            true,
            0.15,          // Medium authority risk
            0.08,          // Medium verifiability risk
            0.75           // Medium base posterior
        );
        
        ModelParameters params2 = new ModelParameters(
            0.60,  // S: Moderate internal signal
            0.80,  // N: Moderate canonical evidence
            0.20,  // α: Slightly more internal weight
            1.10   // β: Small confidence uplift
        );
        
        double psi2 = model.calculatePsi(mediumQuality, params2);
        printDetailedAnalysis(mediumQuality, params2, psi2, "Medium-Quality");
        
        // Example 3: Low-quality claim
        System.out.println("\nExample 3: Low-Quality Claim");
        ClaimData lowQuality = new ClaimData(
            "low-quality",
            false,         // Not verified
            0.35,          // High authority risk
            0.20,          // High verifiability risk
            0.45           // Low base posterior
        );
        
        ModelParameters params3 = new ModelParameters(
            0.40,  // S: Weak internal signal
            0.50,  // N: Weak canonical evidence
            0.30,  // α: More balanced allocation
            1.05   // β: Minimal uplift
        );
        
        double psi3 = model.calculatePsi(lowQuality, params3);
        printDetailedAnalysis(lowQuality, params3, psi3, "Low-Quality");
        
        // Summary
        System.out.println("\n=== Summary ===");
        System.out.printf("High-Quality: Ψ=%.3f (%s)\n", psi1, classify(psi1));
        System.out.printf("Medium-Quality: Ψ=%.3f (%s)\n", psi2, classify(psi2));
        System.out.printf("Low-Quality: Ψ=%.3f (%s)\n", psi3, classify(psi3));
    }
    
    private static void printDetailedAnalysis(ClaimData claim, ModelParameters params, 
                                            double psi, String label) {
        System.out.println("Claim: " + label);
        System.out.println("Input Parameters:");
        System.out.printf("  S=%.2f, N=%.2f, α=%.2f, β=%.2f\n", 
                         params.S(), params.N(), params.alpha(), params.beta());
        System.out.printf("  Rₐ=%.2f, Rᵥ=%.2f, P(H|E)=%.2f\n",
                         claim.riskAuthenticity(), claim.riskVirality(), 
                         claim.probabilityHgivenE());
        
        // Calculate components
        double hybrid = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();
        double penalty = Math.exp(-(0.85 * claim.riskAuthenticity() + 0.15 * claim.riskVirality()));
        double posterior = Math.min(params.beta() * claim.probabilityHgivenE(), 1.0);
        
        System.out.println("Components:");
        System.out.printf("  Hybrid O(α) = %.4f\n", hybrid);
        System.out.printf("  Penalty = %.4f\n", penalty);
        System.out.printf("  Posterior = %.4f\n", posterior);
        System.out.printf("  Ψ = %.3f\n", psi);
        System.out.println("Classification: " + classify(psi));
    }
    
    private static String classify(double psi) {
        if (psi > 0.85) return "Primitive/Empirically Grounded";
        if (psi > 0.70) return "Empirically Grounded";
        return "Interpretive/Contextual";
    }
}
```

### Step 3: Swift Implementation

```swift
// BasicPsiTutorial.swift
import UOIFCore

func runBasicPsiTutorial() {
    print("=== Ψ Framework Tutorial 1: Basic Evaluation ===\n")
    
    // Example 1: High-quality claim
    print("Example 1: High-Quality Claim")
    let highQualityInputs = PsiInputs(
        alpha: 0.15,
        S_symbolic: 0.65,
        N_external: 0.95,
        lambdaAuthority: 0.85,
        lambdaVerifiability: 0.15,
        riskAuthority: 0.05,
        riskVerifiability: 0.02,
        basePosterior: 0.92,
        betaUplift: 1.20
    )
    
    let outcome1 = PsiModel.computePsi(inputs: highQualityInputs)
    printDetailedAnalysis(inputs: highQualityInputs, outcome: outcome1, label: "High-Quality")
    
    // Example 2: Medium-quality claim
    print("\nExample 2: Medium-Quality Claim")
    let mediumQualityInputs = PsiInputs(
        alpha: 0.20,
        S_symbolic: 0.60,
        N_external: 0.80,
        lambdaAuthority: 0.85,
        lambdaVerifiability: 0.15,
        riskAuthority: 0.15,
        riskVerifiability: 0.08,
        basePosterior: 0.75,
        betaUplift: 1.10
    )
    
    let outcome2 = PsiModel.computePsi(inputs: mediumQualityInputs)
    printDetailedAnalysis(inputs: mediumQualityInputs, outcome: outcome2, label: "Medium-Quality")
    
    // Example 3: Low-quality claim
    print("\nExample 3: Low-Quality Claim")
    let lowQualityInputs = PsiInputs(
        alpha: 0.30,
        S_symbolic: 0.40,
        N_external: 0.50,
        lambdaAuthority: 0.85,
        lambdaVerifiability: 0.15,
        riskAuthority: 0.35,
        riskVerifiability: 0.20,
        basePosterior: 0.45,
        betaUplift: 1.05
    )
    
    let outcome3 = PsiModel.computePsi(inputs: lowQualityInputs)
    printDetailedAnalysis(inputs: lowQualityInputs, outcome: outcome3, label: "Low-Quality")
    
    // Summary
    print("\n=== Summary ===")
    print("High-Quality: Ψ=\(String(format: "%.3f", outcome1.psi)) (\(classify(psi: outcome1.psi)))")
    print("Medium-Quality: Ψ=\(String(format: "%.3f", outcome2.psi)) (\(classify(psi: outcome2.psi)))")
    print("Low-Quality: Ψ=\(String(format: "%.3f", outcome3.psi)) (\(classify(psi: outcome3.psi)))")
}

func printDetailedAnalysis(inputs: PsiInputs, outcome: PsiOutcome, label: String) {
    print("Claim: \(label)")
    print("Input Parameters:")
    print("  S=\(String(format: "%.2f", inputs.S_symbolic)), N=\(String(format: "%.2f", inputs.N_external)), α=\(String(format: "%.2f", inputs.alpha)), β=\(String(format: "%.2f", inputs.betaUplift))")
    print("  Rₐ=\(String(format: "%.2f", inputs.riskAuthority)), Rᵥ=\(String(format: "%.2f", inputs.riskVerifiability)), P(H|E)=\(String(format: "%.2f", inputs.basePosterior))")
    
    print("Components:")
    print("  Hybrid O(α) = \(String(format: "%.4f", outcome.hybrid))")
    print("  Penalty = \(String(format: "%.4f", outcome.penalty))")
    print("  Posterior = \(String(format: "%.4f", outcome.posterior))")
    print("  Ψ = \(String(format: "%.3f", outcome.psi))")
    print("  ∂Ψ/∂α = \(String(format: "%.4f", outcome.dPsi_dAlpha))")
    print("Classification: \(classify(psi: outcome.psi))")
}

func classify(psi: Double) -> String {
    if psi > 0.85 { return "Primitive/Empirically Grounded" }
    if psi > 0.70 { return "Empirically Grounded" }
    return "Interpretive/Contextual"
}

// Run the tutorial
runBasicPsiTutorial()
```

### Step 4: Key Insights

1. **Hybrid Evidence Balance**: Lower α values favor canonical evidence (N), which typically increases reliability when canonical sources are strong.

2. **Risk Penalty Impact**: Even small increases in authority or verifiability risk can significantly reduce Ψ scores due to the exponential penalty.

3. **Classification Thresholds**: The three-tier classification system provides actionable guidance for decision-making.

---

## Tutorial 2: Bayesian Inference with HMC

**Goal:** Learn to use Hamiltonian Monte Carlo for full Bayesian inference over model parameters.

### Step 1: Understanding HMC

HMC samples from the posterior distribution of parameters given data:
```
p(θ|D) ∝ p(D|θ) × p(θ)
```

The sampler works in an unconstrained space (z) and transforms to parameter space (θ):
- S = sigmoid(z₀)
- N = sigmoid(z₁)  
- α = sigmoid(z₂)
- β = exp(z₃)

### Step 2: Data Preparation

```java
// HMCTutorial.java
import java.util.*;

public class HMCTutorial {
    public static void main(String[] args) {
        System.out.println("=== Tutorial 2: Bayesian Inference with HMC ===\n");
        
        // Step 1: Create synthetic dataset
        List<ClaimData> dataset = generateSyntheticDataset(50);
        System.out.println("Generated " + dataset.size() + " synthetic claims");
        
        // Show dataset statistics
        long trueCount = dataset.stream().mapToLong(c -> c.isVerifiedTrue() ? 1 : 0).sum();
        double avgRiskAuth = dataset.stream().mapToDouble(ClaimData::riskAuthenticity).average().orElse(0);
        double avgRiskViral = dataset.stream().mapToDouble(ClaimData::riskVirality).average().orElse(0);
        double avgPosterior = dataset.stream().mapToDouble(ClaimData::probabilityHgivenE).average().orElse(0);
        
        System.out.printf("Dataset statistics:\n");
        System.out.printf("  True claims: %d/%d (%.1f%%)\n", trueCount, dataset.size(), 
                         100.0 * trueCount / dataset.size());
        System.out.printf("  Avg authority risk: %.3f\n", avgRiskAuth);
        System.out.printf("  Avg verifiability risk: %.3f\n", avgRiskViral);
        System.out.printf("  Avg base posterior: %.3f\n", avgPosterior);
        
        // Step 2: Create model
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        
        // Step 3: Run HMC sampling
        System.out.println("\nRunning HMC sampling...");
        HmcSampler sampler = new HmcSampler(model, dataset);
        
        // Initial parameters in z-space (unconstrained)
        double[] z0 = {0.5, 1.0, -1.5, 0.2}; // Maps to reasonable θ values
        
        HmcSampler.AdaptiveResult result = sampler.sampleAdaptive(
            500,   // warmup iterations
            1000,  // sampling iterations
            2,     // thinning (keep every 2nd sample)
            42L,   // random seed
            z0,    // initial parameters
            0.01,  // initial step size
            10,    // leapfrog steps
            0.8    // target acceptance rate
        );
        
        // Step 4: Analyze results
        System.out.println("\n=== HMC Results ===");
        System.out.printf("Acceptance rate: %.3f\n", result.acceptanceRate);
        System.out.printf("Final step size: %.4f\n", result.tunedStepSize);
        System.out.printf("Divergences: %d\n", result.divergenceCount);
        System.out.printf("Effective samples: %d\n", result.samples.size());
        
        // Step 5: Posterior statistics
        System.out.println("\n=== Posterior Statistics ===");
        computePosteriorStatistics(result.samples);
        
        // Step 6: Posterior predictive evaluation
        System.out.println("\n=== Posterior Predictive Evaluation ===");
        evaluateNewClaims(model, result.samples);
        
        // Step 7: Model diagnostics
        System.out.println("\n=== Model Diagnostics ===");
        performDiagnostics(result);
    }
    
    private static List<ClaimData> generateSyntheticDataset(int n) {
        List<ClaimData> dataset = new ArrayList<>();
        Random rng = new Random(42);
        
        for (int i = 0; i < n; i++) {
            // Generate realistic claim characteristics
            boolean isTrue = rng.nextBoolean();
            
            // Authority risk: higher for false claims
            double riskAuth = Math.abs(rng.nextGaussian()) * 0.15;
            if (!isTrue) riskAuth += 0.1; // False claims tend to have higher authority risk
            
            // Verifiability risk: random but reasonable
            double riskViral = Math.abs(rng.nextGaussian()) * 0.08;
            
            // Base posterior: higher for true claims
            double probHE = 0.6 + 0.2 * rng.nextGaussian();
            if (isTrue) probHE += 0.15; // True claims have higher base posterior
            probHE = Math.max(0.1, Math.min(0.95, probHE));
            
            dataset.add(new ClaimData("claim-" + i, isTrue, riskAuth, riskViral, probHE));
        }
        
        return dataset;
    }
    
    private static void computePosteriorStatistics(List<ModelParameters> samples) {
        // Extract parameter values
        double[] sValues = samples.stream().mapToDouble(ModelParameters::S).toArray();
        double[] nValues = samples.stream().mapToDouble(ModelParameters::N).toArray();
        double[] alphaValues = samples.stream().mapToDouble(ModelParameters::alpha).toArray();
        double[] betaValues = samples.stream().mapToDouble(ModelParameters::beta).toArray();
        
        System.out.println("Parameter posterior means (95% credible intervals):");
        System.out.printf("  S: %.3f [%.3f, %.3f]\n", 
                         mean(sValues), quantile(sValues, 0.025), quantile(sValues, 0.975));
        System.out.printf("  N: %.3f [%.3f, %.3f]\n",
                         mean(nValues), quantile(nValues, 0.025), quantile(nValues, 0.975));
        System.out.printf("  α: %.3f [%.3f, %.3f]\n",
                         mean(alphaValues), quantile(alphaValues, 0.025), quantile(alphaValues, 0.975));
        System.out.printf("  β: %.3f [%.3f, %.3f]\n",
                         mean(betaValues), quantile(betaValues, 0.025), quantile(betaValues, 0.975));
    }
    
    private static void evaluateNewClaims(HierarchicalBayesianModel model, 
                                        List<ModelParameters> samples) {
        // Test claims with different characteristics
        ClaimData[] testClaims = {
            new ClaimData("test-high", true, 0.05, 0.02, 0.90),
            new ClaimData("test-medium", true, 0.15, 0.08, 0.75),
            new ClaimData("test-low", false, 0.30, 0.15, 0.45)
        };
        
        for (ClaimData claim : testClaims) {
            List<Double> psiScores = new ArrayList<>();
            for (ModelParameters params : samples) {
                double psi = model.calculatePsi(claim, params);
                psiScores.add(psi);
            }
            
            double[] psiArray = psiScores.stream().mapToDouble(Double::doubleValue).toArray();
            double meanPsi = mean(psiArray);
            double lowerCI = quantile(psiArray, 0.025);
            double upperCI = quantile(psiArray, 0.975);
            
            System.out.printf("%s: Ψ=%.3f [%.3f, %.3f] (%s)\n",
                             claim.id(), meanPsi, lowerCI, upperCI, classify(meanPsi));
        }
    }
    
    private static void performDiagnostics(HmcSampler.AdaptiveResult result) {
        if (result.acceptanceRate < 0.6) {
            System.out.println("⚠️  Low acceptance rate - consider smaller step size");
        } else if (result.acceptanceRate > 0.9) {
            System.out.println("⚠️  High acceptance rate - consider larger step size");
        } else {
            System.out.println("✅ Acceptance rate is in good range");
        }
        
        if (result.divergenceCount > 0) {
            System.out.printf("⚠️  %d divergent transitions - consider smaller step size\n", 
                             result.divergenceCount);
        } else {
            System.out.println("✅ No divergent transitions");
        }
        
        System.out.printf("Final step size: %.4f\n", result.tunedStepSize);
        if (result.tunedStepSize < 0.001) {
            System.out.println("⚠️  Very small step size - check model specification");
        }
    }
    
    // Utility methods
    private static double mean(double[] values) {
        return Arrays.stream(values).average().orElse(0.0);
    }
    
    private static double quantile(double[] values, double p) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int index = (int) Math.ceil(p * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
    }
    
    private static String classify(double psi) {
        if (psi > 0.85) return "Primitive/Empirically Grounded";
        if (psi > 0.70) return "Empirically Grounded";
        return "Interpretive/Contextual";
    }
}
```

### Step 3: Understanding the Output

The HMC sampler provides:

1. **Acceptance Rate**: Should be 60-90% for efficient sampling
2. **Step Size**: Automatically tuned during warmup
3. **Divergences**: Should be zero or very few
4. **Posterior Statistics**: Mean and credible intervals for each parameter

### Step 4: Advanced HMC Features

```java
// Advanced HMC configuration
ModelPriors customPriors = new ModelPriors(
    1.0,   // lambda1 - stronger authority penalty
    0.2,   // lambda2 - weaker verifiability penalty
    // ... other hyperparameters
);

HierarchicalBayesianModel customModel = new HierarchicalBayesianModel(customPriors);

// Run with custom settings
HmcSampler.AdaptiveResult result = sampler.sampleAdaptive(
    1000,  // Longer warmup
    5000,  // More samples
    1,     // No thinning
    42L,
    z0,
    0.005, // Smaller initial step size
    15,    // More leapfrog steps
    0.65   // Lower target acceptance for better exploration
);
```

---

## Tutorial 3: Multi-Chain Convergence Analysis

**Goal:** Learn to run multiple HMC chains and assess convergence using R̂ statistics.

### Step 1: Running Multiple Chains

```java
// MultiChainTutorial.java
public class MultiChainTutorial {
    public static void main(String[] args) {
        System.out.println("=== Tutorial 3: Multi-Chain Convergence Analysis ===\n");
        
        // Prepare data and model
        List<ClaimData> dataset = generateLargerDataset(100);
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        
        System.out.println("Running 4 chains with different random seeds...");
        
        // Run multi-chain HMC
        HmcMultiChainRunner.MultiChainResult result = 
            HmcMultiChainRunner.runMultiChain(
                model,
                dataset,
                4,      // Number of chains
                1000,   // Warmup per chain
                2000,   // Sampling per chain
                2,      // Thinning
                42L     // Base seed (each chain gets different seed)
            );
        
        // Analyze convergence
        analyzeConvergence(result);
        
        // Compare chain statistics
        compareChains(result);
        
        // Combine samples if converged
        if (isConverged(result)) {
            List<ModelParameters> combinedSamples = combineChains(result);
            performPosteriorAnalysis(combinedSamples);
        } else {
            System.out.println("⚠️  Chains have not converged - consider longer runs");
        }
    }
    
    private static void analyzeConvergence(HmcMultiChainRunner.MultiChainResult result) {
        System.out.println("\n=== Convergence Diagnostics ===");
        
        String[] paramNames = {"S", "N", "α", "β"};
        for (int i = 0; i < result.rHat.length; i++) {
            System.out.printf("Parameter %s: R̂=%.4f", paramNames[i], result.rHat[i]);
            
            if (result.rHat[i] < 1.01) {
                System.out.println(" ✅ Excellent convergence");
            } else if (result.rHat[i] < 1.05) {
                System.out.println(" ✅ Good convergence");
            } else if (result.rHat[i] < 1.1) {
                System.out.println(" ⚠️  Acceptable convergence");
            } else {
                System.out.println(" ❌ Poor convergence - need more samples");
            }
        }
        
        System.out.println("\nEffective Sample Sizes:");
        for (int i = 0; i < result.effectiveSampleSize.length; i++) {
            System.out.printf("Parameter %s: ESS=%.0f", paramNames[i], result.effectiveSampleSize[i]);
            
            int totalSamples = result.chainSamples.get(0).size() * result.chainSamples.size();
            double essRatio = result.effectiveSampleSize[i] / totalSamples;
            
            if (essRatio > 0.5) {
                System.out.println(" ✅ High efficiency");
            } else if (essRatio > 0.1) {
                System.out.println(" ⚠️  Moderate efficiency");
            } else {
                System.out.println(" ❌ Low efficiency");
            }
        }
    }
    
    private static void compareChains(HmcMultiChainRunner.MultiChainResult result) {
        System.out.println("\n=== Chain Comparison ===");
        
        for (int chain = 0; chain < result.chainSamples.size(); chain++) {
            List<ModelParameters> samples = result.chainSamples.get(chain);
            
            double meanAlpha = samples.stream().mapToDouble(ModelParameters::alpha).average().orElse(0);
            double meanBeta = samples.stream().mapToDouble(ModelParameters::beta).average().orElse(0);
            
            System.out.printf("Chain %d: Acceptance=%.3f, ᾱ=%.3f, β̄=%.3f, Samples=%d\n",
                             chain, result.acceptanceRates[chain], meanAlpha, meanBeta, samples.size());
        }
        
        // Check for chain mixing
        double minAcceptance = Arrays.stream(result.acceptanceRates).min().orElse(0);
        double maxAcceptance = Arrays.stream(result.acceptanceRates).max().orElse(0);
        
        if (maxAcceptance - minAcceptance < 0.1) {
            System.out.println("✅ Chains have similar acceptance rates - good mixing");
        } else {
            System.out.println("⚠️  Chains have different acceptance rates - check initialization");
        }
    }
    
    private static boolean isConverged(HmcMultiChainRunner.MultiChainResult result) {
        return Arrays.stream(result.rHat).allMatch(rhat -> rhat < 1.1);
    }
    
    private static List<ModelParameters> combineChains(HmcMultiChainRunner.MultiChainResult result) {
        List<ModelParameters> combined = new ArrayList<>();
        for (List<ModelParameters> chain : result.chainSamples) {
            combined.addAll(chain);
        }
        return combined;
    }
    
    // ... other methods
}
```

### Step 2: Convergence Criteria

**R̂ (R-hat) Interpretation:**
- R̂ < 1.01: Excellent convergence
- 1.01 ≤ R̂ < 1.05: Good convergence  
- 1.05 ≤ R̂ < 1.1: Acceptable convergence
- R̂ ≥ 1.1: Poor convergence (need more samples)

**Effective Sample Size (ESS):**
- ESS/Total > 0.5: High efficiency
- 0.1 < ESS/Total ≤ 0.5: Moderate efficiency
- ESS/Total ≤ 0.1: Low efficiency

---

## Tutorial 4: Uncertainty Quantification Pipeline

**Goal:** Build a complete uncertainty quantification pipeline with multiple UQ methods.

### Step 1: Basic UQ Pipeline

```python
# uq_tutorial.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from uncertainty_quantification import (
    UncertaintyEstimate, DeepEnsemble, MCDropout, 
    ConformalPredictor, CalibrationMethods, RiskBasedDecisionFramework
)

def tutorial_4_uq_pipeline():
    print("=== Tutorial 4: Uncertainty Quantification Pipeline ===\n")
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic dataset...")
    X, y = make_regression(n_samples=2000, n_features=10, noise=0.2, random_state=42)
    
    # Add some heteroscedastic noise (uncertainty varies with input)
    noise_scale = 0.1 + 0.3 * np.abs(X[:, 0])  # Noise depends on first feature
    y += np.random.normal(0, noise_scale)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training: {X_train.shape[0]}, Calibration: {X_cal.shape[0]}, Test: {X_test.shape[0]}")
    
    # Step 2: Train Deep Ensemble
    print("\nStep 2: Training Deep Ensemble...")
    ensemble = DeepEnsemble(RandomForestRegressor, n_models=10, n_estimators=100, random_state=42)
    ensemble.fit(X_train, y_train)
    
    # Get ensemble predictions
    ensemble_uncertainty = ensemble.predict_with_uncertainty(X_test)
    print(f"Ensemble - Mean prediction range: [{np.min(ensemble_uncertainty.mean):.2f}, {np.max(ensemble_uncertainty.mean):.2f}]")
    print(f"Epistemic uncertainty range: [{np.min(ensemble_uncertainty.epistemic):.4f}, {np.max(ensemble_uncertainty.epistemic):.4f}]")
    
    # Step 3: Add Conformal Prediction
    print("\nStep 3: Adding Conformal Prediction...")
    
    class EnsembleWrapper:
        def __init__(self, ensemble):
            self.ensemble = ensemble
        def predict(self, X):
            return self.ensemble.predict_with_uncertainty(X).mean
    
    wrapper = EnsembleWrapper(ensemble)
    conformal = ConformalPredictor(wrapper, alpha=0.1)  # 90% coverage
    conformal.fit(X_cal, y_cal)
    
    conf_lower, conf_upper = conformal.predict_intervals(X_test)
    
    # Check empirical coverage
    coverage = np.mean((y_test >= conf_lower) & (y_test <= conf_upper))
    print(f"Conformal prediction coverage: {coverage:.3f} (target: 0.90)")
    
    # Step 4: Risk Analysis
    print("\nStep 4: Risk-Based Analysis...")
    risk_framework = RiskBasedDecisionFramework()
    
    # Sample from predictive distributions
    n_risk_samples = min(20, len(X_test))  # Analyze first 20 samples
    risk_results = []
    
    for i in range(n_risk_samples):
        # Generate samples from predictive distribution
        samples = np.random.normal(
            ensemble_uncertainty.mean[i],
            ensemble_uncertainty.std_total[i],
            size=1000
        )
        
        # Compute risk metrics
        var_95 = risk_framework.compute_var(samples, 0.05)
        cvar_95 = risk_framework.compute_cvar(samples, 0.05)
        tail_prob_2 = risk_framework.tail_probability(samples, 2.0)
        tail_prob_3 = risk_framework.tail_probability(samples, 3.0)
        
        risk_results.append({
            'sample_idx': i,
            'prediction': ensemble_uncertainty.mean[i],
            'uncertainty': ensemble_uncertainty.std_total[i],
            'true_value': y_test[i],
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_prob_2': tail_prob_2,
            'tail_prob_3': tail_prob_3,
            'conf_lower': conf_lower[i],
            'conf_upper': conf_upper[i]
        })
    
    # Display risk analysis
    print("\nRisk Analysis Results (first 10 samples):")
    print("Idx | Pred   | True   | Std    | VaR95  | CVaR95 | P(>2) | P(>3) | Conf Interval")
    print("-" * 85)
    for r in risk_results[:10]:
        print(f"{r['sample_idx']:3d} | {r['prediction']:6.2f} | {r['true_value']:6.2f} | "
              f"{r['uncertainty']:6.3f} | {r['var_95']:6.2f} | {r['cvar_95']:6.2f} | "
              f"{r['tail_prob_2']:5.3f} | {r['tail_prob_3']:5.3f} | "
              f"[{r['conf_lower']:5.2f}, {r['conf_upper']:5.2f}]")
    
    # Step 5: Calibration Assessment
    print("\nStep 5: Calibration Assessment...")
    
    # For regression, we assess calibration of prediction intervals
    confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    empirical_coverages = []
    
    for alpha in confidence_levels:
        conf_pred = ConformalPredictor(wrapper, alpha=alpha)
        conf_pred.fit(X_cal, y_cal)
        lower, upper = conf_pred.predict_intervals(X_test)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        empirical_coverages.append(coverage)
    
    print("Calibration Results:")
    print("Target Coverage | Empirical Coverage | Calibration Error")
    print("-" * 55)
    for i, alpha in enumerate(confidence_levels):
        target_coverage = 1 - alpha
        empirical_coverage = empirical_coverages[i]
        error = abs(empirical_coverage - target_coverage)
        print(f"{target_coverage:14.1f} | {empirical_coverage:17.3f} | {error:16.3f}")
    
    mean_calibration_error = np.mean([abs(empirical_coverages[i] - (1 - confidence_levels[i])) 
                                     for i in range(len(confidence_levels))])
    print(f"\nMean Calibration Error: {mean_calibration_error:.4f}")
    
    # Step 6: Decision Making Framework
    print("\nStep 6: Decision Making Framework...")
    
    # Define decision thresholds and costs
    decision_framework = {
        'low_risk_threshold': 1.0,
        'high_risk_threshold': 2.0,
        'action_costs': {
            'no_action': 0.0,
            'investigate': 0.5,
            'immediate_action': 2.0
        }
    }
    
    decisions = []
    for r in risk_results:
        if r['cvar_95'] < decision_framework['low_risk_threshold']:
            decision = 'no_action'
        elif r['cvar_95'] < decision_framework['high_risk_threshold']:
            decision = 'investigate'
        else:
            decision = 'immediate_action'
        
        decisions.append(decision)
    
    decision_counts = {action: decisions.count(action) for action in decision_framework['action_costs']}
    print("Decision Distribution:")
    for action, count in decision_counts.items():
        print(f"  {action}: {count} samples ({100*count/len(decisions):.1f}%)")
    
    # Step 7: Summary and Recommendations
    print("\n=== Summary and Recommendations ===")
    
    if coverage > 0.85:
        print("✅ Conformal prediction is well-calibrated")
    else:
        print("⚠️  Conformal prediction may need adjustment")
    
    if mean_calibration_error < 0.05:
        print("✅ Model shows good calibration across confidence levels")
    else:
        print("⚠️  Model may benefit from calibration techniques")
    
    high_uncertainty_samples = sum(1 for r in risk_results if r['uncertainty'] > np.median([r['uncertainty'] for r in risk_results]))
    print(f"📊 {high_uncertainty_samples}/{len(risk_results)} samples have above-median uncertainty")
    
    return {
        'ensemble_uncertainty': ensemble_uncertainty,
        'conformal_coverage': coverage,
        'risk_results': risk_results,
        'calibration_error': mean_calibration_error,
        'decisions': decisions
    }

if __name__ == "__main__":
    results = tutorial_4_uq_pipeline()
```

### Step 2: Advanced UQ with Neural Networks

```python
# advanced_uq_tutorial.py
import torch
import torch.nn as nn
import torch.optim as optim
from uncertainty_quantification import MCDropout, HeteroscedasticHead

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def advanced_uq_tutorial():
    print("=== Advanced UQ with Neural Networks ===\n")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Train model with dropout
    model = SimpleNet(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training neural network...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # MC Dropout inference
    mc_dropout = MCDropout(model, n_samples=100)
    uncertainty_est = mc_dropout.predict_with_uncertainty(X_test_tensor)
    
    print(f"\nMC Dropout Results:")
    print(f"Mean prediction error: {np.mean(np.abs(uncertainty_est.mean - y_test)):.3f}")
    print(f"Mean epistemic uncertainty: {np.mean(uncertainty_est.epistemic):.4f}")
    
    return uncertainty_est

if __name__ == "__main__":
    advanced_results = advanced_uq_tutorial()
```

This tutorial series provides comprehensive coverage of the Ψ framework and uncertainty quantification tools, from basic concepts to advanced applications. Each tutorial builds upon the previous ones, providing a complete learning path for users at different skill levels.