# Examples and Usage Scenarios

**SPDX-License-Identifier: LicenseRef-Internal-Use-Only**

Practical examples and usage scenarios for the Farmer project components.

## Table of Contents

1. [Java Core Examples](#java-core-examples)
2. [Swift UOIF Examples](#swift-uoif-examples)
3. [Python UQ Examples](#python-uq-examples)
4. [Integration Examples](#integration-examples)

---

## Java Core Examples

### Basic Ψ Calculation

```java
import qualia.*;

// Create model with default priors
HierarchicalBayesianModel model = new HierarchicalBayesianModel();

// Create claim data
ClaimData claim = new ClaimData("claim-001", true, 0.1, 0.05, 0.8);

// Create parameter sample
ModelParameters params = new ModelParameters(0.7, 0.8, 0.6, 1.1);

// Calculate Ψ confidence score
double psi = model.calculatePsi(claim, params);
System.out.println("Ψ confidence: " + psi);

// Expected output: Ψ confidence: 0.6234...
```

### HMC Sampling Pipeline

```java
// Setup model and dataset
HierarchicalBayesianModel model = new HierarchicalBayesianModel();
List<ClaimData> dataset = createSyntheticDataset(100);
HmcSampler hmc = new HmcSampler(model, dataset);

// Initial parameters in unconstrained space
double[] z0 = {logit(0.7), logit(0.6), logit(0.5), Math.log(1.0)};

// Adaptive sampling with tuning
HmcSampler.AdaptiveResult result = hmc.sampleAdaptive(
    1000,  // warmup iterations
    2000,  // sampling iterations
    3,     // thinning
    42L,   // random seed
    z0,    // initial parameters
    0.01,  // initial step size
    20,    // leapfrog steps
    0.75   // target acceptance rate
);

// Analyze results
System.out.println("Acceptance rate: " + result.acceptanceRate);
System.out.println("Tuned step size: " + result.tunedStepSize);
System.out.println("Divergences: " + result.divergenceCount);
System.out.println("Samples collected: " + result.samples.size());

// Calculate mean Ψ across dataset and samples
double meanPsi = 0.0;
for (ModelParameters p : result.samples) {
    double sum = 0.0;
    for (ClaimData c : dataset) {
        sum += model.calculatePsi(c, p);
    }
    meanPsi += sum / dataset.size();
}
meanPsi /= result.samples.size();
System.out.println("Mean Ψ: " + meanPsi);
```

### Multi-Chain HMC Analysis

```java
// Multi-chain runner
HmcMultiChainRunner runner = new HmcMultiChainRunner(
    model, dataset, 4, 1000, 2000, 3, 42L, z0, 0.01, 20, 0.75, 
    new File("hmc-output")
);

// Run all chains
HmcMultiChainRunner.Summary summary = runner.run();

// Access individual chain results
for (int i = 0; i < summary.chains.size(); i++) {
    HmcMultiChainRunner.ChainResult chain = summary.chains.get(i);
    System.out.println("Chain " + i + ":");
    System.out.println("  Samples: " + chain.samples.size());
    System.out.println("  Acceptance: " + chain.acceptanceRate);
    System.out.println("  Step size: " + chain.tunedStepSize);
}

// Diagnostics across chains
Diagnostics diag = model.diagnose(summary.getAllChains());
System.out.println("R̂ diagnostics:");
System.out.println("  S: " + diag.rHatS);
System.out.println("  N: " + diag.rHatN);
System.out.println("  alpha: " + diag.rHatAlpha);
System.out.println("  beta: " + diag.rHatBeta);
```

### Unified Detector with Triad Gating

```java
UnifiedDetector detector = new UnifiedDetector();

// Define dynamics: Simple Harmonic Oscillator
UnifiedDetector.Dynamics sho = (t, y, dy) -> {
    double omega = 1.0;
    dy[0] = y[1];                    // dx/dt = v
    dy[1] = -omega * omega * y[0];   // dv/dt = -ω²x
};

// Energy invariant
double E0 = 0.5; // Initial energy
UnifiedDetector.Invariant energy = new UnifiedDetector.Invariant() {
    @Override public double value(double t, double[] y) {
        double omega = 1.0;
        return 0.5 * (y[1] * y[1] + omega * omega * y[0] * y[0]);
    }
    @Override public double reference() { return E0; }
    @Override public double tolerance() { return E0 * 1e-3; }
};

// Integration parameters
double t = 0.0;
double[] y = {1.0, 0.0};  // Initial state: x=1, v=0
double h = 1e-3;
double eps = 1e-4;
double epsRk4 = 1e-5;
double epsTaylor = 1e-5;
double epsGeom = 1e-5;
long budgetNs = 1_000_000; // 1ms budget

// Integrate with triad gating
try (PrintWriter out = new PrintWriter("unified_results.jsonl")) {
    for (int step = 0; step < 1000; step++) {
        UnifiedDetector.Triad result = detector.triadStep(
            sho, t, y, h, eps, epsRk4, epsTaylor, epsGeom, budgetNs, 
            new UnifiedDetector.Invariant[]{energy}
        );
        
        // Update state
        t = result.tNext;
        y = result.yNext;
        h = result.hUsed;
        
        // Log results
        String json = String.format(Locale.ROOT,
            "{\"step\":%d,\"t\":%.6f,\"x\":%.8f,\"v\":%.8f,\"psi\":%.6f," +
            "\"h\":%.6e,\"eps_rk4\":%.6e,\"eps_taylor\":%.6e,\"eps_geom\":%.6e," +
            "\"geom_drift\":%.6e,\"accepted\":%s}",
            step, t, y[0], y[1], result.psi, result.hUsed, 
            result.epsRk4, result.epsTaylor, result.epsGeom, 
            result.geomDrift, result.accepted);
        out.println(json);
        
        if (!result.accepted) {
            System.out.println("Step " + step + " rejected, Ψ = " + result.psi);
        }
    }
}
```

### MCDA with Ψ Integration

```java
// Define alternatives with Ψ confidence scores
Mcda.Alternative a = new Mcda.Alternative(
    "Model A", 
    Map.of("cost", 100.0, "accuracy", 0.85, "speed", 0.9), 
    0.88  // High confidence
);

Mcda.Alternative b = new Mcda.Alternative(
    "Model B", 
    Map.of("cost", 80.0, "accuracy", 0.82, "speed", 0.85), 
    0.75  // Medium confidence
);

Mcda.Alternative c = new Mcda.Alternative(
    "Model C", 
    Map.of("cost", 120.0, "accuracy", 0.88, "speed", 0.95), 
    0.72  // Lower confidence
);

List<Mcda.Alternative> alternatives = List.of(a, b, c);

// Gate by confidence threshold
double confidenceThreshold = 0.80;
List<Mcda.Alternative> feasible = Mcda.gateByPsi(alternatives, confidenceThreshold);
System.out.println("Feasible alternatives: " + feasible);

// Define criteria with Ψ as primary criterion
List<Mcda.CriterionSpec> criteria = List.of(
    new Mcda.CriterionSpec("psi", Mcda.Direction.BENEFIT, 0.4),      // Ψ confidence
    new Mcda.CriterionSpec("accuracy", Mcda.Direction.BENEFIT, 0.3), // Accuracy
    new Mcda.CriterionSpec("speed", Mcda.Direction.BENEFIT, 0.2),    // Speed
    new Mcda.CriterionSpec("cost", Mcda.Direction.COST, 0.1)         // Cost (minimize)
);

// Normalize criteria
Map<Mcda.Alternative, Map<String, Double>> normalized = 
    Mcda.normalize(feasible, criteria);

// Rank alternatives using different methods
List<Mcda.Ranked> wsmRanking = Mcda.rankByWSM(feasible, criteria);
List<Mcda.Ranked> topsisRanking = Mcda.rankByTOPSIS(feasible, criteria);

System.out.println("WSM ranking:");
wsmRanking.forEach(r -> System.out.println("  " + r.alternative.id + ": " + r.score));

System.out.println("TOPSIS ranking:");
topsisRanking.forEach(r -> System.out.println("  " + r.alternative.id + ": " + r.score));
```

---

## Swift UOIF Examples

### Confidence Assessment

```swift
import UOIFCore

// Basic confidence calculation
let confidence = ConfidenceHeuristics.overall(
    sources: 0.95,      // High source reliability
    hybrid: 0.88,       // Good hybrid model performance
    penalty: 0.82,      // Moderate risk penalty
    posterior: 0.90     // Strong Bayesian posterior
)

print("Overall confidence: \(confidence)")

// Use preset evaluations
let eval2025 = Presets.eval2025Results(alpha: 0.6)
print("2025 Results evaluation: \(eval2025.title)")
print("Ψ score: \(eval2025.confidence.psiOverall)")

let eval2024 = Presets.eval2024(alpha: 0.7)
print("2024 evaluation: \(eval2024.title)")
print("Label: \(eval2024.label)")
```

### Hierarchical Bayesian Model

```swift
// Create model with custom priors
let priors = ModelPriors(
    lambda1: 0.85,      // Authority risk weight
    lambda2: 0.15,      // Verifiability risk weight
    s_alpha: 1.0, s_beta: 1.0,      // Uniform prior for S
    n_alpha: 1.0, n_beta: 1.0,      // Uniform prior for N
    alpha_alpha: 1.0, alpha_beta: 1.0, // Uniform prior for alpha
    beta_mu: 0.0, beta_sigma: 1.0,   // LogNormal prior for beta
    ra_shape: 1.0, ra_scale: 1.0,    // Gamma prior for R_a
    rv_shape: 1.0, rv_scale: 1.0     // Gamma prior for R_v
)

let model = HierarchicalBayesianModel(priors: priors)

// Create claim data
let claim = ClaimData(
    id: "swift-claim-001",
    isVerifiedTrue: true,
    riskAuthenticity: 0.12,
    riskVirality: 0.04,
    probabilityHgivenE: 0.90
)

// Create parameter sample
let params = ModelParameters(
    S: 0.72,
    N: 0.85,
    alpha: 0.6,
    beta: 1.15
)

// Calculate Ψ
let psi = model.calculatePsi(for: claim, with: params)
print("Ψ confidence: \(psi)")

// Calculate likelihood
let logLik = model.logLikelihood(for: claim, with: params)
print("Log-likelihood: \(logLik)")
```

### Physics-Informed Neural Networks

```swift
// Create PINN for heat equation
let pinn = PINN(hiddenWidth: 30)

// Training loop for heat equation: u_t = u_xx
let learningRate = 0.001
let epochs = 1000

for epoch in 0..<epochs {
    var totalLoss = 0.0
    
    // Sample training points
    for _ in 0..<100 {
        let x = Double.random(in: -1...1)
        let t = Double.random(in: 0...1)
        
        // PDE residual: u_t - u_xx
        let residual = PDE.residual_heatEquation(
            model: pinn, x: x, t: t, dx: 1e-4, dt: 1e-4
        )
        
        totalLoss += residual * residual
    }
    
    // Gradient descent update (simplified)
    if epoch % 100 == 0 {
        print("Epoch \(epoch): Loss = \(totalLoss)")
    }
}

// Test solution
let testX = 0.5
let testT = 0.1
let solution = pinn.forward(x: testX, t: testT)
print("u(\(testX), \(testT)) = \(solution)")

// Burgers equation example
let nu = 0.01  // Viscosity
let burgersResidual = PDE.residual_burgersEquation(
    model: pinn, x: testX, t: testT, nu: nu, dx: 1e-4, dt: 1e-4
)
print("Burgers residual: \(burgersResidual)")
```

---

## Python UQ Examples

### Deep Ensemble Uncertainty

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scripts.python.uncertainty_quantification import DeepEnsemble, UncertaintyEstimate

# Create synthetic data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = 2 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(1000) * 0.1

# Split data
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Create deep ensemble
ensemble = DeepEnsemble(
    model_class=RandomForestRegressor,
    n_models=10,
    n_estimators=100,
    max_depth=10
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Predict with uncertainty
estimate = ensemble.predict_with_uncertainty(X_test)

# Analyze results
print(f"Mean predictions shape: {estimate.mean.shape}")
print(f"Epistemic uncertainty range: [{estimate.epistemic.min():.4f}, {estimate.epistemic.max():.4f}]")
print(f"Total uncertainty range: [{estimate.total.min():.4f}, {estimate.total.max():.4f}]")

# Confidence intervals
lower, upper = estimate.confidence_interval(alpha=0.05)
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"95% CI coverage: {coverage:.3f}")

# Visualize uncertainty
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test, estimate.mean, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')

plt.subplot(1, 3, 2)
plt.hist(estimate.epistemic, bins=30, alpha=0.7)
plt.xlabel('Epistemic Uncertainty')
plt.ylabel('Frequency')
plt.title('Epistemic Uncertainty Distribution')

plt.subplot(1, 3, 3)
plt.scatter(estimate.mean, estimate.total, alpha=0.6)
plt.xlabel('Predictions')
plt.ylabel('Total Uncertainty')
plt.title('Uncertainty vs Predictions')

plt.tight_layout()
plt.show()
```

### MCDropout for Neural Networks

```python
import torch
import torch.nn as nn
from scripts.python.uncertainty_quantification import MCDropout

# Define neural network with dropout
class DropoutNet(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Create model and data
model = DropoutNet(dropout_rate=0.1)
X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

# Train model (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# MCDropout for uncertainty
mc_dropout = MCDropout(model, n_samples=100, dropout_rate=0.1)

# Enable dropout for inference
mc_dropout.enable_dropout()

# Predict with uncertainty
X_test_tensor = torch.FloatTensor(X_test)
estimate = mc_dropout.predict_with_uncertainty(X_test_tensor)

print(f"MCDropout uncertainty range: [{estimate.total.min():.4f}, {estimate.total.max():.4f}]")

# Disable dropout for regular inference
mc_dropout.disable_dropout()
```

### Conformal Prediction

```python
from scripts.python.uncertainty_quantification import ConformalPrediction

# Create conformal predictor
cp = ConformalPrediction(
    base_model=RandomForestRegressor(n_estimators=100),
    method="naive",
    alpha=0.1  # 90% prediction intervals
)

# Split data for calibration
X_train_cp, X_cal = X_train[:600], X_train[600:]
y_train_cp, y_cal = y_train[:600], y_train[600:]

# Fit and calibrate
cp.fit(X_cal, y_cal)

# Predict with intervals
lower, upper = cp.predict_with_intervals(X_test)
interval_widths = upper - lower

print(f"Prediction interval widths:")
print(f"  Mean: {interval_widths.mean():.4f}")
print(f"  Std: {interval_widths.std():.4f}")
print(f"  Range: [{interval_widths.min():.4f}, {interval_widths.max():.4f}]")

# Check coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% prediction interval coverage: {coverage:.3f}")

# Quantile predictions
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile_preds = cp.predict_with_quantiles(X_test, quantiles)

print(f"Quantile predictions shape: {quantile_preds.shape}")
```

### Risk-Aware Classification

```python
from scripts.python.uncertainty_quantification import RiskAwareClassifier
from sklearn.ensemble import RandomForestClassifier

# Create classification data
y_class = (y > np.median(y)).astype(int)
y_train_class = y_class[:800]
y_test_class = y_class[800:]

# Create risk-aware classifier
risk_classifier = RiskAwareClassifier(
    base_model=RandomForestClassifier(n_estimators=100),
    uncertainty_threshold=0.15
)

# Train classifier
risk_classifier.fit(X_train, y_train_class)

# Predict with risk assessment
predictions, probabilities, uncertainties = risk_classifier.predict_with_risk(X_test)

# Reject uncertain predictions
rejected = risk_classifier.reject_uncertain(X_test, threshold=0.2)

print(f"Total test samples: {len(X_test)}")
print(f"Rejected samples: {np.sum(rejected)}")
print(f"Accepted samples: {len(X_test) - np.sum(rejected)}")

# Analyze accepted vs rejected
accepted_mask = ~rejected
if np.sum(accepted_mask) > 0:
    accepted_acc = np.mean(predictions[accepted_mask] == y_test_class[accepted_mask])
    print(f"Accuracy on accepted samples: {accepted_acc:.3f}")

if np.sum(rejected) > 0:
    rejected_uncertainty = uncertainties[rejected]
    print(f"Mean uncertainty of rejected: {rejected_uncertainty.mean():.4f}")
```

---

## Integration Examples

### Java + Python Pipeline

```bash
#!/bin/bash

# 1. Run Java HMC sampling
echo "Running HMC sampling..."
java -cp out-qualia qualia.Core hmc_adapt \
    warmup=1000 iters=2000 thin=3 seed=42 \
    jsonl=data/logs/hmc_results.jsonl

# 2. Run Java unified detector
echo "Running unified detector..."
java -cp out-qualia qualia.Core unified \
    h=1e-3 eps=1e-4 triad=true \
    jsonl=data/logs/unified_results.jsonl

# 3. Analyze with Python
echo "Analyzing results with Python..."
python scripts/python/analyze_integration.py \
    --hmc data/logs/hmc_results.jsonl \
    --unified data/logs/unified_results.jsonl \
    --output analysis_results/
```

### Swift + Java Integration

```swift
// Swift confidence assessment
let swiftConfidence = ConfidenceHeuristics.overall(
    sources: 0.95,
    hybrid: 0.88,
    penalty: 0.82,
    posterior: 0.90
)

// Export for Java processing
let confidenceData = [
    "swift_confidence": swiftConfidence,
    "timestamp": Date().timeIntervalSince1970,
    "source": "Swift UOIF"
]

// Save to JSON for Java consumption
if let jsonData = try? JSONSerialization.data(withJSONObject: confidenceData),
   let jsonString = String(data: jsonData, encoding: .utf8) {
    try? jsonString.write(toFile: "swift_confidence.json", atomically: true, encoding: .utf8)
}
```

```java
// Java reads Swift confidence
import com.fasterxml.jackson.databind.ObjectMapper;

ObjectMapper mapper = new ObjectMapper();
Map<String, Object> swiftData = mapper.readValue(
    new File("swift_confidence.json"), Map.class
);

double swiftConfidence = (Double) swiftData.get("swift_confidence");
System.out.println("Swift confidence: " + swiftConfidence);

// Use in Java model
ModelParameters params = new ModelParameters(0.7, 0.8, 0.6, swiftConfidence);
```

### Full Multi-Language Pipeline

```python
# Python orchestrator
import subprocess
import json
import pandas as pd

def run_java_hmc():
    """Run Java HMC sampling"""
    cmd = [
        "java", "-cp", "out-qualia", "qualia.Core", "hmc_adapt",
        "warmup=1000", "iters=2000", "thin=3", "seed=42",
        "jsonl=data/logs/hmc_results.jsonl"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def run_java_unified():
    """Run Java unified detector"""
    cmd = [
        "java", "-cp", "out-qualia", "qualia.Core", "unified",
        "h=1e-3", "eps=1e-4", "triad=true",
        "jsonl=data/logs/unified_results.jsonl"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def run_swift_confidence():
    """Run Swift confidence assessment"""
    cmd = ["swift", "run", "UOIFCLI", "confidence"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def analyze_results():
    """Analyze all results with Python UQ"""
    # Load HMC results
    hmc_results = pd.read_json("data/logs/hmc_results.jsonl", lines=True)
    
    # Load unified detector results
    unified_results = pd.read_json("data/logs/unified_results.jsonl", lines=True)
    
    # Load Swift confidence
    with open("swift_confidence.json", "r") as f:
        swift_data = json.load(f)
    
    # Create ensemble uncertainty estimate
    ensemble = DeepEnsemble(
        model_class=RandomForestRegressor,
        n_models=5,
        n_estimators=100
    )
    
    # Combine all evidence sources
    evidence = {
        "hmc_samples": len(hmc_results),
        "unified_steps": len(unified_results),
        "swift_confidence": swift_data["swift_confidence"],
        "hmc_mean_psi": hmc_results["meanPsi"].mean() if "meanPsi" in hmc_results else 0.0,
        "unified_mean_psi": unified_results["psi"].mean() if "psi" in unified_results else 0.0
    }
    
    print("Integration Results:")
    for key, value in evidence.items():
        print(f"  {key}: {value}")
    
    return evidence

def main():
    """Run complete integration pipeline"""
    print("Starting multi-language integration pipeline...")
    
    # Run Java components
    print("1. Running Java HMC...")
    if not run_java_hmc():
        print("ERROR: Java HMC failed")
        return
    
    print("2. Running Java unified detector...")
    if not run_java_unified():
        print("ERROR: Java unified detector failed")
        return
    
    # Run Swift component
    print("3. Running Swift confidence...")
    if not run_swift_confidence():
        print("ERROR: Swift confidence failed")
        return
    
    # Analyze results
    print("4. Analyzing results...")
    evidence = analyze_results()
    
    print("Pipeline completed successfully!")
    return evidence

if __name__ == "__main__":
    main()
```

---

## License

This examples file is licensed under `LicenseRef-Internal-Use-Only`.

For full API documentation, see [docs/api_reference.md](api_reference.md).
For quick reference, see [docs/quick_reference.md](quick_reference.md).