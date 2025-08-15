# Quick Start Guide: Hybrid Symbolic-Neural Accuracy Functional

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install numpy scipy matplotlib scikit-learn
```

### 2. Basic Usage - Your Example
```python
import numpy as np
from hybrid_accuracy import HybridAccuracyFunctional, HybridAccuracyConfig

# Configuration
config = HybridAccuracyConfig(lambda1=0.75, lambda2=0.25)
haf = HybridAccuracyFunctional(config)

# Your numerical example from the formalization
S = np.array([0.65])      # Symbolic accuracy
N = np.array([0.85])      # Neural accuracy  
alpha = np.array([0.3])   # Adaptive weight
Rcog = np.array([0.20])   # Cognitive penalty
Reff = np.array([0.15])   # Efficiency penalty
P_base = np.array([0.75]) # Base probability
beta = 1.3                # Bias parameter

# Compute V(x)
V = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
print(f"V(x) = {V:.6f}")  # Should output: V(x) = 0.638560
```

### 3. Multi-Time Step Example
```python
# Generate trajectory data
T = 10
S = np.random.uniform(0.6, 0.9, T)      # Symbolic accuracy over time
N = np.random.uniform(0.7, 0.95, T)     # Neural accuracy over time
alpha = np.random.uniform(0.2, 0.8, T)  # Adaptive weights over time
Rcog = np.random.uniform(0.1, 0.3, T)   # Cognitive penalties over time
Reff = np.random.uniform(0.05, 0.25, T) # Efficiency penalties over time
P_base = np.random.uniform(0.6, 0.9, T) # Base probabilities over time

# Compute V(x) for the trajectory
V = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta=1.2)
print(f"Trajectory V(x) = {V:.6f}")
```

### 4. Adaptive Weight Scheduling
```python
from hybrid_accuracy import AdaptiveWeightScheduler

scheduler = AdaptiveWeightScheduler()

# Confidence-based weights
S_conf = np.random.uniform(0.5, 0.9, T)  # Symbolic confidence
N_conf = np.random.uniform(0.6, 0.95, T) # Neural confidence
alpha_conf = scheduler.confidence_based(S_conf, N_conf, temperature=0.5)

# Chaos-based weights (favor symbolic in stable regions)
lyapunov = np.random.uniform(-2.0, 2.0, T)  # Lyapunov exponents
alpha_chaos = scheduler.chaos_based(lyapunov, kappa=1.0)

print(f"Confidence-based α range: [{alpha_conf.min():.3f}, {alpha_conf.max():.3f}]")
print(f"Chaos-based α range: [{alpha_chaos.min():.3f}, {alpha_chaos.max():.3f}]")
```

### 5. Penalty Computation
```python
from hybrid_accuracy import PenaltyComputers

penalty_comp = PenaltyComputers()

# Energy drift penalty (cognitive)
energy_traj = np.cumsum(np.random.normal(0, 0.1, T)) + 100
Rcog = penalty_comp.energy_drift_penalty(energy_traj)

# Computational budget penalty (efficiency)
flops = np.random.uniform(0.5e6, 1.5e6, T)
Reff = penalty_comp.compute_budget_penalty(flops, max_flops=2e6)

print(f"Energy penalty range: [{Rcog.min():.3f}, {Rcog.max():.3f}]")
print(f"Budget penalty range: [{Reff.min():.3f}, {Reff.max():.3f}]")
```

### 6. Broken Neural Scaling Laws
```python
from hybrid_accuracy import BrokenNeuralScaling

# Generate synthetic scaling data
n_values = np.logspace(3, 6, 20)
errors = 0.1 * (n_values ** (-0.3)) + 0.05 * np.random.randn(20)

# Fit BNSL
bnsl = BrokenNeuralScaling.fit_from_data(n_values, errors)

# Predict performance
predicted_accuracy = bnsl.predict_accuracy(1e5)
print(f"Predicted accuracy at n=1e5: {predicted_accuracy:.3f}")
```

### 7. Cross-Modal Interaction
```python
# Enable cross-modal terms
config = HybridAccuracyConfig(
    lambda1=0.7, 
    lambda2=0.3, 
    use_cross_modal=True, 
    w_cross=0.1
)
haf_cross = HybridAccuracyFunctional(config)

# Multiple models
M, T = 2, 5
S = np.random.uniform(0.6, 0.9, (M, T))
N = np.random.uniform(0.7, 0.95, (M, T))
alpha = np.random.uniform(0.2, 0.8, T)
Rcog = np.random.uniform(0.1, 0.3, T)
Reff = np.random.uniform(0.05, 0.25, T)
P_base = np.random.uniform(0.6, 0.9, T)

# Compute with cross-modal interaction
V = haf_cross.compute_V(S, N, alpha, Rcog, Reff, P_base, beta=1.2,
                        cross_modal_indices=(0, 1))
print(f"V(x) with cross-modal: {V}")
```

## 🔧 Configuration Options

### HybridAccuracyConfig Parameters
```python
config = HybridAccuracyConfig(
    lambda1=0.75,           # Cognitive penalty weight
    lambda2=0.25,           # Efficiency penalty weight  
    clip_probability=True,   # Clip P(H|E,β,t) to [0,1]
    use_cross_modal=False,  # Enable cross-modal interaction
    w_cross=0.1            # Cross-modal term weight
)
```

### Adaptive Weight Parameters
```python
# Confidence-based
alpha = scheduler.confidence_based(S_conf, N_conf, temperature=0.5)
# Lower temperature = more extreme weights

# Chaos-based  
alpha = scheduler.chaos_based(lyapunov, kappa=1.0)
# Higher kappa = sharper transition between stable/chaotic
```

## 📊 What Each Component Does

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **HybridAccuracyFunctional** | Main computation | Implements V(x) formula, handles multiple models/time steps |
| **AdaptiveWeightScheduler** | Dynamic weight α(t) | Confidence-based, chaos-based, temperature scaling |
| **PenaltyComputers** | Physics/efficiency penalties | Energy drift, constraints, computational budget |
| **BrokenNeuralScaling** | Performance prediction | Dataset size scaling, parameter fitting |
| **Cross-Modal Interaction** | Model interaction analysis | Commutator computation, bounded cross-terms |

## 🎯 Common Use Cases

### 1. **Model Selection**
```python
# Compare different approaches
V_symbolic = haf.compute_V(S_high, N_low, alpha_high, Rcog_low, Reff_low, P_base, beta)
V_neural = haf.compute_V(S_low, N_high, alpha_low, Rcog_high, Reff_high, P_base, beta)

best_approach = "Symbolic" if V_symbolic > V_neural else "Neural"
print(f"Best approach: {best_approach}")
```

### 2. **Resource Optimization**
```python
# Find optimal dataset size
dataset_sizes = np.logspace(3, 6, 20)
accuracies = [bnsl.predict_accuracy(n) for n in dataset_sizes]
costs = [compute_cost(n) for n in dataset_sizes]

# Balance accuracy vs cost
optimal_idx = np.argmax([acc/cost for acc, cost in zip(accuracies, costs)])
optimal_size = dataset_sizes[optimal_idx]
print(f"Optimal dataset size: {optimal_size:.0e}")
```

### 3. **Performance Monitoring**
```python
# Track V(x) over time
V_trajectory = []
for t in range(T):
    V_t = haf.compute_V(S[:, t], N[:, t], alpha[t], Rcog[t], Reff[t], P_base[t], beta)
    V_trajectory.append(V_t)

# Detect performance degradation
degradation = np.mean(V_trajectory[-5:]) < np.mean(V_trajectory[:5]) * 0.9
if degradation:
    print("Performance degradation detected!")
```

## 🚨 Troubleshooting

### Common Issues
1. **Import Error**: Make sure all dependencies are installed
2. **Shape Mismatch**: Ensure arrays have compatible dimensions
3. **NaN Values**: Check for division by zero in penalty computations
4. **Memory Issues**: Use smaller time windows for very long trajectories

### Getting Help
- Run `python3 simple_test.py` to verify installation
- Check the comprehensive test suite in `test_hybrid_accuracy.py`
- Review the full documentation in `README.md`
- See the implementation summary in `IMPLEMENTATION_SUMMARY.md`

## 🎉 You're Ready!

You now have a **complete, tested implementation** of the hybrid symbolic-neural accuracy functional that:

✅ **Exactly reproduces** your mathematical formalization  
✅ **Handles** real-world data and multiple models  
✅ **Provides** adaptive weight scheduling strategies  
✅ **Includes** physics-informed penalty computation  
✅ **Supports** cross-modal interaction analysis  
✅ **Integrates** broken neural scaling laws  

Start using it immediately for your research, development, or production systems!