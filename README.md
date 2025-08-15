# Hybrid Symbolic-Neural Accuracy Functional

A comprehensive implementation of the hybrid symbolic-neural accuracy functional that combines RK4-based symbolic methods with machine learning approaches for optimal accuracy assessment.

## Mathematical Foundation

The hybrid accuracy functional is defined as:

```
V(x) = (1/T) Σ_{k=1..T} [α(tk)S(x,tk) + (1-α(tk))N(x,tk)] 
       · exp(-[λ1*Rcog(tk) + λ2*Reff(tk)]) 
       · P(H|E,β,tk)
```

Where:
- **S(x,t)**: RK4-based normalized accuracy ∈ [0,1]
- **N(x,t)**: ML/NN-based normalized accuracy ∈ [0,1]
- **α(t)**: Adaptive weight ∈ [0,1] that balances symbolic vs neural
- **Rcog(t)**: Cognitive/theoretical penalty ≥ 0 (e.g., energy drift, constraint violations)
- **Reff(t)**: Efficiency penalty ≥ 0 (e.g., computational cost, memory usage)
- **λ1, λ2**: Penalty weights ≥ 0
- **P(H|E,β,t)**: Calibrated probability of correctness given evidence E and bias β

## Key Features

### 1. Adaptive Weight Scheduling
- **Confidence-based**: Uses model confidence scores with temperature scaling
- **Chaos-based**: Favors neural models in chaotic regions using Lyapunov exponents

### 2. Penalty Computation
- **Cognitive penalties**: Energy drift, constraint violations, ODE residuals
- **Efficiency penalties**: FLOPs, memory usage, inference latency

### 3. Cross-Modal Interaction
- Empirical commutator: `C(m1, m2) = S(m1)N(m2) - S(m2)N(m1)`
- Bounded cross-terms to prevent metric domination

### 4. Broken Neural Scaling Laws (BNSL)
- Predicts neural model performance vs dataset size
- Integrates with efficiency constraints and uncertainty calibration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from hybrid_accuracy import HybridAccuracyFunctional, HybridAccuracyConfig

# Configuration
config = HybridAccuracyConfig(
    lambda1=0.75,  # Cognitive penalty weight
    lambda2=0.25,  # Efficiency penalty weight
    use_cross_modal=True,
    w_cross=0.1
)

# Initialize functional
haf = HybridAccuracyFunctional(config)

# Single time step example
S = np.array([0.65])      # Symbolic accuracy
N = np.array([0.85])      # Neural accuracy
alpha = np.array([0.3])   # Adaptive weight
Rcog = np.array([0.20])   # Cognitive penalty
Reff = np.array([0.15])   # Efficiency penalty
P_base = np.array([0.75]) # Base probability
beta = 1.3                # Bias parameter

# Compute V(x)
V = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
print(f"V(x) = {V:.6f}")  # Should be ~0.638
```

### Adaptive Weight Scheduling

```python
from hybrid_accuracy import AdaptiveWeightScheduler

scheduler = AdaptiveWeightScheduler()

# Confidence-based weights
S_conf = np.random.uniform(0.5, 0.9, 10)
N_conf = np.random.uniform(0.6, 0.95, 10)
alpha_conf = scheduler.confidence_based(S_conf, N_conf, temperature=0.5)

# Chaos-based weights
lyapunov = np.random.uniform(-2.0, 2.0, 10)
alpha_chaos = scheduler.chaos_based(lyapunov, kappa=1.0)
```

### Penalty Computation

```python
from hybrid_accuracy import PenaltyComputers

penalty_comp = PenaltyComputers()

# Energy drift penalty
energy_traj = np.cumsum(np.random.normal(0, 0.1, 100)) + 100
Rcog = penalty_comp.energy_drift_penalty(energy_traj)

# Computational budget penalty
flops = np.random.uniform(0.5e6, 1.5e6, 100)
Reff = penalty_comp.compute_budget_penalty(flops, max_flops=2e6)
```

### Broken Neural Scaling Laws

```python
from hybrid_accuracy import BrokenNeuralScaling

# Fit from historical data
n_values = np.logspace(3, 6, 20)
errors = 0.1 * (n_values ** (-0.3)) + 0.05 * np.random.randn(20)
bnsl = BrokenNeuralScaling.fit_from_data(n_values, errors)

# Predict performance
predicted_accuracy = bnsl.predict_accuracy(1e5)
```

## Advanced Usage

### Multi-Model Evaluation

```python
# Multiple models over time
M, T = 3, 10  # 3 models, 10 time steps
S = np.random.uniform(0.6, 0.9, (M, T))
N = np.random.uniform(0.7, 0.95, (M, T))
alpha = np.random.uniform(0.2, 0.8, T)
Rcog = np.random.uniform(0.1, 0.3, T)
Reff = np.random.uniform(0.05, 0.25, T)
P_base = np.random.uniform(0.6, 0.9, T)

# Compute V(x) for all models
V = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta=1.2)
print(f"V(x) for {M} models: {V}")
```

### Cross-Modal Interaction

```python
# Enable cross-modal terms
config = HybridAccuracyConfig(use_cross_modal=True, w_cross=0.1)
haf_cross = HybridAccuracyFunctional(config)

# Compute with cross-modal interaction between models 0 and 1
V = haf_cross.compute_V(S, N, alpha, Rcog, Reff, P_base, beta,
                        cross_modal_indices=(0, 1))
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest test_hybrid_accuracy.py -v
```

Or run the integration test directly:

```bash
python test_hybrid_accuracy.py
```

## Mathematical Details

### Cross-Modal Commutator

The empirical commutator measures non-commutativity between symbolic and neural representations:

```
C(m1, m2) = S(m1)N(m2) - S(m2)N(m1)
```

This captures the asymmetry in how different models interact with the same data.

### Adaptive Weight Interpretation

- **α(t) → 1**: Favor symbolic methods (stable regions, high confidence)
- **α(t) → 0**: Favor neural methods (chaotic regions, low confidence)

### Penalty Interpretation

- **Rcog(t)**: Physics consistency, energy conservation, constraint satisfaction
- **Reff(t)**: Computational efficiency, memory usage, training cost

### Probability Calibration

The bias parameter β is applied multiplicatively:
```
P(H|E,β,t) = clip(β · P(H|E,t), 0, 1)
```

## Performance Considerations

- **Vectorized operations**: All computations use NumPy for efficiency
- **Memory efficient**: Handles large trajectories without memory issues
- **Scalable**: Supports multiple models and time steps efficiently

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{hybrid_accuracy_functional,
  title={Hybrid Symbolic-Neural Accuracy Functional},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hybrid-accuracy-functional}
}
```

## Acknowledgments

This implementation is based on the formal mathematical framework for hybrid symbolic-neural systems, incorporating concepts from:
- Koopman operator theory
- Broken neural scaling laws
- Adaptive model selection
- Physics-informed machine learning
