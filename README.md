# Hybrid Symbolic-Neural Accuracy Functional

A comprehensive implementation of the discrete-time hybrid accuracy functional that combines RK4-based symbolic methods with neural network approaches, incorporating advanced features like cross-modal analysis, Koopman operator theory, and broken neural scaling laws.

## Overview

The hybrid accuracy functional is defined as:

```
V(x) = (1/T) Σ_{k=1..T} [ α(tk) S(x, tk) + (1 − α(tk)) N(x, tk) ] 
       · exp(−[λ1 Rcog(tk) + λ2 Reff(tk)]) · P(H|E, β, tk)
```

Where:
- **S(x, t)**: RK4-based normalized accuracy in [0,1]
- **N(x, t)**: ML/NN-based normalized accuracy in [0,1]  
- **α(t)**: Adaptive weight in [0,1]
- **Rcog(t)**: Cognitive/theoretical penalty ≥ 0
- **Reff(t)**: Efficiency penalty ≥ 0
- **λ1, λ2**: Penalty weights ≥ 0
- **P(H|E, β, t)**: Calibrated probability of correctness in [0,1]

## Key Features

### ✅ Core Components
- **Hybrid Accuracy Functional**: Main V(x) computation with temporal averaging
- **Adaptive Weighting**: Confidence-based and chaos-based α(t) strategies
- **Penalty Functions**: Physics consistency (Rcog) and computational efficiency (Reff)
- **Calibrated Probabilities**: Platt scaling and temperature scaling with bias correction

### ✅ Advanced Analysis
- **Cross-Modal Non-Commutativity**: Empirical commutator C(m1,m2) = S(m1)N(m2) - S(m2)N(m1)
- **Cognitive-Memory Distance Metric**: Multi-dimensional distance with asymmetric interactions
- **Koopman Operator Analysis**: EDMD-based bifurcation detection and reversal handling
- **Broken Neural Scaling Laws**: Performance prediction with smoothly broken power laws

## Installation

### Quick Start (Pure Python)
```bash
# Clone or download the repository
python3 simple_test.py
```

### Full Installation
```bash
pip install numpy scipy matplotlib
python3 examples_and_tests.py
```

## Usage

### Basic Example

```python
from hybrid_accuracy_functional import HybridAccuracyFunctional
import numpy as np

# Initialize the functional
hybrid_func = HybridAccuracyFunctional(lambda1=0.75, lambda2=0.25)

# Single time step example (reproduces specification)
S = np.array([0.65])      # Symbolic accuracy
N = np.array([0.85])      # Neural accuracy
alpha = np.array([0.3])   # Adaptive weight
Rcog = np.array([0.20])   # Cognitive penalty
Reff = np.array([0.15])   # Efficiency penalty

# Apply bias and compute calibrated probability
P_base = np.array([0.75])
P_corr = hybrid_func.calibrated_probability(P_base, bias=1.3)

# Compute V(x)
result = hybrid_func.compute_V(S, N, alpha, Rcog, Reff, P_corr)
print(f"V(x) = {result:.3f}")  # Expected: ~0.638
```

### Multi-Step Trajectory

```python
# Time series data
T = 10
S = 0.7 + 0.2 * np.sin(2 * np.pi * np.linspace(0, 1, T))
N = 0.8 + 0.1 * np.cos(4 * np.pi * np.linspace(0, 1, T))

# Chaos-based adaptive weighting
lyapunov = 0.5 * np.sin(6 * np.pi * np.linspace(0, 1, T))
alpha = hybrid_func.adaptive_weight_chaos(lyapunov, kappa=2.0)

# Physics and efficiency penalties
Rcog = hybrid_func.cognitive_penalty(
    energy_drift=0.1 + 0.2 * np.linspace(0, 1, T),
    ode_residual=0.01 * np.ones(T)
)
Reff = hybrid_func.efficiency_penalty(
    flops_per_step=1000 + 500 * np.random.rand(T),
    normalize=True
)

P_corr = hybrid_func.calibrated_probability(0.8 + 0.1 * np.random.randn(T))
V_result = hybrid_func.compute_V(S, N, alpha, Rcog, Reff, P_corr)
```

### Cross-Modal Analysis

```python
from hybrid_accuracy_functional import CrossModalAnalysis

# Compute non-commutativity
S_outputs = np.random.uniform(0.5, 0.9, (5, 2))  # S at states m1, m2
N_outputs = np.random.uniform(0.6, 0.95, (5, 2))  # N at states m1, m2
m1_states = np.random.randn(5, 3)
m2_states = np.random.randn(5, 3)

commutator = CrossModalAnalysis.compute_commutator(
    S_outputs, N_outputs, m1_states, m2_states
)
interaction_score = CrossModalAnalysis.cross_modal_interaction(commutator)
```

### Koopman Analysis

```python
from hybrid_accuracy_functional import KoopmanAnalysis

# Generate trajectory data
X = np.random.randn(100, 3)  # Current states
Y = np.random.randn(100, 3)  # Next states

koopman = KoopmanAnalysis()
K_matrix = koopman.fit_koopman_edmd(X, Y)
bifurcation_mask = koopman.detect_bifurcations(threshold=0.1)
```

### Broken Neural Scaling Laws

```python
from hybrid_accuracy_functional import BrokenNeuralScalingLaws

# Fit scaling laws to observed data
dataset_sizes = np.logspace(2, 6, 20)
observed_losses = np.random.exponential(0.1, 20)  # Synthetic data

bnsl = BrokenNeuralScalingLaws()
fitted_params = bnsl.fit(dataset_sizes, observed_losses)
predicted_losses = bnsl.predict_performance(np.logspace(2, 7, 50))

# Find optimal dataset size
cost_fn = lambda n: n / 1000  # Linear cost
optimal_n = bnsl.optimal_dataset_size(cost_fn, target_performance=0.1)
```

## Architecture

### Main Classes

1. **`HybridAccuracyFunctional`**: Core functional implementation
   - `compute_V()`: Main functional computation
   - `adaptive_weight_*()`: Different weighting strategies
   - `*_penalty()`: Penalty function computation
   - `calibrated_probability()`: Probability calibration with bias

2. **`CrossModalAnalysis`**: Non-commutativity analysis
   - `compute_commutator()`: Empirical commutator calculation
   - `cross_modal_interaction()`: Interaction score computation

3. **`CognitiveMemoryMetric`**: Distance metric implementation
   - `distance_squared()`: Multi-dimensional distance computation
   - `interaction_score()`: Asymmetric interaction measurement

4. **`KoopmanAnalysis`**: Operator theory and bifurcation detection
   - `fit_koopman_edmd()`: Extended DMD fitting
   - `detect_bifurcations()`: Eigenvalue analysis near unit circle
   - `reversal_constraint_optimization()`: Constrained nonlinear mapping

5. **`BrokenNeuralScalingLaws`**: Performance prediction
   - `scaling_law()`: Broken power law function
   - `fit()`: Parameter estimation from data
   - `optimal_dataset_size()`: Cost-performance optimization

## Validation Results

The implementation has been thoroughly tested and validates against the specification:

### ✅ Specification Example
- **Input**: S=0.65, N=0.85, α=0.3, Rcog=0.20, Reff=0.15, P=0.75, β=1.3
- **Expected**: V(x) ≈ 0.638
- **Actual**: V(x) = 0.639 ✓

### ✅ Key Properties
- Temporal averaging over multiple time steps
- Adaptive weighting based on confidence or chaos indicators  
- Exponential penalty terms for physics consistency and efficiency
- Calibrated probability with bias correction and clipping
- Cross-modal interaction measurement
- Koopman operator bifurcation detection
- Neural scaling law parameter fitting

## Mathematical Foundation

### Discrete-Time Formulation
The functional uses discrete time steps t₀, ..., tₜ with uniform Δt, ensuring:
- Bounded, normalized accuracy scores S, N ∈ [0,1]
- Adaptive weights α(t) ∈ [0,1] 
- Non-negative penalties Rcog(t), Reff(t) ≥ 0
- Calibrated probabilities P(H|E,β,t) ∈ [0,1]

### Penalty Structure
- **Cognitive penalty**: Physics consistency (energy drift, constraint violations, ODE residuals)
- **Efficiency penalty**: Computational cost (FLOPs, memory, latency)
- **Exponential weighting**: exp(-[λ₁Rcog(t) + λ₂Reff(t)]) ensures smooth penalty application

### Cross-Modal Non-Commutativity
Measures ordering effects via empirical commutator:
```
C(m₁, m₂) = S(m₁)N(m₂) - S(m₂)N(m₁)
```

### Broken Neural Scaling Laws
Models performance vs dataset size with smoothly broken power law:
```
L(n) = A n^(-α) [1 + (n/n₀)^γ]^(-δ) + σ
```

## Performance Considerations

- **Memory**: O(T) for time series of length T
- **Computation**: O(T) per V(x) evaluation
- **Scalability**: Efficient vectorized operations with NumPy
- **Numerical Stability**: Clipping, bounds checking, and robust optimization

## Files

- **`hybrid_accuracy_functional.py`**: Main implementation with all classes
- **`examples_and_tests.py`**: Comprehensive test suite (requires NumPy/SciPy)
- **`simple_test.py`**: Pure Python validation (no dependencies)
- **`requirements.txt`**: Package dependencies
- **`README.md`**: This documentation

## Future Extensions

The modular design supports extensions for:
- Custom dictionary functions for Koopman lifting
- Alternative calibration methods beyond Platt/temperature scaling
- Advanced penalty function formulations
- Integration with specific physics simulators or ML frameworks
- Real-time adaptive parameter tuning

## References

This implementation formalizes concepts from:
- Koopman operator theory and Extended Dynamic Mode Decomposition
- Neural scaling laws and broken power law fitting
- Calibrated probability estimation and bias correction
- Cross-modal learning and non-commutativity analysis
- Hybrid symbolic-neural computation paradigms

---

**Status**: ✅ Complete implementation with full validation
**Compatibility**: Python 3.7+ (Pure Python fallback available)
**License**: Open source implementation of the specified hybrid accuracy functional
