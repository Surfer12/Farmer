# Hybrid Symbolic-Neural Accuracy Functional

This repository implements a sophisticated mathematical framework for balancing symbolic and neural methods in AI systems, with applications ranging from chaotic system modeling to open-source collaboration assessment.

## Mathematical Framework

### Core Functional

The hybrid accuracy functional Ψ(x) is defined as:

```
Ψ(x) = (1/T) * Σ[α(t_k)S(x,t_k) + (1-α(t_k))N(x,t_k)] * 
       exp(-[λ₁R_cog(t_k) + λ₂R_eff(t_k)]) * P(H|E,β,t_k)
```

Where:

- **S(x,t) ∈ [0,1]**: Symbolic accuracy (e.g., RK4 solution fidelity)
- **N(x,t) ∈ [0,1]**: Neural accuracy (e.g., ML/NN prediction fidelity)
- **α(t) ∈ [0,1]**: Adaptive weight favoring N in chaotic regions
- **R_cog(t) ≥ 0**: Cognitive penalty (e.g., physics violations)
- **R_eff(t) ≥ 0**: Efficiency penalty (e.g., computational cost)
- **λ₁, λ₂ ≥ 0**: Regularization weights
- **P(H|E,β,t)**: Calibrated probability of correctness with bias β

### Adaptive Weight Function

The adaptive weight α(t) is computed using:

```
α(t) = σ(-κ * λ_local(t))
```

Where σ is the sigmoid function and λ_local(t) is the local Lyapunov exponent, favoring neural methods in chaotic regions.

### Probability Calibration

The probability is calibrated using a logit shift:

```
P' = σ(logit(P) + log(β))
```

This allows for bias adjustment to account for system responsiveness.

## Implementation

### Core Classes

#### `FunctionalParams`
Configuration dataclass for the functional parameters:
- `lambda1`: Cognitive regularization weight (default: 0.75)
- `lambda2`: Efficiency regularization weight (default: 0.25)
- `kappa`: Adaptive weight parameter (default: 1.0)
- `beta`: Probability bias parameter (default: 1.2)

#### `HybridFunctional`
Main class implementing the hybrid functional:

- `compute_adaptive_weight(t, lyapunov_exponent)`: Computes α(t)
- `compute_hybrid_output(S, N, alpha)`: Computes αS + (1-α)N
- `compute_regularization_penalty(R_cog, R_eff)`: Computes exp(-[λ₁R_cog + λ₂R_eff])
- `compute_probability(base_prob, beta)`: Calibrates probability with bias
- `compute_single_step_psi(...)`: Computes Ψ(x) for single time step
- `compute_multi_step_psi(...)`: Computes Ψ(x) over multiple time steps

## Usage Examples

### Basic Usage

```python
from hybrid_functional import HybridFunctional, FunctionalParams

# Initialize with default parameters
functional = HybridFunctional()

# Compute single-step Ψ(x)
psi = functional.compute_single_step_psi(
    S=0.67,      # Symbolic accuracy
    N=0.87,      # Neural accuracy
    alpha=0.4,   # Adaptive weight
    R_cog=0.17,  # Cognitive penalty
    R_eff=0.11,  # Efficiency penalty
    base_prob=0.81  # Base probability
)

print(f"Ψ(x) = {psi:.3f}")
```

### Custom Parameters

```python
# Custom parameters for different applications
params = FunctionalParams(
    lambda1=0.8,    # Emphasize cognitive accuracy
    lambda2=0.2,    # De-emphasize efficiency
    kappa=1.5,      # More sensitive to chaos
    beta=1.0        # No probability bias
)

functional = HybridFunctional(params)
```

### Multi-step Computation

```python
# Compute over multiple time steps
S = [0.6, 0.7, 0.8]      # Symbolic accuracies
N = [0.8, 0.7, 0.6]      # Neural accuracies
alphas = [0.5, 0.5, 0.5] # Adaptive weights
R_cog = [0.1, 0.1, 0.1]  # Cognitive penalties
R_eff = [0.1, 0.1, 0.1]  # Efficiency penalties
base_probs = [0.9, 0.9, 0.9]  # Base probabilities

psi = functional.compute_multi_step_psi(
    S, N, alphas, R_cog, R_eff, base_probs
)
```

## Numerical Examples

### Example 1: Single Tracking Step

**Inputs:**
- S(x) = 0.67, N(x) = 0.87
- α = 0.4
- R_cognitive = 0.17, R_efficiency = 0.11
- λ₁ = 0.6, λ₂ = 0.4
- P = 0.81, β = 1.2

**Calculation:**
1. Hybrid output: O_hybrid = 0.4 × 0.67 + 0.6 × 0.87 = 0.794
2. Regularization: exp(-(0.6 × 0.17 + 0.4 × 0.11)) ≈ 0.864
3. Probability: P_adj ≈ 0.972
4. Final result: Ψ(x) ≈ 0.794 × 0.864 × 0.972 ≈ 0.667

**Interpretation:** Ψ(x) ≈ 0.67 indicates high responsiveness.

### Example 2: Open Source Contributions

**Inputs:**
- S(x) = 0.74, N(x) = 0.84
- α = 0.5
- R_cognitive = 0.14, R_efficiency = 0.09
- λ₁ = 0.55, λ₂ = 0.45
- P = 0.77, β = 1.3

**Result:** Ψ(x) ≈ 0.70 reflects strong innovation potential.

## Applications

### 1. Chaotic System Modeling
- **Symbolic (S)**: High-fidelity numerical methods (RK4)
- **Neural (N)**: Adaptive ML models (LSTM/GRU)
- **Adaptive weight**: Favors neural methods in chaotic regions
- **Use case**: Multi-pendulum dynamics, weather prediction

### 2. Open Source Collaboration
- **Symbolic (S)**: Methodology quality and tool standards
- **Neural (N)**: Dataset richness and community engagement
- **Adaptive weight**: Balances sharing with innovation
- **Use case**: Assessing contribution impact and innovation potential

### 3. Healthcare and Education
- **Symbolic (S)**: Clinical guidelines and educational frameworks
- **Neural (N)**: Personalized treatment and adaptive learning
- **Adaptive weight**: Balances standardization with personalization
- **Use case**: Phased project implementation and regulatory compliance

## Mathematical Properties

### Boundedness
- Ψ(x) ∈ [0,1] for all valid inputs
- Ensures interpretable and comparable results

### Continuity
- Continuous with respect to all input parameters
- Enables gradient-based optimization

### Interpretability
- Each component has clear physical/operational meaning
- Facilitates debugging and parameter tuning

## Testing

Run the comprehensive test suite:

```bash
pytest test_hybrid_functional.py -v
```

The tests verify:
- Mathematical correctness
- Edge case handling
- Parameter validation
- Numerical examples from the documentation

## Dependencies

- `numpy >= 1.21.0`: Numerical computations
- `matplotlib >= 3.5.0`: Visualization
- `pytest >= 6.0.0`: Testing framework

## Installation

```bash
pip install -r requirements.txt
```

## Theoretical Background

This framework addresses the challenge of balancing symbolic (rule-based) and neural (data-driven) approaches in AI systems. The key insight is that different methods excel in different regimes:

- **Symbolic methods** provide high fidelity and interpretability in well-understood domains
- **Neural methods** adapt to complex, chaotic, or data-rich environments
- **Adaptive weighting** ensures optimal method selection based on local conditions

The regularization terms ensure that accuracy gains don't come at the cost of computational efficiency or theoretical consistency, while the probability calibration accounts for system responsiveness and uncertainty.

## References

- Broken Neural Scaling Laws (BNSL) paper (arXiv:2210.14891v17)
- Multi-pendulum chaotic dynamics
- Hybrid AI system design principles
- Probability calibration in machine learning

## License

This project is open source and available under the MIT License.
