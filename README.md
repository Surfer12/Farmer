# Hybrid Symbolic-Neural Accuracy Functional Ψ(x)

This repository implements the hybrid symbolic-neural accuracy functional framework that combines symbolic reasoning (S(x)) with neural network analysis (N(x)) through adaptive weighting (α(t)) and regularization.

## Mathematical Framework

The hybrid functional Ψ(x) is defined as:

```
Ψ(x) = (1/T) * Σ[α(t_k)S(x,t_k) + (1-α(t_k))N(x,t_k)] * 
        exp(-[λ₁R_cog(t_k) + λ₂R_eff(t_k)]) * P(H|E,β,t_k)
```

Where:
- **S(x,t)**: Symbolic accuracy [0,1] - e.g., normalized RK4 solution fidelity
- **N(x,t)**: Neural accuracy [0,1] - e.g., normalized ML/NN prediction fidelity  
- **α(t)**: Adaptive weight [0,1] - favors N in chaotic regions
- **R_cog(t)**: Cognitive penalty ≥0 - e.g., physics violation
- **R_eff(t)**: Efficiency penalty ≥0 - e.g., normalized FLOPs/latency
- **λ₁, λ₂**: Regularization weights ≥0
- **P(H|E,β,t)**: Calibrated probability of correctness with bias β

## Key Features

### 1. Balanced Intelligence
- Merges symbolic tracking with neural analysis
- Adaptive weighting based on system dynamics
- Continuous monitoring and optimization

### 2. Interpretability
- Infers system states like flow
- Transparent component breakdown
- Human-aligned decision making

### 3. Efficiency
- Handles real-time data processing
- Optimized regularization penalties
- Scalable time series integration

## Implementation

### Core Class: `HybridSymbolicNeural`

```python
hybrid_system = HybridSymbolicNeural(lambda1=0.6, lambda2=0.4)
psi, details = hybrid_system.compute_psi(
    S=0.67, N=0.87, alpha=0.4,
    R_cognitive=0.17, R_efficiency=0.11,
    P=0.81, beta=1.2
)
```

### Methods

- `compute_hybrid_output(S, N, alpha)`: αS + (1-α)N
- `compute_regularization_penalty(R_cog, R_eff)`: exp(-[λ₁R_cog + λ₂R_eff])
- `compute_probability_adjustment(P, beta)`: Logit shift with bias β
- `compute_psi(...)`: Complete Ψ(x) calculation
- `compute_psi_time_series(...)`: Time-averaged integration

## Numerical Examples

### Single Tracking Step
```
Step 1: S(x) = 0.67, N(x) = 0.87
Step 2: α = 0.4, O_hybrid = 0.794
Step 3: R_cognitive = 0.17, R_efficiency = 0.11
         λ₁ = 0.6, λ₂ = 0.4
         P_total = 0.146, exp ≈ 0.864
Step 4: P = 0.81, β = 1.2, P_adj ≈ 0.972
Step 5: Ψ(x) ≈ 0.794 × 0.864 × 0.972 ≈ 0.667
Step 6: Ψ(x) ≈ 0.67 indicates high responsiveness
```

### Open-Source Contribution
```
S(x) = 0.74, N(x) = 0.84
α = 0.5, O_hybrid = 0.79
R_cognitive = 0.14, R_efficiency = 0.09
P = 0.77, β = 1.3, P_adj = 1.0
Ψ(x) ≈ 0.70 reflects strong innovation potential
```

## Applications

### 1. Multi-Pendulum Systems
- Symbolic: RK4 ODE solutions
- Neural: LSTM/GRU predictions
- Adaptive weighting in chaotic regions

### 2. Real-Time Monitoring
- Physiological accuracy tracking
- Data processing efficiency
- Responsive AI systems

### 3. Collaborative AI
- Tool quality assessment
- Release cost optimization
- Community impact evaluation

## Visualization

The implementation includes comprehensive visualizations:

1. **Hybrid Output vs α**: Shows how weighting affects combined accuracy
2. **Regularization Penalty**: 2D surface of cognitive vs efficiency penalties
3. **Probability Adjustment**: Effect of bias β on probability calibration
4. **Ψ(x) Surface**: Complete functional behavior across parameter space
5. **Time Series Analysis**: Dynamic behavior over monitoring cycles

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Run Examples
```bash
python hybrid_symbolic_neural.py
```

### Custom Implementation
```python
from hybrid_symbolic_neural import HybridSymbolicNeural

# Initialize with custom regularization weights
system = HybridSymbolicNeural(lambda1=0.75, lambda2=0.25)

# Compute Ψ(x) for your system
psi, details = system.compute_psi(
    S=your_symbolic_accuracy,
    N=your_neural_accuracy,
    alpha=your_adaptive_weight,
    R_cognitive=your_cognitive_penalty,
    R_efficiency=your_efficiency_penalty,
    P=your_base_probability,
    beta=your_responsiveness_bias
)
```

## Theoretical Foundation

This framework is inspired by:
- **Broken Neural Scaling Laws (BNSL)**: arXiv:2210.14891v17
- **Hybrid AI Systems**: Combining symbolic and neural approaches
- **Adaptive Control Theory**: Dynamic weighting based on system state
- **Bayesian Inference**: Probability calibration with bias adjustment

## Contributing

Contributions are welcome! The framework is designed to be extensible for:
- New regularization schemes
- Alternative probability models
- Additional visualization methods
- Performance optimizations

## License

This project is open source and available under the MIT License.
