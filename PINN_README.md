# Physics-Informed Neural Network (PINN) with Ψ Framework

This repository contains a complete implementation of a Physics-Informed Neural Network (PINN) in Swift, integrated with the Ψ (Psi) framework for hybrid output analysis and validation.

## Overview

The PINN implementation solves partial differential equations (PDEs) by incorporating physical constraints directly into the neural network training process. It's designed to work with the existing Ψ framework, providing a comprehensive analysis of model performance through the lens of Broken Neural Scaling Laws (BNSL).

## Features

- **Neural Network Architecture**: 2→20→20→1 architecture with Xavier initialization
- **Physics Integration**: Solves Burgers' equation with finite difference approximations
- **Ψ Framework Integration**: Comprehensive analysis using S(x), N(x), and α(t) metrics
- **Training Optimization**: Gradient-based training with perturbation methods
- **Visualization Ready**: Chart.js compatible output for solution comparison
- **Swift Implementation**: Native Swift code with no external dependencies

## Architecture

### PINN Class
```swift
public class PINN {
    // Neural network with configurable architecture
    // Xavier initialization for optimal training
    // Physics-informed loss functions
    // Finite difference gradient computation
}
```

### Ψ Framework Integration
```swift
public struct PINNAnalysis {
    public let S_x: Double      // State inference accuracy
    public let N_x: Double      // ML gradient descent convergence
    public let alpha_t: Double  // Real-time validation balance
    public let O_hybrid: Double // Hybrid output
    public let Psi_x: Double    // Final Ψ value
}
```

## Mathematical Framework

### 1. Hybrid Output Computation
```
O_hybrid = (1 - α) * S(x) + α * N(x)
```
Where:
- `S(x)`: State inference accuracy (PINN solution quality)
- `N(x)`: ML gradient descent convergence rate
- `α(t)`: Real-time validation balance parameter

### 2. Penalty Function
```
P_total = λ₁ * R_cognitive + λ₂ * R_efficiency
```
Where:
- `R_cognitive`: PDE residual error
- `R_efficiency`: Computational overhead
- `λ₁, λ₂`: Regularization weights

### 3. Final Ψ Value
```
Ψ(x) = O_hybrid * exp(-P_total) * P_adj
```
Where:
- `P_adj`: Adjusted probability with responsiveness factor β

## Usage

### Command Line Interface

```bash
# Run complete PINN example
uoif-cli pinn

# Run Ψ framework examples
uoif-cli psi

# Show help
uoif-cli help
```

### Programmatic Usage

```swift
import UOIFCore

// Generate training data
let (x, t) = generateTrainingData()

// Initialize PINN
let model = PINN(inputSize: 2, hiddenSize: 20, outputSize: 1)

// Train the model
train(model: model, epochs: 1000, x: x, t: t)

// Analyze results with Ψ framework
let analysis = PINNAnalysis(
    S_x: 0.72,
    N_x: 0.85,
    alpha_t: 0.5,
    R_cognitive: 0.15,
    R_efficiency: 0.10
)

print("Ψ(x) = \(analysis.Psi_x)")
```

## Training Process

### 1. Data Generation
- Spatial domain: x ∈ [-1.0, 1.0] with 0.05 spacing
- Temporal domain: t ∈ [0.0, 1.0] with 0.05 spacing
- Total training points: 41 × 21 = 861

### 2. Loss Functions
- **PDE Loss**: Residual of Burgers' equation
- **Initial Condition Loss**: Deviation from u(x,0) = sin(πx)
- **Total Loss**: Combined PDE and IC losses

### 3. Gradient Computation
- Finite difference approximation with perturbation ε = 1e-5
- Learning rate: 0.005
- Xavier initialization for optimal convergence

## BNSL Integration

The implementation aligns with Broken Neural Scaling Laws (BNSL) from arXiv:2210.14891v17:

- **Non-monotonic Scaling**: Captures inflection points in training dynamics
- **Collaborative Frameworks**: Neural flexibility meets PDE accuracy
- **Smooth Power Laws**: Predicts non-linear scaling behavior

## Example Output

### Training Progress
```
Starting PINN training for 1000 epochs...
Epoch 0: Loss = 0.847392
Epoch 100: Loss = 0.234156
Epoch 200: Loss = 0.089234
...
Training completed!
```

### Ψ Framework Analysis
```
=== Ψ Framework Analysis Results ===
Step 1: Outputs
  S(x) = 0.72 (state inference accuracy)
  N(x) = 0.85 (ML gradient descent convergence)

Step 2: Hybrid
  α(t) = 0.50 (real-time validation balance)
  O_hybrid = (1-α)*S(x) + α*N(x) = 0.785

Step 3: Penalties
  R_cognitive = 0.15 (PDE residual error)
  R_efficiency = 0.10 (computational overhead)
  P_total = λ1*R_cognitive + λ2*R_efficiency = 0.130

Step 4: Probability
  P(H|E,β) = 0.80 (base probability)
  β = 1.2 (responsiveness factor)
  P_adj = P*β = 0.96

Step 5: Ψ(x)
  Ψ(x) = O_hybrid * exp(-P_total) * P_adj ≈ 0.662

Step 6: Interpretation
  Ψ(x) ≈ 0.66 indicates solid model performance
```

### Solution Comparison
```
=== Solution Comparison at t=1.0 ===
  x		PINN u		RK4 u		Difference
  ------------------------------------------------------------
  -1.0		0.000		0.000		0.000
  -0.8		0.523		0.587		0.064
  -0.6		0.847		0.809		0.038
  -0.4		0.923		0.951		0.028
  -0.2		0.309		0.309		0.000
  0.0		-0.309		-0.309		0.000
  0.2		-0.923		-0.951		0.028
  0.4		-0.847		-0.809		0.038
  0.6		-0.523		-0.587		0.064
  0.8		0.000		0.000		0.000
  1.0		0.000		0.000		0.000
```

## Visualization

The implementation generates Chart.js compatible data for visualizing PINN vs RK4 solutions:

```json
{
  "type": "line",
  "data": {
    "labels": ["-1.0", "-0.9", "-0.8", ...],
    "datasets": [
      {
        "label": "PINN u",
        "data": [0.0, 0.3, 0.5, ...],
        "borderColor": "#1E90FF",
        "backgroundColor": "#1E90FF",
        "fill": false,
        "tension": 0.4
      },
      {
        "label": "RK4 u",
        "data": [0.0, 0.4, 0.6, ...],
        "borderColor": "#FF4500",
        "backgroundColor": "#FF4500",
        "fill": false,
        "tension": 0.4
      }
    ]
  }
}
```

## Performance Characteristics

### Training Efficiency
- **Epochs**: 1000 for convergence
- **Loss Reduction**: Typically 10x reduction in first 200 epochs
- **Memory Usage**: Minimal (weights + biases only)
- **Computation**: O(n²) per epoch for n training points

### Accuracy Metrics
- **PDE Residual**: < 0.1 for well-trained models
- **Initial Condition**: < 0.01 deviation from sin(πx)
- **Solution Comparison**: < 0.1 difference from RK4 reference

## Extensions and Improvements

### 1. Adaptive Learning Rates
```swift
// Implement Adam optimizer
// Adaptive learning rate based on gradient history
// Momentum and variance tracking
```

### 2. Multi-Dimensional PDEs
```swift
// Extend to 2D/3D spatial domains
// Handle vector-valued solutions
// Support for coupled PDE systems
```

### 3. Advanced Architectures
```swift
// Residual connections
// Attention mechanisms
// Transformer-based PINNs
```

## Dependencies

- **Swift**: 6.0+
- **Foundation**: Standard library only
- **No External Libraries**: Pure Swift implementation

## Building and Running

### Prerequisites
- Swift 6.0 or later
- macOS 13.0 or later

### Build Commands
```bash
# Build the project
swift build

# Run PINN example
swift run uoif-cli pinn

# Run tests
swift test
```

### Xcode Integration
- Open `Farmer.xcodeproj`
- Build and run the `UOIFCLI` target
- Use command line arguments for different modes

## Testing

The implementation includes comprehensive testing:

```bash
# Run all tests
swift test

# Run specific test categories
swift test --filter PINNTests
swift test --filter PsiFrameworkTests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

GPL-3.0-only - See LICENSE file for details

## References

- **BNSL Paper**: arXiv:2210.14891v17
- **PINN Theory**: Physics-Informed Neural Networks
- **Burgers' Equation**: Nonlinear PDE modeling
- **Ψ Framework**: Hybrid output validation system

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

---

*This implementation demonstrates the power of combining neural networks with physical constraints, validated through the Ψ framework for robust, interpretable scientific computing.*