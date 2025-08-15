# Physics-Informed Neural Networks (PINN) Framework

## Overview

This repository contains a comprehensive implementation of Physics-Informed Neural Networks (PINNs) that integrates seamlessly with the existing Ψ framework. The implementation demonstrates a hybrid approach combining symbolic methods (RK4) with neural networks for solving partial differential equations (PDEs).

## 🚀 Quick Start

### Python Implementation (Recommended for immediate use)

```bash
cd examples
python3 pinn_simple.py
```

This will run a complete demonstration of the PINN framework, including:
- Neural network creation and training
- Mathematical framework validation
- Performance metrics
- Network analysis

### Swift Implementation (Full integration with Ψ framework)

```bash
# Build the Swift package
swift build

# Run tests
swift test

# Run CLI with PINN integration
swift run uoif-cli
```

## 🧮 Mathematical Foundation

### Core Framework

The framework implements the mathematical formulation:

```
Ψ(x) = O_hybrid × exp(-P_total) × P_adj
```

Where:
- **O_hybrid = α·S + (1-α)·N**: Hybrid output combining symbolic (S) and neural (N) components
- **P_total = λ₁·R_cognitive + λ₂·R_efficiency**: Total penalty from cognitive and efficiency regularization
- **P_adj = min(β·P, 1)**: Adjusted probability with responsiveness parameter β

### Component Breakdown

#### 1. Hybrid Output (O_hybrid)
- **S(x)**: State inference for optimized PINN solutions
- **N(x)**: ML gradient descent analysis
- **α(t)**: Real-time validation flows

#### 2. Regularization Penalties
- **R_cognitive**: PDE residual accuracy
- **R_efficiency**: Training loop efficiency
- **λ₁, λ₂**: Weighting parameters

#### 3. Probability Adjustment
- **P(H|E,β)**: Base probability with model responsiveness β
- **P_adj**: Capped adjusted probability

## 📁 Project Structure

```
├── Sources/UOIFCore/
│   ├── PINN.swift              # Core PINN implementation
│   ├── PsiModel.swift          # Existing Ψ framework
│   ├── Types.swift             # Data structures
│   └── HB.swift                # Hierarchical Bayesian model
├── Sources/UOIFCLI/
│   ├── PINNCommand.swift       # CLI commands for PINN
│   └── main.swift              # Main CLI entry point
├── Tests/UOIFCoreTests/
│   └── PINNTests.swift         # Comprehensive test suite
├── examples/
│   ├── pinn_simple.py          # Python implementation (no dependencies)
│   ├── pinn_demo.py            # Full Python implementation with visualization
│   └── requirements.txt        # Python dependencies
└── docs/
    └── PINN_Framework.md       # Detailed documentation
```

## 🔧 Implementation Details

### Neural Network Architecture

The PINN uses a feedforward neural network with configurable layers:

```swift
// Swift
let model = PINN(layerSizes: [2, 20, 20, 1])

// Python
model = PINN(layer_sizes=[2, 20, 20, 1])
```

Default architecture: `[2, 20, 20, 1]` for 2D input (x, t) → 1D output u(x,t)

### Loss Functions

#### PDE Residual Loss
Implements the Burgers' equation residual: `u_t + u·u_x = 0`

#### Initial Condition Loss
Enforces initial condition: `u(x,0) = -sin(πx)`

#### Boundary Condition Loss
Enforces periodic boundary conditions: `u(-1,t) = u(1,t)`

### Training Process

The training follows the mathematical framework step-by-step:

1. **Outputs**: Compute S(x) and N(x)
2. **Hybrid**: Calculate O_hybrid = α·S + (1-α)·N
3. **Penalties**: Apply regularization R_cognitive and R_efficiency
4. **Probability**: Adjust with responsiveness parameter β
5. **Final**: Compute Ψ(x) = O_hybrid × exp(-P_total) × P_adj

## 📊 Usage Examples

### Basic PINN Creation

```swift
// Swift
let model = PINN(layerSizes: [2, 20, 20, 1])
let u = model.forward(x: 0.5, t: 0.3)
```

```python
# Python
model = PINN(layer_sizes=[2, 20, 20, 1])
u = model.forward(0.5, 0.3)
```

### Training with Hybrid Framework

```swift
// Swift
let trainer = HybridTrainer(model: model, learningRate: 0.01, epochs: 1000)
let history = trainer.train(x: x, t: t)
let finalStep = history.last!
print("Final Ψ(x) = \(finalStep.Psi_x)")
```

```python
# Python
trainer = HybridTrainer(model=model, learning_rate=0.01, epochs=1000)
history = trainer.train(x, t)
final_step = history[-1]
print(f"Final Ψ(x) = {final_step.Psi_x}")
```

## ✅ Mathematical Validation

### Example Calculation

Following the research document:

1. **S(x) = 0.72, N(x) = 0.85**
2. **α = 0.5, O_hybrid = 0.785**
3. **R_cognitive = 0.15, R_efficiency = 0.10**
4. **λ₁ = 0.6, λ₂ = 0.4, P_total = 0.13**
5. **P = 0.80, β = 1.2, P_adj ≈ 0.96**
6. **Ψ(x) ≈ 0.785 × 0.878 × 0.96 ≈ 0.662**

### Verification

The implementation validates this calculation:

```python
step = trainer.training_step(x, t)
print(f"Ψ(x) = {step.Psi_x}")  # Should be approximately 0.662
```

## 🧪 Testing

### Swift Tests

```bash
swift test --filter PINNTests
```

### Python Tests

The Python implementation includes built-in validation and can be run directly:

```bash
python3 pinn_simple.py
```

## 📈 Performance Characteristics

### Computational Complexity

- **Forward Pass**: O(L × N²) where L = layers, N = max neurons per layer
- **Derivative Computation**: O(L × N² × D) where D = derivative order
- **Training Step**: O(L × N² × P) where P = training points

### Memory Usage

- **Model Parameters**: O(L × N²)
- **Training Data**: O(P × D) where P = points, D = dimensions
- **Gradient Computation**: O(L × N²) for finite difference approximation

## 🔗 Integration with Ψ Framework

### Seamless Integration

The PINN framework extends the existing Ψ model:

```swift
// Existing Ψ framework
let psiInputs = PsiInputs(alpha: 0.5, S_symbolic: 0.72, N_external: 0.85, ...)
let outcome = PsiModel.computePsi(inputs: psiInputs)

// PINN extension
let pinnModel = PINN(layerSizes: [2, 20, 20, 1])
let pinnTrainer = HybridTrainer(model: pinnModel)
let pinnStep = pinnTrainer.trainingStep(x: x, t: t)
```

### Shared Mathematical Foundation

Both frameworks use the same core equation:
- **Ψ Framework**: For general hybrid modeling
- **PINN Framework**: For physics-informed neural networks

## 🚀 Advanced Features

### Custom PDE Support

The framework can be extended to support other PDEs:

```swift
// Heat equation: u_t = α·u_xx
func heatEquationLoss(model: PINN, x: [Double], t: [Double], alpha: Double) -> Double {
    // Implementation for heat equation
}

// Wave equation: u_tt = c²·u_xx
func waveEquationLoss(model: PINN, x: [Double], t: [Double], c: Double) -> Double {
    // Implementation for wave equation
}
```

### Adaptive Training

The framework supports adaptive learning rates and regularization:

```swift
let trainer = HybridTrainer(
    model: model,
    learningRate: 0.01,  // Initial learning rate
    epochs: 1000
)

// Adaptive regularization based on training progress
let adaptiveLambda = 0.1 * exp(-epoch / 100.0)
```

## 🔮 Future Enhancements

### Planned Features

1. **Automatic Differentiation**: Integration with Swift for TensorFlow
2. **GPU Acceleration**: CUDA/OpenCL support for large-scale problems
3. **Multi-Physics**: Support for coupled PDE systems
4. **Uncertainty Quantification**: Bayesian PINN implementation
5. **Adaptive Meshing**: Dynamic grid refinement

### Research Directions

- **Broken Neural Scaling Laws**: Integration with BNSL framework
- **Hybrid Symbolic-Neural**: Enhanced RK4-PINN coupling
- **Real-time Validation**: Dynamic α(t) adaptation
- **Cognitive Regularization**: Advanced R_cognitive formulations

## 📚 Documentation

- **PINN_Framework.md**: Comprehensive technical documentation
- **Code Comments**: Extensive inline documentation
- **Examples**: Working demonstration scripts
- **Tests**: Validation and verification suite

## 🤝 Contributing

The PINN framework is designed to be extensible. Key areas for contribution:

1. **New PDE Types**: Implement loss functions for additional equations
2. **Optimization**: Improve training algorithms and efficiency
3. **Visualization**: Enhanced plotting and analysis tools
4. **Documentation**: Additional examples and tutorials

## 📄 License

This implementation follows the same licensing as the main Ψ framework:

- **SPDX-License-Identifier**: GPL-3.0-only
- **Copyright**: 2025 Jumping Quail Solutions

## 🎯 Conclusion

The PINN framework provides a robust, mathematically sound implementation of physics-informed neural networks that seamlessly integrates with the existing Ψ framework. It offers:

- **Mathematical Rigor**: Faithful implementation of the research framework
- **Performance**: Efficient Swift implementation with optimization
- **Extensibility**: Easy to extend for new PDEs and applications
- **Integration**: Seamless integration with existing Ψ infrastructure
- **Validation**: Comprehensive testing and mathematical verification

This implementation serves as a foundation for advanced physics-informed machine learning applications while maintaining the mathematical coherence and interpretability of the original Ψ framework.

## 🚀 Getting Started Checklist

- [ ] Run Python demonstration: `python3 examples/pinn_simple.py`
- [ ] Review mathematical framework in documentation
- [ ] Explore Swift implementation structure
- [ ] Run Swift tests: `swift test`
- [ ] Experiment with custom PDE implementations
- [ ] Integrate with existing Ψ framework workflows

For questions or contributions, please refer to the main project documentation and contribution guidelines.