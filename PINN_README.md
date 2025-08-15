# PINN Hybrid Output Optimization Framework

A comprehensive Swift implementation of Physics-Informed Neural Networks (PINNs) with hybrid output optimization, regularization frameworks, and probability-based validation.

## 🚀 Overview

This implementation combines neural learning with physical constraints for accurate dynamics modeling, featuring:

- **Hybrid Output**: S(x) as state inference for optimized PINN solutions; N(x) as ML gradient descent analysis; α(t) for real-time validation flows
- **Regularization**: R_cognitive for PDE residual accuracy; R_efficiency for training loop efficiency
- **Probability**: P(H|E,β) with β for model responsiveness
- **Integration**: Over training epochs and validation steps

## 🏗️ Architecture

### Core Components

1. **DenseLayer**: Neural network layer with Xavier initialization
2. **PINN**: Physics-Informed Neural Network with hybrid output optimization
3. **HybridOutput**: Combines state inference S(x) and ML gradient descent N(x)
4. **Regularization Framework**: Cognitive and efficiency regularization
5. **Probability Model**: P(H|E,β) with responsiveness parameter β
6. **Performance Metric**: Integrated metric Ψ(x) combining all components

### Mathematical Framework

The system implements the following mathematical structure:

```
Ψ(x) = O_hybrid × exp(-(R_cognitive + R_efficiency)) × P_adj

Where:
- O_hybrid = α × S(x) + (1-α) × N(x)
- R_cognitive = λ1 × PDE_residual
- R_efficiency = λ2 × training_efficiency
- P_adj = min(1.0, P(H|E) × β)
```

## 📁 File Structure

```
Sources/UOIFCore/
├── PINN.swift                 # Core PINN implementation
├── PINNVisualization.swift    # SwiftUI visualization components
└── PINNExample.swift          # Comprehensive examples and demonstrations

Tests/UOIFCoreTests/
└── PINNTests.swift            # Unit tests for all components
```

## 🚀 Quick Start

### Basic Usage

```swift
import UOIFCore

// Create a PINN
let pinn = PINN()

// Generate training data
let x = Array(stride(from: -1.0, to: 1.0, by: 0.1))
let t = Array(repeating: 1.0, count: x.count)

// Train the model
train(model: pinn, epochs: 1000, x: x, t: t, printEvery: 50)

// Make predictions
let prediction = pinn.forward(x: 0.5, t: 1.0)
```

### Performance Metrics

```swift
// Create hybrid output
let hybrid = HybridOutput(stateInference: 0.72, mlGradient: 0.85, alpha: 0.5)

// Set up regularization
let cognitiveReg = CognitiveRegularization(pdeResidual: 0.15, weight: 0.6)
let efficiencyReg = EfficiencyRegularization(trainingEfficiency: 0.10, weight: 0.4)

// Create probability model
let probability = ProbabilityModel(hypothesis: 0.80, beta: 1.2)

// Calculate performance metric
let metric = PerformanceMetric(
    hybridOutput: hybrid,
    cognitiveReg: cognitiveReg,
    efficiencyReg: efficiencyReg,
    probability: probability
)

print("Performance: \(metric.value)")
print("Interpretation: \(metric.interpretation)")
```

## 📊 Visualization

The framework includes comprehensive SwiftUI visualization components:

- **PINNSolutionChart**: Compare PINN vs RK4 solutions
- **TrainingProgressChart**: Monitor training progress
- **PerformanceMetricsDisplay**: Detailed performance analysis
- **PINNDashboard**: Complete dashboard combining all components

### Example Visualization

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        PINNDashboard(
            solutionData: SampleDataGenerator.generateSolutionData(),
            trainingData: SampleDataGenerator.generateTrainingData(),
            performanceMetric: runNumericalExample()
        )
    }
}
```

## 🔬 Numerical Example

The implementation includes the exact numerical example from the requirements:

**Step 1: Outputs**
- S(x) = 0.72 (state inference)
- N(x) = 0.85 (ML gradient descent)

**Step 2: Hybrid**
- α = 0.5
- O_hybrid = 0.785

**Step 3: Penalties**
- R_cognitive = 0.15, λ1 = 0.6
- R_efficiency = 0.10, λ2 = 0.4
- P_total = 0.13
- exp(-P_total) ≈ 0.878

**Step 4: Probability**
- P = 0.80, β = 1.2
- P_adj ≈ 0.96

**Step 5: Ψ(x)**
- ≈ 0.785 × 0.878 × 0.96 ≈ 0.662

**Step 6: Interpretation**
- Ψ(x) ≈ 0.66 indicates solid model performance

## 🧪 Testing

Run the comprehensive test suite:

```bash
swift test
```

The tests cover:
- DenseLayer functionality
- PINN initialization and forward pass
- Hybrid output calculations
- Regularization framework
- Probability models
- Performance metrics
- Numerical utilities
- Training procedures
- The complete numerical example

## 📈 Performance Features

### Optimization Features
- **Xavier Initialization**: Proper weight initialization for stable training
- **Batched Processing**: Efficient PDE loss calculation
- **Configurable Step Sizes**: Adjustable finite difference approximations
- **Gradient Approximation**: Robust training with finite difference gradients

### Training Features
- **Epoch Monitoring**: Configurable progress reporting
- **Loss Tracking**: PDE and initial condition loss monitoring
- **Adaptive Learning**: Configurable learning rates
- **Convergence Monitoring**: Performance metric tracking

## 🔧 Configuration

### Hyperparameters

```swift
// Training parameters
let epochs = 1000
let learningRate = 0.005
let batchSize = 20

// Regularization weights
let lambda1 = 0.6  // Cognitive regularization weight
let lambda2 = 0.4  // Efficiency regularization weight

// Probability parameters
let beta = 1.2     // Model responsiveness
let alpha = 0.5    // Hybrid output weight
```

### Network Architecture

```swift
// Default architecture: 2 → 20 → 20 → 1
let pinn = PINN()

// Custom architecture
let customLayers = [
    DenseLayer(inputSize: 2, outputSize: 32),
    DenseLayer(inputSize: 32, outputSize: 64),
    DenseLayer(inputSize: 64, outputSize: 32),
    DenseLayer(inputSize: 32, outputSize: 1)
]
```

## 🌟 Key Features

### Hybrid Intelligence
- **Balanced Approach**: Merges symbolic RK4 with neural PINN
- **State Inference**: S(x) for optimized PINN solutions
- **ML Gradient Descent**: N(x) for neural learning analysis
- **Real-time Validation**: α(t) for dynamic flow validation

### Interpretability
- **Visualization**: SwiftUI charts for solution comparison
- **Performance Metrics**: Comprehensive Ψ(x) analysis
- **Regularization Analysis**: Cognitive and efficiency breakdown
- **Probability Assessment**: Model confidence evaluation

### Efficiency
- **Swift Implementation**: Native performance optimization
- **Batched Processing**: Efficient training loops
- **Memory Management**: Optimized data structures
- **Parallel Computation**: Vectorized operations

### Human Alignment
- **Clear Metrics**: Intuitive performance interpretation
- **Visual Feedback**: Real-time training progress
- **Configurable Parameters**: Adjustable hyperparameters
- **Comprehensive Documentation**: Detailed usage examples

## 🚀 Advanced Usage

### Custom PDEs

```swift
// Custom PDE residual function
func customPDELoss(model: PINN, x: [Double], t: [Double]) -> Double {
    // Implement your custom PDE here
    // Example: ∂u/∂t + c*∂u/∂x = 0 (wave equation)
    return customResidual
}
```

### Dynamic Regularization

```swift
// Adaptive regularization weights
func adaptiveRegularization(epoch: Int) -> (Double, Double) {
    let cognitiveWeight = 0.6 * exp(-Double(epoch) * 0.001)
    let efficiencyWeight = 0.4 * exp(-Double(epoch) * 0.002)
    return (cognitiveWeight, efficiencyWeight)
}
```

### Real-time Monitoring

```swift
// Custom training loop with monitoring
func customTraining(model: PINN, x: [Double], t: [Double]) {
    for epoch in 0..<1000 {
        trainStep(model: model, x: x, t: t)
        
        if epoch % 10 == 0 {
            let metric = calculatePerformanceMetric(model: model, x: x, t: t)
            print("Epoch \(epoch): Ψ(x) = \(metric.value)")
        }
    }
}
```

## 📚 References

This implementation is based on:
- Physics-Informed Neural Networks (PINNs)
- Hybrid output optimization frameworks
- Regularization theory for neural networks
- Probability models for model validation
- SwiftUI for scientific visualization

## 🤝 Contributing

Contributions are welcome! Please ensure:
- All tests pass
- Code follows Swift style guidelines
- New features include appropriate tests
- Documentation is updated

## 📄 License

This project is licensed under the GPL-3.0-only license.

## 🎯 Future Enhancements

Planned improvements:
- GPU acceleration support
- More PDE types
- Advanced optimization algorithms
- Extended visualization options
- Performance benchmarking tools

---

**Built with ❤️ using Swift and SwiftUI for scientific computing and visualization.**