# Hybrid PINN-RK4 System

A Swift implementation of a hybrid Physics-Informed Neural Network (PINN) and Runge-Kutta 4th order (RK4) system with cognitive regularization and real-time validation flows.

## Mathematical Formulation

The system implements the hybrid output formula:

**Ψ(x) = O_hybrid × exp(-P_total) × P(H|E,β)**

Where:
- **S(x)**: State inference from PINN solutions
- **N(x)**: ML gradient descent analysis using RK4
- **α(t)**: Real-time validation flow parameter
- **O_hybrid**: α·S(x) + (1-α)·N(x)
- **R_cognitive**: PDE residual accuracy regularization
- **R_efficiency**: Training loop efficiency regularization
- **P_total**: λ₁·R_cognitive + λ₂·R_efficiency
- **P(H|E,β)**: Probability with model responsiveness β

## Features

### 🧠 Cognitive Intelligence
- **Balanced Learning**: Merges symbolic RK4 with neural PINN approaches
- **Physics-Informed**: Enforces PDE constraints during training
- **Adaptive Parameters**: Real-time adjustment of α(t) based on performance

### 🎯 Optimization
- **Xavier Initialization**: Proper weight initialization for stable training
- **Batch Processing**: Efficient training with configurable batch sizes
- **Finite Difference Gradients**: Accurate gradient approximation
- **Regularization**: Prevents overfitting with cognitive and efficiency penalties

### 📊 Visualization
- **SwiftUI Interface**: Interactive training and solution visualization
- **Real-time Charts**: Compare PINN, RK4, hybrid, and analytical solutions
- **Training Metrics**: Live monitoring of loss, Ψ(x), and regularization terms

## File Structure

```
├── HybridPINNRK4.swift     # Main implementation
├── DemoScript.swift        # Command-line demonstration
└── README.md              # This file
```

## Quick Start

### 1. Command Line Demo

Run the demonstration script to see the numerical example:

```bash
swift DemoScript.swift
```

This will output:
- Step-by-step calculation of Ψ(x)
- Training simulation with adaptive α(t)
- Real-time validation results
- Solution evolution over time

### 2. SwiftUI Application

For the interactive visualization:

1. Open Xcode
2. Create a new SwiftUI project
3. Import `HybridPINNRK4.swift`
4. Use `HybridPINNVisualizationView()` as your main view

```swift
import SwiftUI

@main
struct HybridPINNApp: App {
    var body: some Scene {
        WindowGroup {
            HybridPINNVisualizationView()
        }
    }
}
```

## Numerical Example

The system reproduces the numerical example with these parameters:
- α = 0.5, λ₁ = 0.6, λ₂ = 0.4, β = 1.2

**Expected vs Actual Results:**
```
Step 1: S(x) ≈ 0.72, N(x) ≈ 0.85
Step 2: O_hybrid ≈ 0.785
Step 5: Ψ(x) ≈ 0.66 (indicates solid model performance)
```

## Key Components

### HybridPINNRK4System
Main system class that orchestrates:
- PINN neural network training
- RK4 numerical integration
- Hybrid output computation
- Regularization and probability calculations

### PINN (Physics-Informed Neural Network)
- 4-layer deep neural network (2→32→32→32→1)
- Tanh activation functions
- Xavier weight initialization
- Forward pass for solution approximation

### RK4Solver
- Classical Runge-Kutta 4th order method
- Solves heat equation: ∂u/∂t = ∂²u/∂x²
- Initial condition: u(x,0) = -sin(πx)

### Regularization Components
- **R_cognitive**: Measures PDE residual accuracy
- **R_efficiency**: Accounts for computational cost
- **Adaptive α(t)**: Real-time parameter adjustment

## Training Process

1. **Data Generation**: Spatial-temporal grid points
2. **Forward Pass**: Compute hybrid output Ψ(x)
3. **Loss Calculation**: Compare with analytical solution
4. **Parameter Update**: Finite difference gradient approximation
5. **Regularization**: Apply cognitive and efficiency penalties
6. **Validation**: Real-time performance assessment

## Visualization Features

### Interactive Controls
- **Time Slider**: Explore solutions at different time points
- **Training Controls**: Start/stop training with real-time updates
- **Data Generation**: Create new training datasets

### Charts
- **Solution Comparison**: PINN vs RK4 vs Hybrid vs Analytical
- **Training Progress**: Loss and Ψ(x) evolution over epochs
- **Real-time Metrics**: Current performance indicators

## Performance Interpretation

The system provides automatic interpretation of Ψ(x) values:
- **Ψ(x) ≥ 0.8**: Excellent model performance
- **0.6 ≤ Ψ(x) < 0.8**: Good model performance  
- **0.4 ≤ Ψ(x) < 0.6**: Moderate model performance
- **0.2 ≤ Ψ(x) < 0.4**: Poor model performance
- **Ψ(x) < 0.2**: Very poor model performance

## Advanced Features

### Real-Time Validation
- Continuous assessment of model performance
- Adaptive parameter adjustment based on metrics
- Early stopping when convergence is achieved

### Cognitive Regularization
- Physics-based constraints enforcement
- PDE residual minimization
- Balance between accuracy and efficiency

### Human Alignment
- Interpretable visualizations
- Clear performance metrics
- Intuitive parameter controls

## Dependencies

- **Foundation**: Core Swift functionality
- **SwiftUI**: User interface framework
- **Charts**: Data visualization (iOS 16.0+)
- **Accelerate**: High-performance computations (optional)

## Usage Tips

1. **Training**: Start with default parameters, then adjust based on performance
2. **Visualization**: Use time slider to understand solution evolution
3. **Performance**: Monitor Ψ(x) for model quality assessment
4. **Debugging**: Check individual S(x) and N(x) components if issues arise

## Mathematical Background

The system solves the heat equation:
```
∂u/∂t = ∂²u/∂x²
u(x,0) = -sin(πx)
```

With analytical solution:
```
u(x,t) = -sin(πx) × exp(-π²t)
```

The hybrid approach combines:
- **Neural approximation** (PINN) for flexibility
- **Numerical integration** (RK4) for accuracy
- **Regularization** for stability
- **Probabilistic validation** for reliability

## Future Enhancements

- GPU acceleration with Metal Performance Shaders
- Additional PDE types (wave, Schrödinger, etc.)
- Automatic hyperparameter optimization
- Export capabilities for trained models
- Integration with Core ML for deployment

---

*This implementation demonstrates the power of combining traditional numerical methods with modern machine learning approaches for solving partial differential equations.*
