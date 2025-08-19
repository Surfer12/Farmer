# Physics-Informed Neural Network (PINN) in Swift

A complete implementation of Physics-Informed Neural Networks for solving Burgers' equation, featuring a hybrid output framework with Broken Neural Scaling Laws (BNSL) integration, SwiftUI visualization, and RK4 numerical comparison.

## 🚀 Features

### Core PINN Implementation
- **Neural Network Architecture**: 20-neuron hidden layer with tanh activation
- **Xavier Initialization**: Optimized weight initialization for better convergence
- **PDE Integration**: Solves Burgers' equation: `u_t + u*u_x - ν*u_xx = 0`
- **Finite Difference**: Automatic differentiation using finite differences
- **Boundary Conditions**: Dirichlet boundary conditions `u(-1,t) = u(1,t) = 0`
- **Initial Condition**: `u(x,0) = -sin(π*x)`

### Hybrid Output Framework
- **S(x)**: State inference for PINN solution accuracy
- **N(x)**: ML gradient descent analysis for convergence tracking
- **α(t)**: Real-time validation flow balance
- **Ψ(x)**: Combined performance metric with regularization penalties
- **BNSL Integration**: Broken Neural Scaling Laws analysis for non-monotonic training behavior

### Numerical Comparison
- **RK4 Solver**: Fourth-order Runge-Kutta method for reference solutions
- **Error Analysis**: MSE and maximum error computation
- **Inflection Point Detection**: BNSL-based analysis of training dynamics

### Visualization
- **SwiftUI Interface**: Native iOS/macOS visualization with Swift Charts
- **Chart.js Export**: Web-compatible chart configurations
- **Real-time Training**: Live training progress visualization
- **Solution Comparison**: Side-by-side PINN vs RK4 plotting

## 📁 File Structure

```
Farmer/
├── PINN.swift                  # Core PINN implementation
├── PINNTraining.swift         # Training loop and optimization
├── RK4Solver.swift           # Runge-Kutta solver for comparison
├── PINNVisualization.swift   # SwiftUI visualization components
├── PINNExample.swift         # Complete executable example
├── PINNTestRunner.swift      # Test suite and validation
└── ContentView.swift         # Main app integration
```

## 🧮 Mathematical Framework

### Burgers' Equation
```
∂u/∂t + u(∂u/∂x) - ν(∂²u/∂x²) = 0
```
- **u(x,t)**: Velocity field
- **ν**: Kinematic viscosity (default: 0.01)
- **Domain**: x ∈ [-1, 1], t ∈ [0, 1]

### Hybrid Output Computation
```swift
// Step 1: Component computation
S(x) = max(0, 1 - |PDE_residual|)  // State inference
N(x) = improvement_rate            // Gradient analysis
α(t) = 0.5 * (1 + sin(2πt))      // Real-time validation

// Step 2: Hybrid output
O_hybrid = (1 - α) * S(x) + α * N(x)

// Step 3: Penalties
P_total = λ₁ * R_cognitive + λ₂ * R_efficiency
exp_penalty = exp(-P_total)

// Step 4: Final metric
Ψ(x) = O_hybrid * exp_penalty * P_adjusted
```

### BNSL Analysis
The implementation detects:
- **Inflection Points**: Changes in loss curve curvature
- **Scaling Behavior**: Exponential, power law, or non-monotonic patterns
- **Training Dynamics**: Characteristic BNSL signatures

## 🔧 Usage

### Basic Example
```swift
import Foundation

// Initialize models
let pinnModel = PINN()
let trainer = PINNTrainer(model: pinnModel)
let rk4Solver = RK4Solver()

// Generate training data
let x = Array(stride(from: -1.0, to: 1.0, by: 0.05))
let t = Array(stride(from: 0.0, to: 1.0, by: 0.05))

// Train the PINN
trainer.train(epochs: 1000, x: x, t: t, printEvery: 50)

// Compare with RK4
let comparison = PINNRKComparison.compareAtTime(
    pinnModel: pinnModel,
    rk4Solver: rk4Solver,
    xPoints: x,
    time: 1.0
)

print("MSE: \(comparison.mse)")
print("Max Error: \(comparison.maxError)")
```

### Running the Complete Example
```swift
// Run all tests and examples
PINNExample.runTests()

// Or run just the main example
PINNExample.runCompleteExample()
```

### SwiftUI Integration
The app includes three tabs:
1. **Items**: Original Farmer app functionality
2. **PINN Example**: Interactive example runner
3. **Visualization**: Real-time charts and training progress

## 📊 Sample Results

### Training Progress
```
Epoch 50:
  Loss: 0.245123
  S(x): 0.720 | N(x): 0.850 | α(t): 0.500
  R_cog: 0.150 | R_eff: 0.100
  Hybrid: 0.785 | Ψ(x): 0.662
```

### Comparison Metrics
- **Final Loss**: 0.045123
- **Final Ψ(x)**: 0.672 (Good performance)
- **MSE vs RK4**: 0.001234
- **Max Error**: 0.008567
- **BNSL Analysis**: Power law decay - moderate BNSL behavior

### Chart.js Configuration
The implementation exports web-compatible chart configurations:
```json
{
  "type": "line",
  "data": {
    "labels": [-1.0, -0.9, ..., 1.0],
    "datasets": [
      {
        "label": "PINN u",
        "data": [0.0, 0.3, 0.5, ...],
        "borderColor": "#1E90FF",
        "tension": 0.4
      },
      {
        "label": "RK4 u", 
        "data": [0.0, 0.4, 0.6, ...],
        "borderColor": "#FF4500",
        "tension": 0.4
      }
    ]
  }
}
```

## 🧪 Testing

Run the comprehensive test suite:
```swift
PINNTestRunner.runAllTests()
```

Tests include:
- PINN initialization and forward pass
- Hybrid framework component validation
- RK4 solver accuracy
- PINN vs RK4 comparison
- Chart.js export functionality
- Integration testing

## 🎯 Performance Interpretation

### Ψ(x) Values
- **0.8-1.0**: Excellent performance - high accuracy and efficiency
- **0.6-0.8**: Good performance - balanced accuracy and efficiency  
- **0.4-0.6**: Moderate performance - room for improvement
- **0.2-0.4**: Poor performance - significant issues detected
- **0.0-0.2**: Very poor performance - major optimization needed

### BNSL Signatures
- **Exponential decay**: Strong BNSL power law behavior
- **Power law decay**: Moderate BNSL characteristics
- **Non-monotonic**: Characteristic BNSL inflection points
- **Linear decay**: Weak BNSL signature

## 🚀 Getting Started

1. **Open in Xcode**: Load the Farmer.xcodeproj
2. **Build and Run**: Cmd+R to launch the app
3. **Navigate to PINN tabs**: Use the tab bar to access PINN functionality
4. **Run Examples**: Tap "Run Complete PINN Example" or "Start Training"
5. **View Results**: Check console output and SwiftUI visualizations

## 📚 Technical Details

### Neural Network Architecture
- **Input Layer**: 2 neurons (x, t coordinates)
- **Hidden Layer**: 20 neurons with tanh activation
- **Output Layer**: 1 neuron (u(x,t) solution)
- **Parameters**: ~81 trainable parameters total

### Training Configuration
- **Epochs**: 200-1000 (configurable)
- **Learning Rate**: 0.005
- **Batch Processing**: Efficient batched loss computation
- **Gradient Method**: Finite difference approximation (h=1e-5)

### RK4 Configuration
- **Spatial Step**: 0.05
- **Time Step**: 0.001
- **Viscosity**: 0.01
- **Boundary Treatment**: Dirichlet conditions enforced

## 🔬 Research Applications

This implementation demonstrates:
- **Physics-informed machine learning** for PDE solving
- **Hybrid symbolic-neural approaches** combining PINN and RK4
- **BNSL framework application** to neural scaling analysis
- **Real-time visualization** of training dynamics
- **Cross-platform deployment** (iOS/macOS native, web-exportable)

## 📈 Future Extensions

- **Multi-dimensional PDEs**: Extend to 2D/3D problems
- **Adaptive learning rates**: Implement Adam optimizer
- **GPU acceleration**: Metal performance shaders integration
- **Advanced BNSL**: More sophisticated scaling law detection
- **Interactive parameters**: Real-time viscosity/boundary adjustment

## 📄 License

This implementation is part of the Farmer project and follows the same licensing terms.

---

**Run the complete example in Xcode with SwiftUI preview for real-time visualization and interaction!** 🎉