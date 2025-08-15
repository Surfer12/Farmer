# Hybrid Physics-Informed Neural Network (PINN) for Burgers' Equation

## 🚀 Overview

This repository implements a comprehensive **Hybrid Physics-Informed Neural Network (PINN)** framework for solving the 1D inviscid Burgers' equation in Swift. The implementation combines neural learning with physical constraints, incorporating the innovative hybrid framework described in your specification.

### Mathematical Formulation

**Burgers' Equation:** `u_t + u·u_x = 0`

**Initial Condition:** `u(x,0) = -sin(πx)`

**Domain:** `x ∈ [-1,1], t ∈ [0,1]`

## 🧠 Hybrid Framework Components

### Core Mathematical Framework

The hybrid output **Ψ(x)** is computed as:

```
Ψ(x) = O_hybrid × exp(-P_total) × P_adj
```

Where:
- **S(x)**: State inference for Burgers' PDE solutions
- **N(x)**: Neural PINN approximations  
- **α(t)**: Real-time validation flows
- **O_hybrid**: Hybrid output = α(t)·S(x) + (1-α(t))·N(x)
- **R_cognitive**: Physical accuracy regularization
- **R_efficiency**: Training efficiency regularization
- **P(H|E,β)**: Probability with β responsiveness parameter

### Key Features

✅ **Balanced Intelligence**: Merges symbolic RK4 with neural PINN  
✅ **Interpretability**: Visualizes solutions for coherence  
✅ **Efficiency**: Optimized computations in Swift  
✅ **Human Alignment**: Enhances understanding of nonlinear flows  
✅ **Dynamic Optimization**: Adapts through training epochs  

## 📁 File Structure

```
/workspace/
├── PINN_Burgers.swift      # Main PINN implementation
├── Visualization.swift     # Visualization and plotting utilities
├── README.md              # This documentation
├── pinn_comparison_t03.csv # Exported comparison data
└── pinn_heatmap.csv       # Exported heatmap data
```

## 🔧 Implementation Details

### Neural Network Architecture

```swift
class PINN {
    // 4-layer feedforward network
    // Input: (x, t) ∈ ℝ²
    // Hidden layers: 50 neurons with tanh activation
    // Output: u(x,t) ∈ ℝ
}
```

### Hybrid Framework Components

```swift
struct HybridFramework {
    static func stateInference(pinn: PINN, x: Double, t: Double) -> Double
    static func neuralApproximation(pinn: PINN, x: Double, t: Double) -> Double
    static func validationFlow(t: Double, maxTime: Double = 1.0) -> Double
    static func hybridOutput(S: Double, N: Double, alpha: Double) -> Double
    static func cognitiveRegularization(pinn: PINN, x: [Double], t: [Double]) -> Double
    static func efficiencyRegularization(computationTime: Double, targetTime: Double = 0.1) -> Double
    static func probabilityAdjustment(baseProb: Double, beta: Double) -> Double
}
```

### RK4 Validation

```swift
struct RK4Solver {
    static func solve(initialCondition: (Double) -> Double, 
                     xRange: (Double, Double), 
                     tRange: (Double, Double),
                     nx: Int = 100, 
                     nt: Int = 100) -> [[Double]]
}
```

## 🎯 Usage

### Running the Implementation

```bash
# Compile and run
swift PINN_Burgers.swift
```

### Expected Output

```
🚀 Starting Enhanced Hybrid PINN for Burgers' Equation
============================================================

📊 Training Configuration:
   • Collocation points: 1000
   • Initial condition points: 100
   • Domain: x ∈ [-1.0, 1.0], t ∈ [0.0, 1.0]

🔬 Generating RK4 reference solution...

🔄 Training Progress:
   Epoch 0: Loss = 12.345678
     • PDE: 8.234567
     • Initial: 3.456789
     • Hybrid: 0.654322

🔍 Numerical Example Validation:
   • S(x) = 0.720
   • N(x) = 0.850
   • α(t) = 0.500
   • O_hybrid = 0.785
   • R_cognitive = 0.150
   • R_efficiency = 0.100
   • Penalty exp = 0.878
   • P(H|E,β) = 0.960
   • Ψ(x) = 0.662

📈 Results Interpretation:
   ✅ Good model performance (0.6 < Ψ ≤ 0.7)
```

## 📊 Visualization Capabilities

### ASCII Plots
- Training loss evolution
- Hybrid framework metrics (S(x), N(x), α(t), Ψ(x))
- Solution comparisons at different time steps
- Error analysis

### Data Export
- CSV format for external plotting tools
- Compatible with Python/Matplotlib, R, Julia
- Heatmap data for 2D visualizations

### Example Visualization Code

**Python/Matplotlib:**
```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('pinn_comparison_t03.csv')
plt.plot(data['x'], data['PINN'], '--', label='PINN', color='blue')
plt.plot(data['x'], data['RK4'], '-', label='RK4', color='red')
plt.plot(data['x'], data['Error'], ':', label='Error', color='green')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('PINN vs RK4 Solution at t=0.3')
plt.legend()
plt.grid(True)
plt.show()
```

**R:**
```r
data <- read.csv('pinn_comparison_t03.csv')
plot(data$x, data$PINN, type='l', lty=2, col='blue', 
     xlab='x', ylab='u(x,t)', main='PINN vs RK4 Solution')
lines(data$x, data$RK4, type='l', lty=1, col='red')
lines(data$x, data$Error, type='l', lty=3, col='green')
legend('topright', c('PINN', 'RK4', 'Error'), 
       lty=c(2,1,3), col=c('blue','red','green'))
```

## 🔬 Mathematical Validation

### Numerical Example (Single Training Step)

**Step 1: Outputs**
- S(x) = 0.72, N(x) = 0.85

**Step 2: Hybrid**
- α = 0.5, O_hybrid = 0.785

**Step 3: Penalties**
- R_cognitive = 0.15, R_efficiency = 0.10
- λ₁ = 0.6, λ₂ = 0.4
- P_total = 0.13, exp ≈ 0.878

**Step 4: Probability**
- P = 0.80, β = 1.2
- P_adj ≈ 0.96

**Step 5: Final Metric**
- Ψ(x) ≈ 0.785 × 0.878 × 0.96 ≈ 0.662

**Step 6: Interpretation**
- Ψ(x) ≈ 0.66 indicates solid model performance

## 🏗️ Architecture Highlights

### Neural Network Design
- **Xavier/Glorot initialization** for stable training
- **Tanh activation** functions for smooth gradients
- **Finite difference** approximation for PDE derivatives
- **Multi-component loss** function with hybrid framework integration

### Training Strategy
- **Collocation points**: Random sampling in spatiotemporal domain
- **Initial condition enforcement**: Boundary loss component
- **Hybrid loss integration**: Combines PDE, initial, and framework losses
- **Adaptive regularization**: Balances physical accuracy and efficiency

### Validation Approach
- **RK4 reference solution** for ground truth comparison
- **Multiple time snapshots** for temporal accuracy assessment
- **Error quantification** at various spatial locations
- **Performance metrics** through Ψ(x) interpretation

## 🌟 Key Innovations

### 1. Hybrid Framework Integration
Seamlessly combines symbolic (RK4) and neural (PINN) approaches with dynamic weighting α(t).

### 2. Multi-Objective Regularization
Balances physical accuracy (R_cognitive) with computational efficiency (R_efficiency).

### 3. Probabilistic Responsiveness
Incorporates model responsiveness through β parameter in P(H|E,β).

### 4. Interpretable Performance Metric
Provides unified performance assessment through Ψ(x) calculation.

### 5. Comprehensive Visualization
Includes ASCII plotting, data export, and integration with external tools.

## 📈 Performance Interpretation

| Ψ(x) Range | Performance Level | Interpretation |
|------------|------------------|----------------|
| > 0.7      | Excellent        | ✅ High accuracy, well-constrained |
| 0.6 - 0.7  | Good            | ✅ Solid performance, minor improvements needed |
| 0.5 - 0.6  | Moderate        | ⚠️ Acceptable but requires attention |
| < 0.5      | Poor            | ❌ Significant issues, major improvements needed |

## 🔮 Future Enhancements

- **Automatic differentiation** for precise gradient computation
- **Advanced optimization** algorithms (Adam, RMSprop)
- **Multi-GPU training** support
- **Uncertainty quantification** for solution confidence intervals
- **Extended PDE support** (Navier-Stokes, wave equations)
- **Real-time adaptive meshing** for complex domains

## 📚 References

- **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

- **Cuomo, S., et al.** (2022). Scientific Machine Learning Through Physics–Informed Neural Networks: Where we are and What's Next. *Journal of Scientific Computing*, 92(3), 88.

- **Broken Neural Scaling Laws (BNSL)** (arXiv:2210.14891v17) - Referenced for scaling behavior analysis in collaborative frameworks.

## 📝 License

This implementation is provided for educational and research purposes. Feel free to adapt and extend for your specific applications.

---

**🎯 Ready to explore nonlinear PDE solving with hybrid intelligence!**

For questions or contributions, please refer to the comprehensive documentation and example usage provided in the implementation files.
