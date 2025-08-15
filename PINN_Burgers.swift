import Foundation
import Accelerate

// MARK: - Neural Network Components

/// Dense layer with weights and biases
class DenseLayer {
    var weights: [[Double]]
    var biases: [Double]
    let inputSize: Int
    let outputSize: Int
    
    init(inputSize: Int, outputSize: Int) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        
        // Xavier/Glorot initialization
        let limit = sqrt(6.0 / Double(inputSize + outputSize))
        weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Double.random(in: -limit...limit) }
        }
        biases = Array(repeating: 0.0, count: outputSize)
    }
    
    /// Forward pass with tanh activation
    func forward(_ input: [Double]) -> [Double] {
        var output = biases
        for i in 0..<outputSize {
            for j in 0..<inputSize {
                output[i] += weights[i][j] * input[j]
            }
            output[i] = tanh(output[i]) // Tanh activation
        }
        return output
    }
    
    /// Linear forward pass (for output layer)
    func forwardLinear(_ input: [Double]) -> [Double] {
        var output = biases
        for i in 0..<outputSize {
            for j in 0..<inputSize {
                output[i] += weights[i][j] * input[j]
            }
        }
        return output
    }
}

// MARK: - Physics-Informed Neural Network

/// PINN for solving Burgers' equation: u_t + u*u_x = 0
class PINN {
    private var layers: [DenseLayer]
    private let hiddenSize: Int
    
    init(hiddenSize: Int = 50, numLayers: Int = 4) {
        self.hiddenSize = hiddenSize
        layers = []
        
        // Input layer (x, t) -> hidden
        layers.append(DenseLayer(inputSize: 2, outputSize: hiddenSize))
        
        // Hidden layers
        for _ in 1..<numLayers-1 {
            layers.append(DenseLayer(inputSize: hiddenSize, outputSize: hiddenSize))
        }
        
        // Output layer -> u(x,t)
        layers.append(DenseLayer(inputSize: hiddenSize, outputSize: 1))
    }
    
    /// Forward pass through the network
    func forward(x: Double, t: Double) -> Double {
        var input = [x, t]
        
        // Hidden layers with tanh activation
        for i in 0..<layers.count-1 {
            input = layers[i].forward(input)
        }
        
        // Output layer (linear)
        let output = layers.last!.forwardLinear(input)
        return output[0]
    }
    
    /// Compute partial derivatives using finite differences
    func computeDerivatives(x: Double, t: Double, dx: Double = 1e-5, dt: Double = 1e-5) -> (u: Double, u_x: Double, u_t: Double) {
        let u = forward(x: x, t: t)
        let u_x = (forward(x: x + dx, t: t) - forward(x: x - dx, t: t)) / (2 * dx)
        let u_t = (forward(x: x, t: t + dt) - forward(x: x, t: t - dt)) / (2 * dt)
        return (u, u_x, u_t)
    }
}

// MARK: - Hybrid Framework Components

/// Hybrid framework for combining symbolic and neural approaches
struct HybridFramework {
    /// State inference component S(x)
    static func stateInference(pinn: PINN, x: Double, t: Double) -> Double {
        let derivatives = pinn.computeDerivatives(x: x, t: t)
        // Normalize based on PDE residual quality
        let residual = abs(derivatives.u_t + derivatives.u * derivatives.u_x)
        return exp(-residual) // Higher for better PDE satisfaction
    }
    
    /// Neural approximation component N(x)
    static func neuralApproximation(pinn: PINN, x: Double, t: Double) -> Double {
        let u = pinn.forward(x: x, t: t)
        return sigmoid(u) // Normalized neural output
    }
    
    /// Real-time validation flow α(t)
    static func validationFlow(t: Double, maxTime: Double = 1.0) -> Double {
        return 0.5 + 0.3 * sin(2 * .pi * t / maxTime) // Dynamic weighting
    }
    
    /// Hybrid output O_hybrid
    static func hybridOutput(S: Double, N: Double, alpha: Double) -> Double {
        return alpha * S + (1 - alpha) * N
    }
    
    /// Cognitive regularization R_cognitive
    static func cognitiveRegularization(pinn: PINN, x: [Double], t: [Double]) -> Double {
        var totalResidual = 0.0
        for i in 0..<x.count {
            let derivatives = pinn.computeDerivatives(x: x[i], t: t[i])
            let residual = derivatives.u_t + derivatives.u * derivatives.u_x
            totalResidual += residual * residual
        }
        return totalResidual / Double(x.count)
    }
    
    /// Efficiency regularization R_efficiency
    static func efficiencyRegularization(computationTime: Double, targetTime: Double = 0.1) -> Double {
        return max(0, (computationTime - targetTime) / targetTime)
    }
    
    /// Probability adjustment P(H|E,β)
    static func probabilityAdjustment(baseProb: Double, beta: Double) -> Double {
        return min(1.0, baseProb * beta)
    }
}

// MARK: - RK4 Validation

/// Runge-Kutta 4th order solver for comparison
struct RK4Solver {
    static func solve(initialCondition: (Double) -> Double, 
                     xRange: (Double, Double), 
                     tRange: (Double, Double),
                     nx: Int = 100, 
                     nt: Int = 100) -> [[Double]] {
        
        let dx = (xRange.1 - xRange.0) / Double(nx - 1)
        let dt = (tRange.1 - tRange.0) / Double(nt - 1)
        
        var solution = Array(repeating: Array(repeating: 0.0, count: nx), count: nt)
        
        // Initial condition
        for i in 0..<nx {
            let x = xRange.0 + Double(i) * dx
            solution[0][i] = initialCondition(x)
        }
        
        // Time stepping with RK4
        for n in 1..<nt {
            for i in 1..<nx-1 {
                let u = solution[n-1][i]
                let u_x = (solution[n-1][i+1] - solution[n-1][i-1]) / (2 * dx)
                
                // RK4 step for Burgers' equation: du/dt = -u * du/dx
                let k1 = -u * u_x
                let k2 = -(u + 0.5 * dt * k1) * u_x
                let k3 = -(u + 0.5 * dt * k2) * u_x
                let k4 = -(u + dt * k3) * u_x
                
                solution[n][i] = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            }
            
            // Boundary conditions (periodic or fixed)
            solution[n][0] = solution[n][nx-2]
            solution[n][nx-1] = solution[n][1]
        }
        
        return solution
    }
}

// MARK: - Training and Loss Functions

/// Training manager for PINN
class PINNTrainer {
    private var pinn: PINN
    private let learningRate: Double
    
    init(pinn: PINN, learningRate: Double = 0.001) {
        self.pinn = pinn
        self.learningRate = learningRate
    }
    
    /// Compute total loss with hybrid framework
    func computeLoss(collocationPoints: [(x: Double, t: Double)],
                    initialPoints: [(x: Double, u: Double)],
                    lambda1: Double = 0.6,
                    lambda2: Double = 0.4,
                    beta: Double = 1.2) -> (total: Double, components: (pde: Double, initial: Double, hybrid: Double)) {
        
        // PDE residual loss
        var pdeResidual = 0.0
        for point in collocationPoints {
            let derivatives = pinn.computeDerivatives(x: point.x, t: point.t)
            let residual = derivatives.u_t + derivatives.u * derivatives.u_x
            pdeResidual += residual * residual
        }
        pdeResidual /= Double(collocationPoints.count)
        
        // Initial condition loss
        var initialLoss = 0.0
        for point in initialPoints {
            let predicted = pinn.forward(x: point.x, t: 0.0)
            let error = predicted - point.u
            initialLoss += error * error
        }
        initialLoss /= Double(initialPoints.count)
        
        // Hybrid framework components
        let samplePoint = collocationPoints.first!
        let S = HybridFramework.stateInference(pinn: pinn, x: samplePoint.x, t: samplePoint.t)
        let N = HybridFramework.neuralApproximation(pinn: pinn, x: samplePoint.x, t: samplePoint.t)
        let alpha = HybridFramework.validationFlow(t: samplePoint.t)
        let hybridOutput = HybridFramework.hybridOutput(S: S, N: N, alpha: alpha)
        
        // Regularization terms
        let xPoints = collocationPoints.map { $0.x }
        let tPoints = collocationPoints.map { $0.t }
        let rCognitive = HybridFramework.cognitiveRegularization(pinn: pinn, x: xPoints, t: tPoints)
        let rEfficiency = HybridFramework.efficiencyRegularization(computationTime: 0.05)
        
        let totalPenalty = lambda1 * rCognitive + lambda2 * rEfficiency
        let penaltyExp = exp(-totalPenalty)
        
        // Probability adjustment
        let baseProb = 0.8
        let adjustedProb = HybridFramework.probabilityAdjustment(baseProb: baseProb, beta: beta)
        
        // Final hybrid loss component
        let hybridLoss = 1.0 - (hybridOutput * penaltyExp * adjustedProb)
        
        let totalLoss = pdeResidual + 10.0 * initialLoss + hybridLoss
        
        return (totalLoss, (pdeResidual, initialLoss, hybridLoss))
    }
    
    /// Training step (simplified - in practice would use proper backpropagation)
    func trainStep(collocationPoints: [(x: Double, t: Double)],
                  initialPoints: [(x: Double, u: Double)]) -> Double {
        let loss = computeLoss(collocationPoints: collocationPoints, initialPoints: initialPoints)
        
        // Simple parameter perturbation for demonstration
        // In practice, use automatic differentiation
        for layer in pinn.layers {
            for i in 0..<layer.weights.count {
                for j in 0..<layer.weights[i].count {
                    layer.weights[i][j] += Double.random(in: -learningRate...learningRate)
                }
            }
        }
        
        return loss.total
    }
}

// MARK: - Utility Functions

func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

// MARK: - Example Usage and Validation

/// Main example demonstrating the hybrid PINN framework
func runPINNExample() {
    print("🚀 Starting Hybrid PINN for Burgers' Equation")
    print("=" * 50)
    
    // Initialize PINN
    let pinn = PINN(hiddenSize: 50, numLayers: 4)
    let trainer = PINNTrainer(pinn: pinn, learningRate: 0.001)
    
    // Define domain
    let xRange = (-1.0, 1.0)
    let tRange = (0.0, 1.0)
    
    // Generate collocation points
    var collocationPoints: [(x: Double, t: Double)] = []
    for _ in 0..<1000 {
        let x = Double.random(in: xRange.0...xRange.1)
        let t = Double.random(in: tRange.0...tRange.1)
        collocationPoints.append((x, t))
    }
    
    // Initial condition: u(x,0) = -sin(π*x)
    var initialPoints: [(x: Double, u: Double)] = []
    for i in 0..<100 {
        let x = xRange.0 + Double(i) * (xRange.1 - xRange.0) / 99.0
        let u = -sin(.pi * x)
        initialPoints.append((x, u))
    }
    
    print("📊 Training Configuration:")
    print("   • Collocation points: \(collocationPoints.count)")
    print("   • Initial condition points: \(initialPoints.count)")
    print("   • Domain: x ∈ [\(xRange.0), \(xRange.1)], t ∈ [\(tRange.0), \(tRange.1)]")
    
    // Training loop
    print("\n🔄 Training Progress:")
    let numEpochs = 100
    var losses: [Double] = []
    
    for epoch in 0..<numEpochs {
        let loss = trainer.trainStep(collocationPoints: collocationPoints, 
                                   initialPoints: initialPoints)
        losses.append(loss)
        
        if epoch % 20 == 0 {
            let lossComponents = trainer.computeLoss(collocationPoints: collocationPoints, 
                                                   initialPoints: initialPoints)
            print("   Epoch \(epoch): Loss = \(String(format: "%.6f", loss))")
            print("     • PDE: \(String(format: "%.6f", lossComponents.components.pde))")
            print("     • Initial: \(String(format: "%.6f", lossComponents.components.initial))")
            print("     • Hybrid: \(String(format: "%.6f", lossComponents.components.hybrid))")
        }
    }
    
    // Numerical example validation
    print("\n🔍 Numerical Example Validation:")
    let testX = 0.5, testT = 0.3
    
    let S = HybridFramework.stateInference(pinn: pinn, x: testX, t: testT)
    let N = HybridFramework.neuralApproximation(pinn: pinn, x: testX, t: testT)
    let alpha = HybridFramework.validationFlow(t: testT)
    let hybridOutput = HybridFramework.hybridOutput(S: S, N: N, alpha: alpha)
    
    print("   • S(x) = \(String(format: "%.3f", S))")
    print("   • N(x) = \(String(format: "%.3f", N))")
    print("   • α(t) = \(String(format: "%.3f", alpha))")
    print("   • O_hybrid = \(String(format: "%.3f", hybridOutput))")
    
    let rCognitive = HybridFramework.cognitiveRegularization(pinn: pinn, 
                                                           x: [testX], t: [testT])
    let rEfficiency = HybridFramework.efficiencyRegularization(computationTime: 0.05)
    let lambda1 = 0.6, lambda2 = 0.4
    let totalPenalty = lambda1 * rCognitive + lambda2 * rEfficiency
    let penaltyExp = exp(-totalPenalty)
    
    print("   • R_cognitive = \(String(format: "%.3f", rCognitive))")
    print("   • R_efficiency = \(String(format: "%.3f", rEfficiency))")
    print("   • Penalty exp = \(String(format: "%.3f", penaltyExp))")
    
    let baseProb = 0.8, beta = 1.2
    let adjustedProb = HybridFramework.probabilityAdjustment(baseProb: baseProb, beta: beta)
    
    print("   • P(H|E,β) = \(String(format: "%.3f", adjustedProb))")
    
    let psi = hybridOutput * penaltyExp * adjustedProb
    print("   • Ψ(x) = \(String(format: "%.3f", psi))")
    
    // Interpretation
    print("\n📈 Results Interpretation:")
    if psi > 0.7 {
        print("   ✅ Excellent model performance (Ψ > 0.7)")
    } else if psi > 0.6 {
        print("   ✅ Good model performance (0.6 < Ψ ≤ 0.7)")
    } else if psi > 0.5 {
        print("   ⚠️  Moderate performance (0.5 < Ψ ≤ 0.6)")
    } else {
        print("   ❌ Poor performance (Ψ ≤ 0.5)")
    }
    
    // RK4 comparison
    print("\n🔬 RK4 Validation Comparison:")
    let rk4Solution = RK4Solver.solve(
        initialCondition: { x in -sin(.pi * x) },
        xRange: xRange,
        tRange: (0.0, 0.5),
        nx: 50,
        nt: 50
    )
    
    // Compare at a test point
    let testXIndex = 25, testTIndex = 25
    let rk4Value = rk4Solution[testTIndex][testXIndex]
    let pinnValue = pinn.forward(x: 0.0, t: 0.25)
    let error = abs(rk4Value - pinnValue)
    
    print("   • RK4 solution at (0, 0.25): \(String(format: "%.6f", rk4Value))")
    print("   • PINN solution at (0, 0.25): \(String(format: "%.6f", pinnValue))")
    print("   • Absolute error: \(String(format: "%.6f", error))")
    
    print("\n🎯 Hybrid Framework Summary:")
    print("   • Combines symbolic RK4 validation with neural PINN approximation")
    print("   • Incorporates cognitive and efficiency regularization")
    print("   • Adapts through real-time validation flows α(t)")
    print("   • Provides interpretable performance metric Ψ(x)")
    print("   • Enables balanced intelligence for nonlinear PDE solving")
    
    print("\n✨ Training Complete!")
}

// MARK: - Enhanced Example with Visualization

/// Enhanced example with comprehensive visualization
func runEnhancedPINNExample() {
    print("🚀 Starting Enhanced Hybrid PINN for Burgers' Equation")
    print("=" * 60)
    
    // Initialize PINN
    let pinn = PINN(hiddenSize: 50, numLayers: 4)
    let trainer = PINNTrainer(pinn: pinn, learningRate: 0.001)
    
    // Define domain
    let xRange = (-1.0, 1.0)
    let tRange = (0.0, 1.0)
    
    // Generate collocation points
    var collocationPoints: [(x: Double, t: Double)] = []
    for _ in 0..<1000 {
        let x = Double.random(in: xRange.0...xRange.1)
        let t = Double.random(in: tRange.0...tRange.1)
        collocationPoints.append((x, t))
    }
    
    // Initial condition: u(x,0) = -sin(π*x)
    var initialPoints: [(x: Double, u: Double)] = []
    for i in 0..<100 {
        let x = xRange.0 + Double(i) * (xRange.1 - xRange.0) / 99.0
        let u = -sin(.pi * x)
        initialPoints.append((x, u))
    }
    
    print("📊 Training Configuration:")
    print("   • Collocation points: \(collocationPoints.count)")
    print("   • Initial condition points: \(initialPoints.count)")
    print("   • Domain: x ∈ [\(xRange.0), \(xRange.1)], t ∈ [\(tRange.0), \(tRange.1)]")
    
    // Generate RK4 solution for comparison
    print("\n🔬 Generating RK4 reference solution...")
    let rk4Solution = RK4Solver.solve(
        initialCondition: { x in -sin(.pi * x) },
        xRange: xRange,
        tRange: (0.0, 0.5),
        nx: 100,
        nt: 50
    )
    
    // Training loop with loss tracking
    print("\n🔄 Training Progress:")
    let numEpochs = 100
    var losses: [Double] = []
    
    for epoch in 0..<numEpochs {
        let loss = trainer.trainStep(collocationPoints: collocationPoints, 
                                   initialPoints: initialPoints)
        losses.append(loss)
        
        if epoch % 20 == 0 {
            let lossComponents = trainer.computeLoss(collocationPoints: collocationPoints, 
                                                   initialPoints: initialPoints)
            print("   Epoch \(epoch): Loss = \(String(format: "%.6f", loss))")
            print("     • PDE: \(String(format: "%.6f", lossComponents.components.pde))")
            print("     • Initial: \(String(format: "%.6f", lossComponents.components.initial))")
            print("     • Hybrid: \(String(format: "%.6f", lossComponents.components.hybrid))")
        }
    }
    
    // Numerical example validation
    print("\n🔍 Numerical Example Validation:")
    let testX = 0.5, testT = 0.3
    
    let S = HybridFramework.stateInference(pinn: pinn, x: testX, t: testT)
    let N = HybridFramework.neuralApproximation(pinn: pinn, x: testX, t: testT)
    let alpha = HybridFramework.validationFlow(t: testT)
    let hybridOutput = HybridFramework.hybridOutput(S: S, N: N, alpha: alpha)
    
    print("   • S(x) = \(String(format: "%.3f", S))")
    print("   • N(x) = \(String(format: "%.3f", N))")
    print("   • α(t) = \(String(format: "%.3f", alpha))")
    print("   • O_hybrid = \(String(format: "%.3f", hybridOutput))")
    
    let rCognitive = HybridFramework.cognitiveRegularization(pinn: pinn, 
                                                           x: [testX], t: [testT])
    let rEfficiency = HybridFramework.efficiencyRegularization(computationTime: 0.05)
    let lambda1 = 0.6, lambda2 = 0.4
    let totalPenalty = lambda1 * rCognitive + lambda2 * rEfficiency
    let penaltyExp = exp(-totalPenalty)
    
    print("   • R_cognitive = \(String(format: "%.3f", rCognitive))")
    print("   • R_efficiency = \(String(format: "%.3f", rEfficiency))")
    print("   • Penalty exp = \(String(format: "%.3f", penaltyExp))")
    
    let baseProb = 0.8, beta = 1.2
    let adjustedProb = HybridFramework.probabilityAdjustment(baseProb: baseProb, beta: beta)
    
    print("   • P(H|E,β) = \(String(format: "%.3f", adjustedProb))")
    
    let psi = hybridOutput * penaltyExp * adjustedProb
    print("   • Ψ(x) = \(String(format: "%.3f", psi))")
    
    // Interpretation
    print("\n📈 Results Interpretation:")
    if psi > 0.7 {
        print("   ✅ Excellent model performance (Ψ > 0.7)")
    } else if psi > 0.6 {
        print("   ✅ Good model performance (0.6 < Ψ ≤ 0.7)")
    } else if psi > 0.5 {
        print("   ⚠️  Moderate performance (0.5 < Ψ ≤ 0.6)")
    } else {
        print("   ❌ Poor performance (Ψ ≤ 0.5)")
    }
    
    // Generate comprehensive visualization report
    print("\n🎨 Generating Visualization Report...")
    let visualizationReport = PINNVisualizer.generateVisualizationReport(
        pinn: pinn,
        rk4Solution: rk4Solution,
        losses: losses,
        xRange: xRange,
        tRange: tRange
    )
    print(visualizationReport)
    
    // Export data for external plotting
    print("\n📁 Exporting Data for External Visualization...")
    
    // Export solution comparison at t = 0.3
    let comparison = PINNVisualizer.compareSolutions(
        pinn: pinn,
        rk4Solution: rk4Solution,
        time: 0.3,
        xRange: xRange,
        tRange: (0.0, 0.5)
    )
    
    PINNVisualizer.exportToCSV(
        series: [comparison.pinnSeries, comparison.rk4Series, comparison.errorSeries],
        filename: "/workspace/pinn_comparison_t03.csv"
    )
    
    // Export heatmap data
    let heatmapData = PINNVisualizer.generateHeatmapData(
        pinn: pinn,
        xRange: xRange,
        tRange: tRange,
        nx: 50,
        nt: 50
    )
    
    PINNVisualizer.exportHeatmapToCSV(
        data: heatmapData.data,
        xAxis: heatmapData.xAxis,
        tAxis: heatmapData.tAxis,
        filename: "/workspace/pinn_heatmap.csv"
    )
    
    print("\n🎯 Hybrid Framework Summary:")
    print("   • Combines symbolic RK4 validation with neural PINN approximation")
    print("   • Incorporates cognitive and efficiency regularization")
    print("   • Adapts through real-time validation flows α(t)")
    print("   • Provides interpretable performance metric Ψ(x)")
    print("   • Enables balanced intelligence for nonlinear PDE solving")
    print("   • Includes comprehensive visualization and data export capabilities")
    
    print("\n✨ Enhanced Training Complete!")
    print("📊 Check exported CSV files for publication-quality plots")
}

// MARK: - Main Execution

// Run the enhanced example with visualization
runEnhancedPINNExample()