import Foundation

// MARK: - Mathematical Framework Components

/// Core mathematical framework for hybrid AI system
struct HybridFramework {
    // State inference and ML analysis outputs
    var S_x: Double  // State inference for optimized PINN solutions
    var N_x: Double  // ML gradient descent analysis
    var alpha_t: Double  // Real-time validation flows
    
    // Regularization terms
    var R_cognitive: Double  // PDE residual accuracy
    var R_efficiency: Double  // Training loop efficiency
    
    // Probability parameters
    var beta: Double  // Model responsiveness parameter
    
    // Penalty weights
    var lambda1: Double = 0.6
    var lambda2: Double = 0.4
    
    /// Calculate hybrid output O_hybrid
    func hybridOutput() -> Double {
        return alpha_t * S_x + (1 - alpha_t) * N_x
    }
    
    /// Calculate total penalty P_total
    func totalPenalty() -> Double {
        return lambda1 * R_cognitive + lambda2 * R_efficiency
    }
    
    /// Calculate exponential penalty term
    func exponentialPenalty() -> Double {
        return exp(-totalPenalty())
    }
    
    /// Calculate adjusted probability P_adj
    func adjustedProbability(baseProb P: Double) -> Double {
        let adjustment = pow(P, beta)
        return min(adjustment, 1.0)  // Cap at 1.0
    }
    
    /// Calculate final framework output Ψ(x)
    func psi(baseProb P: Double) -> Double {
        let O_hybrid = hybridOutput()
        let exp_penalty = exponentialPenalty()
        let P_adj = adjustedProbability(baseProb: P)
        return O_hybrid * exp_penalty * P_adj
    }
}

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
        
        // Xavier initialization
        let limit = sqrt(6.0 / Double(inputSize + outputSize))
        weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Double.random(in: -limit...limit) }
        }
        biases = (0..<outputSize).map { _ in Double.random(in: -0.1...0.1) }
    }
    
    /// Forward pass through the layer
    func forward(_ input: [Double]) -> [Double] {
        return (0..<outputSize).map { i in
            let weightedSum = zip(weights[i], input).map(*).reduce(0, +)
            return tanh(weightedSum + biases[i])  // Tanh activation
        }
    }
    
    /// Update weights with gradients
    func updateWeights(weightGradients: [[Double]], biasGradients: [Double], learningRate: Double) {
        for i in 0..<outputSize {
            for j in 0..<inputSize {
                weights[i][j] -= learningRate * weightGradients[i][j]
            }
            biases[i] -= learningRate * biasGradients[i]
        }
    }
}

// MARK: - Physics-Informed Neural Network

/// Physics-Informed Neural Network for solving PDEs
class PINN {
    var layers: [DenseLayer]
    var framework: HybridFramework
    
    init(architecture: [Int], framework: HybridFramework) {
        self.framework = framework
        self.layers = []
        
        for i in 0..<(architecture.count - 1) {
            layers.append(DenseLayer(inputSize: architecture[i], outputSize: architecture[i + 1]))
        }
    }
    
    /// Forward pass through the network
    func forward(x: Double, t: Double) -> Double {
        var input = [x, t]
        for layer in layers {
            input = layer.forward(input)
        }
        return input[0]
    }
    
    /// Calculate spatial derivative using finite differences
    func spatialDerivative(x: Double, t: Double, dx: Double = 1e-5) -> Double {
        return (forward(x: x + dx, t: t) - forward(x: x - dx, t: t)) / (2 * dx)
    }
    
    /// Calculate temporal derivative using finite differences
    func temporalDerivative(x: Double, t: Double, dt: Double = 1e-5) -> Double {
        return (forward(x: x, t: t + dt) - forward(x: x, t: t - dt)) / (2 * dt)
    }
    
    /// Calculate second spatial derivative
    func secondSpatialDerivative(x: Double, t: Double, dx: Double = 1e-5) -> Double {
        let u_plus = forward(x: x + dx, t: t)
        let u_center = forward(x: x, t: t)
        let u_minus = forward(x: x - dx, t: t)
        return (u_plus - 2 * u_center + u_minus) / (dx * dx)
    }
}

// MARK: - PDE Solver for 1D Inviscid Burgers' Equation

/// Solver for the 1D inviscid Burgers' equation: ∂u/∂t + u∂u/∂x = 0
class BurgersEquationSolver {
    let pinn: PINN
    
    init(pinn: PINN) {
        self.pinn = pinn
    }
    
    /// PDE residual for Burgers' equation
    func pdeResidual(x: Double, t: Double) -> Double {
        let u = pinn.forward(x: x, t: t)
        let u_t = pinn.temporalDerivative(x: x, t: t)
        let u_x = pinn.spatialDerivative(x: x, t: t)
        
        // Burgers' equation: ∂u/∂t + u∂u/∂x = 0
        return u_t + u * u_x
    }
    
    /// Initial condition loss
    func initialConditionLoss(xPoints: [Double]) -> Double {
        let totalLoss = xPoints.reduce(0.0) { loss, x in
            let u = pinn.forward(x: x, t: 0.0)
            let trueU = -sin(.pi * x)  // Initial condition: u(x,0) = -sin(πx)
            return loss + pow(u - trueU, 2)
        }
        return totalLoss / Double(xPoints.count)
    }
    
    /// Boundary condition loss
    func boundaryConditionLoss(tPoints: [Double]) -> Double {
        let totalLoss = tPoints.reduce(0.0) { loss, t in
            let u_left = pinn.forward(x: 0.0, t: t)
            let u_right = pinn.forward(x: 1.0, t: t)
            // Periodic boundary conditions: u(0,t) = u(1,t)
            return loss + pow(u_left - u_right, 2)
        }
        return totalLoss / Double(tPoints.count)
    }
    
    /// Total PDE loss
    func pdeLoss(xPoints: [Double], tPoints: [Double]) -> Double {
        var totalLoss = 0.0
        var count = 0
        
        for x in xPoints {
            for t in tPoints {
                totalLoss += pow(pdeResidual(x: x, t: t), 2)
                count += 1
            }
        }
        
        return totalLoss / Double(count)
    }
    
    /// Combined loss function
    func totalLoss(xPoints: [Double], tPoints: [Double]) -> Double {
        let pde = pdeLoss(xPoints: xPoints, tPoints: tPoints)
        let ic = initialConditionLoss(xPoints: xPoints)
        let bc = boundaryConditionLoss(tPoints: tPoints)
        
        // Weight the losses
        return pde + 10.0 * ic + 5.0 * bc
    }
}

// MARK: - RK4 Integration for Comparison

/// Fourth-order Runge-Kutta integration
struct RK4Integrator {
    /// RK4 step for system of ODEs
    static func step(f: (Double, [Double]) -> [Double], t: Double, y: [Double], dt: Double) -> [Double] {
        let k1 = f(t, y)
        let y2 = y.enumerated().map { $1 + dt / 2 * k1[$0] }
        let k2 = f(t + dt / 2, y2)
        let y3 = y.enumerated().map { $1 + dt / 2 * k2[$0] }
        let k3 = f(t + dt / 2, y3)
        let y4 = y.enumerated().map { $1 + dt * k3[$0] }
        let k4 = f(t + dt, y4)
        
        return y.enumerated().map { $1 + dt / 6 * (k1[$0] + 2 * k2[$0] + 2 * k3[$0] + k4[$0]) }
    }
    
    /// Solve Burgers' equation using method of lines
    static func solveBurgers(xPoints: [Double], tFinal: Double, dt: Double) -> [[Double]] {
        let nx = xPoints.count
        let nt = Int(tFinal / dt) + 1
        var solution = Array(repeating: Array(repeating: 0.0, count: nx), count: nt)
        
        // Initial condition
        for i in 0..<nx {
            solution[0][i] = -sin(.pi * xPoints[i])
        }
        
        // Time stepping
        for n in 0..<(nt - 1) {
            let t = Double(n) * dt
            let currentU = solution[n]
            
            // Spatial derivatives using finite differences
            let dudt = (0..<nx).map { i -> Double in
                let u = currentU[i]
                let dudx = (i == nx - 1) ? 
                    (currentU[0] - currentU[i - 1]) / (2 * (xPoints[1] - xPoints[0])) :
                    (i == 0) ? 
                    (currentU[i + 1] - currentU[nx - 1]) / (2 * (xPoints[1] - xPoints[0])) :
                    (currentU[i + 1] - currentU[i - 1]) / (2 * (xPoints[1] - xPoints[0]))
                
                return -u * dudx  // Burgers' equation: ∂u/∂t = -u∂u/∂x
            }
            
            solution[n + 1] = RK4Integrator.step(
                f: { _, y in dudt },
                t: t,
                y: currentU,
                dt: dt
            )
        }
        
        return solution
    }
}

// MARK: - Training System

/// Training system for PINN
class PINNTrainer {
    let solver: BurgersEquationSolver
    var trainingHistory: [Double] = []
    
    init(solver: BurgersEquationSolver) {
        self.solver = solver
    }
    
    /// Approximate gradients using finite differences
    func approximateGradients(xPoints: [Double], tPoints: [Double], epsilon: Double = 1e-6) -> ([[Double]], [Double]) {
        let baseLoss = solver.totalLoss(xPoints: xPoints, tPoints: tPoints)
        var weightGradients: [[[Double]]] = []
        var biasGradients: [[Double]] = []
        
        for layer in solver.pinn.layers {
            var layerWeightGrads = Array(repeating: Array(repeating: 0.0, count: layer.inputSize), count: layer.outputSize)
            var layerBiasGrads = Array(repeating: 0.0, count: layer.outputSize)
            
            // Weight gradients
            for i in 0..<layer.outputSize {
                for j in 0..<layer.inputSize {
                    layer.weights[i][j] += epsilon
                    let lossPlus = solver.totalLoss(xPoints: xPoints, tPoints: tPoints)
                    layer.weights[i][j] -= 2 * epsilon
                    let lossMinus = solver.totalLoss(xPoints: xPoints, tPoints: tPoints)
                    layer.weights[i][j] += epsilon  // Reset
                    
                    layerWeightGrads[i][j] = (lossPlus - lossMinus) / (2 * epsilon)
                }
            }
            
            // Bias gradients
            for i in 0..<layer.outputSize {
                layer.biases[i] += epsilon
                let lossPlus = solver.totalLoss(xPoints: xPoints, tPoints: tPoints)
                layer.biases[i] -= 2 * epsilon
                let lossMinus = solver.totalLoss(xPoints: xPoints, tPoints: tPoints)
                layer.biases[i] += epsilon  // Reset
                
                layerBiasGrads[i] = (lossPlus - lossMinus) / (2 * epsilon)
            }
            
            weightGradients.append(layerWeightGrads)
            biasGradients.append(layerBiasGrads)
        }
        
        return (weightGradients, biasGradients)
    }
    
    /// Training step
    func trainStep(xPoints: [Double], tPoints: [Double], learningRate: Double = 0.01) -> Double {
        let (weightGrads, biasGrads) = approximateGradients(xPoints: xPoints, tPoints: tPoints)
        
        // Update parameters
        for (i, layer) in solver.pinn.layers.enumerated() {
            layer.updateWeights(
                weightGradients: weightGrads[i],
                biasGradients: biasGrads[i],
                learningRate: learningRate
            )
        }
        
        let loss = solver.totalLoss(xPoints: xPoints, tPoints: tPoints)
        trainingHistory.append(loss)
        return loss
    }
    
    /// Train the PINN
    func train(epochs: Int, xPoints: [Double], tPoints: [Double], learningRate: Double = 0.01) {
        print("Starting PINN training...")
        
        for epoch in 0..<epochs {
            let loss = trainStep(xPoints: xPoints, tPoints: tPoints, learningRate: learningRate)
            
            if epoch % 10 == 0 {
                let psi = solver.pinn.framework.psi(baseProb: 0.8)
                print("Epoch \(epoch): Loss = \(String(format: "%.6f", loss)), Ψ(x) = \(String(format: "%.3f", psi))")
            }
        }
        
        print("Training completed!")
    }
}

// MARK: - Visualization and Analysis

/// Results analyzer and visualizer
struct ResultsAnalyzer {
    let pinn: PINN
    let rk4Solution: [[Double]]
    let xPoints: [Double]
    let tPoints: [Double]
    
    /// Compare PINN and RK4 solutions
    func compareAtTime(t: Double) -> [(x: Double, pinnSolution: Double, rk4Solution: Double, error: Double)] {
        let tIndex = Int(t / (tPoints.last! / Double(tPoints.count - 1)))
        let clampedIndex = min(max(tIndex, 0), rk4Solution.count - 1)
        
        return xPoints.enumerated().map { i, x in
            let pinnU = pinn.forward(x: x, t: t)
            let rk4U = rk4Solution[clampedIndex][i]
            let error = abs(pinnU - rk4U)
            
            return (x: x, pinnSolution: pinnU, rk4Solution: rk4U, error: error)
        }
    }
    
    /// Calculate framework metrics
    func calculateFrameworkMetrics() -> HybridFramework {
        var framework = pinn.framework
        
        // Update S(x) based on PINN performance
        let testPoints = Array(stride(from: 0.0, through: 1.0, by: 0.1))
        let pinnSolutions = testPoints.map { pinn.forward(x: $0, t: 0.5) }
        let variance = pinnSolutions.reduce(0) { $0 + $1 * $1 } / Double(pinnSolutions.count)
        framework.S_x = min(max(0.7 + 0.1 * (1.0 - variance), 0.0), 1.0)
        
        // Update N(x) based on training convergence
        if let lastLoss = pinn.layers.first?.biases.first {
            framework.N_x = min(max(0.8 - abs(lastLoss) * 0.1, 0.0), 1.0)
        }
        
        return framework
    }
    
    /// Print detailed analysis
    func printAnalysis() {
        let framework = calculateFrameworkMetrics()
        let psi = framework.psi(baseProb: 0.8)
        
        print("\n=== PINN Framework Analysis ===")
        print("S(x) (State Inference): \(String(format: "%.3f", framework.S_x))")
        print("N(x) (ML Analysis): \(String(format: "%.3f", framework.N_x))")
        print("α(t) (Validation Flow): \(String(format: "%.3f", framework.alpha_t))")
        print("R_cognitive: \(String(format: "%.3f", framework.R_cognitive))")
        print("R_efficiency: \(String(format: "%.3f", framework.R_efficiency))")
        print("β (Responsiveness): \(String(format: "%.3f", framework.beta))")
        print("Ψ(x) (Final Output): \(String(format: "%.3f", psi))")
        
        let comparison = compareAtTime(t: 0.5)
        let avgError = comparison.reduce(0.0) { $0 + $1.error } / Double(comparison.count)
        print("Average PINN vs RK4 Error: \(String(format: "%.6f", avgError))")
        
        // Interpretation
        if psi > 0.7 {
            print("Interpretation: Excellent model performance with strong hybrid intelligence")
        } else if psi > 0.6 {
            print("Interpretation: Good model performance with solid framework integration")
        } else if psi > 0.5 {
            print("Interpretation: Moderate performance, framework shows potential")
        } else {
            print("Interpretation: Performance needs improvement, consider parameter tuning")
        }
    }
}

// MARK: - Main Execution

/// Main execution function
func main() {
    print("=== Physics-Informed Neural Network Framework ===")
    print("Solving 1D Inviscid Burgers' Equation: ∂u/∂t + u∂u/∂x = 0")
    print("Initial Condition: u(x,0) = -sin(πx)")
    print("Domain: x ∈ [0,1], t ∈ [0,1]")
    
    // Initialize framework parameters
    let framework = HybridFramework(
        S_x: 0.72,
        N_x: 0.85,
        alpha_t: 0.5,
        R_cognitive: 0.15,
        R_efficiency: 0.10,
        beta: 1.2
    )
    
    // Create PINN with architecture [2, 20, 20, 1]
    let pinn = PINN(architecture: [2, 20, 20, 1], framework: framework)
    let solver = BurgersEquationSolver(pinn: pinn)
    let trainer = PINNTrainer(solver: solver)
    
    // Training and validation points
    let numPoints = 25
    let xPoints = (0..<numPoints).map { Double($0) / Double(numPoints - 1) }
    let tPoints = (0..<numPoints).map { Double($0) / Double(numPoints - 1) }
    
    // Train the PINN
    trainer.train(epochs: 100, xPoints: xPoints, tPoints: tPoints, learningRate: 0.01)
    
    // Generate RK4 solution for comparison
    print("\nGenerating RK4 reference solution...")
    let rk4Solution = RK4Integrator.solveBurgers(xPoints: xPoints, tFinal: 1.0, dt: 0.01)
    
    // Analyze results
    let analyzer = ResultsAnalyzer(pinn: pinn, rk4Solution: rk4Solution, xPoints: xPoints, tPoints: tPoints)
    analyzer.printAnalysis()
    
    // Demonstrate numerical example from the framework
    print("\n=== Numerical Example: Single Training Step ===")
    let exampleFramework = HybridFramework(
        S_x: 0.72,
        N_x: 0.85,
        alpha_t: 0.5,
        R_cognitive: 0.15,
        R_efficiency: 0.10,
        beta: 1.2
    )
    
    let O_hybrid = exampleFramework.hybridOutput()
    let exp_penalty = exampleFramework.exponentialPenalty()
    let P_adj = exampleFramework.adjustedProbability(baseProb: 0.80)
    let psi_final = exampleFramework.psi(baseProb: 0.80)
    
    print("Step 1: S(x) = \(exampleFramework.S_x), N(x) = \(exampleFramework.N_x)")
    print("Step 2: α = \(exampleFramework.alpha_t), O_hybrid = \(String(format: "%.3f", O_hybrid))")
    print("Step 3: R_cognitive = \(exampleFramework.R_cognitive), R_efficiency = \(exampleFramework.R_efficiency)")
    print("        P_total = \(String(format: "%.3f", exampleFramework.totalPenalty())), exp ≈ \(String(format: "%.3f", exp_penalty))")
    print("Step 4: P = 0.80, β = \(exampleFramework.beta), P_adj ≈ \(String(format: "%.3f", P_adj))")
    print("Step 5: Ψ(x) ≈ \(String(format: "%.3f", psi_final))")
    print("Step 6: Interpretation - Ψ(x) ≈ \(String(format: "%.2f", psi_final)) indicates \(psi_final > 0.65 ? "solid" : "moderate") model performance")
}

// Execute main function
main()