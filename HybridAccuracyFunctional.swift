import Foundation
import SwiftUI

// MARK: - Dense Layer Implementation
class DenseLayer {
    var weights: [[Double]]
    var biases: [Double]
    var momentumW: [[Double]]
    var momentumB: [Double]
    
    init(inputSize: Int, outputSize: Int) {
        // Xavier initialization
        let bound = sqrt(6.0 / Double(inputSize + outputSize))
        self.weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Double.random(in: -bound...bound) }
        }
        self.biases = Array(repeating: 0.0, count: outputSize)
        
        // Initialize momentum arrays
        self.momentumW = Array(repeating: Array(repeating: 0.0, count: inputSize), count: outputSize)
        self.momentumB = Array(repeating: 0.0, count: outputSize)
    }
    
    func forward(_ input: [Double]) -> [Double] {
        return (0..<weights.count).map { i in
            let weightedSum = zip(weights[i], input).map(*).reduce(0, +)
            return weightedSum + biases[i]
        }
    }
    
    func updateWeights(gradients: [[Double]], biasGradients: [Double], 
                      learningRate: Double, momentum: Double = 0.9) {
        // Update weights with momentum
        for i in 0..<weights.count {
            for j in 0..<weights[i].count {
                momentumW[i][j] = momentum * momentumW[i][j] - learningRate * gradients[i][j]
                weights[i][j] += momentumW[i][j]
            }
        }
        
        // Update biases with momentum
        for i in 0..<biases.count {
            momentumB[i] = momentum * momentumB[i] - learningRate * biasGradients[i]
            biases[i] += momentumB[i]
        }
    }
}

// MARK: - Physics-Informed Neural Network
class PINN {
    var layers: [DenseLayer]
    
    init(layerSizes: [Int] = [2, 50, 50, 1]) {
        self.layers = []
        for i in 0..<(layerSizes.count - 1) {
            layers.append(DenseLayer(inputSize: layerSizes[i], outputSize: layerSizes[i + 1]))
        }
    }
    
    func forward(_ input: [Double]) -> [Double] {
        var output = input
        for (index, layer) in layers.enumerated() {
            output = layer.forward(output)
            // Apply tanh activation to all layers except the last
            if index < layers.count - 1 {
                output = output.map { tanh($0) }
            }
        }
        return output
    }
    
    func predict(x: Double, t: Double) -> Double {
        let input = [x, t]
        let output = forward(input)
        return output[0]
    }
}

// MARK: - Hybrid Accuracy Functional Configuration
struct HybridConfig {
    let lambda1: Double
    let lambda2: Double
    let beta: Double
    let kappa: Double
    
    init(lambda1: Double = 0.75, lambda2: Double = 0.25, 
         beta: Double = 1.2, kappa: Double = 1.0) {
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.beta = beta
        self.kappa = kappa
    }
}

// MARK: - Hybrid Accuracy Functional Implementation
class HybridAccuracyFunctional {
    let config: HybridConfig
    let pinn: PINN
    
    init(config: HybridConfig = HybridConfig()) {
        self.config = config
        self.pinn = PINN()
    }
    
    // Symbolic accuracy S(x,t) ∈ [0,1] - RK4 solution fidelity
    func symbolicAccuracy(x: Double, t: Double) -> Double {
        // Simulate RK4 accuracy - higher for stable regions
        let baseAccuracy = 0.9 - 0.2 * exp(-abs(x))
        let noise = 0.05 * Double.random(in: -1...1)
        return max(0, min(1, baseAccuracy + noise))
    }
    
    // Neural accuracy N(x,t) ∈ [0,1] - ML/NN prediction fidelity
    func neuralAccuracy(x: Double, t: Double) -> Double {
        // Simulate neural network accuracy - adaptive to chaos
        let baseAccuracy = 0.85 + 0.1 * tanh(2 * abs(x))
        let noise = 0.03 * Double.random(in: -1...1)
        return max(0, min(1, baseAccuracy + noise))
    }
    
    // Adaptive weight α(t) ∈ [0,1] - favoring N in chaotic regions
    func adaptiveWeight(t: Double, lyapunovExp: Double? = nil) -> Double {
        let lyapExp = lyapunovExp ?? (0.5 * sin(2 * Double.pi * t) + 0.3 * Double.random(in: -0.1...0.1))
        return 1.0 / (1.0 + exp(config.kappa * lyapExp)) // sigmoid(-κ * λ_local)
    }
    
    // Cognitive penalty R_cog(t) ≥ 0 - physics violation penalty
    func cognitivePenalty(x: Double, t: Double) -> Double {
        // Higher penalties for extreme conditions
        return 0.1 + 0.15 * exp(-0.5 * x * x) * (1 + 0.1 * abs(t))
    }
    
    // Efficiency penalty R_eff(t) ≥ 0 - computational cost penalty
    func efficiencyPenalty(x: Double, t: Double) -> Double {
        // Varies with problem complexity
        return 0.08 + 0.12 * (1 + abs(x)) * (1 + 0.05 * t)
    }
    
    // Calibrated probability P(H|E,β,t) ∈ [0,1] - Platt scaling with bias
    func calibratedProbability(baseProb: Double, t: Double) -> Double {
        let baseProbClipped = max(1e-7, min(1 - 1e-7, baseProb))
        let logitP = log(baseProbClipped / (1 - baseProbClipped))
        let adjustedLogit = logitP + log(config.beta)
        return max(0, min(1, 1.0 / (1.0 + exp(-adjustedLogit))))
    }
    
    // Compute hybrid accuracy functional Ψ(x)
    func computePsi(x: Double, t: Double, baseProb: Double = 0.85) -> Double {
        let S = symbolicAccuracy(x: x, t: t)
        let N = neuralAccuracy(x: x, t: t)
        let alpha = adaptiveWeight(t: t)
        
        let rCog = cognitivePenalty(x: x, t: t)
        let rEff = efficiencyPenalty(x: x, t: t)
        
        let pCalibrated = calibratedProbability(baseProb: baseProb, t: t)
        
        // Hybrid output
        let hybridOutput = alpha * S + (1 - alpha) * N
        
        // Regularization term
        let regularization = exp(-(config.lambda1 * rCog + config.lambda2 * rEff))
        
        // Final functional
        let psi = hybridOutput * regularization * pCalibrated
        
        return psi
    }
    
    // Compute detailed breakdown for analysis
    func computePsiDetailed(x: Double, t: Double, baseProb: Double = 0.85) -> [String: Double] {
        let S = symbolicAccuracy(x: x, t: t)
        let N = neuralAccuracy(x: x, t: t)
        let alpha = adaptiveWeight(t: t)
        let rCog = cognitivePenalty(x: x, t: t)
        let rEff = efficiencyPenalty(x: x, t: t)
        let pCalibrated = calibratedProbability(baseProb: baseProb, t: t)
        let hybridOutput = alpha * S + (1 - alpha) * N
        let regularization = exp(-(config.lambda1 * rCog + config.lambda2 * rEff))
        let psi = hybridOutput * regularization * pCalibrated
        
        return [
            "psi": psi,
            "S": S,
            "N": N,
            "alpha": alpha,
            "R_cog": rCog,
            "R_eff": rEff,
            "P_calibrated": pCalibrated,
            "hybrid_output": hybridOutput,
            "regularization": regularization
        ]
    }
}

// MARK: - Burgers Equation Solver
class BurgersSolver {
    let nu: Double // Viscosity parameter
    let pinn: PINN
    
    init(nu: Double = 0.01 / Double.pi) {
        self.nu = nu
        self.pinn = PINN()
    }
    
    // Initial condition: u(x, 0) = -sin(πx)
    func initialCondition(x: Double) -> Double {
        return -sin(Double.pi * x)
    }
    
    // Finite difference for spatial derivatives
    func finiteDiff(f: (Double) -> Double, at x: Double, dx: Double = 1e-6) -> Double {
        return (f(x + dx) - f(x - dx)) / (2 * dx)
    }
    
    // Second derivative using finite differences
    func secondDerivative(values: [Double], dx: Double) -> [Double] {
        var result = Array(repeating: 0.0, count: values.count)
        for i in 1..<(values.count - 1) {
            result[i] = (values[i + 1] - 2 * values[i] + values[i - 1]) / (dx * dx)
        }
        return result
    }
    
    // RK4 solver for viscous Burgers equation
    func solveRK4(xGrid: [Double], tFinal: Double, nt: Int = 1000) -> (tGrid: [Double], solution: [[Double]]) {
        let nx = xGrid.count
        let dx = xGrid[1] - xGrid[0]
        let dt = tFinal / Double(nt)
        let tGrid = (0...nt).map { Double($0) * dt }
        
        var solution = Array(repeating: Array(repeating: 0.0, count: nx), count: nt + 1)
        
        // Initial condition
        for i in 0..<nx {
            solution[0][i] = initialCondition(x: xGrid[i])
        }
        
        // RK4 time integration
        for n in 0..<nt {
            let u = solution[n]
            
            // Compute spatial derivatives
            let u_x = (1..<(nx-1)).map { i in (u[i + 1] - u[i - 1]) / (2 * dx) }
            let u_xx = secondDerivative(values: u, dx: dx)
            
            // RHS function for Burgers equation: u_t = -u*u_x + ν*u_xx
            func rhs(_ u_vals: [Double]) -> [Double] {
                var result = Array(repeating: 0.0, count: nx)
                for i in 1..<(nx-1) {
                    let ux = (u_vals[i + 1] - u_vals[i - 1]) / (2 * dx)
                    let uxx = (u_vals[i + 1] - 2 * u_vals[i] + u_vals[i - 1]) / (dx * dx)
                    result[i] = -u_vals[i] * ux + nu * uxx
                }
                return result
            }
            
            // RK4 step
            let k1 = rhs(u).map { $0 * dt }
            let u_k1 = zip(u, k1).map { $0 + $1 / 2 }
            
            let k2 = rhs(u_k1).map { $0 * dt }
            let u_k2 = zip(u, k2).map { $0 + $1 / 2 }
            
            let k3 = rhs(u_k2).map { $0 * dt }
            let u_k3 = zip(u, k3).map { $0 + $1 }
            
            let k4 = rhs(u_k3).map { $0 * dt }
            
            // Update solution
            for i in 0..<nx {
                solution[n + 1][i] = u[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
            }
        }
        
        return (tGrid, solution)
    }
}

// MARK: - Numerical Examples and Demonstrations
class HybridFunctionalDemo {
    let functional: HybridAccuracyFunctional
    let burgersSolver: BurgersSolver
    
    init() {
        let config = HybridConfig(lambda1: 0.75, lambda2: 0.25, beta: 1.2)
        self.functional = HybridAccuracyFunctional(config: config)
        self.burgersSolver = BurgersSolver()
    }
    
    // Reproduce the numerical example from the document
    func reproduceNumericalExample() -> [String: Double] {
        print("Reproducing Numerical Example from Document:")
        print("=" + String(repeating: "=", count: 49))
        
        // Step 1: Outputs
        let S = 0.67
        let N = 0.87
        print("Step 1 - Outputs: S(x) = \(S), N(x) = \(N)")
        
        // Step 2: Hybrid
        let alpha = 0.4
        let oHybrid = alpha * S + (1 - alpha) * N
        print("Step 2 - Hybrid: α = \(alpha), O_hybrid = \(String(format: "%.3f", oHybrid))")
        
        // Step 3: Penalties
        let rCognitive = 0.17
        let rEfficiency = 0.11
        let lambda1 = 0.6
        let lambda2 = 0.4
        let pTotal = lambda1 * rCognitive + lambda2 * rEfficiency
        let expTerm = exp(-pTotal)
        print("Step 3 - Penalties: R_cog = \(rCognitive), R_eff = \(rEfficiency)")
        print("         λ₁ = \(lambda1), λ₂ = \(lambda2), P_total = \(String(format: "%.3f", pTotal))")
        print("         exp(-P_total) ≈ \(String(format: "%.3f", expTerm))")
        
        // Step 4: Probability
        let P = 0.81
        let beta = 1.2
        let logitP = log(P / (1 - P))
        let pAdj = 1.0 / (1.0 + exp(-(logitP + log(beta))))
        print("Step 4 - Probability: P = \(P), β = \(beta)")
        print("         P_adj ≈ \(String(format: "%.3f", pAdj))")
        
        // Step 5: Ψ(x)
        let psi = oHybrid * expTerm * pAdj
        print("Step 5 - Ψ(x): ≈ \(String(format: "%.3f", oHybrid)) × \(String(format: "%.3f", expTerm)) × \(String(format: "%.3f", pAdj)) ≈ \(String(format: "%.3f", psi))")
        
        // Step 6: Interpret
        print("Step 6 - Interpretation: Ψ(x) ≈ \(String(format: "%.2f", psi)) indicates high responsiveness")
        print("")
        
        return [
            "S": S,
            "N": N,
            "alpha": alpha,
            "O_hybrid": oHybrid,
            "R_cognitive": rCognitive,
            "R_efficiency": rEfficiency,
            "exp_term": expTerm,
            "P_adj": pAdj,
            "psi": psi
        ]
    }
    
    // Demonstrate various scenarios
    func demonstrateScenarios() {
        print("Demonstrating Hybrid Accuracy Functional Framework:")
        print("=" + String(repeating: "=", count: 54))
        
        let scenarios: [(String, Double, Double)] = [
            ("Chaotic System (Multi-pendulum)", 0.5, 1.0),
            ("Stable System", 0.1, 0.5),
            ("High Chaos Region", 1.2, 0.8),
            ("Open-Source Contribution", 0.74, 0.5),
            ("Collaboration Benefits", 0.6, 1.2)
        ]
        
        for (name, x, t) in scenarios {
            print("\n\(name):")
            print(String(repeating: "-", count: name.count))
            
            let results = functional.computePsiDetailed(x: x, t: t)
            
            print("  Ψ(x) = \(String(format: "%.3f", results["psi"]!))")
            print("  Components:")
            print("    Symbolic Accuracy (S): \(String(format: "%.3f", results["S"]!))")
            print("    Neural Accuracy (N): \(String(format: "%.3f", results["N"]!))")
            print("    Adaptive Weight (α): \(String(format: "%.3f", results["alpha"]!))")
            print("    Cognitive Penalty (R_cog): \(String(format: "%.3f", results["R_cog"]!))")
            print("    Efficiency Penalty (R_eff): \(String(format: "%.3f", results["R_eff"]!))")
            print("    Calibrated Probability (P): \(String(format: "%.3f", results["P_calibrated"]!))")
        }
    }
    
    // Demonstrate Burgers equation solving
    func demonstrateBurgersSolver() {
        print("\nViscous Burgers Equation Demonstration:")
        print("=" + String(repeating: "=", count: 40))
        
        let xGrid = stride(from: -1.0, through: 1.0, by: 0.1).map { $0 }
        let (tGrid, solution) = burgersSolver.solveRK4(xGrid: xGrid, tFinal: 1.0, nt: 100)
        
        print("Solved viscous Burgers equation with:")
        print("  Viscosity (ν): \(String(format: "%.6f", burgersSolver.nu))")
        print("  Spatial points: \(xGrid.count)")
        print("  Time points: \(tGrid.count)")
        print("  Final time: \(tGrid.last!)")
        
        // Compute some statistics
        let finalSolution = solution.last!
        let maxValue = finalSolution.max()!
        let minValue = finalSolution.min()!
        
        print("  Final solution range: [\(String(format: "%.3f", minValue)), \(String(format: "%.3f", maxValue))]")
    }
}

// MARK: - Main Execution
func runHybridFunctionalDemo() {
    let demo = HybridFunctionalDemo()
    
    // Set random seed for reproducibility
    srand48(42)
    
    // Reproduce numerical example
    let _ = demo.reproduceNumericalExample()
    
    // Demonstrate various scenarios
    demo.demonstrateScenarios()
    
    // Demonstrate Burgers solver
    demo.demonstrateBurgersSolver()
    
    print("\nHybrid Accuracy Functional demonstration completed!")
}

// Execute the demonstration
runHybridFunctionalDemo()