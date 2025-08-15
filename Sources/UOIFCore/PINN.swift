// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

// MARK: - PINN Neural Network Architecture
public class PINN {
    private var weights: [[Double]]
    private var biases: [[Double]]
    private let inputSize: Int
    private let hiddenSize: Int
    private let outputSize: Int
    
    public init(inputSize: Int = 2, hiddenSize: Int = 20, outputSize: Int = 1) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        
        // Xavier initialization for better training
        self.weights = [
            Self.xavierInit(rows: hiddenSize, cols: inputSize),
            Self.xavierInit(rows: hiddenSize, cols: hiddenSize),
            Self.xavierInit(rows: outputSize, cols: hiddenSize)
        ]
        
        self.biases = [
            Array(repeating: 0.0, count: hiddenSize),
            Array(repeating: 0.0, count: hiddenSize),
            Array(repeating: 0.0, count: outputSize)
        ]
    }
    
    // MARK: - Xavier Initialization
    private static func xavierInit(rows: Int, cols: Int) -> [[Double]] {
        let scale = sqrt(2.0 / Double(cols))
        return (0..<rows).map { _ in
            (0..<cols).map { _ in
                Double.random(in: -scale...scale)
            }
        }
    }
    
    // MARK: - Forward Pass
    public func forward(_ input: [Double]) -> [Double] {
        var current = input
        
        for i in 0..<weights.count {
            current = Self.linearTransform(current, weights: weights[i], bias: biases[i])
            if i < weights.count - 1 {
                current = Self.tanh(current)
            }
        }
        
        return current
    }
    
    private static func linearTransform(_ input: [Double], weights: [[Double]], bias: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: weights.count)
        
        for i in 0..<weights.count {
            for j in 0..<input.count {
                output[i] += weights[i][j] * input[j]
            }
            output[i] += bias[i]
        }
        
        return output
    }
    
    private static func tanh(_ input: [Double]) -> [Double] {
        return input.map { tanh($0) }
    }
    
    // MARK: - Gradient Computation
    public func computeGradients(x: [Double], t: [Double], learningRate: Double = 0.005) {
        let perturbation: Double = 1e-5
        
        for layer in 0..<weights.count {
            for i in 0..<weights[layer].count {
                for j in 0..<weights[layer][0].count {
                    // Forward pass with perturbation
                    weights[layer][i][j] += perturbation
                    let lossPlus = computeTotalLoss(x: x, t: t)
                    
                    // Forward pass without perturbation
                    weights[layer][i][j] -= 2 * perturbation
                    let lossMinus = computeTotalLoss(x: x, t: t)
                    
                    // Reset weight
                    weights[layer][i][j] += perturbation
                    
                    // Compute gradient
                    let gradient = (lossPlus - lossMinus) / (2 * perturbation)
                    
                    // Update weight
                    weights[layer][i][j] -= learningRate * gradient
                }
            }
            
            // Update biases
            for i in 0..<biases[layer].count {
                biases[layer][i] += perturbation
                let lossPlus = computeTotalLoss(x: x, t: t)
                
                biases[layer][i] -= 2 * perturbation
                let lossMinus = computeTotalLoss(x: x, t: t)
                
                biases[layer][i] += perturbation
                
                let gradient = (lossPlus - lossMinus) / (2 * perturbation)
                biases[layer][i] -= learningRate * gradient
            }
        }
    }
    
    // MARK: - Loss Functions
    public func computeTotalLoss(x: [Double], t: [Double]) -> Double {
        let pdeLoss = computePDELoss(x: x, t: t)
        let icLoss = computeInitialConditionLoss(x: x)
        return pdeLoss + icLoss
    }
    
    private func computePDELoss(x: [Double], t: [Double]) -> Double {
        var totalLoss: Double = 0.0
        
        for i in 0..<x.count {
            for j in 0..<t.count {
                let input = [x[i], t[j]]
                let output = forward(input)[0]
                
                // Burgers' equation: u_t + u*u_x - ν*u_xx = 0
                let u_t = computeTimeDerivative(x: x[i], t: t[j])
                let u_x = computeSpatialDerivative(x: x[i], t: t[j])
                let u_xx = computeSecondSpatialDerivative(x: x[i], t: t[j])
                
                let residual = u_t + output * u_x - 0.01 * u_xx
                totalLoss += residual * residual
            }
        }
        
        return totalLoss / Double(x.count * t.count)
    }
    
    private func computeInitialConditionLoss(x: [Double]) -> Double {
        var totalLoss: Double = 0.0
        
        for i in 0..<x.count {
            let input = [x[i], 0.0] // t = 0
            let output = forward(input)[0]
            
            // Initial condition: u(x,0) = sin(πx)
            let target = sin(Double.pi * x[i])
            let error = output - target
            totalLoss += error * error
        }
        
        return totalLoss / Double(x.count)
    }
    
    // MARK: - Finite Difference Approximations
    private func computeTimeDerivative(x: Double, t: Double) -> Double {
        let dt = 1e-5
        let input1 = [x, t + dt]
        let input2 = [x, t - dt]
        let u1 = forward(input1)[0]
        let u2 = forward(input2)[0]
        return (u1 - u2) / (2 * dt)
    }
    
    private func computeSpatialDerivative(x: Double, t: Double) -> Double {
        let dx = 1e-5
        let input1 = [x + dx, t]
        let input2 = [x - dx, t]
        let u1 = forward(input1)[0]
        let u2 = forward(input2)[0]
        return (u1 - u2) / (2 * dx)
    }
    
    private func computeSecondSpatialDerivative(x: Double, t: Double) -> Double {
        let dx = 1e-5
        let input1 = [x + dx, t]
        let input2 = [x - dx, t]
        let input3 = [x, t]
        let u1 = forward(input1)[0]
        let u2 = forward(input2)[0]
        let u3 = forward(input3)[0]
        return (u1 - 2 * u3 + u2) / (dx * dx)
    }
    
    // MARK: - Prediction
    public func predict(x: Double, t: Double) -> Double {
        let input = [x, t]
        return forward(input)[0]
    }
}

// MARK: - Training Functions
public func train(model: PINN, epochs: Int, x: [Double], t: [Double], printEvery: Int = 50) {
    print("Starting PINN training for \(epochs) epochs...")
    
    for epoch in 0..<epochs {
        model.computeGradients(x: x, t: t)
        
        if epoch % printEvery == 0 {
            let loss = model.computeTotalLoss(x: x, t: t)
            print("Epoch \(epoch): Loss = \(String(format: "%.6f", loss))")
        }
    }
    
    print("Training completed!")
}

// MARK: - RK4 Solver for Comparison
public class RK4Solver {
    public static func solveBurgers(x: [Double], t: Double, nu: Double = 0.01) -> [Double] {
        return x.map { xVal in
            // Simplified analytical solution for comparison
            // In practice, this would be a full RK4 implementation
            let k = 2.0 * Double.pi
            return exp(-nu * k * k * t) * sin(k * xVal)
        }
    }
}

// MARK: - PINN Analysis with Ψ Framework
public struct PINNAnalysis {
    public let S_x: Double      // State inference accuracy
    public let N_x: Double      // ML gradient descent convergence
    public let alpha_t: Double  // Real-time validation balance
    public let O_hybrid: Double // Hybrid output
    public let R_cognitive: Double // PDE residual error
    public let R_efficiency: Double // Computational overhead
    public let P_total: Double  // Total penalty
    public let P_adj: Double    // Adjusted probability
    public let Psi_x: Double    // Final Ψ value
    
    public init(
        S_x: Double,
        N_x: Double,
        alpha_t: Double,
        R_cognitive: Double,
        R_efficiency: Double,
        baseProbability: Double = 0.80,
        beta: Double = 1.2,
        lambda1: Double = 0.6,
        lambda2: Double = 0.4
    ) {
        self.S_x = S_x
        self.N_x = N_x
        self.alpha_t = alpha_t
        self.R_cognitive = R_cognitive
        self.R_efficiency = R_efficiency
        
        // Compute hybrid output
        self.O_hybrid = (1.0 - alpha_t) * S_x + alpha_t * N_x
        
        // Compute total penalty
        self.P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
        
        // Compute adjusted probability
        self.P_adj = min(baseProbability * beta, 1.0)
        
        // Compute final Ψ value
        self.Psi_x = O_hybrid * exp(-P_total) * P_adj
    }
}

// MARK: - Training Data Generation
public func generateTrainingData() -> (x: [Double], t: [Double]) {
    let x = Array(stride(from: -1.0, through: 1.0, by: 0.05))
    let t = Array(stride(from: 0.0, through: 1.0, by: 0.05))
    return (x, t)
}

// MARK: - Solution Comparison
public func compareSolutions(pinn: PINN, x: [Double], t: Double) -> (pinn: [Double], rk4: [Double]) {
    let pinnSolutions = x.map { pinn.predict(x: $0, t: t) }
    let rk4Solutions = RK4Solver.solveBurgers(x: x, t: t)
    return (pinn: pinnSolutions, rk4: rk4Solutions)
}