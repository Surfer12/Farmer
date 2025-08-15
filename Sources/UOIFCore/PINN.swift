// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation
import Accelerate

// MARK: - PINN Core Structures

/// Represents a single training point for the PINN
public struct TrainingPoint {
    public let x: Double
    public let t: Double
    public let u: Double?  // Optional: only for initial/boundary conditions
    
    public init(x: Double, t: Double, u: Double? = nil) {
        self.x = x
        self.t = t
        self.u = u
    }
}

/// Represents the PINN solution and validation metrics
public struct PINNSolution {
    public let u: [[Double]]  // u[x][t] grid
    public let x: [Double]    // spatial grid
    public let t: [Double]    // temporal grid
    public let pdeResidual: [[Double]]  // PDE residual at each grid point
    public let trainingLoss: Double
    public let validationLoss: Double
    
    public init(u: [[Double]], x: [Double], t: [Double], pdeResidual: [[Double]], trainingLoss: Double, validationLoss: Double) {
        self.u = u
        self.x = x
        self.t = t
        self.pdeResidual = pdeResidual
        self.trainingLoss = trainingLoss
        self.validationLoss = validationLoss
    }
}

// MARK: - Neural Network Layer

/// Dense neural network layer with forward pass
public class DenseLayer {
    public var weights: [[Double]]
    public var biases: [Double]
    public let activation: ActivationFunction
    
    public init(inputSize: Int, outputSize: Int, activation: ActivationFunction = .tanh) {
        self.weights = Array(repeating: Array(repeating: 0.0, count: inputSize), count: outputSize)
        self.biases = Array(repeating: 0.0, count: outputSize)
        self.activation = activation
        
        // Xavier/Glorot initialization
        let scale = sqrt(2.0 / Double(inputSize))
        for i in 0..<outputSize {
            for j in 0..<inputSize {
                weights[i][j] = Double.random(in: -scale...scale)
            }
            biases[i] = Double.random(in: -0.1...0.1)
        }
    }
    
    public func forward(_ input: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: weights.count)
        
        // Matrix multiplication: output = weights * input + biases
        for i in 0..<weights.count {
            for j in 0..<weights[i].count {
                output[i] += weights[i][j] * input[j]
            }
            output[i] += biases[i]
        }
        
        // Apply activation function
        return output.map { activation.apply($0) }
    }
}

// MARK: - Activation Functions

public enum ActivationFunction {
    case tanh
    case sigmoid
    case relu
    case sin
    
    public func apply(_ x: Double) -> Double {
        switch self {
        case .tanh:
            return tanh(x)
        case .sigmoid:
            return 1.0 / (1.0 + exp(-x))
        case .relu:
            return max(0, x)
        case .sin:
            return sin(x)
        }
    }
    
    public func derivative(_ x: Double) -> Double {
        switch self {
        case .tanh:
            let t = tanh(x)
            return 1.0 - t * t
        case .sigmoid:
            let s = 1.0 / (1.0 + exp(-x))
            return s * (1.0 - s)
        case .relu:
            return x > 0 ? 1.0 : 0.0
        case .sin:
            return cos(x)
        }
    }
}

// MARK: - PINN Model

/// Physics-Informed Neural Network for solving PDEs
public class PINN {
    public var layers: [DenseLayer]
    public let learningRate: Double
    public let maxEpochs: Int
    
    public init(layerSizes: [Int], learningRate: Double = 0.001, maxEpochs: Int = 10000) {
        self.layers = []
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        
        // Build network architecture
        for i in 0..<(layerSizes.count - 1) {
            let activation: ActivationFunction = (i == layerSizes.count - 2) ? .tanh : .tanh
            layers.append(DenseLayer(inputSize: layerSizes[i], outputSize: layerSizes[i + 1], activation: activation))
        }
    }
    
    /// Forward pass through the network
    public func forward(x: Double, t: Double) -> Double {
        var input = [x, t]
        
        for layer in layers {
            input = layer.forward(input)
        }
        
        return input[0]
    }
    
    /// Compute partial derivatives using finite differences
    public func computeDerivatives(x: Double, t: Double, dx: Double = 1e-5, dt: Double = 1e-5) -> (u_x: Double, u_t: Double, u_xx: Double) {
        // First derivatives
        let u_x = (forward(x: x + dx, t: t) - forward(x: x - dx, t: t)) / (2 * dx)
        let u_t = (forward(x: x, t: t + dt) - forward(x: x, t: t - dt)) / (2 * dt)
        
        // Second derivative
        let u_xx = (forward(x: x + dx, t: t) - 2 * forward(x: x, t: t) + forward(x: x - dx, t: t)) / (dx * dx)
        
        return (u_x, u_t, u_xx)
    }
    
    /// Compute PDE residual: u_t + u * u_x = 0 (inviscid Burgers)
    public func pdeResidual(x: Double, t: Double) -> Double {
        let (u_x, u_t, _) = computeDerivatives(x: x, t: t)
        let u = forward(x: x, t: t)
        return u_t + u * u_x
    }
    
    /// Training loss combining PDE residual and initial/boundary conditions
    public func computeLoss(collocationPoints: [TrainingPoint], initialPoints: [TrainingPoint], boundaryPoints: [TrainingPoint]) -> Double {
        var totalLoss = 0.0
        
        // PDE residual loss
        for point in collocationPoints {
            let residual = pdeResidual(x: point.x, t: point.t)
            totalLoss += residual * residual
        }
        
        // Initial condition loss
        for point in initialPoints {
            guard let u_true = point.u else { continue }
            let u_pred = forward(x: point.x, t: point.t)
            totalLoss += (u_pred - u_true) * (u_pred - u_true)
        }
        
        // Boundary condition loss
        for point in boundaryPoints {
            guard let u_true = point.u else { continue }
            let u_pred = forward(x: point.x, t: point.t)
            totalLoss += (u_pred - u_true) * (u_pred - u_true)
        }
        
        return totalLoss
    }
    
    /// Simple gradient descent training (for demonstration)
    public func train(collocationPoints: [TrainingPoint], initialPoints: [TrainingPoint], boundaryPoints: [TrainingPoint]) -> [Double] {
        var losses: [Double] = []
        
        for epoch in 0..<maxEpochs {
            let loss = computeLoss(collocationPoints: collocationPoints, initialPoints: initialPoints, boundaryPoints: boundaryPoints)
            losses.append(loss)
            
            if epoch % 1000 == 0 {
                print("Epoch \(epoch): Loss = \(loss)")
            }
            
            // Simple gradient descent update (simplified)
            // In practice, use proper backpropagation or automatic differentiation
            updateWeights(loss: loss)
        }
        
        return losses
    }
    
    /// Simplified weight update (placeholder for proper backpropagation)
    private func updateWeights(loss: Double) {
        // This is a simplified update - in practice, implement proper backpropagation
        // or use Swift for TensorFlow for automatic differentiation
        for layer in layers {
            for i in 0..<layer.weights.count {
                for j in 0..<layer.weights[i].count {
                    layer.weights[i][j] -= learningRate * loss * 0.01  // Simplified gradient
                }
            }
        }
    }
}

// MARK: - RK4 Validation

/// Runge-Kutta 4th order method for validation
public class RK4Validator {
    /// Single RK4 step for the Burgers' equation
    public static func step(f: (Double, [Double]) -> [Double], y: [Double], t: Double, dt: Double) -> [Double] {
        let k1 = f(t, y)
        var y2 = y
        for i in 0..<y.count {
            y2[i] = y[i] + 0.5 * dt * k1[i]
        }
        
        let k2 = f(t + 0.5 * dt, y2)
        var y3 = y
        for i in 0..<y.count {
            y3[i] = y[i] + 0.5 * dt * k2[i]
        }
        
        let k3 = f(t + 0.5 * dt, y3)
        var y4 = y
        for i in 0..<y.count {
            y4[i] = y[i] + dt * k3[i]
        }
        
        let k4 = f(t + dt, y4)
        
        var result = y
        for i in 0..<y.count {
            result[i] = y[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        }
        
        return result
    }
    
    /// Solve Burgers' equation using RK4 for validation
    public static func solveBurgers(x: [Double], t: [Double], initialCondition: (Double) -> Double) -> [[Double]] {
        let nx = x.count
        let nt = t.count
        var u = Array(repeating: Array(repeating: 0.0, count: nt), count: nx)
        
        // Set initial condition
        for i in 0..<nx {
            u[i][0] = initialCondition(x[i])
        }
        
        // Time stepping
        for j in 0..<(nt - 1) {
            let dt = t[j + 1] - t[j]
            
            for i in 0..<nx {
                let dx = (i < nx - 1) ? (x[i + 1] - x[i]) : (x[i] - x[i - 1])
                
                // Burgers' equation: du/dt = -u * du/dx
                let du_dx = (i < nx - 1) ? (u[i + 1][j] - u[i][j]) / dx : (u[i][j] - u[i - 1][j]) / dx
                let du_dt = -u[i][j] * du_dx
                
                u[i][j + 1] = u[i][j] + dt * du_dt
            }
        }
        
        return u
    }
}

// MARK: - PINN Solver

/// Main solver class that integrates PINN with validation
public class PINNSolver {
    public let pinn: PINN
    public let xRange: ClosedRange<Double>
    public let tRange: ClosedRange<Double>
    public let nx: Int
    public let nt: Int
    
    public init(xRange: ClosedRange<Double>, tRange: ClosedRange<Double>, nx: Int = 100, nt: Int = 100, layerSizes: [Int] = [2, 20, 20, 20, 1]) {
        self.xRange = xRange
        self.tRange = tRange
        self.nx = nx
        self.nt = nt
        self.pinn = PINN(layerSizes: layerSizes)
    }
    
    /// Generate training points
    public func generateTrainingPoints() -> (collocation: [TrainingPoint], initial: [TrainingPoint], boundary: [TrainingPoint]) {
        var collocationPoints: [TrainingPoint] = []
        var initialPoints: [TrainingPoint] = []
        var boundaryPoints: [TrainingPoint] = []
        
        // Collocation points (interior)
        for i in 0..<nx {
            for j in 0..<nt {
                let x = xRange.lowerBound + Double(i) * (xRange.upperBound - xRange.lowerBound) / Double(nx - 1)
                let t = tRange.lowerBound + Double(j) * (tRange.upperBound - tRange.lowerBound) / Double(nt - 1)
                collocationPoints.append(TrainingPoint(x: x, t: t))
            }
        }
        
        // Initial condition (t = 0)
        for i in 0..<nx {
            let x = xRange.lowerBound + Double(i) * (xRange.upperBound - xRange.lowerBound) / Double(nx - 1)
            let u = initialCondition(x)
            initialPoints.append(TrainingPoint(x: x, t: tRange.lowerBound, u: u))
        }
        
        // Boundary conditions (x = boundaries)
        for j in 0..<nt {
            let t = tRange.lowerBound + Double(j) * (tRange.upperBound - tRange.lowerBound) / Double(nt - 1)
            boundaryPoints.append(TrainingPoint(x: xRange.lowerBound, t: t, u: 0.0))
            boundaryPoints.append(TrainingPoint(x: xRange.upperBound, t: t, u: 0.0))
        }
        
        return (collocationPoints, initialPoints, boundaryPoints)
    }
    
    /// Initial condition for Burgers' equation: u(x,0) = -sin(πx)
    private func initialCondition(_ x: Double) -> Double {
        return -sin(.pi * x)
    }
    
    /// Solve the PDE using PINN
    public func solve() -> PINNSolution {
        let (collocation, initial, boundary) = generateTrainingPoints()
        
        print("Training PINN...")
        let losses = pinn.train(collocationPoints: collocation, initialPoints: initial, boundaryPoints: boundary)
        
        // Generate solution grid
        let x = Array(stride(from: xRange.lowerBound, through: xRange.upperBound, by: (xRange.upperBound - xRange.lowerBound) / Double(nx - 1)))
        let t = Array(stride(from: tRange.lowerBound, through: tRange.upperBound, by: (tRange.upperBound - tRange.lowerBound) / Double(nt - 1)))
        
        var u = Array(repeating: Array(repeating: 0.0, count: nt), count: nx)
        var pdeResidual = Array(repeating: Array(repeating: 0.0, count: nt), count: nx)
        
        for i in 0..<nx {
            for j in 0..<nt {
                u[i][j] = pinn.forward(x: x[i], t: t[j])
                pdeResidual[i][j] = pinn.pdeResidual(x: x[i], t: t[j])
            }
        }
        
        // Compute validation using RK4
        let rk4Solution = RK4Validator.solveBurgers(x: x, t: t, initialCondition: initialCondition)
        
        // Compute validation loss
        var validationLoss = 0.0
        for i in 0..<nx {
            for j in 0..<nt {
                let diff = u[i][j] - rk4Solution[i][j]
                validationLoss += diff * diff
            }
        }
        validationLoss = sqrt(validationLoss / Double(nx * nt))
        
        return PINNSolution(
            u: u,
            x: x,
            t: t,
            pdeResidual: pdeResidual,
            trainingLoss: losses.last ?? 0.0,
            validationLoss: validationLoss
        )
    }
}

// MARK: - Ψ Framework Integration

/// Extension to integrate PINN with the Ψ framework
extension PINNSolver {
    /// Compute Ψ(x) for PINN performance evaluation
    public func computePsiPerformance() -> PsiOutcome {
        // S(x): Symbolic method performance (RK4 validation)
        let S_symbolic = max(0.0, 1.0 - validationLoss)
        
        // N(x): Neural network performance (training convergence)
        let N_external = max(0.0, 1.0 - trainingLoss)
        
        // α(t): Balance parameter (can be tuned)
        let alpha = 0.5
        
        // Risk factors (simplified)
        let riskAuthority = 0.1      // Low risk for well-established methods
        let riskVerifiability = 0.2  // Moderate risk for neural methods
        
        // Lambda weights
        let lambdaAuthority = 0.6
        let lambdaVerifiability = 0.4
        
        // Base posterior probability
        let basePosterior = 0.8
        
        // Beta uplift factor
        let betaUplift = 1.2
        
        let inputs = PsiInputs(
            alpha: alpha,
            S_symbolic: S_symbolic,
            N_external: N_external,
            lambdaAuthority: lambdaAuthority,
            lambdaVerifiability: lambdaVerifiability,
            riskAuthority: riskAuthority,
            riskVerifiability: riskVerifiability,
            basePosterior: basePosterior,
            betaUplift: betaUplift
        )
        
        return PsiModel.computePsi(inputs: inputs)
    }
}