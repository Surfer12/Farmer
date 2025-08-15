import Foundation

// MARK: - Core PINN Components

/// Dense neural network layer with Xavier initialization
public class DenseLayer {
    public var weights: [[Double]]
    public var biases: [Double]
    
    public init(inputSize: Int, outputSize: Int) {
        let bound = sqrt(6.0 / Double(inputSize + outputSize))  // Xavier initialization
        weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Double.random(in: -bound...bound) }
        }
        biases = (0..<outputSize).map { _ in Double.random(in: -bound...bound) }
    }
    
    public func forward(_ input: [Double]) -> [Double] {
        var output = [Double](repeating: 0.0, count: weights.count)
        
        for i in 0..<weights.count {
            for j in 0..<input.count {
                output[i] += weights[i][j] * input[j]
            }
            output[i] += biases[i]
            output[i] = tanh(output[i])  // Activation function
        }
        
        return output
    }
}

/// Physics-Informed Neural Network with hybrid output optimization
public class PINN {
    public var layers: [DenseLayer]
    
    public init() {
        layers = [
            DenseLayer(inputSize: 2, outputSize: 20),
            DenseLayer(inputSize: 20, outputSize: 20),
            DenseLayer(inputSize: 20, outputSize: 1)
        ]
    }
    
    public func forward(x: Double, t: Double) -> Double {
        var input = [x, t]
        for layer in layers {
            input = layer.forward(input)
        }
        return input[0]
    }
}

// MARK: - Hybrid Output Optimization

/// Hybrid output combining state inference S(x) and ML gradient descent N(x)
public struct HybridOutput {
    public let stateInference: Double      // S(x)
    public let mlGradient: Double         // N(x)
    public let alpha: Double              // α(t) for real-time validation
    public let hybridValue: Double        // O_hybrid
    
    public init(stateInference: Double, mlGradient: Double, alpha: Double = 0.5) {
        self.stateInference = stateInference
        self.mlGradient = mlGradient
        self.alpha = alpha
        self.hybridValue = alpha * stateInference + (1 - alpha) * mlGradient
    }
}

// MARK: - Regularization Framework

/// Cognitive regularization for PDE residual accuracy
public struct CognitiveRegularization {
    public let pdeResidual: Double
    public let weight: Double
    
    public init(pdeResidual: Double, weight: Double = 0.6) {
        self.pdeResidual = pdeResidual
        self.weight = weight
    }
    
    public var value: Double {
        return weight * pdeResidual
    }
}

/// Efficiency regularization for training loop optimization
public struct EfficiencyRegularization {
    public let trainingEfficiency: Double
    public let weight: Double
    
    public init(trainingEfficiency: Double, weight: Double = 0.4) {
        self.trainingEfficiency = trainingEfficiency
        self.weight = weight
    }
    
    public var value: Double {
        return weight * trainingEfficiency
    }
}

// MARK: - Probability and Validation

/// Probability model P(H|E,β) with β for model responsiveness
public struct ProbabilityModel {
    public let hypothesis: Double          // P(H|E)
    public let beta: Double               // β for model responsiveness
    public let adjustedProbability: Double
    
    public init(hypothesis: Double, beta: Double = 1.2) {
        self.hypothesis = hypothesis
        self.beta = beta
        self.adjustedProbability = min(1.0, hypothesis * beta)
    }
}

// MARK: - Integrated Performance Metric

/// Comprehensive performance metric Ψ(x) combining all components
public struct PerformanceMetric {
    public let hybridOutput: HybridOutput
    public let cognitiveReg: CognitiveRegularization
    public let efficiencyReg: EfficiencyRegularization
    public let probability: ProbabilityModel
    
    public init(hybridOutput: HybridOutput, 
                cognitiveReg: CognitiveRegularization, 
                efficiencyReg: EfficiencyRegularization, 
                probability: ProbabilityModel) {
        self.hybridOutput = hybridOutput
        self.cognitiveReg = cognitiveReg
        self.efficiencyReg = efficiencyReg
        self.probability = probability
    }
    
    /// Calculate the integrated performance metric Ψ(x)
    public var value: Double {
        let regularizationFactor = exp(-(cognitiveReg.value + efficiencyReg.value))
        return hybridOutput.hybridValue * regularizationFactor * probability.adjustedProbability
    }
    
    /// Interpret the performance metric
    public var interpretation: String {
        switch value {
        case 0.8...:
            return "Excellent model performance with high accuracy and efficiency"
        case 0.6..<0.8:
            return "Good model performance with solid accuracy"
        case 0.4..<0.6:
            return "Moderate model performance, consider optimization"
        case 0.2..<0.4:
            return "Poor model performance, significant improvements needed"
        default:
            return "Very poor model performance, review implementation"
        }
    }
}

// MARK: - Numerical Utilities

/// Finite difference approximation with configurable step size
public func finiteDiff(f: (Double) -> Double, at: Double, dx: Double = 1e-6) -> Double {
    return (f(at + dx) - f(at - dx)) / (2 * dx)
}

/// PDE residual loss calculation (batched for efficiency)
public func pdeLoss(model: PINN, x: [Double], t: [Double]) -> Double {
    let batchSize = 20
    var totalLoss = 0.0
    
    for batchStart in stride(from: 0, to: x.count, by: batchSize) {
        let batchEnd = min(batchStart + batchSize, x.count)
        let batchX = Array(x[batchStart..<batchEnd])
        let batchT = Array(t[batchStart..<batchEnd])
        
        for i in 0..<batchX.count {
            let xVal = batchX[i]
            let tVal = batchT[i]
            
            // Calculate PDE residual: ∂u/∂t + u*∂u/∂x = 0
            let u = model.forward(x: xVal, t: tVal)
            
            // Finite difference approximations
            let dt = 1e-6
            let dx = 1e-6
            
            let ut = (model.forward(x: xVal, t: tVal + dt) - model.forward(x: xVal, t: tVal - dt)) / (2 * dt)
            let ux = (model.forward(x: xVal + dx, t: tVal) - model.forward(x: xVal - dx, t: tVal)) / (2 * dx)
            
            let residual = ut + u * ux
            totalLoss += residual * residual
        }
    }
    
    return totalLoss / Double(x.count)
}

/// Initial condition loss calculation
public func icLoss(model: PINN, x: [Double]) -> Double {
    return x.reduce(0.0) { loss, val in
        let u = model.forward(x: val, t: 0.0)
        let trueU = -sin(.pi * val)
        return loss + pow(u - trueU, 2)
    } / Double(x.count)
}

// MARK: - Training and Optimization

/// Training step with gradient approximation
public func trainStep(model: PINN, x: [Double], t: [Double], learningRate: Double = 0.005) {
    let perturbation = 1e-5
    
    let computeLoss = {
        pdeLoss(model: model, x: x, t: t) + icLoss(model: model, x: x)
    }
    
    for layer in model.layers {
        // Update weights
        for i in 0..<layer.weights.count {
            for j in 0..<layer.weights[i].count {
                let originalWeight = layer.weights[i][j]
                
                // Forward perturbation
                layer.weights[i][j] = originalWeight + perturbation
                let lossPlus = computeLoss()
                
                // Backward perturbation
                layer.weights[i][j] = originalWeight - perturbation
                let lossMinus = computeLoss()
                
                // Gradient approximation
                let gradient = (lossPlus - lossMinus) / (2 * perturbation)
                
                // Update weight
                layer.weights[i][j] = originalWeight - learningRate * gradient
            }
        }
        
        // Update biases
        for i in 0..<layer.biases.count {
            let originalBias = layer.biases[i]
            
            // Forward perturbation
            layer.biases[i] = originalBias + perturbation
            let lossPlus = computeLoss()
            
            // Backward perturbation
            layer.biases[i] = originalBias - perturbation
            let lossMinus = computeLoss()
            
            // Gradient approximation
            let gradient = (lossPlus - lossMinus) / (2 * perturbation)
            
            // Update bias
            layer.biases[i] = originalBias - learningRate * gradient
        }
    }
}

/// Complete training loop with monitoring
public func train(model: PINN, epochs: Int = 1000, x: [Double], t: [Double], printEvery: Int = 50) {
    for epoch in 0..<epochs {
        trainStep(model: model, x: x, t: t)
        
        if epoch % printEvery == 0 {
            let pdeLossValue = pdeLoss(model: model, x: x, t: t)
            let icLossValue = icLoss(model: model, x: x)
            let totalLoss = pdeLossValue + icLossValue
            
            print("Epoch \(epoch): PDE Loss: \(pdeLossValue), IC Loss: \(icLossValue), Total: \(totalLoss)")
        }
    }
}

// MARK: - Example Usage and Testing

/// Example implementation of the numerical example from the requirements
public func runNumericalExample() -> PerformanceMetric {
    // Step 1: Outputs
    let s_x = 0.72  // S(x) - state inference
    let n_x = 0.85  // N(x) - ML gradient descent
    
    // Step 2: Hybrid
    let alpha = 0.5
    let o_hybrid = alpha * s_x + (1 - alpha) * n_x  // Should be 0.785
    
    // Step 3: Penalties
    let r_cognitive = 0.15
    let r_efficiency = 0.10
    let lambda1 = 0.6
    let lambda2 = 0.4
    let p_total = lambda1 * r_cognitive + lambda2 * r_efficiency  // Should be 0.13
    let exp_factor = exp(-p_total)  // Should be ≈ 0.878
    
    // Step 4: Probability
    let p = 0.80
    let beta = 1.2
    let p_adj = min(1.0, p * beta)  // Should be ≈ 0.96
    
    // Step 5: Ψ(x)
    let psi = o_hybrid * exp_factor * p_adj  // Should be ≈ 0.662
    
    // Create the performance metric
    let hybridOutput = HybridOutput(stateInference: s_x, mlGradient: n_x, alpha: alpha)
    let cognitiveReg = CognitiveRegularization(pdeResidual: r_cognitive, weight: lambda1)
    let efficiencyReg = EfficiencyRegularization(trainingEfficiency: r_efficiency, weight: lambda2)
    let probability = ProbabilityModel(hypothesis: p, beta: beta)
    
    let metric = PerformanceMetric(
        hybridOutput: hybridOutput,
        cognitiveReg: cognitiveReg,
        efficiencyReg: efficiencyReg,
        probability: probability
    )
    
    print("Numerical Example Results:")
    print("S(x) = \(s_x)")
    print("N(x) = \(n_x)")
    print("α = \(alpha)")
    print("O_hybrid = \(o_hybrid)")
    print("R_cognitive = \(r_cognitive), λ1 = \(lambda1)")
    print("R_efficiency = \(r_efficiency), λ2 = \(lambda2)")
    print("P_total = \(p_total)")
    print("exp(-P_total) ≈ \(exp_factor)")
    print("P = \(p), β = \(beta)")
    print("P_adj ≈ \(p_adj)")
    print("Ψ(x) ≈ \(psi)")
    print("Interpretation: \(metric.interpretation)")
    
    return metric
}