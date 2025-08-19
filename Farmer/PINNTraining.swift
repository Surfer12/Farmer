import Foundation

// MARK: - Training Implementation
class PINNTrainer {
    private let model: PINN
    private var trainingHistory: [(epoch: Int, loss: Double, psi: Double)] = []
    
    init(model: PINN) {
        self.model = model
    }
    
    // Compute gradients using finite differences
    func computeGradients(x: [Double], t: [Double], h: Double = 1e-5) -> PINNGradients {
        var gradients = PINNGradients(hiddenSize: 20, inputSize: 2, outputSize: 1)
        let batchSize = min(x.count, t.count)
        
        // Compute current loss
        let currentLoss = computeTotalLoss(x: x, t: t)
        
        // Approximate gradients for weights1
        for i in 0..<20 {
            for j in 0..<2 {
                let originalWeight = getWeight1(i: i, j: j)
                
                // Forward perturbation
                setWeight1(i: i, j: j, value: originalWeight + h)
                let lossForward = computeTotalLoss(x: x, t: t)
                
                // Backward perturbation
                setWeight1(i: i, j: j, value: originalWeight - h)
                let lossBackward = computeTotalLoss(x: x, t: t)
                
                // Restore original weight
                setWeight1(i: i, j: j, value: originalWeight)
                
                // Compute gradient
                gradients.dW1[i][j] = (lossForward - lossBackward) / (2 * h)
            }
        }
        
        // Approximate gradients for biases1
        for i in 0..<20 {
            let originalBias = getBias1(i: i)
            
            setBias1(i: i, value: originalBias + h)
            let lossForward = computeTotalLoss(x: x, t: t)
            
            setBias1(i: i, value: originalBias - h)
            let lossBackward = computeTotalLoss(x: x, t: t)
            
            setBias1(i: i, value: originalBias)
            
            gradients.dB1[i] = (lossForward - lossBackward) / (2 * h)
        }
        
        // Approximate gradients for weights2
        for i in 0..<1 {
            for j in 0..<20 {
                let originalWeight = getWeight2(i: i, j: j)
                
                setWeight2(i: i, j: j, value: originalWeight + h)
                let lossForward = computeTotalLoss(x: x, t: t)
                
                setWeight2(i: i, j: j, value: originalWeight - h)
                let lossBackward = computeTotalLoss(x: x, t: t)
                
                setWeight2(i: i, j: j, value: originalWeight)
                
                gradients.dW2[i][j] = (lossForward - lossBackward) / (2 * h)
            }
        }
        
        // Approximate gradients for biases2
        for i in 0..<1 {
            let originalBias = getBias2(i: i)
            
            setBias2(i: i, value: originalBias + h)
            let lossForward = computeTotalLoss(x: x, t: t)
            
            setBias2(i: i, value: originalBias - h)
            let lossBackward = computeTotalLoss(x: x, t: t)
            
            setBias2(i: i, value: originalBias)
            
            gradients.dB2[i] = (lossForward - lossBackward) / (2 * h)
        }
        
        return gradients
    }
    
    // Compute total loss (PDE + IC + BC)
    func computeTotalLoss(x: [Double], t: [Double]) -> Double {
        let batchSize = min(x.count, t.count)
        var pdeLoss = 0.0
        var icLoss = 0.0
        var bcLoss = 0.0
        
        // PDE loss
        for i in 0..<batchSize {
            let residual = model.pdeResidual(x[i], t[i])
            pdeLoss += residual * residual
        }
        pdeLoss /= Double(batchSize)
        
        // Initial condition loss
        for i in 0..<x.count {
            let predicted = model.forward(x[i], 0.0)
            let actual = model.initialCondition(x[i])
            let diff = predicted - actual
            icLoss += diff * diff
        }
        icLoss /= Double(x.count)
        
        // Boundary condition loss
        for i in 0..<t.count {
            let bc1 = model.forward(-1.0, t[i]) - model.boundaryCondition(t[i])
            let bc2 = model.forward(1.0, t[i]) - model.boundaryCondition(t[i])
            bcLoss += bc1 * bc1 + bc2 * bc2
        }
        bcLoss /= Double(t.count)
        
        return pdeLoss + icLoss + bcLoss
    }
    
    // Training loop with hybrid framework integration
    func train(epochs: Int, x: [Double], t: [Double], printEvery: Int = 50) {
        var previousLoss = Double.infinity
        
        print("Starting PINN Training with Hybrid Framework...")
        print("Epochs: \(epochs), Data points: \(x.count) x \(t.count)")
        print("="*60)
        
        for epoch in 1...epochs {
            // Compute gradients and update parameters
            let gradients = computeGradients(x: x, t: t)
            model.updateParameters(gradients: gradients)
            
            // Compute current loss
            let currentLoss = computeTotalLoss(x: x, t: t)
            
            // Hybrid framework analysis
            let midX = x[x.count / 2]
            let midT = t[t.count / 2]
            
            let sX = HybridFramework.stateInference(midX, midT, model: model)
            let nX = HybridFramework.gradientDescentAnalysis(loss: currentLoss, previousLoss: previousLoss)
            let alphaT = HybridFramework.realTimeValidation(Double(epoch) / Double(epochs))
            
            let rCognitive = min(1.0, currentLoss) // Normalize cognitive load
            let rEfficiency = min(1.0, abs(currentLoss - previousLoss) / max(previousLoss, 1e-10))
            
            let (hybrid, psi) = HybridFramework.computeHybridOutput(
                sX: sX, nX: nX, alphaT: alphaT,
                rCognitive: rCognitive, rEfficiency: rEfficiency
            )
            
            // Store training history
            trainingHistory.append((epoch: epoch, loss: currentLoss, psi: psi))
            
            // Print progress
            if epoch % printEvery == 0 {
                print("Epoch \(epoch):")
                print("  Loss: \(String(format: "%.6f", currentLoss))")
                print("  S(x): \(String(format: "%.3f", sX)) | N(x): \(String(format: "%.3f", nX)) | α(t): \(String(format: "%.3f", alphaT))")
                print("  R_cog: \(String(format: "%.3f", rCognitive)) | R_eff: \(String(format: "%.3f", rEfficiency))")
                print("  Hybrid: \(String(format: "%.3f", hybrid)) | Ψ(x): \(String(format: "%.3f", psi))")
                print("  " + "-" * 50)
            }
            
            previousLoss = currentLoss
        }
        
        print("\nTraining completed!")
        print("Final Loss: \(String(format: "%.6f", previousLoss))")
        print("Final Ψ(x): \(String(format: "%.3f", trainingHistory.last?.psi ?? 0.0))")
    }
    
    // Get training history
    func getTrainingHistory() -> [(epoch: Int, loss: Double, psi: Double)] {
        return trainingHistory
    }
}

// MARK: - Parameter Access Methods (for gradient computation)
extension PINN {
    func getWeight1(i: Int, j: Int) -> Double {
        return weights1[i][j]
    }
    
    func setWeight1(i: Int, j: Int, value: Double) {
        weights1[i][j] = value
    }
    
    func getBias1(i: Int) -> Double {
        return biases1[i]
    }
    
    func setBias1(i: Int, value: Double) {
        biases1[i] = value
    }
    
    func getWeight2(i: Int, j: Int) -> Double {
        return weights2[i][j]
    }
    
    func setWeight2(i: Int, j: Int, value: Double) {
        weights2[i][j] = value
    }
    
    func getBias2(i: Int) -> Double {
        return biases2[i]
    }
    
    func setBias2(i: Int, value: Double) {
        biases2[i] = value
    }
}

// MARK: - String Repetition Extension
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}