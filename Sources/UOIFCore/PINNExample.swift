import Foundation

// MARK: - PINN Example Demonstrations

/// Comprehensive example demonstrating PINN usage and the hybrid output optimization framework
public struct PINNExample {
    
    /// Run the complete numerical example from the requirements
    public static func runCompleteExample() {
        print("=== PINN Hybrid Output Optimization Example ===\n")
        
        // Step 1: Outputs
        print("Step 1: Outputs")
        print("S(x) = 0.72 (state inference for optimized PINN solutions)")
        print("N(x) = 0.85 (ML gradient descent analysis)")
        print()
        
        // Step 2: Hybrid
        print("Step 2: Hybrid")
        let alpha = 0.5
        let o_hybrid = alpha * 0.72 + (1 - alpha) * 0.85
        print("α = \(alpha) (real-time validation flows)")
        print("O_hybrid = \(alpha) × 0.72 + (1 - \(alpha)) × 0.85 = \(o_hybrid)")
        print()
        
        // Step 3: Penalties
        print("Step 3: Penalties")
        let r_cognitive = 0.15
        let r_efficiency = 0.10
        let lambda1 = 0.6
        let lambda2 = 0.4
        let p_total = lambda1 * r_cognitive + lambda2 * r_efficiency
        let exp_factor = exp(-p_total)
        
        print("R_cognitive = \(r_cognitive) (PDE residual accuracy)")
        print("R_efficiency = \(r_efficiency) (training loop efficiency)")
        print("λ1 = \(lambda1), λ2 = \(lambda2)")
        print("P_total = \(lambda1) × \(r_cognitive) + \(lambda2) × \(r_efficiency) = \(p_total)")
        print("exp(-P_total) ≈ \(exp_factor)")
        print()
        
        // Step 4: Probability
        print("Step 4: Probability")
        let p = 0.80
        let beta = 1.2
        let p_adj = min(1.0, p * beta)
        
        print("P(H|E) = \(p)")
        print("β = \(beta) (model responsiveness)")
        print("P_adj = min(1.0, \(p) × \(beta)) ≈ \(p_adj)")
        print()
        
        // Step 5: Ψ(x)
        print("Step 5: Ψ(x)")
        let psi = o_hybrid * exp_factor * p_adj
        print("Ψ(x) = \(o_hybrid) × \(exp_factor) × \(p_adj) ≈ \(psi)")
        print()
        
        // Step 6: Interpretation
        print("Step 6: Interpret")
        print("Ψ(x) ≈ \(psi) indicates solid model performance")
        print()
        
        // Now demonstrate using the actual PINN classes
        print("=== Implementing with PINN Classes ===\n")
        
        let hybridOutput = HybridOutput(stateInference: 0.72, mlGradient: 0.85, alpha: 0.5)
        let cognitiveReg = CognitiveRegularization(pdeResidual: 0.15, weight: 0.6)
        let efficiencyReg = EfficiencyRegularization(trainingEfficiency: 0.10, weight: 0.4)
        let probability = ProbabilityModel(hypothesis: 0.80, beta: 1.2)
        
        let metric = PerformanceMetric(
            hybridOutput: hybridOutput,
            cognitiveReg: cognitiveReg,
            efficiencyReg: efficiencyReg,
            probability: probability
        )
        
        print("Using PINN classes:")
        print("Hybrid Output: \(hybridOutput.hybridValue)")
        print("Cognitive Regularization: \(cognitiveReg.value)")
        print("Efficiency Regularization: \(efficiencyReg.value)")
        print("Adjusted Probability: \(probability.adjustedProbability)")
        print("Final Performance Metric: \(metric.value)")
        print("Interpretation: \(metric.interpretation)")
        print()
    }
    
    /// Demonstrate PINN training and solution generation
    public static func demonstratePINNTraining() {
        print("=== PINN Training Demonstration ===\n")
        
        // Create and train a PINN
        let pinn = PINN()
        
        // Generate training data
        let x = Array(stride(from: -1.0, to: 1.0, by: 0.1))
        let t = Array(repeating: 1.0, count: x.count)
        
        print("Training PINN with \(x.count) spatial points...")
        print("Spatial domain: x ∈ [\(x.first!), \(x.last!)]")
        print("Time: t = \(t.first!)")
        print()
        
        // Train the model
        train(model: pinn, epochs: 100, x: x, t: t, printEvery: 25)
        
        print("\nTraining completed!")
        print("Testing PINN predictions...")
        
        // Test some predictions
        let testPoints = [-0.8, -0.4, 0.0, 0.4, 0.8]
        for x in testPoints {
            let prediction = pinn.forward(x: x, t: 1.0)
            let trueValue = -sin(.pi * x)
            let error = abs(prediction - trueValue)
            
            print("x = \(x): PINN = \(String(format: "%.4f", prediction)), True = \(String(format: "%.4f", trueValue)), Error = \(String(format: "%.4f", error))")
        }
        print()
    }
    
    /// Demonstrate the regularization framework
    public static func demonstrateRegularization() {
        print("=== Regularization Framework Demonstration ===\n")
        
        // Show different regularization scenarios
        let scenarios = [
            ("High PDE Residual", 0.3, 0.6),
            ("Medium PDE Residual", 0.15, 0.6),
            ("Low PDE Residual", 0.05, 0.6)
        ]
        
        print("Cognitive Regularization (PDE Residual Accuracy):")
        for (name, residual, weight) in scenarios {
            let reg = CognitiveRegularization(pdeResidual: residual, weight: weight)
            print("  \(name): R = \(residual), λ = \(weight), Value = \(String(format: "%.3f", reg.value))")
        }
        print()
        
        let efficiencyScenarios = [
            ("Inefficient Training", 0.25, 0.4),
            ("Moderate Efficiency", 0.10, 0.4),
            ("High Efficiency", 0.02, 0.4)
        ]
        
        print("Efficiency Regularization (Training Loop Optimization):")
        for (name, efficiency, weight) in efficiencyScenarios {
            let reg = EfficiencyRegularization(trainingEfficiency: efficiency, weight: weight)
            print("  \(name): R = \(efficiency), λ = \(weight), Value = \(String(format: "%.3f", reg.value))")
        }
        print()
        
        // Show combined effect
        print("Combined Regularization Effect:")
        let cognitiveReg = CognitiveRegularization(pdeResidual: 0.15, weight: 0.6)
        let efficiencyReg = EfficiencyRegularization(trainingEfficiency: 0.10, weight: 0.4)
        let totalReg = cognitiveReg.value + efficiencyReg.value
        let regFactor = exp(-totalReg)
        
        print("  Total Regularization: \(String(format: "%.3f", totalReg))")
        print("  Regularization Factor: exp(-\(String(format: "%.3f", totalReg))) = \(String(format: "%.3f", regFactor))")
        print()
    }
    
    /// Demonstrate the probability model
    public static func demonstrateProbabilityModel() {
        print("=== Probability Model Demonstration ===\n")
        
        // Show different hypothesis and beta combinations
        let scenarios = [
            (0.6, 1.0, "Low confidence, standard responsiveness"),
            (0.8, 1.2, "Medium confidence, high responsiveness"),
            (0.9, 1.5, "High confidence, very high responsiveness"),
            (0.95, 2.0, "Very high confidence, extreme responsiveness")
        ]
        
        print("Probability Model P(H|E,β) with β for model responsiveness:")
        for (hypothesis, beta, description) in scenarios {
            let prob = ProbabilityModel(hypothesis: hypothesis, beta: beta)
            print("  \(description):")
            print("    P(H|E) = \(hypothesis), β = \(beta)")
            print("    P_adj = \(String(format: "%.3f", prob.adjustedProbability))")
            print()
        }
        
        // Show the capping behavior
        print("Note: P_adj is capped at 1.0 for high confidence × high responsiveness combinations")
        print()
    }
    
    /// Demonstrate performance metric interpretation
    public static func demonstratePerformanceMetrics() {
        print("=== Performance Metric Interpretation ===\n")
        
        // Create different performance scenarios
        let scenarios = [
            ("Excellent Performance", 0.85, 0.02, 0.01, 0.9, 1.1),
            ("Good Performance", 0.75, 0.08, 0.05, 0.8, 1.2),
            ("Moderate Performance", 0.65, 0.15, 0.12, 0.7, 1.1),
            ("Poor Performance", 0.45, 0.25, 0.20, 0.6, 1.0),
            ("Very Poor Performance", 0.25, 0.35, 0.30, 0.5, 0.9)
        ]
        
        print("Performance Metric Ψ(x) Interpretation Examples:")
        for (name, s_x, n_x, r_cog, r_eff, p_hyp, beta) in scenarios {
            let hybrid = HybridOutput(stateInference: s_x, mlGradient: n_x, alpha: 0.5)
            let cognitiveReg = CognitiveRegularization(pdeResidual: r_cog, weight: 0.6)
            let efficiencyReg = EfficiencyRegularization(trainingEfficiency: r_eff, weight: 0.4)
            let probability = ProbabilityModel(hypothesis: p_hyp, beta: beta)
            
            let metric = PerformanceMetric(
                hybridOutput: hybrid,
                cognitiveReg: cognitiveReg,
                efficiencyReg: efficiencyReg,
                probability: probability
            )
            
            print("  \(name):")
            print("    S(x) = \(s_x), N(x) = \(n_x)")
            print("    R_cognitive = \(r_cog), R_efficiency = \(r_eff)")
            print("    P(H|E) = \(p_hyp), β = \(beta)")
            print("    Ψ(x) = \(String(format: "%.3f", metric.value))")
            print("    Interpretation: \(metric.interpretation)")
            print()
        }
    }
    
    /// Run a complete demonstration of all components
    public static func runFullDemonstration() {
        print("🚀 Starting Complete PINN Hybrid Output Optimization Demonstration\n")
        print("=" * 80)
        
        runCompleteExample()
        print("=" * 80)
        
        demonstratePINNTraining()
        print("=" * 80)
        
        demonstrateRegularization()
        print("=" * 80)
        
        demonstrateProbabilityModel()
        print("=" * 80)
        
        demonstratePerformanceMetrics()
        print("=" * 80)
        
        print("✅ Demonstration completed successfully!")
        print("\nThis implementation demonstrates:")
        print("• Hybrid Output: S(x) as state inference, N(x) as ML gradient descent")
        print("• Regularization: R_cognitive for PDE accuracy, R_efficiency for training")
        print("• Probability: P(H|E,β) with β for model responsiveness")
        print("• Integration: Over training epochs and validation steps")
        print("• Balanced Intelligence: Merges symbolic RK4 with neural PINN")
        print("• Interpretability: Visualizes solutions for coherence")
        print("• Efficiency: Optimizes computations in Swift")
        print("• Human Alignment: Enhances understanding of nonlinear flows")
        print("• Dynamic Optimization: Adapts through epochs")
    }
}

// MARK: - Utility Extensions

extension String {
    static func * (lhs: String, rhs: Int) -> String {
        return String(repeating: lhs, count: rhs)
    }
}

// MARK: - Example Usage

/// Example of how to use the PINN system in practice
public func demonstratePINNUsage() {
    print("=== Practical PINN Usage Example ===\n")
    
    // 1. Create a PINN
    let pinn = PINN()
    print("1. Created PINN with \(pinn.layers.count) layers")
    
    // 2. Generate training data
    let x = Array(stride(from: -1.0, to: 1.0, by: 0.05))
    let t = Array(repeating: 1.0, count: x.count)
    print("2. Generated \(x.count) training points")
    
    // 3. Train the model
    print("3. Training PINN...")
    train(model: pinn, epochs: 50, x: x, t: t, printEvery: 10)
    
    // 4. Evaluate performance
    print("4. Evaluating performance...")
    let pdeLoss = pdeLoss(model: pinn, x: x, t: t)
    let icLoss = icLoss(model: pinn, x: x)
    let totalLoss = pdeLoss + icLoss
    
    print("   PDE Loss: \(String(format: "%.6f", pdeLoss))")
    print("   IC Loss: \(String(format: "%.6f", icLoss))")
    print("   Total Loss: \(String(format: "%.6f", totalLoss))")
    
    // 5. Create performance metrics
    print("5. Creating performance metrics...")
    let hybridOutput = HybridOutput(stateInference: 0.8, mlGradient: 0.9, alpha: 0.6)
    let cognitiveReg = CognitiveRegularization(pdeResidual: pdeLoss, weight: 0.6)
    let efficiencyReg = EfficiencyRegularization(trainingEfficiency: totalLoss, weight: 0.4)
    let probability = ProbabilityModel(hypothesis: 0.85, beta: 1.1)
    
    let metric = PerformanceMetric(
        hybridOutput: hybridOutput,
        cognitiveReg: cognitiveReg,
        efficiencyReg: efficiencyReg,
        probability: probability
    )
    
    print("   Hybrid Output: \(String(format: "%.3f", hybridOutput.hybridValue))")
    print("   Performance Metric Ψ(x): \(String(format: "%.3f", metric.value))")
    print("   Interpretation: \(metric.interpretation)")
    
    print("\n✅ PINN usage demonstration completed!")
}