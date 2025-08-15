#!/usr/bin/env swift

import Foundation

// MARK: - Demonstration Script for Hybrid PINN-RK4 System

func main() {
    print("🚀 Hybrid PINN-RK4 System Demonstration")
    print("========================================")
    
    // Initialize the system
    let system = HybridPINNRK4System()
    
    // Set parameters to match the numerical example
    system.alpha = 0.5
    system.lambda1 = 0.6
    system.lambda2 = 0.4
    system.beta = 1.2
    
    print("\n📊 System Configuration:")
    print("α(t) = \(system.alpha) (real-time validation flows)")
    print("λ₁ = \(system.lambda1) (cognitive regularization weight)")
    print("λ₂ = \(system.lambda2) (efficiency regularization weight)")
    print("β = \(system.beta) (model responsiveness)")
    
    // Demonstrate single training step (numerical example)
    demonstrateNumericalExample(system: system)
    
    // Generate training data
    print("\n🔄 Generating Training Data...")
    let trainingData = generateTrainingData()
    print("Generated \(trainingData.count) training points")
    
    // Run training simulation
    runTrainingSimulation(system: system, trainingData: trainingData)
    
    // Demonstrate real-time validation
    demonstrateRealTimeValidation(system: system)
    
    // Show solution comparison at different time points
    demonstrateSolutionEvolution(system: system)
    
    print("\n✅ Demonstration Complete!")
    print("To use in Xcode: Import this file and run HybridPINNVisualizationView()")
}

func demonstrateNumericalExample(system: HybridPINNRK4System) {
    print("\n🔢 Numerical Example: Single Training Step")
    print("==========================================")
    
    let x = 0.0
    let t = 0.1
    let epoch = 1
    
    let result = system.computeHybridOutput(x: x, t: t, epoch: epoch)
    
    print("Input: x = \(x), t = \(t), epoch = \(epoch)")
    print()
    
    print("Step 1 - Individual Outputs:")
    print("  S(x) = \(String(format: "%.3f", result.S_x)) (state inference)")
    print("  N(x) = \(String(format: "%.3f", result.N_x)) (ML gradient analysis)")
    
    print("\nStep 2 - Hybrid Combination:")
    print("  α = \(system.alpha)")
    print("  O_hybrid = α·S(x) + (1-α)·N(x) = \(String(format: "%.3f", result.O_hybrid))")
    
    print("\nStep 3 - Regularization Penalties:")
    print("  R_cognitive = \(String(format: "%.3f", result.R_cognitive)) (PDE residual accuracy)")
    print("  R_efficiency = \(String(format: "%.3f", result.R_efficiency)) (training loop efficiency)")
    print("  P_total = λ₁·R_cognitive + λ₂·R_efficiency = \(String(format: "%.3f", result.P_total))")
    print("  exp(-P_total) ≈ \(String(format: "%.3f", result.expTerm))")
    
    print("\nStep 4 - Probability Computation:")
    print("  P_base = \(String(format: "%.3f", result.P_base))")
    print("  β = \(system.beta)")
    print("  P_adjusted = P(H|E,β) ≈ \(String(format: "%.3f", result.P_adjusted))")
    
    print("\nStep 5 - Final Hybrid Output:")
    print("  Ψ(x) = O_hybrid × exp(-P_total) × P_adjusted")
    print("  Ψ(x) = \(String(format: "%.3f", result.O_hybrid)) × \(String(format: "%.3f", result.expTerm)) × \(String(format: "%.3f", result.P_adjusted))")
    print("  Ψ(x) ≈ \(String(format: "%.3f", result.psi))")
    
    print("\nStep 6 - Interpretation:")
    print("  \(result.interpretation)")
    
    // Compare with your expected values
    print("\n📈 Comparison with Expected Values:")
    print("Expected S(x) ≈ 0.72, Actual: \(String(format: "%.3f", result.S_x))")
    print("Expected N(x) ≈ 0.85, Actual: \(String(format: "%.3f", result.N_x))")
    print("Expected O_hybrid ≈ 0.785, Actual: \(String(format: "%.3f", result.O_hybrid))")
    print("Expected Ψ(x) ≈ 0.66, Actual: \(String(format: "%.3f", result.psi))")
}

func generateTrainingData() -> [(x: Double, t: Double)] {
    var data: [(x: Double, t: Double)] = []
    
    // Generate spatial-temporal grid
    for x in stride(from: -1.0, through: 1.0, by: 0.2) {
        for t in stride(from: 0.0, through: 1.0, by: 0.1) {
            data.append((x: x, t: t))
        }
    }
    
    return data
}

func runTrainingSimulation(system: HybridPINNRK4System, trainingData: [(x: Double, t: Double)]) {
    print("\n🎯 Training Simulation")
    print("=====================")
    
    let epochs = 10
    var previousLoss: Double = Double.infinity
    
    for epoch in 0..<epochs {
        let metrics = system.trainEpoch(trainingData: trainingData, learningRate: 0.01)
        
        // Adaptive alpha adjustment
        system.adaptiveUpdateAlpha(basedOnPerformance: metrics.averagePsi)
        
        let improvement = previousLoss - metrics.loss
        let improvementPercent = improvement / previousLoss * 100
        
        print("Epoch \(String(format: "%2d", epoch + 1)): " +
              "Loss = \(String(format: "%.4f", metrics.loss)) " +
              "Ψ̄ = \(String(format: "%.3f", metrics.averagePsi)) " +
              "α = \(String(format: "%.3f", system.getCurrentAlpha())) " +
              "Δ = \(improvement > 0 ? "+" : "")\(String(format: "%.1f", improvementPercent))%")
        
        previousLoss = metrics.loss
    }
    
    let finalHistory = system.getTrainingHistory()
    if let finalMetrics = finalHistory.last {
        print("\n📊 Final Training Metrics:")
        print("  Final Loss: \(String(format: "%.4f", finalMetrics.loss))")
        print("  Average Ψ(x): \(String(format: "%.3f", finalMetrics.averagePsi))")
        print("  Cognitive Regularization: \(String(format: "%.3f", finalMetrics.averageCognitiveReg))")
        print("  Efficiency Regularization: \(String(format: "%.3f", finalMetrics.averageEfficiencyReg))")
    }
}

func demonstrateRealTimeValidation(system: HybridPINNRK4System) {
    print("\n⚡ Real-Time Validation Flow")
    print("===========================")
    
    let testPoints = [
        (x: -0.5, t: 0.2),
        (x: 0.0, t: 0.5),
        (x: 0.5, t: 0.8)
    ]
    
    for (i, point) in testPoints.enumerated() {
        let result = system.computeHybridOutput(x: point.x, t: point.t, epoch: 10)
        let analytical = -sin(.pi * point.x) * exp(-.pi * .pi * point.t)
        let error = abs(result.psi - analytical)
        
        print("Test Point \(i + 1): x=\(String(format: "%4.1f", point.x)), t=\(String(format: "%.1f", point.t))")
        print("  Ψ(x) = \(String(format: "%.3f", result.psi))")
        print("  Analytical = \(String(format: "%.3f", analytical))")
        print("  Error = \(String(format: "%.4f", error))")
        print("  Performance: \(result.interpretation)")
        print()
    }
}

func demonstrateSolutionEvolution(system: HybridPINNRK4System) {
    print("🌊 Solution Evolution Over Time")
    print("===============================")
    
    let x = 0.5
    let timePoints = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("At x = \(x):")
    print("Time  | PINN   | RK4    | Hybrid | Analytical")
    print("------|--------|--------|--------|----------")
    
    for t in timePoints {
        let result = system.computeHybridOutput(x: x, t: t, epoch: 10)
        let rk4 = system.rk4Solver.solve(x: x, t: t)
        let analytical = -sin(.pi * x) * exp(-.pi * .pi * t)
        
        print(String(format: "%.1f   | %.3f  | %.3f  | %.3f  | %.3f",
                     t, result.S_x, rk4, result.psi, analytical))
    }
}

// Extension to make RK4Solver accessible
extension HybridPINNRK4System {
    var rk4Solver: RK4Solver {
        return RK4Solver()
    }
}

// Run the demonstration
main()