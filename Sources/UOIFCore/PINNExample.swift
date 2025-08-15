// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

/// Example usage and demonstration of the PINN framework
public class PINNExample {
    
    /// Run a complete PINN example solving the Burgers' equation
    public static func runBurgersExample() {
        print("=== PINN Burgers' Equation Solver ===")
        print("Solving: u_t + u * u_x = 0")
        print("Initial condition: u(x,0) = -sin(πx)")
        print("Domain: x ∈ [-1, 1], t ∈ [0, 1]")
        print()
        
        // Create PINN solver
        let solver = PINNSolver(
            xRange: -1.0...1.0,
            tRange: 0.0...1.0,
            nx: 50,  // Reduced for faster demonstration
            nt: 50,
            layerSizes: [2, 20, 20, 20, 1]
        )
        
        // Solve the PDE
        let solution = solver.solve()
        
        print("\n=== Solution Summary ===")
        print("Grid size: \(solution.x.count) × \(solution.t.count)")
        print("Final training loss: \(solution.trainingLoss)")
        print("Validation loss (vs RK4): \(solution.validationLoss)")
        print()
        
        // Compute Ψ performance metrics
        let psiOutcome = solver.computePsiPerformance()
        
        print("=== Ψ Framework Integration ===")
        print("Hybrid output: \(psiOutcome.hybrid)")
        print("Penalty factor: \(psiOutcome.penalty)")
        print("Posterior probability: \(psiOutcome.posterior)")
        print("Final Ψ(x): \(psiOutcome.psi)")
        print("Gradient dΨ/dα: \(psiOutcome.dPsi_dAlpha)")
        print()
        
        // Analyze solution at specific points
        analyzeSolution(solution)
        
        // Compare with analytical solution where possible
        compareWithAnalytical(solution)
    }
    
    /// Analyze the PINN solution at key points
    private static func analyzeSolution(_ solution: PINNSolution) {
        print("=== Solution Analysis ===")
        
        // Check initial condition preservation
        let initialIndex = 0
        var initialError = 0.0
        for i in 0..<solution.x.count {
            let x = solution.x[i]
            let u_pred = solution.u[i][initialIndex]
            let u_true = -sin(.pi * x)
            initialError += (u_pred - u_true) * (u_pred - u_true)
        }
        initialError = sqrt(initialError / Double(solution.x.count))
        print("Initial condition error: \(initialError)")
        
        // Check boundary conditions
        var boundaryError = 0.0
        for j in 0..<solution.t.count {
            let leftBoundary = solution.u[0][j]
            let rightBoundary = solution.u[solution.x.count - 1][j]
            boundaryError += leftBoundary * leftBoundary + rightBoundary * rightBoundary
        }
        boundaryError = sqrt(boundaryError / Double(solution.t.count))
        print("Boundary condition error: \(boundaryError)")
        
        // Check PDE residual
        var maxResidual = 0.0
        var avgResidual = 0.0
        for i in 0..<solution.x.count {
            for j in 0..<solution.t.count {
                let residual = abs(solution.pdeResidual[i][j])
                maxResidual = max(maxResidual, residual)
                avgResidual += residual
            }
        }
        avgResidual /= Double(solution.x.count * solution.t.count)
        print("Max PDE residual: \(maxResidual)")
        print("Average PDE residual: \(avgResidual)")
        print()
    }
    
    /// Compare PINN solution with analytical solution where available
    private static func compareWithAnalytical(_ solution: PINNSolution) {
        print("=== Analytical Comparison ===")
        print("Note: Burgers' equation develops shocks, making analytical solutions complex")
        print("We compare with the initial condition and known properties")
        
        // Check shock formation (simplified)
        let midTimeIndex = solution.t.count / 2
        let midTime = solution.t[midTimeIndex]
        
        print("At t = \(midTime):")
        
        // Find maximum and minimum values
        var maxU = -Double.infinity
        var minU = Double.infinity
        for i in 0..<solution.x.count {
            let u = solution.u[i][midTimeIndex]
            maxU = max(maxU, u)
            minU = min(minU, u)
        }
        
        print("  Max u: \(maxU)")
        print("  Min u: \(minU)")
        print("  Range: \(maxU - minU)")
        
        // Check symmetry (Burgers' equation preserves some symmetry)
        let centerIndex = solution.x.count / 2
        let leftIndex = centerIndex - 5
        let rightIndex = centerIndex + 5
        
        if leftIndex >= 0 && rightIndex < solution.x.count {
            let leftU = solution.u[leftIndex][midTimeIndex]
            let rightU = solution.u[rightIndex][midTimeIndex]
            let symmetryError = abs(leftU + rightU)  // Should be close to 0 for odd initial condition
            print("  Symmetry error: \(symmetryError)")
        }
        print()
    }
    
    /// Demonstrate the mathematical framework from your description
    public static func demonstrateMathematicalFramework() {
        print("=== Mathematical Framework Demonstration ===")
        print("Based on your Ψ(x) = O_hybrid × exp(-P_total) × P_adj framework")
        print()
        
        // Example values from your walkthrough
        let S_x = 0.72  // State inference
        let N_x = 0.85  // Neural PINN approximation
        let alpha = 0.5  // Real-time validation flow
        
        // Step 1: Hybrid Output
        let O_hybrid = alpha * S_x + (1 - alpha) * N_x
        print("Step 1: Hybrid Output")
        print("  S(x) = \(S_x) (state inference)")
        print("  N(x) = \(N_x) (neural PINN)")
        print("  α(t) = \(alpha) (validation flow)")
        print("  O_hybrid = \(alpha) × \(S_x) + (1 - \(alpha)) × \(N_x) = \(O_hybrid)")
        print()
        
        // Step 2: Regularization Penalties
        let R_cognitive = 0.15  // Physical accuracy in residuals
        let R_efficiency = 0.10  // Training efficiency
        let lambda1 = 0.6
        let lambda2 = 0.4
        
        let P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
        let penalty_exp = exp(-P_total)
        
        print("Step 2: Regularization")
        print("  R_cognitive = \(R_cognitive) (physical accuracy)")
        print("  R_efficiency = \(R_efficiency) (training efficiency)")
        print("  λ₁ = \(lambda1), λ₂ = \(lambda2)")
        print("  P_total = \(lambda1) × \(R_cognitive) + \(lambda2) × \(R_efficiency) = \(P_total)")
        print("  exp(-P_total) = \(penalty_exp)")
        print()
        
        // Step 3: Probability Adjustment
        let P_base = 0.80  // Base probability
        let beta = 1.2     // Model responsiveness
        let P_adj = min(beta * P_base, 1.0)
        
        print("Step 3: Probability")
        print("  P(H|E) = \(P_base)")
        print("  β = \(beta) (responsiveness)")
        print("  P_adj = min(\(beta) × \(P_base), 1.0) = \(P_adj)")
        print()
        
        // Step 4: Final Ψ(x)
        let psi_x = O_hybrid * penalty_exp * P_adj
        
        print("Step 4: Final Result")
        print("  Ψ(x) = O_hybrid × exp(-P_total) × P_adj")
        print("  Ψ(x) = \(O_hybrid) × \(penalty_exp) × \(P_adj)")
        print("  Ψ(x) ≈ \(psi_x)")
        print()
        
        // Step 5: Interpretation
        print("Step 5: Interpretation")
        if psi_x > 0.7 {
            print("  Ψ(x) ≈ \(String(format: "%.2f", psi_x)) indicates excellent model performance")
        } else if psi_x > 0.5 {
            print("  Ψ(x) ≈ \(String(format: "%.2f", psi_x)) indicates solid model performance")
        } else if psi_x > 0.3 {
            print("  Ψ(x) ≈ \(String(format: "%.2f", psi_x)) indicates moderate model performance")
        } else {
            print("  Ψ(x) ≈ \(String(format: "%.2f", psi_x)) indicates poor model performance")
        }
        print()
    }
    
    /// Run a performance comparison between different network architectures
    public static func runArchitectureComparison() {
        print("=== Architecture Comparison ===")
        
        let architectures = [
            ("Small", [2, 10, 10, 1]),
            ("Medium", [2, 20, 20, 1]),
            ("Large", [2, 30, 30, 30, 1]),
            ("Deep", [2, 15, 15, 15, 15, 15, 1])
        ]
        
        var results: [(name: String, trainingLoss: Double, validationLoss: Double, psi: Double)] = []
        
        for (name, layers) in architectures {
            print("Testing \(name) architecture: \(layers)")
            
            let solver = PINNSolver(
                xRange: -1.0...1.0,
                tRange: 0.0...1.0,
                nx: 30,  // Smaller grid for faster comparison
                nt: 30,
                layerSizes: layers
            )
            
            let solution = solver.solve()
            let psiOutcome = solver.computePsiPerformance()
            
            results.append((
                name: name,
                trainingLoss: solution.trainingLoss,
                validationLoss: solution.validationLoss,
                psi: psiOutcome.psi
            ))
            
            print("  Training Loss: \(solution.trainingLoss)")
            print("  Validation Loss: \(solution.validationLoss)")
            print("  Ψ(x): \(psiOutcome.psi)")
            print()
        }
        
        // Find best architecture
        let best = results.min { $0.psi > $1.psi } ?? results[0]
        print("Best Architecture: \(best.name)")
        print("  Ψ(x): \(best.psi)")
        print("  Training Loss: \(best.trainingLoss)")
        print("  Validation Loss: \(best.validationLoss)")
    }
}

// MARK: - Utility Functions

/// Utility functions for PINN analysis
public struct PINNUtilities {
    
    /// Compute L2 norm of a 2D array
    public static func l2Norm(_ array: [[Double]]) -> Double {
        var sum = 0.0
        for row in array {
            for val in row {
                sum += val * val
            }
        }
        return sqrt(sum)
    }
    
    /// Compute relative error between two solutions
    public static func relativeError(_ predicted: [[Double]], _ reference: [[Double]]) -> Double {
        let diff = l2Norm(predicted) - l2Norm(reference)
        return abs(diff) / l2Norm(reference)
    }
    
    /// Export solution to CSV format for external plotting
    public static func exportToCSV(_ solution: PINNSolution, filename: String) -> String {
        var csv = "x,t,u,pde_residual\n"
        
        for i in 0..<solution.x.count {
            for j in 0..<solution.t.count {
                let x = solution.x[i]
                let t = solution.t[j]
                let u = solution.u[i][j]
                let residual = solution.pdeResidual[i][j]
                
                csv += "\(x),\(t),\(u),\(residual)\n"
            }
        }
        
        return csv
    }
    
    /// Generate a simple text-based visualization of the solution
    public static func textVisualization(_ solution: PINNSolution, maxWidth: Int = 50) -> String {
        var output = "Solution Visualization (u(x,t))\n"
        output += String(repeating: "=", count: maxWidth) + "\n"
        
        let stepX = max(1, solution.x.count / maxWidth)
        let stepT = max(1, solution.t.count / 20)
        
        for j in stride(from: 0, to: solution.t.count, by: stepT) {
            output += "t=\(String(format: "%.2f", solution.t[j])): "
            
            for i in stride(from: 0, to: solution.x.count, by: stepX) {
                let u = solution.u[i][j]
                if u > 0.5 {
                    output += "█"
                } else if u > 0.0 {
                    output += "▓"
                } else if u > -0.5 {
                    output += "▒"
                } else {
                    output += "░"
                }
            }
            output += "\n"
        }
        
        return output
    }
}