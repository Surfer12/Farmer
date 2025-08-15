// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation
import UOIFCore

print("UOIF CLI - Unified Output Interface Framework")
print("=============================================\n")

// Check command line arguments
let args = CommandLine.arguments
if args.count > 1 {
    switch args[1] {
    case "pinn":
        print("Running PINN (Physics-Informed Neural Network) Example...\n")
        PINNExample.runCompleteExample()
        
    case "psi":
        print("Running Ψ Framework Example...\n")
        runPsiExample()
        
    case "help":
        printHelp()
        
    default:
        print("Unknown command: \(args[1])")
        printHelp()
    }
} else {
    print("No command specified. Available commands:")
    print("  pinn  - Run PINN example with Ψ framework analysis")
    print("  psi   - Run Ψ framework example")
    print("  help  - Show this help message")
    print("\nUse: uoif-cli <command>")
}

// MARK: - Ψ Framework Example
func runPsiExample() {
    print("=== Ψ Framework Example ===\n")
    
    // Example 1: PINN Analysis
    let pinnInputs = PsiInputs(
        alpha: 0.5,
        S_symbolic: 0.72,
        N_external: 0.85,
        lambdaAuthority: 0.6,
        lambdaVerifiability: 0.4,
        riskAuthority: 0.15,
        riskVerifiability: 0.10,
        basePosterior: 0.80,
        betaUplift: 1.2
    )
    
    let pinnOutcome = PsiModel.computePsi(inputs: pinnInputs)
    
    print("PINN Analysis Results:")
    print("  Hybrid Output: \(String(format: "%.3f", pinnOutcome.hybrid))")
    print("  Penalty Factor: \(String(format: "%.3f", pinnOutcome.penalty))")
    print("  Posterior: \(String(format: "%.3f", pinnOutcome.posterior))")
    print("  Ψ Value: \(String(format: "%.3f", pinnOutcome.psi))")
    print("  dΨ/dα: \(String(format: "%.3f", pinnOutcome.dPsi_dAlpha))")
    print()
    
    // Example 2: Different α values
    print("Ψ Values for Different α Values:")
    for alpha in stride(from: 0.0, through: 1.0, by: 0.2) {
        var inputs = pinnInputs
        inputs.alpha = alpha
        let outcome = PsiModel.computePsi(inputs: inputs)
        print("  α = \(String(format: "%.1f", alpha)): Ψ = \(String(format: "%.3f", outcome.psi))")
    }
    print()
}

func printHelp() {
    print("UOIF CLI Commands:")
    print("  pinn  - Run Physics-Informed Neural Network example")
    print("         Includes training, Ψ framework analysis, and visualization")
    print("  psi   - Run Ψ framework mathematical examples")
    print("         Demonstrates hybrid output computation and analysis")
    print("  help  - Show this help message")
    print("\nExamples:")
    print("  uoif-cli pinn    # Run complete PINN example")
    print("  uoif-cli psi     # Run Ψ framework examples")
}

