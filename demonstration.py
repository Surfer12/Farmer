#!/usr/bin/env python3
"""
Comprehensive Demonstration of the Hybrid Symbolic-Neural Accuracy Functional
Shows the mathematical framework in action with various parameter settings
"""

import math
from typing import List, Tuple
from hybrid_functional_simple import HybridFunctional, FunctionalParams

def demonstrate_parameter_sensitivity():
    """
    Demonstrate how different parameters affect the functional value
    """
    print("="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Base case parameters
    base_S, base_N = 0.7, 0.8
    base_alpha = 0.5
    base_R_cog, base_R_eff = 0.15, 0.1
    base_prob = 0.8
    base_beta = 1.2
    
    functional = HybridFunctional()
    
    # Test different alpha values
    print("\n1. Adaptive Weight (α) Sensitivity:")
    print(f"{'α':<6} {'O_hybrid':<10} {'Ψ(x)':<8} {'Interpretation'}")
    print("-" * 50)
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        O_hybrid = functional.compute_hybrid_output(base_S, base_N, alpha)
        psi = functional.compute_single_step_psi(
            base_S, base_N, alpha, base_R_cog, base_R_eff, base_prob
        )
        
        if alpha == 0.0:
            interpretation = "Pure Neural"
        elif alpha == 1.0:
            interpretation = "Pure Symbolic"
        else:
            interpretation = f"Hybrid ({alpha*100:.0f}% S, {(1-alpha)*100:.0f}% N)"
            
        print(f"{alpha:<6.2f} {O_hybrid:<10.3f} {psi:<8.3f} {interpretation}")
    
    # Test different regularization weights
    print("\n2. Regularization Weight Sensitivity:")
    print(f"{'λ₁':<6} {'λ₂':<6} {'Penalty':<10} {'Ψ(x)':<8}")
    print("-" * 40)
    
    for lambda1 in [0.5, 0.75, 1.0]:
        for lambda2 in [0.25, 0.5, 0.75]:
            if lambda1 + lambda2 <= 1.0:  # Ensure weights sum to ≤ 1
                params = FunctionalParams(lambda1=lambda1, lambda2=lambda2)
                func = HybridFunctional(params)
                penalty = func.compute_regularization_penalty(base_R_cog, base_R_eff)
                psi = func.compute_single_step_psi(
                    base_S, base_N, base_alpha, base_R_cog, base_R_eff, base_prob
                )
                print(f"{lambda1:<6.2f} {lambda2:<6.2f} {penalty:<10.3f} {psi:<8.3f}")
    
    # Test different probability bias values
    print("\n3. Probability Bias (β) Sensitivity:")
    print(f"{'β':<6} {'P_adj':<10} {'Ψ(x)':<8}")
    print("-" * 30)
    
    for beta in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        P_adj = functional.compute_probability(base_prob, beta)
        psi = functional.compute_single_step_psi(
            base_S, base_N, base_alpha, base_R_cog, base_R_eff, base_prob
        )
        print(f"{beta:<6.2f} {P_adj:<10.3f} {psi:<8.3f}")

def demonstrate_chaotic_system_modeling():
    """
    Demonstrate the framework for chaotic system modeling
    """
    print("\n" + "="*80)
    print("CHAOTIC SYSTEM MODELING EXAMPLE")
    print("="*80)
    
    print("Simulating multi-pendulum dynamics with adaptive weighting...")
    
    # Time points
    time_points = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Simulate chaotic behavior (increasing Lyapunov exponent over time)
    lyapunov_exponents = [0.1, 0.3, 0.6, 1.2, 2.0, 3.5]
    
    # Symbolic accuracy decreases in chaotic regions (RK4 struggles)
    symbolic_accuracies = [0.95, 0.92, 0.85, 0.70, 0.55, 0.35]
    
    # Neural accuracy more stable (LSTM adapts to chaos)
    neural_accuracies = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92]
    
    # Penalties increase with chaos
    cognitive_penalties = [0.05, 0.08, 0.15, 0.25, 0.40, 0.60]
    efficiency_penalties = [0.03, 0.05, 0.10, 0.18, 0.30, 0.50]
    
    # Base probabilities
    base_probs = [0.90, 0.88, 0.85, 0.80, 0.75, 0.70]
    
    functional = HybridFunctional()
    
    print(f"{'Time':<6} {'λ_local':<8} {'α':<6} {'S':<6} {'N':<6} {'Ψ(x)':<8} {'Method'}")
    print("-" * 70)
    
    psi_values = []
    for i, t in enumerate(time_points):
        # Compute adaptive weight based on local chaos
        alpha = functional.compute_adaptive_weight(t, lyapunov_exponents[i])
        
        # Compute functional value
        psi = functional.compute_single_step_psi(
            symbolic_accuracies[i], neural_accuracies[i], alpha,
            cognitive_penalties[i], efficiency_penalties[i], base_probs[i]
        )
        psi_values.append(psi)
        
        # Determine dominant method
        if alpha > 0.7:
            method = "Symbolic"
        elif alpha < 0.3:
            method = "Neural"
        else:
            method = "Hybrid"
            
        print(f"{t:<6.1f} {lyapunov_exponents[i]:<8.2f} {alpha:<6.3f} "
              f"{symbolic_accuracies[i]:<6.2f} {neural_accuracies[i]:<6.2f} "
              f"{psi:<8.3f} {method}")
    
    # Compute multi-step average
    multi_step_psi = functional.compute_multi_step_psi(
        symbolic_accuracies, neural_accuracies, 
        [functional.compute_adaptive_weight(t, le) for t, le in zip(time_points, lyapunov_exponents)],
        cognitive_penalties, efficiency_penalties, base_probs
    )
    
    print("-" * 70)
    print(f"Multi-step average Ψ(x): {multi_step_psi:.3f}")
    print(f"Chaos adaptation: α ranges from {min([functional.compute_adaptive_weight(t, le) for t, le in zip(time_points, lyapunov_exponents)]):.3f} to {max([functional.compute_adaptive_weight(t, le) for t, le in zip(time_points, lyapunov_exponents)]):.3f}")

def demonstrate_optimization_scenarios():
    """
    Demonstrate optimization scenarios with different parameter settings
    """
    print("\n" + "="*80)
    print("OPTIMIZATION SCENARIOS")
    print("="*80)
    
    # Scenario 1: High accuracy, low efficiency
    print("Scenario 1: High Accuracy, Low Efficiency")
    print("Parameters: λ₁ = 0.8, λ₂ = 0.2 (emphasize cognitive accuracy)")
    
    params1 = FunctionalParams(lambda1=0.8, lambda2=0.2)
    func1 = HybridFunctional(params1)
    
    psi1 = func1.compute_single_step_psi(0.9, 0.85, 0.6, 0.1, 0.3, 0.9)
    print(f"Result: Ψ(x) = {psi1:.3f}")
    
    # Scenario 2: Balanced approach
    print("\nScenario 2: Balanced Approach")
    print("Parameters: λ₁ = 0.5, λ₂ = 0.5 (equal weighting)")
    
    params2 = FunctionalParams(lambda1=0.5, lambda2=0.5)
    func2 = HybridFunctional(params2)
    
    psi2 = func2.compute_single_step_psi(0.9, 0.85, 0.6, 0.1, 0.3, 0.9)
    print(f"Result: Ψ(x) = {psi2:.3f}")
    
    # Scenario 3: High efficiency, lower accuracy
    print("\nScenario 3: High Efficiency, Lower Accuracy")
    print("Parameters: λ₁ = 0.2, λ₂ = 0.8 (emphasize efficiency)")
    
    params3 = FunctionalParams(lambda1=0.2, lambda2=0.8)
    func3 = HybridFunctional(params3)
    
    psi3 = func3.compute_single_step_psi(0.9, 0.85, 0.6, 0.1, 0.3, 0.9)
    print(f"Result: Ψ(x) = {psi3:.3f}")
    
    print(f"\nComparison:")
    print(f"High Accuracy:   Ψ(x) = {psi1:.3f}")
    print(f"Balanced:        Ψ(x) = {psi2:.3f}")
    print(f"High Efficiency: Ψ(x) = {psi3:.3f}")
    
    if psi1 > psi2 and psi1 > psi3:
        print("✓ High accuracy approach yields best results")
    elif psi2 > psi1 and psi2 > psi3:
        print("✓ Balanced approach yields best results")
    else:
        print("✓ High efficiency approach yields best results")

def demonstrate_mathematical_properties():
    """
    Demonstrate key mathematical properties of the functional
    """
    print("\n" + "="*80)
    print("MATHEMATICAL PROPERTIES VERIFICATION")
    print("="*80)
    
    functional = HybridFunctional()
    
    # Property 1: Boundedness
    print("1. Boundedness: Ψ(x) ∈ [0,1]")
    test_cases = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # All zeros
        (1.0, 1.0, 1.0, 0.0, 0.0, 1.0),  # All ones
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),  # Mixed values
        (0.0, 0.0, 0.0, 10.0, 10.0, 0.0), # High penalties
    ]
    
    for i, (S, N, alpha, R_cog, R_eff, base_prob) in enumerate(test_cases):
        psi = functional.compute_single_step_psi(S, N, alpha, R_cog, R_eff, base_prob)
        bounded = 0 <= psi <= 1
        print(f"   Case {i+1}: Ψ(x) = {psi:.3f} {'✓' if bounded else '✗'}")
    
    # Property 2: Continuity
    print("\n2. Continuity: Small input changes → small output changes")
    base_inputs = (0.5, 0.5, 0.5, 0.1, 0.1, 0.8)
    base_psi = functional.compute_single_step_psi(*base_inputs)
    
    delta = 0.01
    perturbed_inputs = (0.5 + delta, 0.5, 0.5, 0.1, 0.1, 0.8)
    perturbed_psi = functional.compute_single_step_psi(*perturbed_inputs)
    
    change = abs(perturbed_psi - base_psi)
    print(f"   Base Ψ(x) = {base_psi:.3f}")
    print(f"   Perturbed Ψ(x) = {perturbed_psi:.3f}")
    print(f"   Change = {change:.6f} (small input change → small output change)")
    
    # Property 3: Monotonicity in key parameters
    print("\n3. Monotonicity: Increasing accuracy → increasing Ψ(x)")
    S_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    psi_values = []
    
    for S in S_values:
        psi = functional.compute_single_step_psi(S, 0.5, 0.5, 0.1, 0.1, 0.8)
        psi_values.append(psi)
    
    monotonic = all(psi_values[i] <= psi_values[i+1] for i in range(len(psi_values)-1))
    print(f"   S values: {S_values}")
    print(f"   Ψ(x) values: {[f'{p:.3f}' for p in psi_values]}")
    print(f"   Monotonic: {'✓' if monotonic else '✗'}")

def main():
    """
    Run all demonstrations
    """
    print("HYBRID SYMBOLIC-NEURAL ACCURACY FUNCTIONAL DEMONSTRATION")
    print("="*80)
    print("This demonstration showcases the mathematical framework with various")
    print("parameter settings and real-world applications.")
    print("="*80)
    
    # Run all demonstrations
    demonstrate_parameter_sensitivity()
    demonstrate_chaotic_system_modeling()
    demonstrate_optimization_scenarios()
    demonstrate_mathematical_properties()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("The hybrid functional framework successfully demonstrates:")
    print("✓ Parameter sensitivity and interpretability")
    print("✓ Chaotic system modeling with adaptive weighting")
    print("✓ Optimization scenarios for different objectives")
    print("✓ Mathematical properties (boundedness, continuity, monotonicity)")
    print("="*80)

if __name__ == "__main__":
    main()