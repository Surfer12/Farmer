#!/usr/bin/env python3
"""
Simplified Test for Hybrid Accuracy Functional

This test demonstrates the core functionality using only built-in Python libraries.
For full functionality, install numpy, scipy, and matplotlib.
"""

import math
from typing import List, Tuple

def simple_array_mean(arr: List[float]) -> float:
    """Simple mean calculation."""
    return sum(arr) / len(arr)

def simple_array_exp(arr: List[float]) -> List[float]:
    """Simple element-wise exponential."""
    return [math.exp(x) for x in arr]

def simple_array_clip(arr: List[float], min_val: float, max_val: float) -> List[float]:
    """Simple clipping function."""
    return [max(min_val, min(max_val, x)) for x in arr]

def compute_V_simple(S: List[float], N: List[float], alpha: List[float],
                    Rcog: List[float], Reff: List[float], P_corr: List[float],
                    lambda1: float = 0.75, lambda2: float = 0.25) -> float:
    """
    Simplified version of the hybrid accuracy functional V(x).
    
    V(x) = (1/T) Σ_{k=1..T} [ α(tk) S(x, tk) + (1 − α(tk)) N(x, tk) ] 
           · exp(−[λ1 Rcog(tk) + λ2 Reff(tk)]) · P(H|E, β, tk)
    """
    if not all(len(arr) == len(S) for arr in [N, alpha, Rcog, Reff, P_corr]):
        raise ValueError("All input arrays must have the same length")
    
    T = len(S)
    terms = []
    
    for i in range(T):
        # Hybrid accuracy term
        hybrid = alpha[i] * S[i] + (1.0 - alpha[i]) * N[i]
        
        # Exponential penalty term
        penalty_exponent = -(lambda1 * Rcog[i] + lambda2 * Reff[i])
        reg = math.exp(penalty_exponent)
        
        # Clip probability to [0,1] for safety
        P_clipped = max(0.0, min(1.0, P_corr[i]))
        
        # Combined term for this time step
        term = hybrid * reg * P_clipped
        terms.append(term)
    
    # Return temporal average
    return simple_array_mean(terms)

def test_specification_example():
    """Test the exact numerical example from the specification."""
    print("=" * 60)
    print("SPECIFICATION EXAMPLE TEST")
    print("=" * 60)
    
    # Input values from specification
    S = [0.65]
    N = [0.85]
    alpha = [0.3]
    Rcog = [0.20]
    Reff = [0.15]
    
    # Apply bias to probability: 0.75 * 1.3 = 0.975 (clipped to [0,1])
    P_base = 0.75
    bias = 1.3
    P_corr = [min(1.0, P_base * bias)]  # 0.975
    
    lambda1, lambda2 = 0.75, 0.25
    
    print(f"Input parameters:")
    print(f"  S (symbolic accuracy) = {S[0]}")
    print(f"  N (neural accuracy) = {N[0]}")
    print(f"  α (adaptive weight) = {alpha[0]}")
    print(f"  Rcog (cognitive penalty) = {Rcog[0]}")
    print(f"  Reff (efficiency penalty) = {Reff[0]}")
    print(f"  P(H|E,β) = {P_corr[0]:.3f}")
    print()
    
    # Step-by-step calculation
    hybrid_term = alpha[0] * S[0] + (1 - alpha[0]) * N[0]
    penalty_exp = -(lambda1 * Rcog[0] + lambda2 * Reff[0])
    penalty_factor = math.exp(penalty_exp)
    
    print(f"Step-by-step calculation:")
    print(f"  Hybrid = α·S + (1-α)·N = {alpha[0]}·{S[0]} + {1-alpha[0]}·{N[0]} = {hybrid_term}")
    print(f"  Penalty exponent = -({lambda1}·{Rcog[0]} + {lambda2}·{Reff[0]}) = {penalty_exp}")
    print(f"  Penalty factor = exp({penalty_exp}) = {penalty_factor:.4f}")
    print(f"  Final term = {hybrid_term} · {penalty_factor:.4f} · {P_corr[0]:.3f} = {hybrid_term * penalty_factor * P_corr[0]:.3f}")
    print()
    
    # Compute V(x)
    result = compute_V_simple(S, N, alpha, Rcog, Reff, P_corr, lambda1, lambda2)
    expected = 0.638
    
    print(f"Result: V(x) = {result:.3f}")
    print(f"Expected: ~{expected}")
    print(f"Match within tolerance: {abs(result - expected) < 0.01}")
    print()
    
    return result

def test_multi_step():
    """Test with multiple time steps."""
    print("=" * 60)
    print("MULTI-STEP TRAJECTORY TEST")
    print("=" * 60)
    
    # Create synthetic multi-step data
    T = 5
    S = [0.7, 0.75, 0.65, 0.8, 0.72]
    N = [0.8, 0.78, 0.85, 0.75, 0.82]
    alpha = [0.3, 0.4, 0.6, 0.5, 0.45]
    Rcog = [0.1, 0.15, 0.2, 0.12, 0.18]
    Reff = [0.05, 0.08, 0.1, 0.06, 0.09]
    P_corr = [0.9, 0.85, 0.88, 0.92, 0.87]
    
    print(f"Multi-step trajectory with {T} time steps:")
    print(f"  S range: [{min(S):.2f}, {max(S):.2f}]")
    print(f"  N range: [{min(N):.2f}, {max(N):.2f}]")
    print(f"  α range: [{min(alpha):.2f}, {max(alpha):.2f}]")
    print()
    
    # Compute V(x)
    result = compute_V_simple(S, N, alpha, Rcog, Reff, P_corr)
    
    print(f"Step-by-step breakdown:")
    lambda1, lambda2 = 0.75, 0.25
    for i in range(T):
        hybrid_i = alpha[i] * S[i] + (1 - alpha[i]) * N[i]
        penalty_i = math.exp(-(lambda1 * Rcog[i] + lambda2 * Reff[i]))
        term_i = hybrid_i * penalty_i * P_corr[i]
        print(f"  t={i}: hybrid={hybrid_i:.3f}, penalty={penalty_i:.3f}, P={P_corr[i]:.2f} → term={term_i:.3f}")
    
    print(f"\nOverall V(x) = {result:.4f}")
    print()
    
    return result

def test_adaptive_weighting():
    """Test different adaptive weighting strategies."""
    print("=" * 60)
    print("ADAPTIVE WEIGHTING TEST")
    print("=" * 60)
    
    # Test scenario: chaos-based weighting simulation
    # α = σ(-κ·λ_lyapunov) where σ is sigmoid function
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(x))
    
    lyapunov_exponents = [-1.0, 0.0, 1.0, 2.0]
    kappa = 2.0
    
    print("Chaos-based adaptive weighting:")
    print("α = σ(-κ·λ_lyapunov) where σ is sigmoid, κ=2.0")
    print("High chaos (positive λ) → low α → favor neural method")
    print()
    
    for i, lyap in enumerate(lyapunov_exponents):
        alpha_val = sigmoid(-kappa * lyap)
        interpretation = "favor symbolic" if alpha_val > 0.5 else "favor neural"
        print(f"  λ_lyap={lyap:4.1f} → α={alpha_val:.3f} ({interpretation})")
    
    print()
    return [sigmoid(-kappa * lyap) for lyap in lyapunov_exponents]

def test_penalty_functions():
    """Test penalty function computation."""
    print("=" * 60)
    print("PENALTY FUNCTIONS TEST")
    print("=" * 60)
    
    # Cognitive penalties (physics consistency)
    print("Cognitive penalties (Rcog):")
    energy_drift = [0.01, 0.05, 0.1, 0.2]
    constraint_violation = [0.0, 0.02, 0.08, 0.15]
    ode_residual = [0.005, 0.01, 0.03, 0.1]
    
    Rcog = [e + c + o for e, c, o in zip(energy_drift, constraint_violation, ode_residual)]
    
    print(f"  Energy drift: {energy_drift}")
    print(f"  Constraint violation: {constraint_violation}")
    print(f"  ODE residual: {ode_residual}")
    print(f"  Total Rcog: {[round(r, 3) for r in Rcog]}")
    print()
    
    # Efficiency penalties (computational cost)
    print("Efficiency penalties (Reff, normalized):")
    flops = [1000, 2000, 5000, 10000]
    memory = [100, 150, 300, 500]
    latency = [0.1, 0.2, 0.5, 1.0]
    
    # Normalize each component to [0,1]
    max_flops, max_memory, max_latency = max(flops), max(memory), max(latency)
    Reff = [f/max_flops + m/max_memory + l/max_latency 
            for f, m, l in zip(flops, memory, latency)]
    
    print(f"  FLOPs: {flops}")
    print(f"  Memory: {memory}")
    print(f"  Latency: {latency}")
    print(f"  Normalized Reff: {[round(r, 3) for r in Reff]}")
    print()
    
    return Rcog, Reff

def main():
    """Run all simplified tests."""
    print("HYBRID SYMBOLIC-NEURAL ACCURACY FUNCTIONAL")
    print("Simplified Test Suite (Pure Python)")
    print("=" * 80)
    print()
    
    # Run tests
    test_results = {}
    
    test_results['specification'] = test_specification_example()
    test_results['multi_step'] = test_multi_step()
    test_results['adaptive'] = test_adaptive_weighting()
    test_results['penalties'] = test_penalty_functions()
    
    # Summary
    print("=" * 80)
    print("SIMPLIFIED TESTING COMPLETE")
    print("=" * 80)
    print("✓ Core hybrid accuracy functional V(x)")
    print("✓ Specification example reproduced (V ≈ 0.638)")
    print("✓ Multi-step trajectory handling")
    print("✓ Adaptive weighting strategies")
    print("✓ Penalty function computation")
    print()
    print("Key Formula:")
    print("V(x) = (1/T) Σ [ α(t) S(t) + (1-α(t)) N(t) ] · exp(-[λ1 Rcog(t) + λ2 Reff(t)]) · P(H|E,β,t)")
    print()
    print("For full functionality including:")
    print("- Cross-modal non-commutativity analysis")
    print("- Cognitive-memory distance metrics") 
    print("- Koopman operator analysis")
    print("- Broken Neural Scaling Laws")
    print("Install: pip install numpy scipy matplotlib")
    print("Then run: python hybrid_accuracy_functional.py")
    
    return test_results

if __name__ == "__main__":
    results = main()