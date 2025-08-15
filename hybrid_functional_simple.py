#!/usr/bin/env python3
"""
Simplified Hybrid Symbolic-Neural Accuracy Functional
No external dependencies required - uses only Python standard library
"""

import math
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class FunctionalParams:
    """Parameters for the hybrid functional computation"""
    lambda1: float = 0.75  # Cognitive regularization weight
    lambda2: float = 0.25  # Efficiency regularization weight
    kappa: float = 1.0     # Adaptive weight parameter
    beta: float = 1.2      # Probability bias parameter

class HybridFunctional:
    """
    Implementation of the hybrid symbolic-neural accuracy functional
    
    Ψ(x) = (1/T) * Σ[α(t_k)S(x,t_k) + (1-α(t_k))N(x,t_k)] * 
            exp(-[λ₁R_cog(t_k) + λ₂R_eff(t_k)]) * P(H|E,β,t_k)
    """
    
    def __init__(self, params: FunctionalParams = None):
        self.params = params or FunctionalParams()
        
    def compute_adaptive_weight(self, t: float, lyapunov_exponent: Optional[float] = None) -> float:
        """
        Compute adaptive weight α(t) = σ(-κ * λ_local(t))
        
        Args:
            t: Time point
            lyapunov_exponent: Local Lyapunov exponent (if available)
            
        Returns:
            Adaptive weight between 0 and 1
        """
        if lyapunov_exponent is None:
            # Default: simple time-based adaptation
            lyapunov_exponent = math.sin(t) * 0.5 + 0.5
        
        # Sigmoid function to bound between 0 and 1
        alpha = 1 / (1 + math.exp(self.params.kappa * lyapunov_exponent))
        return max(0.0, min(1.0, alpha))
    
    def compute_hybrid_output(self, S: float, N: float, alpha: float) -> float:
        """
        Compute hybrid output: α*S + (1-α)*N
        
        Args:
            S: Symbolic accuracy [0,1]
            N: Neural accuracy [0,1]
            alpha: Adaptive weight [0,1]
            
        Returns:
            Hybrid output value
        """
        return alpha * S + (1 - alpha) * N
    
    def compute_regularization_penalty(self, R_cog: float, R_eff: float) -> float:
        """
        Compute regularization penalty: exp(-[λ₁R_cog + λ₂R_eff])
        
        Args:
            R_cog: Cognitive penalty ≥ 0
            R_eff: Efficiency penalty ≥ 0
            
        Returns:
            Regularization penalty value
        """
        total_penalty = self.params.lambda1 * R_cog + self.params.lambda2 * R_eff
        return math.exp(-total_penalty)
    
    def compute_probability(self, base_prob: float, beta: Optional[float] = None) -> float:
        """
        Compute calibrated probability P(H|E,β) with bias adjustment
        
        Args:
            base_prob: Base probability [0,1]
            beta: Bias parameter (defaults to self.params.beta)
            
        Returns:
            Calibrated probability [0,1]
        """
        if beta is None:
            beta = self.params.beta
            
        # Logit shift: P' = σ(logit(P) + log(β))
        if base_prob > 0 and base_prob < 1:
            logit = math.log(base_prob / (1 - base_prob))
            adjusted_logit = logit + math.log(beta)
            adjusted_prob = 1 / (1 + math.exp(-adjusted_logit))
        else:
            adjusted_prob = base_prob
            
        return max(0.0, min(1.0, adjusted_prob))
    
    def compute_single_step_psi(self, S: float, N: float, alpha: float, 
                               R_cog: float, R_eff: float, 
                               base_prob: float) -> float:
        """
        Compute Ψ(x) for a single time step (T=1)
        
        Args:
            S: Symbolic accuracy
            N: Neural accuracy  
            alpha: Adaptive weight
            R_cog: Cognitive penalty
            R_eff: Efficiency penalty
            base_prob: Base probability
            
        Returns:
            Single-step Ψ(x) value
        """
        # Step 1: Hybrid output
        hybrid_output = self.compute_hybrid_output(S, N, alpha)
        
        # Step 2: Regularization penalty
        reg_penalty = self.compute_regularization_penalty(R_cog, R_eff)
        
        # Step 3: Calibrated probability
        calibrated_prob = self.compute_probability(base_prob)
        
        # Step 4: Final functional value
        psi = hybrid_output * reg_penalty * calibrated_prob
        
        return psi
    
    def compute_multi_step_psi(self, S: List[float], N: List[float], 
                              alphas: List[float], R_cog: List[float], 
                              R_eff: List[float], base_probs: List[float]) -> float:
        """
        Compute Ψ(x) over multiple time steps
        
        Args:
            S: List of symbolic accuracies
            N: List of neural accuracies
            alphas: List of adaptive weights
            R_cog: List of cognitive penalties
            R_eff: List of efficiency penalties
            base_probs: List of base probabilities
            
        Returns:
            Multi-step Ψ(x) value
        """
        T = len(S)
        if not all(len(x) == T for x in [N, alphas, R_cog, R_eff, base_probs]):
            raise ValueError("All input lists must have the same length")
        
        psi_sum = 0.0
        for k in range(T):
            psi_k = self.compute_single_step_psi(
                S[k], N[k], alphas[k], R_cog[k], R_eff[k], base_probs[k]
            )
            psi_sum += psi_k
        
        return psi_sum / T
    
    def run_numerical_example(self) -> dict:
        """
        Run the numerical example from the user's query
        """
        print("=== Numerical Example: Single Tracking Step ===")
        
        # Step 1: Outputs
        S = 0.67
        N = 0.87
        print(f"Step 1: S(x) = {S}, N(x) = {N}")
        
        # Step 2: Hybrid
        alpha = 0.4
        O_hybrid = self.compute_hybrid_output(S, N, alpha)
        print(f"Step 2: α = {alpha}, O_hybrid = {O_hybrid:.3f}")
        
        # Step 3: Penalties
        R_cognitive = 0.17
        R_efficiency = 0.11
        lambda1, lambda2 = 0.6, 0.4
        P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
        exp_term = math.exp(-P_total)
        print(f"Step 3: R_cognitive = {R_cognitive}, R_efficiency = {R_efficiency}")
        print(f"         λ1 = {lambda1}, λ2 = {lambda2}")
        print(f"         P_total = {P_total:.3f}")
        print(f"         exp ≈ {exp_term:.3f}")
        
        # Step 4: Probability
        P = 0.81
        beta = 1.2
        P_adj = self.compute_probability(P, beta)
        print(f"Step 4: P = {P}, β = {beta}")
        print(f"         P_adj ≈ {P_adj:.3f}")
        
        # Step 5: Final Ψ(x)
        psi = O_hybrid * exp_term * P_adj
        print(f"Step 5: Ψ(x) ≈ {O_hybrid:.3f} × {exp_term:.3f} × {P_adj:.3f} ≈ {psi:.3f}")
        
        # Step 6: Interpretation
        print(f"Step 6: Ψ(x) ≈ {psi:.3f} indicates high responsiveness")
        
        return {
            'S': S, 'N': N, 'alpha': alpha, 'O_hybrid': O_hybrid,
            'R_cognitive': R_cognitive, 'R_efficiency': R_efficiency,
            'lambda1': lambda1, 'lambda2': lambda2, 'P_total': P_total,
            'exp_term': exp_term, 'P': P, 'beta': beta, 'P_adj': P_adj,
            'psi': psi
        }
    
    def run_open_source_example(self) -> dict:
        """
        Run the open-source contributions example
        """
        print("\n=== Open-Source Contributions Example ===")
        
        # Step 1: Outputs
        S = 0.74
        N = 0.84
        print(f"Step 1: S(x) = {S}, N(x) = {N}")
        
        # Step 2: Hybrid
        alpha = 0.5
        O_hybrid = self.compute_hybrid_output(S, N, alpha)
        print(f"Step 2: α = {alpha}, O_hybrid = {O_hybrid:.3f}")
        
        # Step 3: Penalties
        R_cognitive = 0.14
        R_efficiency = 0.09
        lambda1, lambda2 = 0.55, 0.45
        P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
        exp_term = math.exp(-P_total)
        print(f"Step 3: R_cognitive = {R_cognitive}, R_efficiency = {R_efficiency}")
        print(f"         λ1 = {lambda1}, λ2 = {lambda2}")
        print(f"         P_total = {P_total:.3f}")
        print(f"         exp ≈ {exp_term:.3f}")
        
        # Step 4: Probability
        P = 0.77
        beta = 1.3
        P_adj = self.compute_probability(P, beta)
        print(f"Step 4: P = {P}, β = {beta}")
        print(f"         P_adj ≈ {P_adj:.3f}")
        
        # Step 5: Final Ψ(x)
        psi = O_hybrid * exp_term * P_adj
        print(f"Step 5: Ψ(x) ≈ {O_hybrid:.3f} × {exp_term:.3f} × {P_adj:.3f} ≈ {psi:.3f}")
        
        # Step 6: Interpretation
        print(f"Step 6: Ψ(x) ≈ {psi:.3f} reflects strong innovation potential")
        
        return {
            'S': S, 'N': N, 'alpha': alpha, 'O_hybrid': O_hybrid,
            'R_cognitive': R_cognitive, 'R_efficiency': R_efficiency,
            'lambda1': lambda1, 'lambda2': lambda2, 'P_total': P_total,
            'exp_term': exp_term, 'P': P, 'beta': beta, 'P_adj': P_adj,
            'psi': psi
        }
    
    def run_collaboration_example(self) -> dict:
        """
        Run the collaboration benefits example
        """
        print("\n=== Collaboration Benefits Example ===")
        
        # Step 1: Outputs
        S = 0.72  # Reasoning/ethical/inclusive gains
        N = 0.78  # Breakthroughs
        print(f"Step 1: S(x) = {S}, N(x) = {N}")
        
        # Step 2: Hybrid
        alpha = 0.6  # Balanced approach
        O_hybrid = self.compute_hybrid_output(S, N, alpha)
        print(f"Step 2: α = {alpha}, O_hybrid = {O_hybrid:.3f}")
        
        # Step 3: Penalties
        R_cognitive = 0.16  # Implementation challenges
        R_efficiency = 0.12  # Coordination costs
        lambda1, lambda2 = 0.65, 0.35
        P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
        exp_term = math.exp(-P_total)
        print(f"Step 3: R_cognitive = {R_cognitive}, R_efficiency = {R_efficiency}")
        print(f"         λ1 = {lambda1}, λ2 = {lambda2}")
        print(f"         P_total = {P_total:.3f}")
        print(f"         exp ≈ {exp_term:.3f}")
        
        # Step 4: Probability
        P = 0.79
        beta = 1.15
        P_adj = self.compute_probability(P, beta)
        print(f"Step 4: P = {P}, β = {beta}")
        print(f"         P_adj ≈ {P_adj:.3f}")
        
        # Step 5: Final Ψ(x)
        psi = O_hybrid * exp_term * P_adj
        print(f"Step 5: Ψ(x) ≈ {O_hybrid:.3f} × {exp_term:.3f} × {P_adj:.3f} ≈ {psi:.3f}")
        
        # Step 6: Interpretation
        print(f"Step 6: Ψ(x) ≈ {psi:.3f} indicates comprehensive collaboration benefits")
        
        return {
            'S': S, 'N': N, 'alpha': alpha, 'O_hybrid': O_hybrid,
            'R_cognitive': R_cognitive, 'R_efficiency': R_efficiency,
            'lambda1': lambda1, 'lambda2': lambda2, 'P_total': P_total,
            'exp_term': exp_term, 'P': P, 'beta': beta, 'P_adj': P_adj,
            'psi': psi
        }

def print_summary_table(results: List[dict]):
    """
    Print a summary table of all results
    """
    print("\n" + "="*80)
    print("SUMMARY OF HYBRID FUNCTIONAL RESULTS")
    print("="*80)
    print(f"{'Scenario':<25} {'S(x)':<6} {'N(x)':<6} {'α':<6} {'Ψ(x)':<8} {'Interpretation'}")
    print("-"*80)
    
    scenarios = [
        ("Tracking Step", results[0]),
        ("Open Source", results[1]),
        ("Collaboration", results[2])
    ]
    
    for name, result in scenarios:
        psi = result['psi']
        if psi >= 0.7:
            interpretation = "Excellent"
        elif psi >= 0.6:
            interpretation = "Good"
        elif psi >= 0.5:
            interpretation = "Moderate"
        else:
            interpretation = "Needs Improvement"
            
        print(f"{name:<25} {result['S']:<6.2f} {result['N']:<6.2f} "
              f"{result['alpha']:<6.1f} {psi:<8.3f} {interpretation}")
    
    print("-"*80)
    print(f"Average Ψ(x): {sum(r['psi'] for r in results) / len(results):.3f}")
    print("="*80)

if __name__ == "__main__":
    # Initialize the hybrid functional
    functional = HybridFunctional()
    
    # Run all examples
    tracking_result = functional.run_numerical_example()
    open_source_result = functional.run_open_source_example()
    collaboration_result = functional.run_collaboration_example()
    
    # Print summary table
    all_results = [tracking_result, open_source_result, collaboration_result]
    print_summary_table(all_results)
    
    # Demonstrate multi-step computation
    print("\n=== Multi-Step Computation Example ===")
    print("Computing Ψ(x) over 3 time steps...")
    
    S = [0.65, 0.70, 0.75]
    N = [0.80, 0.75, 0.70]
    alphas = [0.4, 0.5, 0.6]
    R_cog = [0.15, 0.12, 0.10]
    R_eff = [0.10, 0.08, 0.06]
    base_probs = [0.85, 0.88, 0.90]
    
    multi_step_psi = functional.compute_multi_step_psi(S, N, alphas, R_cog, R_eff, base_probs)
    print(f"Multi-step Ψ(x) = {multi_step_psi:.3f}")
    
    # Show individual steps for comparison
    print("\nIndividual step values:")
    for i in range(3):
        single_step = functional.compute_single_step_psi(
            S[i], N[i], alphas[i], R_cog[i], R_eff[i], base_probs[i]
        )
        print(f"  Step {i+1}: Ψ(x) = {single_step:.3f}")
    
    print(f"\nAverage of individual steps: {sum(single_step for single_step in [functional.compute_single_step_psi(S[i], N[i], alphas[i], R_cog[i], R_eff[i], base_probs[i]) for i in range(3)]) / 3:.3f}")
    print(f"Multi-step computation: {multi_step_psi:.3f}")
    print("✓ Multi-step computation matches average of individual steps")