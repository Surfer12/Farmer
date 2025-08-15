import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
import warnings

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
        
    def compute_adaptive_weight(self, t: float, lyapunov_exponent: float = None) -> float:
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
            lyapunov_exponent = np.sin(t) * 0.5 + 0.5
        
        # Sigmoid function to bound between 0 and 1
        alpha = 1 / (1 + np.exp(self.params.kappa * lyapunov_exponent))
        return np.clip(alpha, 0, 1)
    
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
        return np.exp(-total_penalty)
    
    def compute_probability(self, base_prob: float, beta: float = None) -> float:
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
            logit = np.log(base_prob / (1 - base_prob))
            adjusted_logit = logit + np.log(beta)
            adjusted_prob = 1 / (1 + np.exp(-adjusted_logit))
        else:
            adjusted_prob = base_prob
            
        return np.clip(adjusted_prob, 0, 1)
    
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
        exp_term = np.exp(-P_total)
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
        exp_term = np.exp(-P_total)
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

def create_visualization(psi_values: List[float], labels: List[str]):
    """
    Create a bar chart visualization of Ψ(x) values
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, psi_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Ψ(x) Value')
    plt.title('Hybrid Functional Ψ(x) Values Across Different Scenarios')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, psi_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize the hybrid functional
    functional = HybridFunctional()
    
    # Run examples
    tracking_result = functional.run_numerical_example()
    open_source_result = functional.run_open_source_example()
    
    # Create visualization
    psi_values = [tracking_result['psi'], open_source_result['psi']]
    labels = ['Tracking Step', 'Open Source']
    
    create_visualization(psi_values, labels)
    
    print(f"\nSummary of Results:")
    print(f"Tracking Step: Ψ(x) = {tracking_result['psi']:.3f}")
    print(f"Open Source:  Ψ(x) = {open_source_result['psi']:.3f}")