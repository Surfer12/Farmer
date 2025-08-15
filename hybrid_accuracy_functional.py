"""
Hybrid Symbolic-Neural Accuracy Functional Implementation

This module implements the hybrid functional Ψ(x) that balances symbolic (RK4-derived) 
and neural (ML/NN) accuracies with regularization and probability calibration for 
theoretical and computational fidelity in chaotic systems.

Based on the formalization:
Ψ(x) = (1/T) Σ[k=1 to T] [α(t_k)S(x,t_k) + (1-α(t_k))N(x,t_k)] 
       * exp(-[λ₁R_cog(t_k) + λ₂R_eff(t_k)]) * P(H|E,β,t_k)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.special import expit  # sigmoid function


@dataclass
class HybridConfig:
    """Configuration for the hybrid accuracy functional."""
    lambda_1: float = 0.75  # Weight for cognitive penalty
    lambda_2: float = 0.25  # Weight for efficiency penalty
    beta: float = 1.2       # Bias parameter for probability calibration
    kappa: float = 1.0      # Sensitivity parameter for adaptive weight
    T: int = 1              # Number of time steps (1 for single-step evaluation)


class HybridAccuracyFunctional:
    """
    Implementation of the Hybrid Symbolic-Neural Accuracy Functional.
    
    This class provides methods to compute Ψ(x) for various scenarios including
    chaotic systems, collaboration benefits, and open-source contributions.
    """
    
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
    
    def symbolic_accuracy(self, x: np.ndarray, t: np.ndarray, 
                         solution_func: Optional[Callable] = None) -> np.ndarray:
        """
        Compute symbolic accuracy S(x,t) ∈ [0,1].
        
        For RK4 solutions, this is 1 - normalized_error from ODE solving.
        """
        if solution_func is None:
            # Default: simulate RK4 accuracy for demonstration
            # In practice, this would be computed from actual RK4 solution fidelity
            base_accuracy = 0.9 - 0.2 * np.exp(-np.abs(x))  # Higher for stable regions
            noise = 0.05 * np.random.normal(0, 1, x.shape)
            return np.clip(base_accuracy + noise, 0, 1)
        else:
            return solution_func(x, t)
    
    def neural_accuracy(self, x: np.ndarray, t: np.ndarray,
                       model_func: Optional[Callable] = None) -> np.ndarray:
        """
        Compute neural accuracy N(x,t) ∈ [0,1].
        
        For ML/NN predictions, this is R² or 1 - RMSE from LSTM/GRU.
        """
        if model_func is None:
            # Default: simulate neural network accuracy
            # Neural networks often perform better in chaotic regions
            base_accuracy = 0.85 + 0.1 * np.tanh(2 * np.abs(x))  # Adaptive to chaos
            noise = 0.03 * np.random.normal(0, 1, x.shape)
            return np.clip(base_accuracy + noise, 0, 1)
        else:
            return model_func(x, t)
    
    def adaptive_weight(self, t: np.ndarray, lyapunov_exp: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute adaptive weight α(t) ∈ [0,1].
        
        α(t) = σ(-κ * λ_local(t)), favoring N in chaotic regions (high Lyapunov exponent).
        """
        if lyapunov_exp is None:
            # Simulate varying chaos levels over time
            lyapunov_exp = 0.5 * np.sin(2 * np.pi * t) + 0.3 * np.random.normal(0, 0.1, t.shape)
        
        return expit(-self.config.kappa * lyapunov_exp)
    
    def cognitive_penalty(self, x: np.ndarray, t: np.ndarray,
                         physics_violation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cognitive penalty R_cog(t) ≥ 0.
        
        Physics violation penalty (energy drift or ODE residual in pendulums).
        """
        if physics_violation is None:
            # Simulate physics violations - higher for extreme conditions
            return 0.1 + 0.15 * np.exp(-0.5 * (x**2)) * (1 + 0.1 * np.abs(t))
        else:
            return physics_violation
    
    def efficiency_penalty(self, x: np.ndarray, t: np.ndarray,
                          computational_cost: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute efficiency penalty R_eff(t) ≥ 0.
        
        Normalized FLOPs or latency per prediction step.
        """
        if computational_cost is None:
            # Simulate computational costs - varies with problem complexity
            return 0.08 + 0.12 * (1 + np.abs(x)) * (1 + 0.05 * t)
        else:
            return computational_cost
    
    def calibrated_probability(self, base_prob: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compute calibrated probability P(H|E,β,t) ∈ [0,1].
        
        Platt-scaled with bias β: P' = σ(logit(P) + log(β))
        """
        # Avoid numerical issues with logit
        base_prob_clipped = np.clip(base_prob, 1e-7, 1 - 1e-7)
        logit_p = np.log(base_prob_clipped / (1 - base_prob_clipped))
        adjusted_logit = logit_p + np.log(self.config.beta)
        return np.clip(expit(adjusted_logit), 0, 1)
    
    def compute_psi(self, x: np.ndarray, t: np.ndarray,
                   S_func: Optional[Callable] = None,
                   N_func: Optional[Callable] = None,
                   base_prob: Optional[np.ndarray] = None) -> float:
        """
        Compute the hybrid accuracy functional Ψ(x).
        
        Args:
            x: Input array (e.g., initial conditions or angles)
            t: Time array
            S_func: Optional function for symbolic accuracy
            N_func: Optional function for neural accuracy  
            base_prob: Optional base probability array
            
        Returns:
            Ψ(x): Hybrid accuracy value ∈ [0,1]
        """
        # Compute components
        S = self.symbolic_accuracy(x, t, S_func)
        N = self.neural_accuracy(x, t, N_func)
        alpha = self.adaptive_weight(t)
        
        R_cog = self.cognitive_penalty(x, t)
        R_eff = self.efficiency_penalty(x, t)
        
        if base_prob is None:
            base_prob = 0.8 + 0.15 * np.random.random(x.shape)  # Default probability
        
        P_calibrated = self.calibrated_probability(base_prob, t)
        
        # Compute hybrid output
        hybrid_output = alpha * S + (1 - alpha) * N
        
        # Compute regularization term
        regularization = np.exp(-(self.config.lambda_1 * R_cog + self.config.lambda_2 * R_eff))
        
        # Compute final functional
        integrand = hybrid_output * regularization * P_calibrated
        
        # Average over time steps
        psi = np.mean(integrand)
        
        return psi
    
    def compute_psi_detailed(self, x: np.ndarray, t: np.ndarray) -> dict:
        """
        Compute Ψ(x) with detailed breakdown of all components.
        
        Returns a dictionary with all intermediate values for analysis.
        """
        S = self.symbolic_accuracy(x, t)
        N = self.neural_accuracy(x, t)
        alpha = self.adaptive_weight(t)
        
        R_cog = self.cognitive_penalty(x, t)
        R_eff = self.efficiency_penalty(x, t)
        
        base_prob = 0.8 + 0.15 * np.random.random(x.shape)
        P_calibrated = self.calibrated_probability(base_prob, t)
        
        hybrid_output = alpha * S + (1 - alpha) * N
        regularization = np.exp(-(self.config.lambda_1 * R_cog + self.config.lambda_2 * R_eff))
        
        psi = np.mean(hybrid_output * regularization * P_calibrated)
        
        return {
            'psi': psi,
            'S': S,
            'N': N,
            'alpha': alpha,
            'R_cog': R_cog,
            'R_eff': R_eff,
            'base_prob': base_prob,
            'P_calibrated': P_calibrated,
            'hybrid_output': hybrid_output,
            'regularization': regularization
        }


def reproduce_numerical_example():
    """Reproduce the numerical example from the document."""
    print("Reproducing Numerical Example from Document:")
    print("=" * 50)
    
    # Step 1: Outputs
    S = 0.67
    N = 0.87
    print(f"Step 1 - Outputs: S(x) = {S}, N(x) = {N}")
    
    # Step 2: Hybrid
    alpha = 0.4
    O_hybrid = alpha * S + (1 - alpha) * N
    print(f"Step 2 - Hybrid: α = {alpha}, O_hybrid = {O_hybrid:.3f}")
    
    # Step 3: Penalties
    R_cognitive = 0.17
    R_efficiency = 0.11
    lambda1 = 0.6
    lambda2 = 0.4
    P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
    exp_term = np.exp(-P_total)
    print(f"Step 3 - Penalties: R_cog = {R_cognitive}, R_eff = {R_efficiency}")
    print(f"         λ₁ = {lambda1}, λ₂ = {lambda2}, P_total = {P_total:.3f}")
    print(f"         exp(-P_total) ≈ {exp_term:.3f}")
    
    # Step 4: Probability
    P = 0.81
    beta = 1.2
    # P_adj = σ(logit(P) + log(β))
    logit_P = np.log(P / (1 - P))
    P_adj = expit(logit_P + np.log(beta))
    print(f"Step 4 - Probability: P = {P}, β = {beta}")
    print(f"         P_adj ≈ {P_adj:.3f}")
    
    # Step 5: Ψ(x)
    psi = O_hybrid * exp_term * P_adj
    print(f"Step 5 - Ψ(x): ≈ {O_hybrid:.3f} × {exp_term:.3f} × {P_adj:.3f} ≈ {psi:.3f}")
    
    # Step 6: Interpret
    print(f"Step 6 - Interpretation: Ψ(x) ≈ {psi:.2f} indicates high responsiveness")
    print()
    
    return psi


def demonstrate_framework():
    """Demonstrate the framework with various scenarios."""
    print("Demonstrating Hybrid Accuracy Functional Framework:")
    print("=" * 55)
    
    # Initialize framework
    config = HybridConfig(lambda_1=0.75, lambda_2=0.25, beta=1.2)
    framework = HybridAccuracyFunctional(config)
    
    # Test scenarios
    scenarios = [
        ("Chaotic System (Multi-pendulum)", np.array([0.1, 0.5, -0.3]), np.array([0.0, 0.5, 1.0])),
        ("Stable System", np.array([0.0, 0.1]), np.array([0.0, 1.0])),
        ("High Chaos Region", np.array([1.5, -1.2]), np.array([0.0, 0.8])),
    ]
    
    for name, x, t in scenarios:
        print(f"\n{name}:")
        print("-" * len(name))
        
        # Compute detailed breakdown
        results = framework.compute_psi_detailed(x, t)
        
        print(f"  Ψ(x) = {results['psi']:.3f}")
        print(f"  Components:")
        print(f"    Symbolic Accuracy (S): {np.mean(results['S']):.3f}")
        print(f"    Neural Accuracy (N): {np.mean(results['N']):.3f}")  
        print(f"    Adaptive Weight (α): {np.mean(results['alpha']):.3f}")
        print(f"    Cognitive Penalty (R_cog): {np.mean(results['R_cog']):.3f}")
        print(f"    Efficiency Penalty (R_eff): {np.mean(results['R_eff']):.3f}")
        print(f"    Calibrated Probability (P): {np.mean(results['P_calibrated']):.3f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Reproduce the numerical example
    reproduce_numerical_example()
    
    # Demonstrate the framework
    demonstrate_framework()