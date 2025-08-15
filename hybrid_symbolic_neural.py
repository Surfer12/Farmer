import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import math

class HybridSymbolicNeural:
    """
    Implementation of the hybrid symbolic-neural accuracy functional Ψ(x)
    
    Ψ(x) = (1/T) * Σ[α(t_k)S(x,t_k) + (1-α(t_k))N(x,t_k)] * 
            exp(-[λ₁R_cog(t_k) + λ₂R_eff(t_k)]) * P(H|E,β,t_k)
    """
    
    def __init__(self, lambda1: float = 0.6, lambda2: float = 0.4):
        self.lambda1 = lambda1  # Weight for cognitive regularization
        self.lambda2 = lambda2  # Weight for efficiency regularization
        
    def compute_hybrid_output(self, S: float, N: float, alpha: float) -> float:
        """Compute hybrid output: αS + (1-α)N"""
        return alpha * S + (1 - alpha) * N
    
    def compute_regularization_penalty(self, R_cognitive: float, R_efficiency: float) -> float:
        """Compute regularization penalty: exp(-[λ₁R_cog + λ₂R_eff])"""
        penalty_total = self.lambda1 * R_cognitive + self.lambda2 * R_efficiency
        return np.exp(-penalty_total)
    
    def compute_probability_adjustment(self, P: float, beta: float) -> float:
        """Compute probability adjustment with bias β"""
        # Logit shift: P' = σ(logit(P) + log(β))
        if P <= 0 or P >= 1:
            return np.clip(P, 0, 1)
        
        logit_P = np.log(P / (1 - P))
        adjusted_logit = logit_P + np.log(beta)
        P_adjusted = 1 / (1 + np.exp(-adjusted_logit))
        return np.clip(P_adjusted, 0, 1)
    
    def compute_psi(self, S: float, N: float, alpha: float, 
                    R_cognitive: float, R_efficiency: float, 
                    P: float, beta: float) -> Tuple[float, dict]:
        """
        Compute the complete hybrid functional Ψ(x)
        
        Args:
            S: Symbolic accuracy [0,1]
            N: Neural accuracy [0,1] 
            alpha: Adaptive weight [0,1]
            R_cognitive: Cognitive penalty ≥0
            R_efficiency: Efficiency penalty ≥0
            P: Base probability [0,1]
            beta: Responsiveness bias >0
            
        Returns:
            Ψ(x): Hybrid accuracy functional
        """
        # Step 1: Hybrid output
        O_hybrid = self.compute_hybrid_output(S, N, alpha)
        
        # Step 2: Regularization penalty
        reg_penalty = self.compute_regularization_penalty(R_cognitive, R_efficiency)
        
        # Step 3: Probability adjustment
        P_adjusted = self.compute_probability_adjustment(P, beta)
        
        # Step 4: Final Ψ(x)
        psi = O_hybrid * reg_penalty * P_adjusted
        
        return psi, {
            'O_hybrid': O_hybrid,
            'reg_penalty': reg_penalty,
            'P_adjusted': P_adjusted,
            'penalty_total': self.lambda1 * R_cognitive + self.lambda2 * R_efficiency
        }
    
    def compute_psi_time_series(self, S_series: List[float], N_series: List[float],
                               alpha_series: List[float], R_cog_series: List[float],
                               R_eff_series: List[float], P_series: List[float],
                               beta: float) -> Tuple[float, List[float]]:
        """
        Compute Ψ(x) over time series with averaging
        """
        T = len(S_series)
        psi_values = []
        
        for k in range(T):
            psi_k, _ = self.compute_psi(
                S_series[k], N_series[k], alpha_series[k],
                R_cog_series[k], R_eff_series[k], P_series[k], beta
            )
            psi_values.append(psi_k)
        
        # Average over time
        psi_avg = np.mean(psi_values)
        
        return psi_avg, psi_values

def demonstrate_numerical_example():
    """Demonstrate the numerical example from the query"""
    print("=== Numerical Example: Single Tracking Step ===")
    
    # Initialize the hybrid system
    hybrid_system = HybridSymbolicNeural(lambda1=0.6, lambda2=0.4)
    
    # Step 1: Outputs
    S = 0.67  # Symbolic accuracy
    N = 0.87  # Neural accuracy
    print(f"Step 1: S(x) = {S}, N(x) = {N}")
    
    # Step 2: Hybrid
    alpha = 0.4
    O_hybrid = hybrid_system.compute_hybrid_output(S, N, alpha)
    print(f"Step 2: α = {alpha}, O_hybrid = {O_hybrid:.3f}")
    
    # Step 3: Penalties
    R_cognitive = 0.17
    R_efficiency = 0.11
    penalty_total = hybrid_system.lambda1 * R_cognitive + hybrid_system.lambda2 * R_efficiency
    reg_exp = np.exp(-penalty_total)
    print(f"Step 3: R_cognitive = {R_cognitive}, R_efficiency = {R_efficiency}")
    print(f"         λ₁ = {hybrid_system.lambda1}, λ₂ = {hybrid_system.lambda2}")
    print(f"         P_total = {penalty_total:.3f}, exp ≈ {reg_exp:.3f}")
    
    # Step 4: Probability
    P = 0.81
    beta = 1.2
    P_adjusted = hybrid_system.compute_probability_adjustment(P, beta)
    print(f"Step 4: P = {P}, β = {beta}, P_adj ≈ {P_adjusted:.3f}")
    
    # Step 5: Final Ψ(x)
    psi, details = hybrid_system.compute_psi(S, N, alpha, R_cognitive, R_efficiency, P, beta)
    print(f"Step 5: Ψ(x) ≈ {O_hybrid:.3f} × {reg_exp:.3f} × {P_adjusted:.3f} ≈ {psi:.3f}")
    
    # Step 6: Interpretation
    print(f"Step 6: Interpret Ψ(x) ≈ {psi:.3f} indicates high responsiveness")
    
    return psi, details

def demonstrate_open_source_contribution():
    """Demonstrate open-source contribution example"""
    print("\n=== Open-Source Contribution Example ===")
    
    hybrid_system = HybridSymbolicNeural(lambda1=0.55, lambda2=0.45)
    
    # Parameters from the example
    S = 0.74
    N = 0.84
    alpha = 0.5
    R_cognitive = 0.14
    R_efficiency = 0.09
    P = 0.77
    beta = 1.3
    
    psi, details = hybrid_system.compute_psi(S, N, alpha, R_cognitive, R_efficiency, P, beta)
    
    print(f"S(x) = {S}, N(x) = {N}")
    print(f"α = {alpha}, O_hybrid = {details['O_hybrid']:.3f}")
    print(f"R_cognitive = {R_cognitive}, R_efficiency = {R_efficiency}")
    print(f"P = {P}, β = {beta}, P_adj = {details['P_adjusted']:.3f}")
    print(f"Ψ(x) ≈ {psi:.3f} reflects strong innovation potential")
    
    return psi

def visualize_hybrid_functional():
    """Create visualization of the hybrid functional components"""
    print("\n=== Visualization ===")
    
    # Create sample data
    alphas = np.linspace(0, 1, 100)
    S_fixed = 0.7
    N_fixed = 0.8
    
    hybrid_outputs = [alpha * S_fixed + (1 - alpha) * N_fixed for alpha in alphas]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Hybrid output vs alpha
    plt.subplot(2, 2, 1)
    plt.plot(alphas, hybrid_outputs, 'b-', linewidth=2)
    plt.xlabel('α (Adaptive Weight)')
    plt.ylabel('Hybrid Output')
    plt.title('Hybrid Output: αS + (1-α)N')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Regularization penalty
    plt.subplot(2, 2, 2)
    R_cog_range = np.linspace(0, 1, 100)
    R_eff_range = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(R_cog_range, R_eff_range)
    Z = np.exp(-(0.6 * X + 0.4 * Y))
    
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('R_cognitive')
    plt.ylabel('R_efficiency')
    plt.title('Regularization Penalty')
    
    # Subplot 3: Probability adjustment
    plt.subplot(2, 2, 3)
    P_range = np.linspace(0.1, 0.9, 100)
    betas = [0.8, 1.0, 1.2, 1.5]
    
    for beta in betas:
        P_adj = [1 / (1 + np.exp(-(np.log(p/(1-p)) + np.log(beta))) for p in P_range]
        P_adj = [np.clip(p, 0, 1) for p in P_adj]
        plt.plot(P_range, P_adj, label=f'β = {beta}')
    
    plt.xlabel('Base Probability P')
    plt.ylabel('Adjusted Probability P\'')
    plt.title('Probability Adjustment with Bias β')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Complete Ψ(x) surface
    plt.subplot(2, 2, 4)
    alpha_mesh, S_mesh = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0.5, 1, 50))
    N_mesh = 0.8 * np.ones_like(S_mesh)
    
    # Fixed parameters for this visualization
    R_cog_fixed, R_eff_fixed = 0.2, 0.15
    P_fixed, beta_fixed = 0.8, 1.1
    
    psi_surface = np.zeros_like(alpha_mesh)
    for i in range(alpha_mesh.shape[0]):
        for j in range(alpha_mesh.shape[1]):
            hybrid_system = HybridSymbolicNeural()
            psi_surface[i, j], _ = hybrid_system.compute_psi(
                S_mesh[i, j], N_mesh[i, j], alpha_mesh[i, j],
                R_cog_fixed, R_eff_fixed, P_fixed, beta_fixed
            )
    
    contour = plt.contourf(alpha_mesh, S_mesh, psi_surface, levels=20, cmap='plasma')
    plt.colorbar(contour)
    plt.xlabel('α (Adaptive Weight)')
    plt.ylabel('S(x) (Symbolic Accuracy)')
    plt.title('Ψ(x) Surface (Fixed N, R, P)')
    
    plt.tight_layout()
    plt.savefig('hybrid_functional_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'hybrid_functional_visualization.png'")
    
    return plt

def time_series_example():
    """Demonstrate time series integration"""
    print("\n=== Time Series Integration Example ===")
    
    # Simulate time series data
    T = 10
    time_steps = np.arange(T)
    
    # Generate realistic time series
    S_series = [0.6 + 0.1 * np.sin(2 * np.pi * t / T) + 0.05 * np.random.randn() for t in time_steps]
    N_series = [0.8 + 0.1 * np.cos(2 * np.pi * t / T) + 0.05 * np.random.randn() for t in time_steps]
    alpha_series = [0.3 + 0.4 * (1 + np.sin(2 * np.pi * t / T)) / 2 for t in time_steps]
    R_cog_series = [0.2 + 0.1 * np.random.rand() for t in time_steps]
    R_eff_series = [0.15 + 0.1 * np.random.rand() for t in time_steps]
    P_series = [0.75 + 0.1 * np.random.rand() for t in time_steps]
    
    # Clip values to valid ranges
    S_series = [np.clip(s, 0, 1) for s in S_series]
    N_series = [np.clip(n, 0, 1) for n in N_series]
    alpha_series = [np.clip(a, 0, 1) for a in alpha_series]
    R_cog_series = [max(r, 0) for r in R_cog_series]
    R_eff_series = [max(r, 0) for r in R_eff_series]
    P_series = [np.clip(p, 0, 1) for p in P_series]
    
    hybrid_system = HybridSymbolicNeural()
    psi_avg, psi_series = hybrid_system.compute_psi_time_series(
        S_series, N_series, alpha_series, R_cog_series, R_eff_series, P_series, beta=1.2
    )
    
    print(f"Time-averaged Ψ(x) = {psi_avg:.3f}")
    
    # Plot time series
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(time_steps, S_series, 'b-', label='S(x)', linewidth=2)
    plt.plot(time_steps, N_series, 'r-', label='N(x)', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Symbolic vs Neural Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(time_steps, alpha_series, 'g-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('α')
    plt.title('Adaptive Weight Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 3)
    plt.plot(time_steps, R_cog_series, 'orange', label='R_cognitive', linewidth=2)
    plt.plot(time_steps, R_eff_series, 'purple', label='R_efficiency', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Penalty')
    plt.title('Regularization Penalties Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    plt.plot(time_steps, P_series, 'brown', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('P')
    plt.title('Base Probability Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 5)
    plt.plot(time_steps, psi_series, 'k-', linewidth=2)
    plt.axhline(y=psi_avg, color='r', linestyle='--', label=f'Average: {psi_avg:.3f}')
    plt.xlabel('Time Step')
    plt.ylabel('Ψ(x)')
    plt.title('Hybrid Functional Ψ(x) Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    # Show the final Ψ(x) components
    final_psi, final_details = hybrid_system.compute_psi(
        S_series[-1], N_series[-1], alpha_series[-1],
        R_cog_series[-1], R_eff_series[-1], P_series[-1], 1.2
    )
    
    components = ['Hybrid Output', 'Reg Penalty', 'P Adjusted']
    values = [final_details['O_hybrid'], final_details['reg_penalty'], final_details['P_adjusted']]
    
    plt.bar(components, values, color=['blue', 'green', 'orange'])
    plt.ylabel('Value')
    plt.title('Final Ψ(x) Components')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    print("Time series analysis saved as 'time_series_analysis.png'")
    
    return psi_avg, psi_series

if __name__ == "__main__":
    # Run demonstrations
    print("Hybrid Symbolic-Neural Accuracy Functional Implementation")
    print("=" * 60)
    
    # Numerical example
    psi_single, details_single = demonstrate_numerical_example()
    
    # Open source contribution example
    psi_contribution = demonstrate_open_source_contribution()
    
    # Time series example
    psi_avg, psi_series = time_series_example()
    
    # Visualization
    plt = visualize_hybrid_functional()
    
    print("\n" + "=" * 60)
    print("Summary of Results:")
    print(f"Single step Ψ(x): {psi_single:.3f}")
    print(f"Contribution Ψ(x): {psi_contribution:.3f}")
    print(f"Time-averaged Ψ(x): {psi_avg:.3f}")
    print("\nAll visualizations have been saved as PNG files.")