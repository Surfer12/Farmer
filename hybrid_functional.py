import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

class HybridAccuracyFunctional:
    """
    Clean Formalization of the Hybrid Symbolic–Neural Accuracy Functional
    
    Ψ(x) = (1/T) Σ[k=1 to T] [α(t_k)S(x,t_k) + (1-α(t_k))N(x,t_k)] 
           × exp(-[λ₁R_cog(t_k) + λ₂R_eff(t_k)]) × P(H|E,β,t_k)
    """
    
    def __init__(self, lambda1=0.75, lambda2=0.25, beta=1.2):
        self.lambda1 = lambda1  # Weight for cognitive penalty
        self.lambda2 = lambda2  # Weight for efficiency penalty
        self.beta = beta        # Responsiveness bias
        
    def symbolic_accuracy(self, x, t):
        """S(x,t) ∈ [0,1]: Symbolic accuracy (e.g., RK4 solution fidelity)"""
        # Normalized accuracy based on theoretical solution quality
        return np.clip(0.9 - 0.2 * np.abs(np.sin(np.pi * x * t)), 0, 1)
    
    def neural_accuracy(self, x, t):
        """N(x,t) ∈ [0,1]: Neural accuracy (e.g., ML/NN prediction fidelity)"""
        # Simulated neural network accuracy with adaptive behavior
        chaos_factor = np.abs(x) * t  # Higher in chaotic regions
        return np.clip(0.8 + 0.1 * np.cos(2 * np.pi * chaos_factor), 0, 1)
    
    def adaptive_weight(self, t, kappa=2.0):
        """α(t) ∈ [0,1]: Adaptive weight favoring neural in chaotic regions"""
        # Sigmoid function: α(t) = σ(-κ·λ_local(t))
        lambda_local = np.sin(np.pi * t)  # Simulated Lyapunov exponent
        return 1 / (1 + np.exp(kappa * lambda_local))
    
    def cognitive_penalty(self, t):
        """R_cog(t) ≥ 0: Cognitive penalty (physics violation, energy drift)"""
        # Penalty increases with time due to accumulated errors
        return 0.1 + 0.1 * t**2
    
    def efficiency_penalty(self, t):
        """R_eff(t) ≥ 0: Efficiency penalty (FLOPs, latency)"""
        # Computational cost penalty
        return 0.05 + 0.05 * np.log(1 + t)
    
    def calibrated_probability(self, x, t, base_prob=0.8):
        """P(H|E,β,t) ∈ [0,1]: Calibrated probability with bias correction"""
        # Platt scaling with bias: P' = σ(logit(P) + log(β))
        logit_p = np.log(base_prob / (1 - base_prob))
        adjusted_logit = logit_p + np.log(self.beta)
        prob_adjusted = 1 / (1 + np.exp(-adjusted_logit))
        return np.clip(prob_adjusted, 0, 1)
    
    def compute_psi_single(self, x, t):
        """Compute Ψ(x) for a single time step"""
        S = self.symbolic_accuracy(x, t)
        N = self.neural_accuracy(x, t)
        alpha = self.adaptive_weight(t)
        
        # Hybrid output
        hybrid = alpha * S + (1 - alpha) * N
        
        # Regularization term
        R_cog = self.cognitive_penalty(t)
        R_eff = self.efficiency_penalty(t)
        reg_term = np.exp(-(self.lambda1 * R_cog + self.lambda2 * R_eff))
        
        # Probability term
        prob_term = self.calibrated_probability(x, t)
        
        # Final functional value
        psi = hybrid * reg_term * prob_term
        
        return {
            'psi': psi,
            'S': S, 'N': N, 'alpha': alpha,
            'hybrid': hybrid,
            'R_cog': R_cog, 'R_eff': R_eff, 'reg_term': reg_term,
            'prob_term': prob_term
        }
    
    def compute_psi_temporal(self, x, time_points):
        """Compute Ψ(x) averaged over multiple time points"""
        results = []
        for t in time_points:
            results.append(self.compute_psi_single(x, t))
        
        # Average over time
        psi_avg = np.mean([r['psi'] for r in results])
        
        return psi_avg, results

def numerical_example():
    """Reproduce the numerical example from the specification"""
    print("=== Numerical Example: Single Tracking Step ===")
    
    # Given values from specification
    S_val = 0.67
    N_val = 0.87
    alpha_val = 0.4
    R_cog = 0.17
    R_eff = 0.11
    lambda1 = 0.6
    lambda2 = 0.4
    P_base = 0.81
    beta = 1.2
    
    print(f"Step 1: Outputs - S(x)={S_val}, N(x)={N_val}")
    
    # Step 2: Hybrid
    O_hybrid = alpha_val * S_val + (1 - alpha_val) * N_val
    print(f"Step 2: Hybrid - α={alpha_val}, O_hybrid={O_hybrid:.3f}")
    
    # Step 3: Penalties
    P_total = lambda1 * R_cog + lambda2 * R_eff
    exp_term = np.exp(-P_total)
    print(f"Step 3: Penalties - R_cog={R_cog}, R_eff={R_eff}")
    print(f"         λ₁={lambda1}, λ₂={lambda2}, P_total={P_total:.3f}, exp≈{exp_term:.3f}")
    
    # Step 4: Probability
    logit_p = np.log(P_base / (1 - P_base))
    P_adj = 1 / (1 + np.exp(-(logit_p + np.log(beta))))
    print(f"Step 4: Probability - P={P_base}, β={beta}, P_adj≈{P_adj:.3f}")
    
    # Step 5: Final Ψ(x)
    psi_final = O_hybrid * exp_term * P_adj
    print(f"Step 5: Ψ(x) ≈ {O_hybrid:.3f} × {exp_term:.3f} × {P_adj:.3f} ≈ {psi_final:.3f}")
    
    # Step 6: Interpretation
    print(f"Step 6: Interpret - Ψ(x) ≈ {psi_final:.2f} indicates {'high' if psi_final > 0.6 else 'moderate'} responsiveness")
    
    return psi_final

def demonstrate_functional():
    """Demonstrate the functional with realistic parameters"""
    print("\n=== Functional Demonstration ===")
    
    functional = HybridAccuracyFunctional()
    
    # Single step evaluation
    x_test = 0.5
    t_test = 1.0
    
    result = functional.compute_psi_single(x_test, t_test)
    
    print(f"Input: x={x_test}, t={t_test}")
    print(f"S(x,t) = {result['S']:.3f}")
    print(f"N(x,t) = {result['N']:.3f}")
    print(f"α(t) = {result['alpha']:.3f}")
    print(f"Hybrid = {result['hybrid']:.3f}")
    print(f"R_cog = {result['R_cog']:.3f}, R_eff = {result['R_eff']:.3f}")
    print(f"Regularization = {result['reg_term']:.3f}")
    print(f"Probability = {result['prob_term']:.3f}")
    print(f"Ψ(x) = {result['psi']:.3f}")
    
    # Temporal averaging
    time_points = np.linspace(0.1, 2.0, 10)
    psi_avg, _ = functional.compute_psi_temporal(x_test, time_points)
    print(f"\nTemporal average Ψ(x) = {psi_avg:.3f}")

def visualize_functional():
    """Create visualization of the functional behavior"""
    functional = HybridAccuracyFunctional()
    
    # Parameter space exploration
    x_range = np.linspace(-1, 1, 50)
    t_range = np.linspace(0.1, 2.0, 30)
    
    X, T = np.meshgrid(x_range, t_range)
    PSI = np.zeros_like(X)
    
    for i in range(len(t_range)):
        for j in range(len(x_range)):
            result = functional.compute_psi_single(X[i,j], T[i,j])
            PSI[i,j] = result['psi']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 2D heatmap
    im1 = axes[0,0].contourf(X, T, PSI, levels=20, cmap='viridis')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('t')
    axes[0,0].set_title('Ψ(x,t) Heatmap')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Cross-sections
    t_fixed = 1.0
    psi_x = [functional.compute_psi_single(x, t_fixed)['psi'] for x in x_range]
    axes[0,1].plot(x_range, psi_x)
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('Ψ(x)')
    axes[0,1].set_title(f'Ψ(x) at t={t_fixed}')
    axes[0,1].grid(True)
    
    x_fixed = 0.0
    psi_t = [functional.compute_psi_single(x_fixed, t)['psi'] for t in t_range]
    axes[1,0].plot(t_range, psi_t)
    axes[1,0].set_xlabel('t')
    axes[1,0].set_ylabel('Ψ(x)')
    axes[1,0].set_title(f'Ψ(x) at x={x_fixed}')
    axes[1,0].grid(True)
    
    # Component analysis
    t_test = 1.0
    components = {'S': [], 'N': [], 'alpha': [], 'hybrid': []}
    for x in x_range:
        result = functional.compute_psi_single(x, t_test)
        for key in components:
            components[key].append(result[key])
    
    axes[1,1].plot(x_range, components['S'], label='S(x)', linestyle='-')
    axes[1,1].plot(x_range, components['N'], label='N(x)', linestyle='--')
    axes[1,1].plot(x_range, components['alpha'], label='α(x)', linestyle=':')
    axes[1,1].plot(x_range, components['hybrid'], label='Hybrid', linestyle='-.')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('Component Values')
    axes[1,1].set_title('Component Analysis')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/workspace/hybrid_functional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return PSI

if __name__ == "__main__":
    # Run numerical example
    psi_example = numerical_example()
    
    # Demonstrate functional
    demonstrate_functional()
    
    # Create visualization
    print("\n=== Creating Visualization ===")
    psi_grid = visualize_functional()
    print("Visualization saved as 'hybrid_functional_analysis.png'")