import math
import random

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
        return math.exp(-penalty_total)
    
    def compute_probability_adjustment(self, P: float, beta: float) -> float:
        """Compute probability adjustment with bias β"""
        # Logit shift: P' = σ(logit(P) + log(β))
        if P <= 0 or P >= 1:
            return max(0, min(1, P))
        
        logit_P = math.log(P / (1 - P))
        adjusted_logit = logit_P + math.log(beta)
        P_adjusted = 1 / (1 + math.exp(-adjusted_logit))
        return max(0, min(1, P_adjusted))
    
    def compute_psi(self, S: float, N: float, alpha: float, 
                    R_cognitive: float, R_efficiency: float, 
                    P: float, beta: float):
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
    
    def compute_psi_time_series(self, S_series, N_series, alpha_series, 
                               R_cog_series, R_eff_series, P_series, beta: float):
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
        psi_avg = sum(psi_values) / len(psi_values)
        
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
    reg_exp = math.exp(-penalty_total)
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

def time_series_example():
    """Demonstrate time series integration"""
    print("\n=== Time Series Integration Example ===")
    
    # Simulate time series data
    T = 10
    time_steps = list(range(T))
    
    # Generate realistic time series
    S_series = [0.6 + 0.1 * math.sin(2 * math.pi * t / T) + 0.05 * random.gauss(0, 1) for t in time_steps]
    N_series = [0.8 + 0.1 * math.cos(2 * math.pi * t / T) + 0.05 * random.gauss(0, 1) for t in time_steps]
    alpha_series = [0.3 + 0.4 * (1 + math.sin(2 * math.pi * t / T)) / 2 for t in time_steps]
    R_cog_series = [0.2 + 0.1 * random.random() for t in time_steps]
    R_eff_series = [0.15 + 0.1 * random.random() for t in time_steps]
    P_series = [0.75 + 0.1 * random.random() for t in time_steps]
    
    # Clip values to valid ranges
    S_series = [max(0, min(1, s)) for s in S_series]
    N_series = [max(0, min(1, n)) for n in N_series]
    alpha_series = [max(0, min(1, a)) for a in alpha_series]
    R_cog_series = [max(0, r) for r in R_cog_series]
    R_eff_series = [max(0, r) for r in R_eff_series]
    P_series = [max(0, min(1, p)) for p in P_series]
    
    hybrid_system = HybridSymbolicNeural()
    psi_avg, psi_series = hybrid_system.compute_psi_time_series(
        S_series, N_series, alpha_series, R_cog_series, R_eff_series, P_series, beta=1.2
    )
    
    print(f"Time-averaged Ψ(x) = {psi_avg:.3f}")
    
    # Display time series data
    print(f"\nTime Series Data (first 5 steps):")
    print(f"Time Step | S(x)    | N(x)    | α       | R_cog   | R_eff   | P       | Ψ(x)")
    print(f"----------|---------|---------|---------|---------|---------|---------|---------")
    for i in range(min(5, T)):
        print(f"{i:9d} | {S_series[i]:7.3f} | {N_series[i]:7.3f} | {alpha_series[i]:7.3f} | "
              f"{R_cog_series[i]:7.3f} | {R_eff_series[i]:7.3f} | {P_series[i]:7.3f} | {psi_series[i]:7.3f}")
    
    return psi_avg, psi_series

def demonstrate_parameter_sensitivity():
    """Demonstrate how Ψ(x) changes with different parameters"""
    print("\n=== Parameter Sensitivity Analysis ===")
    
    hybrid_system = HybridSymbolicNeural()
    
    # Base parameters
    S_base, N_base = 0.7, 0.8
    alpha_base = 0.5
    R_cog_base, R_eff_base = 0.2, 0.15
    P_base, beta_base = 0.8, 1.1
    
    # Test different alpha values
    print("Effect of α (Adaptive Weight):")
    print("α     | Ψ(x)  | Hybrid Output")
    print("------|-------|---------------")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        psi, details = hybrid_system.compute_psi(
            S_base, N_base, alpha, R_cog_base, R_eff_base, P_base, beta_base
        )
        print(f"{alpha:5.2f} | {psi:5.3f} | {details['O_hybrid']:13.3f}")
    
    # Test different regularization weights
    print(f"\nEffect of λ₁, λ₂ (Regularization Weights):")
    print("λ₁    | λ₂    | Ψ(x)  | Reg Penalty")
    print("------|-------|-------|-------------")
    for lambda1 in [0.3, 0.5, 0.7, 0.9]:
        for lambda2 in [0.7, 0.5, 0.3, 0.1]:
            if abs(lambda1 + lambda2 - 1.0) < 0.01:  # Keep sum ≈ 1
                test_system = HybridSymbolicNeural(lambda1, lambda2)
                psi, details = test_system.compute_psi(
                    S_base, N_base, alpha_base, R_cog_base, R_eff_base, P_base, beta_base
                )
                print(f"{lambda1:5.1f} | {lambda2:5.1f} | {psi:5.3f} | {details['reg_penalty']:11.3f}")

def demonstrate_application_scenarios():
    """Demonstrate different application scenarios"""
    print("\n=== Application Scenarios ===")
    
    scenarios = [
        {
            'name': 'Multi-Pendulum System',
            'description': 'Chaotic dynamics with RK4 + LSTM',
            'S': 0.65, 'N': 0.85, 'alpha': 0.3, 'R_cog': 0.25, 'R_eff': 0.18, 'P': 0.75, 'beta': 1.1
        },
        {
            'name': 'Real-Time Monitoring',
            'description': 'Physiological tracking with ML analysis',
            'S': 0.72, 'N': 0.88, 'alpha': 0.4, 'R_cog': 0.15, 'R_eff': 0.12, 'P': 0.82, 'beta': 1.2
        },
        {
            'name': 'Collaborative AI',
            'description': 'Tool quality + community impact',
            'S': 0.78, 'N': 0.82, 'alpha': 0.6, 'R_cog': 0.12, 'R_eff': 0.08, 'P': 0.79, 'beta': 1.3
        }
    ]
    
    hybrid_system = HybridSymbolicNeural()
    
    print("Scenario                | Ψ(x)  | Hybrid | Reg Penalty | P Adjusted")
    print("-----------------------|--------|--------|-------------|------------")
    
    for scenario in scenarios:
        psi, details = hybrid_system.compute_psi(
            scenario['S'], scenario['N'], scenario['alpha'],
            scenario['R_cog'], scenario['R_eff'], scenario['P'], scenario['beta']
        )
        
        print(f"{scenario['name']:<23} | {psi:5.3f} | {details['O_hybrid']:6.3f} | "
              f"{details['reg_penalty']:11.3f} | {details['P_adjusted']:10.3f}")

if __name__ == "__main__":
    # Run demonstrations
    print("Hybrid Symbolic-Neural Accuracy Functional Implementation")
    print("Pure Python Version (No External Dependencies)")
    print("=" * 70)
    
    # Numerical example
    psi_single, details_single = demonstrate_numerical_example()
    
    # Open source contribution example
    psi_contribution = demonstrate_open_source_contribution()
    
    # Time series example
    psi_avg, psi_series = time_series_example()
    
    # Parameter sensitivity
    demonstrate_parameter_sensitivity()
    
    # Application scenarios
    demonstrate_application_scenarios()
    
    print("\n" + "=" * 70)
    print("Summary of Results:")
    print(f"Single step Ψ(x): {psi_single:.3f}")
    print(f"Contribution Ψ(x): {psi_contribution:.3f}")
    print(f"Time-averaged Ψ(x): {psi_avg:.3f}")
    print("\nFramework successfully demonstrates:")
    print("✓ Hybrid symbolic-neural integration")
    print("✓ Adaptive weighting with α(t)")
    print("✓ Regularization penalties (cognitive + efficiency)")
    print("✓ Probability calibration with bias β")
    print("✓ Time series integration")
    print("✓ Parameter sensitivity analysis")
    print("✓ Multiple application scenarios")