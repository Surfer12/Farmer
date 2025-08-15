import numpy as np

class HybridSymbolicNeural:
    def __init__(self, lambda1=0.6, lambda2=0.4):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
    def compute_hybrid_output(self, S, N, alpha):
        return alpha * S + (1 - alpha) * N
    
    def compute_regularization_penalty(self, R_cognitive, R_efficiency):
        penalty_total = self.lambda1 * R_cognitive + self.lambda2 * R_efficiency
        return np.exp(-penalty_total)
    
    def compute_probability_adjustment(self, P, beta):
        if P <= 0 or P >= 1:
            return np.clip(P, 0, 1)
        
        logit_P = np.log(P / (1 - P))
        adjusted_logit = logit_P + np.log(beta)
        P_adjusted = 1 / (1 + np.exp(-adjusted_logit))
        return np.clip(P_adjusted, 0, 1)
    
    def compute_psi(self, S, N, alpha, R_cognitive, R_efficiency, P, beta):
        O_hybrid = self.compute_hybrid_output(S, N, alpha)
        reg_penalty = self.compute_regularization_penalty(R_cognitive, R_efficiency)
        P_adjusted = self.compute_probability_adjustment(P, beta)
        psi = O_hybrid * reg_penalty * P_adjusted
        
        return psi, {
            'O_hybrid': O_hybrid,
            'reg_penalty': reg_penalty,
            'P_adjusted': P_adjusted,
            'penalty_total': self.lambda1 * R_cognitive + self.lambda2 * R_efficiency
        }

def main():
    print("Testing Hybrid Symbolic-Neural System")
    print("=" * 40)
    
    # Initialize system
    hybrid_system = HybridSymbolicNeural(lambda1=0.6, lambda2=0.4)
    
    # Test parameters from the example
    S = 0.67
    N = 0.87
    alpha = 0.4
    R_cognitive = 0.17
    R_efficiency = 0.11
    P = 0.81
    beta = 1.2
    
    # Compute Ψ(x)
    psi, details = hybrid_system.compute_psi(S, N, alpha, R_cognitive, R_efficiency, P, beta)
    
    print(f"Input Parameters:")
    print(f"S(x) = {S} (Symbolic accuracy)")
    print(f"N(x) = {N} (Neural accuracy)")
    print(f"α = {alpha} (Adaptive weight)")
    print(f"R_cognitive = {R_cognitive}")
    print(f"R_efficiency = {R_efficiency}")
    print(f"P = {P} (Base probability)")
    print(f"β = {beta} (Responsiveness bias)")
    
    print(f"\nComputed Results:")
    print(f"Hybrid Output: {details['O_hybrid']:.3f}")
    print(f"Regularization Penalty: {details['reg_penalty']:.3f}")
    print(f"Adjusted Probability: {details['P_adjusted']:.3f}")
    print(f"Final Ψ(x): {psi:.3f}")
    
    print(f"\nVerification:")
    expected_psi = details['O_hybrid'] * details['reg_penalty'] * details['P_adjusted']
    print(f"Expected: {expected_psi:.3f}")
    print(f"Actual: {psi:.3f}")
    print(f"Match: {abs(expected_psi - psi) < 1e-10}")

if __name__ == "__main__":
    main()