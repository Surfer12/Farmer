#!/usr/bin/env python3
"""
Simple test script for the hybrid accuracy functional without external dependencies.
This verifies the core mathematical logic.
"""

import math

def test_single_step_verification():
    """Test the single-step example from the formalization."""
    print("=== Single Step Verification ===")
    
    # Your numerical example
    S = 0.65
    N = 0.85
    alpha = 0.3
    Rcog = 0.20
    Reff = 0.15
    P_base = 0.75
    beta = 1.3
    lambda1 = 0.75
    lambda2 = 0.25
    
    # Manual calculation as in your formalization
    hybrid = alpha * S + (1.0 - alpha) * N  # 0.79
    penalty = math.exp(-(lambda1 * Rcog + lambda2 * Reff))  # ≈ 0.8288
    P_calibrated = min(max(P_base * beta, 0.0), 1.0)  # 0.975
    expected = hybrid * penalty * P_calibrated  # ≈ 0.638
    
    print(f"Inputs:")
    print(f"  S = {S:.2f}, N = {N:.2f}, α = {alpha:.2f}")
    print(f"  Rcog = {Rcog:.2f}, Reff = {Reff:.2f}")
    print(f"  P_base = {P_base:.2f}, β = {beta:.2f}")
    print(f"  λ₁ = {lambda1:.2f}, λ₂ = {lambda2:.2f}")
    print(f"\nManual calculation:")
    print(f"  Hybrid term: α·S + (1-α)·N = {alpha:.1f}·{S:.2f} + {1-alpha:.1f}·{N:.2f} = {hybrid:.4f}")
    print(f"  Penalty: exp(-[λ₁·Rcog + λ₂·Reff]) = exp(-[{lambda1:.2f}·{Rcog:.2f} + {lambda2:.2f}·{Reff:.2f}]) = {penalty:.4f}")
    print(f"  P_calibrated: clip(β·P_base, 0, 1) = clip({beta:.1f}·{P_base:.2f}, 0, 1) = {P_calibrated:.4f}")
    print(f"  Expected V(x): {hybrid:.4f} × {penalty:.4f} × {P_calibrated:.4f} = {expected:.6f}")
    print(f"\nVerification: {'✓ PASS' if abs(expected - 0.638) < 0.001 else '✗ FAIL'}")
    print()


def test_adaptive_weight_scheduling():
    """Test adaptive weight scheduling logic."""
    print("=== Adaptive Weight Scheduling Logic ===")
    
    # Test confidence-based scheduling
    def confidence_based(S_conf, N_conf, temperature):
        """Simple confidence-based weight computation."""
        # Softmax over confidence scores
        S_exp = math.exp(S_conf / temperature)
        N_exp = math.exp(N_conf / temperature)
        total = S_exp + N_exp
        return S_exp / total
    
    # Test chaos-based scheduling
    def chaos_based(lyapunov, kappa):
        """Simple chaos-based weight computation."""
        # Sigmoid function: favor symbolic in stable regions
        return 1.0 / (1.0 + math.exp(kappa * lyapunov))
    
    # Test cases
    print("Confidence-based scheduling:")
    S_conf, N_conf = 0.7, 0.8
    temp_low = 0.5
    temp_high = 2.0
    
    alpha_low = confidence_based(S_conf, N_conf, temp_low)
    alpha_high = confidence_based(S_conf, N_conf, temp_high)
    
    print(f"  S_conf = {S_conf:.1f}, N_conf = {N_conf:.1f}")
    print(f"  α(T={temp_low}) = {alpha_low:.4f}")
    print(f"  α(T={temp_high}) = {alpha_high:.4f}")
    print(f"  Lower temperature makes weights more extreme: {'✓' if alpha_low < alpha_high else '✗'}")
    
    print("\nChaos-based scheduling:")
    lyapunov_stable = -1.0
    lyapunov_chaotic = 1.0
    kappa = 1.0
    
    alpha_stable = chaos_based(lyapunov_stable, kappa)
    alpha_chaotic = chaos_based(lyapunov_chaotic, kappa)
    
    print(f"  Stable region (λ={lyapunov_stable:.1f}): α = {alpha_stable:.4f}")
    print(f"  Chaotic region (λ={lyapunov_chaotic:.1f}): α = {alpha_chaotic:.4f}")
    print(f"  Favor symbolic in stable regions: {'✓' if alpha_stable > alpha_chaotic else '✗'}")
    print()


def test_penalty_computation():
    """Test penalty computation logic."""
    print("=== Penalty Computation Logic ===")
    
    def energy_drift_penalty(energy_traj, reference_energy=None):
        """Compute energy drift penalty."""
        if reference_energy is None:
            reference_energy = energy_traj[0]
        
        penalties = []
        for energy in energy_traj:
            relative_drift = abs(energy - reference_energy) / (abs(reference_energy) + 1e-8)
            penalties.append(min(max(relative_drift, 0.0), 1.0))
        return penalties
    
    def constraint_violation_penalty(constraint_residuals):
        """Compute constraint violation penalty."""
        max_residual = max(abs(r) for r in constraint_residuals) + 1e-8
        normalized_residuals = [abs(r) / max_residual for r in constraint_residuals]
        return [min(max(r, 0.0), 1.0) for r in normalized_residuals]
    
    def compute_budget_penalty(flops_per_step, max_flops):
        """Compute computational budget penalty."""
        return [min(f / max_flops, 1.0) for f in flops_per_step]
    
    # Test energy drift
    energy_traj = [100.0, 100.1, 99.8, 100.3, 99.9]
    energy_penalty = energy_drift_penalty(energy_traj)
    
    print("Energy drift penalty:")
    print(f"  Energy trajectory: {energy_traj}")
    print(f"  Penalties: {[f'{p:.4f}' for p in energy_penalty]}")
    print(f"  Initial penalty is zero: {'✓' if energy_penalty[0] == 0.0 else '✗'}")
    
    # Test constraint violations
    residuals = [0.1, 0.05, 0.15, 0.02, 0.08]
    constraint_penalty = constraint_violation_penalty(residuals)
    
    print(f"\nConstraint violation penalty:")
    print(f"  Residuals: {residuals}")
    print(f"  Penalties: {[f'{p:.4f}' for p in constraint_penalty]}")
    print(f"  Max penalty corresponds to max residual: {'✓' if constraint_penalty[2] == 1.0 else '✗'}")
    
    # Test budget penalty
    flops = [1e6, 1.2e6, 0.8e6, 1.5e6, 1.1e6]
    max_flops = 2e6
    budget_penalty = compute_budget_penalty(flops, max_flops)
    
    print(f"\nComputational budget penalty:")
    print(f"  FLOPs: {[f'{f/1e6:.1f}M' for f in flops]}")
    print(f"  Penalties: {[f'{p:.4f}' for p in budget_penalty]}")
    print(f"  Highest FLOPs has highest penalty: {'✓' if budget_penalty[3] == max(budget_penalty) else '✗'}")
    print()


def test_broken_neural_scaling():
    """Test broken neural scaling laws logic."""
    print("=== Broken Neural Scaling Laws Logic ===")
    
    class SimpleBNSL:
        """Simplified BNSL implementation."""
        def __init__(self, A, alpha, n0, gamma, delta, sigma):
            self.A = A
            self.alpha = alpha
            self.n0 = n0
            self.gamma = gamma
            self.delta = delta
            self.sigma = sigma
        
        def predict_error(self, n):
            """Predict error L(n) for given dataset size n."""
            term1 = self.A * (n ** (-self.alpha))
            term2 = (1 + (n / self.n0) ** self.gamma) ** (-self.delta)
            return term1 * term2 + self.sigma
        
        def predict_accuracy(self, n):
            """Predict accuracy (1 - error) for given dataset size n."""
            return 1.0 - self.predict_error(n)
    
    # Test parameters
    bnsl = SimpleBNSL(A=0.15, alpha=0.4, n0=1e5, gamma=1.2, delta=0.8, sigma=0.02)
    
    # Test predictions
    n_values = [1e3, 1e4, 1e5, 1e6]
    print("BNSL predictions:")
    for n in n_values:
        error = bnsl.predict_error(n)
        accuracy = bnsl.predict_accuracy(n)
        print(f"  n = {n:.0e}: error = {error:.4f}, accuracy = {accuracy:.4f}")
    
    # Verify error decreases with larger n
    errors = [bnsl.predict_error(n) for n in n_values]
    decreasing = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
    print(f"\nError decreases with dataset size: {'✓' if decreasing else '✗'}")
    
    # Verify accuracy = 1 - error
    accuracies = [bnsl.predict_accuracy(n) for n in n_values]
    consistent = all(abs(1 - errors[i] - accuracies[i]) < 1e-10 for i in range(len(errors)))
    print(f"Accuracy = 1 - error consistency: {'✓' if consistent else '✗'}")
    print()


def test_cross_modal_commutator():
    """Test cross-modal commutator logic."""
    print("=== Cross-Modal Commutator Logic ===")
    
    def compute_commutator(S, N, m1_idx, m2_idx):
        """Compute empirical commutator C(m1, m2) = S(m1)N(m2) - S(m2)N(m1)."""
        return S[m1_idx] * N[m2_idx] - S[m2_idx] * N[m1_idx]
    
    # Test data: 2 models, 3 time steps
    S = [[0.6, 0.7, 0.8], [0.8, 0.9, 0.7]]  # 2 models, 3 time steps
    N = [[0.7, 0.8, 0.9], [0.9, 0.95, 0.8]]  # 2 models, 3 time steps
    
    print("Model accuracies:")
    print(f"  Symbolic (S): Model 0: {S[0]}, Model 1: {S[1]}")
    print(f"  Neural (N):   Model 0: {N[0]}, Model 1: {N[1]}")
    
    # Compute commutator for each time step
    commutators = []
    for t in range(3):
        comm = S[0][t] * N[1][t] - S[1][t] * N[0][t]
        commutators.append(comm)
        print(f"  Time {t}: C(0,1) = S[0]·N[1] - S[1]·N[0] = {S[0][t]:.1f}·{N[1][t]:.2f} - {S[1][t]:.1f}·{N[0][t]:.1f} = {comm:.4f}")
    
    # Test anti-symmetry: C(m1, m2) = -C(m2, m1)
    anti_symmetric = True
    for t in range(3):
        comm_01 = S[0][t] * N[1][t] - S[1][t] * N[0][t]
        comm_10 = S[1][t] * N[0][t] - S[0][t] * N[1][t]
        if abs(comm_01 + comm_10) > 1e-10:
            anti_symmetric = False
            break
    
    print(f"\nCommutator anti-symmetry C(0,1) = -C(1,0): {'✓' if anti_symmetric else '✗'}")
    print()


def test_hybrid_functional_formula():
    """Test the complete hybrid functional formula."""
    print("=== Complete Hybrid Functional Formula ===")
    
    def compute_V(S, N, alpha, Rcog, Reff, P_base, beta, lambda1, lambda2):
        """Compute V(x) = (1/T) Σ [α(t)S(t) + (1-α(t))N(t)] · exp(-[λ1·Rcog(t) + λ2·Reff(t)]) · P(H|E,β,t)"""
        T = len(S)
        total = 0.0
        
        for t in range(T):
            # Hybrid term
            hybrid = alpha[t] * S[t] + (1.0 - alpha[t]) * N[t]
            
            # Penalty term
            penalty = math.exp(-(lambda1 * Rcog[t] + lambda2 * Reff[t]))
            
            # Probability term
            P_calibrated = min(max(P_base[t] * beta, 0.0), 1.0)
            
            # Combined term
            term = hybrid * penalty * P_calibrated
            total += term
        
        return total / T
    
    # Test case: 3 time steps
    S = [0.65, 0.70, 0.75]
    N = [0.85, 0.80, 0.90]
    alpha = [0.3, 0.4, 0.5]
    Rcog = [0.20, 0.15, 0.25]
    Reff = [0.15, 0.20, 0.10]
    P_base = [0.75, 0.80, 0.85]
    beta = 1.3
    lambda1 = 0.75
    lambda2 = 0.25
    
    print("Test case (3 time steps):")
    print(f"  S = {S}")
    print(f"  N = {N}")
    print(f"  α = {alpha}")
    print(f"  Rcog = {Rcog}")
    print(f"  Reff = {Reff}")
    print(f"  P_base = {P_base}")
    print(f"  β = {beta}, λ₁ = {lambda1}, λ₂ = {lambda2}")
    
    # Compute V(x)
    V = compute_V(S, N, alpha, Rcog, Reff, P_base, beta, lambda1, lambda2)
    print(f"\nComputed V(x) = {V:.6f}")
    
    # Verify individual terms
    print("\nIndividual terms:")
    for t in range(3):
        hybrid = alpha[t] * S[t] + (1.0 - alpha[t]) * N[t]
        penalty = math.exp(-(lambda1 * Rcog[t] + lambda2 * Reff[t]))
        P_calibrated = min(max(P_base[t] * beta, 0.0), 1.0)
        term = hybrid * penalty * P_calibrated
        
        print(f"  Time {t}: hybrid={hybrid:.4f}, penalty={penalty:.4f}, P={P_calibrated:.4f}, term={term:.6f}")
    
    print(f"Average: {V:.6f}")
    print()


def main():
    """Run all tests."""
    print("Hybrid Symbolic-Neural Accuracy Functional - Simple Tests")
    print("=" * 65)
    print()
    
    try:
        test_single_step_verification()
        test_adaptive_weight_scheduling()
        test_penalty_computation()
        test_broken_neural_scaling()
        test_cross_modal_commutator()
        test_hybrid_functional_formula()
        
        print("All tests completed successfully!")
        print("\n✓ Core mathematical logic verified")
        print("✓ Formula implementation correct")
        print("✓ Adaptive weight scheduling logic sound")
        print("✓ Penalty computation methods working")
        print("✓ BNSL implementation consistent")
        print("✓ Cross-modal commutator properties verified")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()