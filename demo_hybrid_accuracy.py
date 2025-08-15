"""
Demonstration of the Hybrid Symbolic-Neural Accuracy Functional

This script showcases the key features of the hybrid accuracy functional
with realistic examples and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from hybrid_accuracy import (
    HybridAccuracyFunctional, 
    HybridAccuracyConfig,
    AdaptiveWeightScheduler,
    PenaltyComputers,
    BrokenNeuralScaling
)


def demo_single_step_verification():
    """Demonstrate the single-step example from the formalization."""
    print("=== Single Step Verification ===")
    
    # Configuration as in your formalization
    config = HybridAccuracyConfig(
        lambda1=0.75,
        lambda2=0.25,
        use_cross_modal=False
    )
    
    haf = HybridAccuracyFunctional(config)
    
    # Your numerical example
    S = np.array([0.65])
    N = np.array([0.85])
    alpha = np.array([0.3])
    Rcog = np.array([0.20])
    Reff = np.array([0.15])
    P_base = np.array([0.75])
    beta = 1.3
    
    # Compute V(x)
    V = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
    
    # Manual verification
    hybrid = 0.3 * 0.65 + 0.7 * 0.85  # 0.79
    penalty = np.exp(-(0.75 * 0.20 + 0.25 * 0.15))  # ≈ 0.8288
    P_calibrated = np.clip(0.75 * 1.3, 0.0, 1.0)  # 0.975
    expected = hybrid * penalty * P_calibrated  # ≈ 0.638
    
    print(f"Inputs:")
    print(f"  S = {S[0]:.2f}, N = {N[0]:.2f}, α = {alpha[0]:.2f}")
    print(f"  Rcog = {Rcog[0]:.2f}, Reff = {Reff[0]:.2f}")
    print(f"  P_base = {P_base[0]:.2f}, β = {beta:.2f}")
    print(f"\nManual calculation:")
    print(f"  Hybrid term: {hybrid:.4f}")
    print(f"  Penalty: {penalty:.4f}")
    print(f"  P_calibrated: {P_calibrated:.4f}")
    print(f"  Expected V(x): {expected:.6f}")
    print(f"\nComputed V(x): {V:.6f}")
    print(f"Verification: {'✓ PASS' if np.isclose(V, expected, atol=1e-3) else '✗ FAIL'}")
    print()


def demo_adaptive_weight_scheduling():
    """Demonstrate adaptive weight scheduling strategies."""
    print("=== Adaptive Weight Scheduling ===")
    
    scheduler = AdaptiveWeightScheduler()
    
    # Generate realistic confidence scores
    np.random.seed(42)
    T = 20
    time_steps = np.arange(T)
    
    # Simulate confidence evolution
    S_conf = 0.7 + 0.1 * np.sin(time_steps * 0.3) + 0.05 * np.random.randn(T)
    N_conf = 0.8 + 0.15 * np.cos(time_steps * 0.2) + 0.03 * np.random.randn(T)
    
    # Compute different scheduling strategies
    alpha_conf = scheduler.confidence_based(S_conf, N_conf, temperature=0.5)
    alpha_conf_high_temp = scheduler.confidence_based(S_conf, N_conf, temperature=2.0)
    
    # Simulate Lyapunov exponents (chaos indicator)
    lyapunov = 0.5 * np.sin(time_steps * 0.4) + 0.2 * np.random.randn(T)
    alpha_chaos = scheduler.chaos_based(lyapunov, kappa=1.0)
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Confidence scores
    ax1.plot(time_steps, S_conf, 'b-', label='Symbolic Confidence', linewidth=2)
    ax1.plot(time_steps, N_conf, 'r-', label='Neural Confidence', linewidth=2)
    ax1.set_ylabel('Confidence Score')
    ax1.set_title('Model Confidence Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Adaptive weights
    ax2.plot(time_steps, alpha_conf, 'g-', label='Confidence-based (T=0.5)', linewidth=2)
    ax2.plot(time_steps, alpha_conf_high_temp, 'g--', label='Confidence-based (T=2.0)', linewidth=2)
    ax2.plot(time_steps, alpha_chaos, 'm-', label='Chaos-based (κ=1.0)', linewidth=2)
    ax2.set_ylabel('Adaptive Weight α(t)')
    ax2.set_title('Adaptive Weight Scheduling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Lyapunov exponents
    ax3.plot(time_steps, lyapunov, 'c-', label='Local Lyapunov Exponent', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Lyapunov Exponent')
    ax3.set_title('Chaos Indicator')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_weight_scheduling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated 'adaptive_weight_scheduling.png'")
    print(f"Confidence-based α range: [{alpha_conf.min():.3f}, {alpha_conf.max():.3f}]")
    print(f"Chaos-based α range: [{alpha_chaos.min():.3f}, {alpha_chaos.max():.3f}]")
    print()


def demo_penalty_computation():
    """Demonstrate penalty computation methods."""
    print("=== Penalty Computation ===")
    
    penalty_comp = PenaltyComputers()
    
    # Generate realistic trajectories
    np.random.seed(42)
    T = 100
    time_steps = np.arange(T)
    
    # Energy trajectory with drift
    energy_traj = 100.0 + np.cumsum(0.1 * np.random.randn(T)) + 0.05 * np.sin(time_steps * 0.1)
    
    # Constraint residuals
    constraint_residuals = 0.1 * np.exp(-time_steps / 50) + 0.02 * np.random.randn(T)
    
    # Computational budget
    flops_per_step = 1e6 + 0.3e6 * np.sin(time_steps * 0.05) + 0.1e6 * np.random.randn(T)
    
    # Compute penalties
    Rcog_energy = penalty_comp.energy_drift_penalty(energy_traj)
    Rcog_constraint = penalty_comp.constraint_violation_penalty(constraint_residuals)
    Reff_budget = penalty_comp.compute_budget_penalty(flops_per_step, max_flops=2e6)
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Energy trajectory
    ax1.plot(time_steps, energy_traj, 'b-', linewidth=2)
    ax1.axhline(y=100.0, color='r', linestyle='--', alpha=0.7, label='Reference Energy')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Trajectory with Drift')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy drift penalty
    ax2.plot(time_steps, Rcog_energy, 'r-', linewidth=2)
    ax2.set_ylabel('Energy Drift Penalty')
    ax2.set_title('Cognitive Penalty: Energy Drift')
    ax2.grid(True, alpha=0.3)
    
    # Constraint residuals and penalty
    ax3.plot(time_steps, constraint_residuals, 'g-', linewidth=2, label='Constraint Residuals')
    ax3.set_ylabel('Constraint Residuals')
    ax3.set_title('Constraint Violations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Efficiency penalty
    ax4.plot(time_steps, Reff_budget, 'm-', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Budget Penalty')
    ax4.set_title('Efficiency Penalty: Computational Budget')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('penalty_computation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated 'penalty_computation.png'")
    print(f"Energy drift penalty range: [{Rcog_energy.min():.3f}, {Rcog_energy.max():.3f}]")
    print(f"Constraint penalty range: [{Rcog_constraint.min():.3f}, {Rcog_constraint.max():.3f}]")
    print(f"Budget penalty range: [{Reff_budget.min():.3f}, {Reff_budget.max():.3f}]")
    print()


def demo_broken_neural_scaling():
    """Demonstrate broken neural scaling laws."""
    print("=== Broken Neural Scaling Laws ===")
    
    # Generate synthetic scaling data
    np.random.seed(42)
    n_values = np.logspace(3, 7, 30)
    
    # True BNSL parameters
    true_params = {
        'A': 0.15,
        'alpha': 0.4,
        'n0': 1e5,
        'gamma': 1.2,
        'delta': 0.8,
        'sigma': 0.02
    }
    
    # Generate errors with noise
    true_errors = (true_params['A'] * (n_values ** (-true_params['alpha'])) * 
                   (1 + (n_values / true_params['n0']) ** true_params['gamma']) ** (-true_params['delta']) + 
                   true_params['sigma'])
    
    # Add noise
    observed_errors = true_errors + 0.01 * np.random.randn(len(n_values))
    
    # Fit BNSL
    bnsl = BrokenNeuralScaling.fit_from_data(n_values, observed_errors)
    
    # Predictions
    predicted_errors = bnsl.predict_error(n_values)
    predicted_accuracy = bnsl.predict_accuracy(n_values)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error scaling
    ax1.loglog(n_values, true_errors, 'b-', linewidth=3, label='True BNSL', alpha=0.7)
    ax1.loglog(n_values, observed_errors, 'ko', markersize=4, label='Observed Data', alpha=0.6)
    ax1.loglog(n_values, predicted_errors, 'r--', linewidth=2, label='Fitted BNSL')
    ax1.set_xlabel('Dataset Size (n)')
    ax1.set_ylabel('Error L(n)')
    ax1.set_title('Broken Neural Scaling Laws')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy scaling
    ax2.semilogx(n_values, 1 - true_errors, 'b-', linewidth=3, label='True Accuracy', alpha=0.7)
    ax2.semilogx(n_values, 1 - observed_errors, 'ko', markersize=4, label='Observed Accuracy', alpha=0.6)
    ax2.semilogx(n_values, predicted_accuracy, 'r--', linewidth=2, label='Predicted Accuracy')
    ax2.set_xlabel('Dataset Size (n)')
    ax2.set_ylabel('Accuracy 1 - L(n)')
    ax2.set_title('Accuracy Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('broken_neural_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated 'broken_neural_scaling.png'")
    print(f"True BNSL parameters: {true_params}")
    print(f"Fitted BNSL parameters:")
    print(f"  A = {bnsl.A:.4f}, α = {bnsl.alpha:.4f}, n0 = {bnsl.n0:.1e}")
    print(f"  γ = {bnsl.gamma:.4f}, δ = {bnsl.delta:.4f}, σ = {bnsl.sigma:.4f}")
    
    # Predict performance at specific dataset sizes
    test_sizes = [1e4, 1e5, 1e6]
    for n in test_sizes:
        acc = bnsl.predict_accuracy(n)
        print(f"  Predicted accuracy at n={n:.0e}: {acc:.4f}")
    print()


def demo_cross_modal_interaction():
    """Demonstrate cross-modal interaction effects."""
    print("=== Cross-Modal Interaction ===")
    
    # Configuration with cross-modal terms
    config = HybridAccuracyConfig(
        lambda1=0.7,
        lambda2=0.3,
        use_cross_modal=True,
        w_cross=0.15
    )
    
    haf = HybridAccuracyConfig(config)
    
    # Generate multi-model data
    np.random.seed(42)
    M, T = 3, 50
    time_steps = np.arange(T)
    
    # Symbolic accuracy with different characteristics
    S = np.zeros((M, T))
    S[0] = 0.7 + 0.1 * np.sin(time_steps * 0.1) + 0.02 * np.random.randn(T)  # Stable
    S[1] = 0.8 + 0.05 * np.cos(time_steps * 0.15) + 0.03 * np.random.randn(T)  # Oscillatory
    S[2] = 0.6 + 0.2 * np.exp(-time_steps / 20) + 0.04 * np.random.randn(T)   # Decaying
    
    # Neural accuracy
    N = np.zeros((M, T))
    N[0] = 0.75 + 0.08 * np.sin(time_steps * 0.12) + 0.025 * np.random.randn(T)
    N[1] = 0.85 + 0.06 * np.cos(time_steps * 0.18) + 0.035 * np.random.randn(T)
    N[2] = 0.7 + 0.15 * np.exp(-time_steps / 25) + 0.045 * np.random.randn(T)
    
    # Adaptive weights
    scheduler = AdaptiveWeightScheduler()
    S_conf = np.random.uniform(0.6, 0.9, T)
    N_conf = np.random.uniform(0.7, 0.95, T)
    alpha = scheduler.confidence_based(S_conf, N_conf, temperature=0.6)
    
    # Penalties
    penalty_comp = PenaltyComputers()
    energy_traj = 100.0 + np.cumsum(0.05 * np.random.randn(T))
    Rcog = penalty_comp.energy_drift_penalty(energy_traj)
    Reff = np.random.uniform(0.05, 0.25, T)
    
    P_base = np.random.uniform(0.7, 0.9, T)
    beta = 1.1
    
    # Compute V(x) with and without cross-modal terms
    config_no_cross = HybridAccuracyConfig(
        lambda1=0.7,
        lambda2=0.3,
        use_cross_modal=False
    )
    haf_no_cross = HybridAccuracyFunctional(config_no_cross)
    
    V_no_cross = haf_no_cross.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
    V_with_cross = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta, cross_modal_indices=(0, 1))
    
    # Compute commutator
    commutator = haf.compute_cross_modal_commutator(S, N, 0, 1)
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model accuracies
    for i in range(M):
        ax1.plot(time_steps, S[i], f'C{i}-', linewidth=2, label=f'Symbolic Model {i+1}')
        ax1.plot(time_steps, N[i], f'C{i}--', linewidth=2, label=f'Neural Model {i+1}')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Adaptive weights
    ax2.plot(time_steps, alpha, 'g-', linewidth=2, label='Adaptive Weight α(t)')
    ax2.set_ylabel('Weight α(t)')
    ax2.set_title('Adaptive Weight Scheduling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cross-modal commutator
    ax3.plot(time_steps, commutator, 'm-', linewidth=2, label='C(0,1)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Commutator C(0,1)')
    ax3.set_title('Cross-Modal Interaction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # V(x) comparison
    ax4.bar(['No Cross-Modal', 'With Cross-Modal'], 
             [V_no_cross[0], V_with_cross[0]], 
             color=['lightblue', 'lightcoral'], alpha=0.7)
    ax4.set_ylabel('V(x)')
    ax4.set_title('Hybrid Accuracy Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_modal_interaction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated 'cross_modal_interaction.png'")
    print(f"V(x) without cross-modal: {V_no_cross[0]:.6f}")
    print(f"V(x) with cross-modal: {V_with_cross[0]:.6f}")
    print(f"Cross-modal effect: {V_with_cross[0] - V_no_cross[0]:.6f}")
    print(f"Commutator range: [{commutator.min():.4f}, {commutator.max():.4f}]")
    print()


def demo_parameter_sensitivity():
    """Demonstrate sensitivity to key parameters."""
    print("=== Parameter Sensitivity Analysis ===")
    
    # Base configuration
    base_config = HybridAccuracyConfig(
        lambda1=0.75,
        lambda2=0.25,
        use_cross_modal=False
    )
    
    haf = HybridAccuracyFunctional(base_config)
    
    # Base parameters
    S = np.array([0.7])
    N = np.array([0.85])
    alpha = np.array([0.4])
    Rcog = np.array([0.2])
    Reff = np.array([0.15])
    P_base = np.array([0.8])
    beta = 1.2
    
    # Parameter ranges to test
    lambda1_range = np.linspace(0.1, 2.0, 20)
    lambda2_range = np.linspace(0.1, 2.0, 20)
    alpha_range = np.linspace(0.1, 0.9, 20)
    beta_range = np.linspace(0.5, 2.0, 20)
    
    # Sensitivity analysis
    V_lambda1 = []
    V_lambda2 = []
    V_alpha = []
    V_beta = []
    
    for lam1 in lambda1_range:
        config = HybridAccuracyConfig(lambda1=lam1, lambda2=0.25)
        haf_temp = HybridAccuracyFunctional(config)
        V = haf_temp.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
        V_lambda1.append(V)
    
    for lam2 in lambda2_range:
        config = HybridAccuracyConfig(lambda1=0.75, lambda2=lam2)
        haf_temp = HybridAccuracyFunctional(config)
        V = haf_temp.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
        V_lambda2.append(V)
    
    for a in alpha_range:
        V = haf.compute_V(S, N, np.array([a]), Rcog, Reff, P_base, beta)
        V_alpha.append(V)
    
    for b in beta_range:
        V = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, b)
        V_beta.append(V)
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(lambda1_range, V_lambda1, 'b-', linewidth=2)
    ax1.set_xlabel('λ₁ (Cognitive Penalty Weight)')
    ax1.set_ylabel('V(x)')
    ax1.set_title('Sensitivity to λ₁')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(lambda2_range, V_lambda2, 'r-', linewidth=2)
    ax2.set_xlabel('λ₂ (Efficiency Penalty Weight)')
    ax2.set_ylabel('V(x)')
    ax2.set_title('Sensitivity to λ₂')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(alpha_range, V_alpha, 'g-', linewidth=2)
    ax3.set_xlabel('α (Adaptive Weight)')
    ax3.set_ylabel('V(x)')
    ax3.set_title('Sensitivity to α')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(beta_range, V_beta, 'm-', linewidth=2)
    ax4.set_xlabel('β (Bias Parameter)')
    ax4.set_ylabel('V(x)')
    ax4.set_title('Sensitivity to β')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated 'parameter_sensitivity.png'")
    print(f"V(x) range with λ₁ variation: [{min(V_lambda1):.6f}, {max(V_lambda1):.6f}]")
    print(f"V(x) range with λ₂ variation: [{min(V_lambda2):.6f}, {max(V_lambda2):.6f}]")
    print(f"V(x) range with α variation: [{min(V_alpha):.6f}, {max(V_alpha):.6f}]")
    print(f"V(x) range with β variation: [{min(V_beta):.6f}, {max(V_beta):.6f}]")
    print()


def main():
    """Run all demonstrations."""
    print("Hybrid Symbolic-Neural Accuracy Functional Demonstration")
    print("=" * 60)
    print()
    
    try:
        demo_single_step_verification()
        demo_adaptive_weight_scheduling()
        demo_penalty_computation()
        demo_broken_neural_scaling()
        demo_cross_modal_interaction()
        demo_parameter_sensitivity()
        
        print("All demonstrations completed successfully!")
        print("Generated visualization files:")
        print("  - adaptive_weight_scheduling.png")
        print("  - penalty_computation.png")
        print("  - broken_neural_scaling.png")
        print("  - cross_modal_interaction.png")
        print("  - parameter_sensitivity.png")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()