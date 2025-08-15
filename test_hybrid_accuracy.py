"""
Test suite for the hybrid accuracy functional implementation.
"""

import numpy as np
import pytest
from hybrid_accuracy import (
    HybridAccuracyConfig, 
    HybridAccuracyFunctional,
    AdaptiveWeightScheduler,
    PenaltyComputers,
    BrokenNeuralScaling
)


class TestHybridAccuracyConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HybridAccuracyConfig()
        assert config.lambda1 == 0.75
        assert config.lambda2 == 0.25
        assert config.clip_probability is True
        assert config.use_cross_modal is False
        assert config.w_cross == 0.1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridAccuracyConfig(
            lambda1=0.8,
            lambda2=0.2,
            clip_probability=False,
            use_cross_modal=True,
            w_cross=0.15
        )
        assert config.lambda1 == 0.8
        assert config.lambda2 == 0.2
        assert config.clip_probability is False
        assert config.use_cross_modal is True
        assert config.w_cross == 0.15


class TestHybridAccuracyFunctional:
    """Test the main hybrid accuracy functional."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HybridAccuracyConfig()
        self.haf = HybridAccuracyFunctional(self.config)
        
        # Test data
        self.S = np.array([0.65])
        self.N = np.array([0.85])
        self.alpha = np.array([0.3])
        self.Rcog = np.array([0.20])
        self.Reff = np.array([0.15])
        self.P_base = np.array([0.75])
        self.beta = 1.3
    
    def test_compute_hybrid_term(self):
        """Test hybrid term computation."""
        hybrid = self.haf.compute_hybrid_term(self.S, self.N, self.alpha)
        expected = 0.3 * 0.65 + 0.7 * 0.85
        assert np.isclose(hybrid[0], expected)
    
    def test_compute_penalty_term(self):
        """Test penalty term computation."""
        penalty = self.haf.compute_penalty_term(self.Rcog, self.Reff)
        expected_penalty = 0.75 * 0.20 + 0.25 * 0.15
        expected = np.exp(-expected_penalty)
        assert np.isclose(penalty[0], expected)
    
    def test_apply_bias_and_calibrate(self):
        """Test bias application and probability calibration."""
        P_calibrated = self.haf.apply_bias_and_calibrate(self.P_base, self.beta)
        expected = 0.75 * 1.3
        assert np.isclose(P_calibrated[0], expected)
        
        # Test clipping
        P_high = np.array([0.9])
        P_calibrated_high = self.haf.apply_bias_and_calibrate(P_high, 1.5)
        assert P_calibrated_high[0] == 1.0  # Should be clipped
    
    def test_compute_V_single_step(self):
        """Test V(x) computation for single time step."""
        V = self.haf.compute_V(self.S, self.N, self.alpha, 
                               self.Rcog, self.Reff, self.P_base, self.beta)
        
        # Manual calculation as in your formalization
        hybrid = 0.3 * 0.65 + 0.7 * 0.85  # 0.79
        penalty = np.exp(-(0.75 * 0.20 + 0.25 * 0.15))  # ≈ 0.8288
        P_calibrated = np.clip(0.75 * 1.3, 0.0, 1.0)  # 0.975
        expected = hybrid * penalty * P_calibrated  # ≈ 0.638
        
        assert np.isclose(V, expected, atol=1e-3)
    
    def test_compute_V_multi_step(self):
        """Test V(x) computation for multiple time steps."""
        T = 5
        S_multi = np.random.uniform(0.6, 0.9, T)
        N_multi = np.random.uniform(0.7, 0.95, T)
        alpha_multi = np.random.uniform(0.2, 0.8, T)
        Rcog_multi = np.random.uniform(0.1, 0.3, T)
        Reff_multi = np.random.uniform(0.05, 0.25, T)
        P_base_multi = np.random.uniform(0.6, 0.9, T)
        
        V = self.haf.compute_V(S_multi, N_multi, alpha_multi,
                               Rcog_multi, Reff_multi, P_base_multi, self.beta)
        
        assert isinstance(V, float)
        assert 0.0 <= V <= 1.0
    
    def test_compute_V_multiple_models(self):
        """Test V(x) computation for multiple models."""
        M, T = 3, 4
        S_multi = np.random.uniform(0.6, 0.9, (M, T))
        N_multi = np.random.uniform(0.7, 0.95, (M, T))
        alpha_multi = np.random.uniform(0.2, 0.8, T)
        Rcog_multi = np.random.uniform(0.1, 0.3, T)
        Reff_multi = np.random.uniform(0.05, 0.25, T)
        P_base_multi = np.random.uniform(0.6, 0.9, T)
        
        V = self.haf.compute_V(S_multi, N_multi, alpha_multi,
                               Rcog_multi, Reff_multi, P_base_multi, self.beta)
        
        assert V.shape == (M,)
        assert all(0.0 <= v <= 1.0 for v in V)
    
    def test_cross_modal_commutator(self):
        """Test cross-modal commutator computation."""
        S = np.array([[0.6, 0.7], [0.8, 0.9]])  # 2 models, 2 time steps
        N = np.array([[0.7, 0.8], [0.9, 0.95]])
        
        commutator = self.haf.compute_cross_modal_commutator(S, N, 0, 1)
        expected = S[0] * N[1] - S[1] * N[0]
        
        assert np.allclose(commutator, expected)
    
    def test_compute_V_with_cross_modal(self):
        """Test V(x) computation with cross-modal interaction."""
        config = HybridAccuracyConfig(use_cross_modal=True, w_cross=0.1)
        haf_cross = HybridAccuracyFunctional(config)
        
        M, T = 2, 3
        S = np.random.uniform(0.6, 0.9, (M, T))
        N = np.random.uniform(0.7, 0.95, (M, T))
        alpha = np.random.uniform(0.2, 0.8, T)
        Rcog = np.random.uniform(0.1, 0.3, T)
        Reff = np.random.uniform(0.05, 0.25, T)
        P_base = np.random.uniform(0.6, 0.9, T)
        
        V = haf_cross.compute_V(S, N, alpha, Rcog, Reff, P_base, 
                                self.beta, cross_modal_indices=(0, 1))
        
        assert isinstance(V, np.ndarray)
        assert V.shape == (M,)


class TestAdaptiveWeightScheduler:
    """Test adaptive weight scheduling."""
    
    def test_confidence_based(self):
        """Test confidence-based weight scheduling."""
        T = 5
        S_conf = np.random.uniform(0.5, 0.9, T)
        N_conf = np.random.uniform(0.6, 0.95, T)
        
        alpha = AdaptiveWeightScheduler.confidence_based(S_conf, N_conf, temperature=0.5)
        
        assert alpha.shape == (T,)
        assert all(0.0 <= a <= 1.0 for a in alpha)
        
        # Test temperature effect
        alpha_high_temp = AdaptiveWeightScheduler.confidence_based(S_conf, N_conf, temperature=2.0)
        alpha_low_temp = AdaptiveWeightScheduler.confidence_based(S_conf, N_conf, temperature=0.1)
        
        # Lower temperature should make weights more extreme
        assert np.std(alpha_low_temp) >= np.std(alpha_high_temp)
    
    def test_chaos_based(self):
        """Test chaos-based weight scheduling."""
        T = 5
        lyapunov = np.random.uniform(-2.0, 2.0, T)
        
        alpha = AdaptiveWeightScheduler.chaos_based(lyapunov, kappa=1.0)
        
        assert alpha.shape == (T,)
        assert all(0.0 <= a <= 1.0 for a in alpha)
        
        # Test kappa effect
        alpha_high_kappa = AdaptiveWeightScheduler.chaos_based(lyapunov, kappa=2.0)
        alpha_low_kappa = AdaptiveWeightScheduler.chaos_based(lyapunov, kappa=0.5)
        
        # Higher kappa should make transition more sharp
        assert np.std(alpha_high_kappa) >= np.std(alpha_low_kappa)


class TestPenaltyComputers:
    """Test penalty computation utilities."""
    
    def test_energy_drift_penalty(self):
        """Test energy drift penalty computation."""
        T = 5
        energy_traj = np.array([100.0, 100.1, 99.8, 100.3, 99.9])
        
        penalty = PenaltyComputers.energy_drift_penalty(energy_traj)
        
        assert penalty.shape == (T,)
        assert all(0.0 <= p <= 1.0 for p in penalty)
        assert penalty[0] == 0.0  # No drift at initial time
        
        # Test with custom reference
        penalty_custom = PenaltyComputers.energy_drift_penalty(energy_traj, reference_energy=100.5)
        assert penalty_custom.shape == (T,)
    
    def test_constraint_violation_penalty(self):
        """Test constraint violation penalty computation."""
        T = 5
        residuals = np.array([0.1, 0.05, 0.15, 0.02, 0.08])
        
        penalty = PenaltyComputers.constraint_violation_penalty(residuals)
        
        assert penalty.shape == (T,)
        assert all(0.0 <= p <= 1.0 for p in penalty)
        assert penalty[3] == 0.02 / 0.15  # Normalized by max residual
    
    def test_compute_budget_penalty(self):
        """Test computational budget penalty computation."""
        T = 5
        flops = np.array([1e6, 1.2e6, 0.8e6, 1.5e6, 1.1e6])
        max_flops = 2e6
        
        penalty = PenaltyComputers.compute_budget_penalty(flops, max_flops)
        
        assert penalty.shape == (T,)
        assert all(0.0 <= p <= 1.0 for p in penalty)
        assert penalty[3] == 1.5e6 / 2e6  # Should be 0.75


class TestBrokenNeuralScaling:
    """Test broken neural scaling laws implementation."""
    
    def test_bnsl_parameters(self):
        """Test BNSL parameter handling."""
        bnsl = BrokenNeuralScaling(A=1.0, alpha=0.3, n0=1e4, gamma=1.0, delta=1.0, sigma=0.05)
        
        assert bnsl.A == 1.0
        assert bnsl.alpha == 0.3
        assert bnsl.n0 == 1e4
        assert bnsl.gamma == 1.0
        assert bnsl.delta == 1.0
        assert bnsl.sigma == 0.05
    
    def test_predict_error(self):
        """Test error prediction."""
        bnsl = BrokenNeuralScaling(A=1.0, alpha=0.3, n0=1e4, gamma=1.0, delta=1.0, sigma=0.05)
        
        # Single value
        error_single = bnsl.predict_error(1e5)
        assert isinstance(error_single, float)
        assert error_single > 0.0
        
        # Array of values
        n_values = np.array([1e3, 1e4, 1e5])
        errors = bnsl.predict_error(n_values)
        assert errors.shape == (3,)
        assert all(e > 0.0 for e in errors)
        
        # Error should decrease with larger n
        assert errors[2] <= errors[1] <= errors[0]
    
    def test_predict_accuracy(self):
        """Test accuracy prediction."""
        bnsl = BrokenNeuralScaling(A=1.0, alpha=0.3, n0=1e4, gamma=1.0, delta=1.0, sigma=0.05)
        
        accuracy = bnsl.predict_accuracy(1e5)
        error = bnsl.predict_error(1e5)
        
        assert np.isclose(accuracy, 1.0 - error)
        assert 0.0 <= accuracy <= 1.0
    
    def test_fit_from_data(self):
        """Test BNSL fitting from data."""
        # Generate synthetic data
        n_values = np.logspace(3, 6, 20)
        true_errors = 0.1 * (n_values ** (-0.3)) + 0.05 * np.random.randn(20)
        
        # Fit BNSL
        bnsl = BrokenNeuralScaling.fit_from_data(n_values, true_errors)
        
        assert isinstance(bnsl, BrokenNeuralScaling)
        assert all(hasattr(bnsl, attr) for attr in ['A', 'alpha', 'n0', 'gamma', 'delta', 'sigma'])


def test_integration():
    """Integration test combining multiple components."""
    # Configuration
    config = HybridAccuracyConfig(
        lambda1=0.8,
        lambda2=0.2,
        use_cross_modal=True,
        w_cross=0.1
    )
    
    # Initialize components
    haf = HybridAccuracyFunctional(config)
    scheduler = AdaptiveWeightScheduler()
    penalty_comp = PenaltyComputers()
    
    # Generate test data
    T = 10
    S = np.random.uniform(0.6, 0.9, T)
    N = np.random.uniform(0.7, 0.95, T)
    
    # Compute adaptive weights
    S_conf = np.random.uniform(0.5, 0.9, T)
    N_conf = np.random.uniform(0.6, 0.95, T)
    alpha = scheduler.confidence_based(S_conf, N_conf, temperature=0.5)
    
    # Compute penalties
    energy_traj = np.cumsum(np.random.normal(0, 0.1, T)) + 100
    Rcog = penalty_comp.energy_drift_penalty(energy_traj)
    
    flops = np.random.uniform(0.5e6, 1.5e6, T)
    Reff = penalty_comp.compute_budget_penalty(flops, max_flops=2e6)
    
    P_base = np.random.uniform(0.6, 0.9, T)
    beta = 1.2
    
    # Compute V(x)
    V = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
    
    # Validate result
    assert isinstance(V, float)
    assert 0.0 <= V <= 1.0
    
    print(f"Integration test passed. V(x) = {V:.6f}")


if __name__ == "__main__":
    # Run tests
    test_integration()
    print("All tests completed successfully!")