import pytest
import numpy as np
from hybrid_functional import HybridFunctional, FunctionalParams, create_visualization

class TestFunctionalParams:
    """Test the FunctionalParams dataclass"""
    
    def test_default_params(self):
        params = FunctionalParams()
        assert params.lambda1 == 0.75
        assert params.lambda2 == 0.25
        assert params.kappa == 1.0
        assert params.beta == 1.2
    
    def test_custom_params(self):
        params = FunctionalParams(lambda1=0.8, lambda2=0.2, kappa=1.5, beta=1.0)
        assert params.lambda1 == 0.8
        assert params.lambda2 == 0.2
        assert params.kappa == 1.5
        assert params.beta == 1.0

class TestHybridFunctional:
    """Test the HybridFunctional class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.functional = HybridFunctional()
        self.params = FunctionalParams(lambda1=0.6, lambda2=0.4, kappa=1.0, beta=1.2)
        self.functional_custom = HybridFunctional(self.params)
    
    def test_compute_adaptive_weight(self):
        """Test adaptive weight computation"""
        # Test with default lyapunov exponent
        alpha = self.functional.compute_adaptive_weight(0.0)
        assert 0 <= alpha <= 1
        
        # Test with custom lyapunov exponent
        alpha = self.functional.compute_adaptive_weight(0.0, lyapunov_exponent=0.5)
        assert 0 <= alpha <= 1
        
        # Test edge cases
        alpha = self.functional.compute_adaptive_weight(0.0, lyapunov_exponent=0.0)
        assert alpha > 0.5  # Should favor symbolic when chaos is low
        
        alpha = self.functional.compute_adaptive_weight(0.0, lyapunov_exponent=10.0)
        assert alpha < 0.5  # Should favor neural when chaos is high
    
    def test_compute_hybrid_output(self):
        """Test hybrid output computation"""
        # Test basic case
        result = self.functional.compute_hybrid_output(0.5, 0.8, 0.6)
        expected = 0.6 * 0.5 + 0.4 * 0.8
        assert np.isclose(result, expected)
        
        # Test edge cases
        result = self.functional.compute_hybrid_output(1.0, 0.0, 1.0)
        assert result == 1.0  # Full symbolic
        
        result = self.functional.compute_hybrid_output(0.0, 1.0, 0.0)
        assert result == 1.0  # Full neural
        
        result = self.functional.compute_hybrid_output(0.5, 0.5, 0.5)
        assert result == 0.5  # Equal weighting
    
    def test_compute_regularization_penalty(self):
        """Test regularization penalty computation"""
        # Test basic case
        penalty = self.functional.compute_regularization_penalty(0.1, 0.2)
        expected = np.exp(-(0.75 * 0.1 + 0.25 * 0.2))
        assert np.isclose(penalty, expected)
        
        # Test with custom parameters
        penalty = self.functional_custom.compute_regularization_penalty(0.1, 0.2)
        expected = np.exp(-(0.6 * 0.1 + 0.4 * 0.2))
        assert np.isclose(penalty, expected)
        
        # Test edge cases
        penalty = self.functional.compute_regularization_penalty(0.0, 0.0)
        assert penalty == 1.0  # No penalty
        
        penalty = self.functional.compute_regularization_penalty(100.0, 100.0)
        assert penalty < 1e-10  # Very high penalty
    
    def test_compute_probability(self):
        """Test probability computation with bias adjustment"""
        # Test basic case
        prob = self.functional.compute_probability(0.5)
        assert 0 <= prob <= 1
        
        # Test with custom beta
        prob = self.functional.compute_probability(0.5, beta=2.0)
        assert prob > 0.5  # Higher beta should increase probability
        
        # Test edge cases
        prob = self.functional.compute_probability(0.0)
        assert prob == 0.0
        
        prob = self.functional.compute_probability(1.0)
        assert prob == 1.0
        
        # Test bias adjustment
        prob = self.functional.compute_probability(0.5, beta=1.0)
        assert np.isclose(prob, 0.5)  # No bias change
    
    def test_compute_single_step_psi(self):
        """Test single-step Ψ(x) computation"""
        # Test with the numerical example from the query
        psi = self.functional.compute_single_step_psi(
            S=0.67, N=0.87, alpha=0.4,
            R_cog=0.17, R_eff=0.11, base_prob=0.81
        )
        
        # Verify the calculation step by step
        hybrid = 0.4 * 0.67 + 0.6 * 0.87  # 0.794
        reg_penalty = np.exp(-(0.75 * 0.17 + 0.25 * 0.11))  # ≈ 0.864
        prob = self.functional.compute_probability(0.81, 1.2)  # ≈ 0.972
        
        expected = hybrid * reg_penalty * prob
        assert np.isclose(psi, expected, rtol=1e-3)
        
        # Test that result is bounded
        assert 0 <= psi <= 1
    
    def test_compute_multi_step_psi(self):
        """Test multi-step Ψ(x) computation"""
        # Test with 3 time steps
        S = [0.6, 0.7, 0.8]
        N = [0.8, 0.7, 0.6]
        alphas = [0.5, 0.5, 0.5]
        R_cog = [0.1, 0.1, 0.1]
        R_eff = [0.1, 0.1, 0.1]
        base_probs = [0.9, 0.9, 0.9]
        
        psi = self.functional.compute_multi_step_psi(S, N, alphas, R_cog, R_eff, base_probs)
        
        # Should be average of single-step values
        single_steps = []
        for i in range(3):
            single_step = self.functional.compute_single_step_psi(
                S[i], N[i], alphas[i], R_cog[i], R_eff[i], base_probs[i]
            )
            single_steps.append(single_step)
        
        expected = np.mean(single_steps)
        assert np.isclose(psi, expected)
    
    def test_multi_step_validation(self):
        """Test multi-step input validation"""
        # Test mismatched lengths
        with pytest.raises(ValueError):
            self.functional.compute_multi_step_psi(
                [0.5, 0.6], [0.7], [0.5], [0.1], [0.1], [0.9]
            )
    
    def test_numerical_example(self):
        """Test the numerical example from the user's query"""
        result = self.functional.run_numerical_example()
        
        # Verify all expected keys are present
        expected_keys = ['S', 'N', 'alpha', 'O_hybrid', 'R_cognitive', 
                        'R_efficiency', 'lambda1', 'lambda2', 'P_total', 
                        'exp_term', 'P', 'beta', 'P_adj', 'psi']
        
        for key in expected_keys:
            assert key in result
        
        # Verify the final Ψ(x) value matches expected calculation
        expected_psi = 0.794 * 0.864 * 0.972
        assert np.isclose(result['psi'], expected_psi, rtol=1e-3)
    
    def test_open_source_example(self):
        """Test the open-source contributions example"""
        result = self.functional.run_open_source_example()
        
        # Verify all expected keys are present
        expected_keys = ['S', 'N', 'alpha', 'O_hybrid', 'R_cognitive', 
                        'R_efficiency', 'lambda1', 'lambda2', 'P_total', 
                        'exp_term', 'P', 'beta', 'P_adj', 'psi']
        
        for key in expected_keys:
            assert key in result
        
        # Verify the final Ψ(x) value is reasonable
        assert 0 <= result['psi'] <= 1

class TestVisualization:
    """Test visualization functions"""
    
    def test_create_visualization(self):
        """Test visualization creation"""
        psi_values = [0.667, 0.702]
        labels = ['Tracking Step', 'Open Source']
        
        # This should not raise an error
        try:
            create_visualization(psi_values, labels)
        except Exception as e:
            pytest.fail(f"Visualization creation failed: {e}")
    
    def test_visualization_input_validation(self):
        """Test visualization input validation"""
        # Test with empty lists
        with pytest.raises(IndexError):
            create_visualization([], [])
        
        # Test with mismatched lengths
        with pytest.raises(IndexError):
            create_visualization([0.5], ['Label1', 'Label2'])

def test_mathematical_properties():
    """Test mathematical properties of the functional"""
    functional = HybridFunctional()
    
    # Test that Ψ(x) is bounded between 0 and 1 for valid inputs
    for S in [0.0, 0.5, 1.0]:
        for N in [0.0, 0.5, 1.0]:
            for alpha in [0.0, 0.5, 1.0]:
                for R_cog in [0.0, 0.1, 0.5]:
                    for R_eff in [0.0, 0.1, 0.5]:
                        for base_prob in [0.1, 0.5, 0.9]:
                            psi = functional.compute_single_step_psi(
                                S, N, alpha, R_cog, R_eff, base_prob
                            )
                            assert 0 <= psi <= 1, f"Psi {psi} not bounded for inputs: S={S}, N={N}, alpha={alpha}, R_cog={R_cog}, R_eff={R_eff}, base_prob={base_prob}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])