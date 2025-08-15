"""
Hybrid Symbolic-Neural Accuracy Functional

A comprehensive implementation of the discrete-time hybrid accuracy functional
that combines RK4-based symbolic methods with neural network approaches.

Key Components:
- V(x): Main hybrid accuracy functional
- Adaptive weighting α(t)
- Penalty functions Rcog(t), Reff(t)
- Calibrated probability P(H|E,β,t)
- Cross-modal non-commutativity measurement
- Cognitive-memory distance metric
- Koopman reversal detection
- Broken Neural Scaling Laws integration
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from scipy.optimize import minimize
from scipy.linalg import eig
import warnings


class HybridAccuracyFunctional:
    """
    Main class implementing the hybrid symbolic-neural accuracy functional.
    
    The functional is defined as:
    V(x) = (1/T) Σ_{k=1..T} [ α(tk) S(x, tk) + (1 − α(tk)) N(x, tk) ] 
           · exp(−[λ1 Rcog(tk) + λ2 Reff(tk)]) · P(H|E, β, tk)
    """
    
    def __init__(self, lambda1: float = 0.75, lambda2: float = 0.25):
        """
        Initialize the hybrid accuracy functional.
        
        Args:
            lambda1: Weight for cognitive penalty (physics consistency)
            lambda2: Weight for efficiency penalty (computational cost)
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.calibration_params = None
        
    def compute_V(self, 
                  S: np.ndarray, 
                  N: np.ndarray, 
                  alpha: np.ndarray,
                  Rcog: np.ndarray, 
                  Reff: np.ndarray, 
                  P_corr: np.ndarray) -> float:
        """
        Compute the hybrid accuracy functional V(x).
        
        Args:
            S: RK4-based normalized accuracy in [0,1], shape (T,)
            N: ML/NN-based normalized accuracy in [0,1], shape (T,)
            alpha: Adaptive weight in [0,1], shape (T,)
            Rcog: Cognitive/theoretical penalty ≥ 0, shape (T,)
            Reff: Efficiency penalty ≥ 0, shape (T,)
            P_corr: Calibrated probability of correctness in [0,1], shape (T,)
            
        Returns:
            Hybrid accuracy functional value
        """
        # Ensure all inputs are numpy arrays
        S = np.asarray(S)
        N = np.asarray(N)
        alpha = np.asarray(alpha)
        Rcog = np.asarray(Rcog)
        Reff = np.asarray(Reff)
        P_corr = np.asarray(P_corr)
        
        # Validate inputs are in correct ranges
        assert np.all((S >= 0) & (S <= 1)), "S must be in [0,1]"
        assert np.all((N >= 0) & (N <= 1)), "N must be in [0,1]"
        assert np.all((alpha >= 0) & (alpha <= 1)), "alpha must be in [0,1]"
        assert np.all(Rcog >= 0), "Rcog must be >= 0"
        assert np.all(Reff >= 0), "Reff must be >= 0"
        
        # Hybrid accuracy term
        hybrid = alpha * S + (1.0 - alpha) * N
        
        # Exponential penalty term
        penalty_exponent = -(self.lambda1 * Rcog + self.lambda2 * Reff)
        reg = np.exp(penalty_exponent)
        
        # Clip probability to [0,1] for safety
        P_corr_clipped = np.clip(P_corr, 0.0, 1.0)
        
        # Combined term for each time step
        term = hybrid * reg * P_corr_clipped
        
        # Return temporal average
        return term.mean()
    
    def adaptive_weight_confidence(self, 
                                 S_conf: np.ndarray, 
                                 N_conf: np.ndarray,
                                 temperature: float = 1.0) -> np.ndarray:
        """
        Compute adaptive weight α(t) based on model confidences.
        
        Args:
            S_conf: Confidence scores for symbolic method
            N_conf: Confidence scores for neural method
            temperature: Softmax temperature parameter
            
        Returns:
            Adaptive weights α(t) in [0,1]
        """
        # Softmax with temperature
        logits = np.stack([S_conf / temperature, N_conf / temperature], axis=-1)
        softmax_weights = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        
        # Return weight for symbolic method (S)
        return softmax_weights[..., 0]
    
    def adaptive_weight_chaos(self, 
                            lyapunov_exponents: np.ndarray,
                            kappa: float = 2.0) -> np.ndarray:
        """
        Compute adaptive weight α(t) based on local Lyapunov exponents.
        Favors neural method (N) in more chaotic regions.
        
        Args:
            lyapunov_exponents: Local Lyapunov exponents
            kappa: Scaling parameter for sigmoid
            
        Returns:
            Adaptive weights α(t) in [0,1]
        """
        # Sigmoid function: α = σ(-κ·λ_lyapunov)
        # High chaos (positive λ) → low α → favor N
        # Low chaos (negative λ) → high α → favor S
        return 1.0 / (1.0 + np.exp(kappa * lyapunov_exponents))
    
    def cognitive_penalty(self, 
                         energy_drift: Optional[np.ndarray] = None,
                         constraint_violation: Optional[np.ndarray] = None,
                         ode_residual: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cognitive/theoretical penalty Rcog(t).
        
        Args:
            energy_drift: Energy conservation violation
            constraint_violation: Physics constraint violations
            ode_residual: ODE residual error
            
        Returns:
            Cognitive penalty values
        """
        penalty = np.zeros_like(energy_drift if energy_drift is not None else [0])
        
        if energy_drift is not None:
            penalty += np.abs(energy_drift)
            
        if constraint_violation is not None:
            penalty += np.abs(constraint_violation)
            
        if ode_residual is not None:
            penalty += np.abs(ode_residual)
            
        return penalty
    
    def efficiency_penalty(self, 
                          flops_per_step: Optional[np.ndarray] = None,
                          memory_usage: Optional[np.ndarray] = None,
                          latency: Optional[np.ndarray] = None,
                          normalize: bool = True) -> np.ndarray:
        """
        Compute efficiency penalty Reff(t).
        
        Args:
            flops_per_step: FLOPs per computation step
            memory_usage: Memory usage
            latency: Computation latency
            normalize: Whether to normalize penalties to [0,1]
            
        Returns:
            Efficiency penalty values
        """
        penalty = np.zeros_like(flops_per_step if flops_per_step is not None else [0])
        
        if flops_per_step is not None:
            if normalize and np.max(flops_per_step) > 0:
                penalty += flops_per_step / np.max(flops_per_step)
            else:
                penalty += flops_per_step
                
        if memory_usage is not None:
            if normalize and np.max(memory_usage) > 0:
                penalty += memory_usage / np.max(memory_usage)
            else:
                penalty += memory_usage
                
        if latency is not None:
            if normalize and np.max(latency) > 0:
                penalty += latency / np.max(latency)
            else:
                penalty += latency
                
        return penalty
    
    def calibrated_probability(self, 
                             raw_probability: np.ndarray,
                             bias: float = 1.0,
                             calibration_method: str = "platt") -> np.ndarray:
        """
        Compute calibrated probability P(H|E,β,t) with bias correction.
        
        Args:
            raw_probability: Raw probability estimates
            bias: Multiplicative bias factor
            calibration_method: Calibration method ("platt", "temperature", "none")
            
        Returns:
            Calibrated and bias-corrected probabilities in [0,1]
        """
        # Apply calibration if parameters are available
        if self.calibration_params is not None and calibration_method == "platt":
            # Platt scaling: P_cal = 1 / (1 + exp(A * P_raw + B))
            A, B = self.calibration_params
            logits = A * raw_probability + B
            calibrated = 1.0 / (1.0 + np.exp(-logits))
        elif self.calibration_params is not None and calibration_method == "temperature":
            # Temperature scaling
            temperature = self.calibration_params[0]
            logits = np.log(raw_probability / (1 - raw_probability + 1e-8))
            calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        else:
            calibrated = raw_probability
        
        # Apply bias multiplicatively then clip to [0,1]
        biased = calibrated * bias
        return np.clip(biased, 0.0, 1.0)
    
    def fit_calibration(self, 
                       raw_probabilities: np.ndarray,
                       true_labels: np.ndarray,
                       method: str = "platt") -> None:
        """
        Fit calibration parameters on validation data.
        
        Args:
            raw_probabilities: Raw probability predictions
            true_labels: Ground truth binary labels
            method: Calibration method ("platt" or "temperature")
        """
        if method == "platt":
            # Fit Platt scaling parameters
            def platt_loss(params):
                A, B = params
                logits = A * raw_probabilities + B
                pred_probs = 1.0 / (1.0 + np.exp(-logits))
                # Binary cross-entropy loss
                eps = 1e-8
                pred_probs = np.clip(pred_probs, eps, 1 - eps)
                return -np.mean(true_labels * np.log(pred_probs) + 
                              (1 - true_labels) * np.log(1 - pred_probs))
            
            result = minimize(platt_loss, [1.0, 0.0], method='BFGS')
            self.calibration_params = result.x
            
        elif method == "temperature":
            # Fit temperature scaling
            def temp_loss(temp):
                temp = temp[0]
                logits = np.log(raw_probabilities / (1 - raw_probabilities + 1e-8))
                pred_probs = 1.0 / (1.0 + np.exp(-logits / temp))
                # Binary cross-entropy loss
                eps = 1e-8
                pred_probs = np.clip(pred_probs, eps, 1 - eps)
                return -np.mean(true_labels * np.log(pred_probs) + 
                              (1 - true_labels) * np.log(1 - pred_probs))
            
            result = minimize(temp_loss, [1.0], method='BFGS')
            self.calibration_params = result.x


class CrossModalAnalysis:
    """
    Analysis tools for cross-modal interactions and non-commutativity.
    """
    
    @staticmethod
    def compute_commutator(S_outputs: np.ndarray, 
                          N_outputs: np.ndarray,
                          m1_states: np.ndarray,
                          m2_states: np.ndarray) -> np.ndarray:
        """
        Compute empirical commutator C(m1, m2) = S(m1)N(m2) - S(m2)N(m1).
        
        Args:
            S_outputs: Symbolic method outputs at different states
            N_outputs: Neural method outputs at different states
            m1_states: First set of states
            m2_states: Second set of states
            
        Returns:
            Commutator values
        """
        # Evaluate S at m1 and m2, N at m1 and m2
        S_m1 = S_outputs[..., 0]  # S evaluated at m1
        S_m2 = S_outputs[..., 1]  # S evaluated at m2
        N_m1 = N_outputs[..., 0]  # N evaluated at m1
        N_m2 = N_outputs[..., 1]  # N evaluated at m2
        
        # Commutator: S(m1)N(m2) - S(m2)N(m1)
        return S_m1 * N_m2 - S_m2 * N_m1
    
    @staticmethod
    def cross_modal_interaction(commutator_values: np.ndarray,
                               weight: float = 1.0) -> float:
        """
        Compute cross-modal interaction score.
        
        Args:
            commutator_values: Commutator values over time
            weight: Weight for the interaction term
            
        Returns:
            Interaction score
        """
        return weight * np.mean(commutator_values)


class CognitiveMemoryMetric:
    """
    Cognitive-memory distance metric with asymmetric interactions.
    """
    
    def __init__(self, 
                 w_t: float = 1.0, 
                 w_c: float = 1.0, 
                 w_e: float = 1.0, 
                 w_a: float = 1.0,
                 w_cross: float = 0.1):
        """
        Initialize the cognitive-memory metric.
        
        Args:
            w_t: Weight for temporal distance
            w_c: Weight for cognitive distance
            w_e: Weight for episodic distance
            w_a: Weight for action distance
            w_cross: Weight for cross-modal interaction
        """
        self.w_t = w_t
        self.w_c = w_c
        self.w_e = w_e
        self.w_a = w_a
        self.w_cross = w_cross
    
    def distance_squared(self,
                        t1: float, m1: np.ndarray, e1: np.ndarray, a1: np.ndarray,
                        t2: float, m2: np.ndarray, e2: np.ndarray, a2: np.ndarray) -> float:
        """
        Compute squared cognitive-memory distance.
        
        Args:
            t1, t2: Time points
            m1, m2: Memory/cognitive states
            e1, e2: Episodic representations
            a1, a2: Action representations
            
        Returns:
            Squared distance
        """
        d_squared = (
            self.w_t * (t1 - t2)**2 +
            self.w_c * np.sum((m1 - m2)**2) +
            self.w_e * np.sum((e1 - e2)**2) +
            self.w_a * np.sum((a1 - a2)**2)
        )
        return d_squared
    
    def interaction_score(self, 
                         S_m1: float, N_m1: float,
                         S_m2: float, N_m2: float) -> float:
        """
        Compute asymmetric interaction score.
        
        Args:
            S_m1, N_m1: Symbolic and neural outputs at state m1
            S_m2, N_m2: Symbolic and neural outputs at state m2
            
        Returns:
            Interaction score
        """
        commutator = S_m1 * N_m2 - S_m2 * N_m1
        return self.w_cross * commutator


class KoopmanAnalysis:
    """
    Koopman operator analysis for detecting insight bifurcations and reversals.
    """
    
    def __init__(self, dictionary_functions: Optional[Callable] = None):
        """
        Initialize Koopman analysis.
        
        Args:
            dictionary_functions: Functions for lifting state to dictionary space
        """
        self.dictionary_functions = dictionary_functions
        self.K_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
    
    def fit_koopman_edmd(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit Koopman operator using Extended Dynamic Mode Decomposition (EDMD).
        
        Args:
            X: State data at time t, shape (n_samples, n_features)
            Y: State data at time t+1, shape (n_samples, n_features)
            
        Returns:
            Koopman matrix K
        """
        # If dictionary functions provided, lift the data
        if self.dictionary_functions is not None:
            Psi_X = self.dictionary_functions(X)
            Psi_Y = self.dictionary_functions(Y)
        else:
            Psi_X = X
            Psi_Y = Y
        
        # EDMD: K = Psi_Y @ Psi_X^+ (pseudoinverse)
        self.K_matrix = Psi_Y.T @ np.linalg.pinv(Psi_X.T)
        
        # Compute eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = eig(self.K_matrix)
        
        return self.K_matrix
    
    def detect_bifurcations(self, threshold: float = 0.05) -> np.ndarray:
        """
        Detect eigenvalues near the unit circle (potential bifurcations).
        
        Args:
            threshold: Threshold for detecting eigenvalues near |λ| = 1
            
        Returns:
            Boolean array indicating bifurcation eigenvalues
        """
        if self.eigenvalues is None:
            raise ValueError("Must fit Koopman operator first")
        
        magnitudes = np.abs(self.eigenvalues)
        return np.abs(magnitudes - 1.0) < threshold
    
    def reversal_constraint_optimization(self,
                                       X_window: np.ndarray,
                                       Y_window: np.ndarray,
                                       physics_penalty_fn: Callable,
                                       max_iterations: int = 100) -> Dict[str, Any]:
        """
        Learn constrained nonlinear map for reversal windows.
        
        Args:
            X_window: Input states in bifurcation window
            Y_window: Target states in bifurcation window
            physics_penalty_fn: Function computing physics penalty
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        # Placeholder for alternating minimization
        # Step A: Fix dictionary, fit K (least squares)
        # Step B: Fit nonlinear map with physics constraints
        
        results = {
            'converged': False,
            'iterations': 0,
            'final_error': np.inf,
            'nonlinear_map_params': None
        }
        
        # This would implement the alternating minimization described
        # in the specification. For now, return placeholder.
        warnings.warn("Koopman reversal optimization not fully implemented")
        
        return results


class BrokenNeuralScalingLaws:
    """
    Implementation of Broken Neural Scaling Laws for performance prediction.
    """
    
    def __init__(self):
        """Initialize BNSL with default parameters."""
        self.fitted_params = None
    
    def scaling_law(self, n: np.ndarray, A: float, alpha: float, 
                   n0: float, gamma: float, delta: float, sigma: float) -> np.ndarray:
        """
        Broken power law scaling function.
        
        L(n) = A n^{-α} [1 + (n/n0)^γ]^{-δ} + σ
        
        Args:
            n: Dataset sizes
            A, alpha, n0, gamma, delta, sigma: Scaling law parameters
            
        Returns:
            Expected loss values
        """
        term1 = A * (n ** (-alpha))
        term2 = (1 + (n / n0) ** gamma) ** (-delta)
        return term1 * term2 + sigma
    
    def fit(self, dataset_sizes: np.ndarray, observed_losses: np.ndarray) -> Dict[str, float]:
        """
        Fit BNSL parameters to observed scaling data.
        
        Args:
            dataset_sizes: Array of dataset sizes used in training
            observed_losses: Corresponding observed loss values
            
        Returns:
            Dictionary of fitted parameters
        """
        def objective(params):
            A, alpha, n0, gamma, delta, sigma = params
            predicted = self.scaling_law(dataset_sizes, A, alpha, n0, gamma, delta, sigma)
            return np.mean((observed_losses - predicted) ** 2)
        
        # Initial parameter guess
        initial_guess = [1.0, 0.5, np.median(dataset_sizes), 1.0, 0.5, 0.1]
        
        # Bounds to ensure reasonable parameters
        bounds = [
            (1e-6, 10.0),    # A > 0
            (0.01, 2.0),     # alpha > 0
            (1.0, None),     # n0 > 0
            (0.01, 5.0),     # gamma > 0
            (0.01, 2.0),     # delta > 0
            (0.0, 1.0)       # sigma >= 0
        ]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        self.fitted_params = {
            'A': result.x[0],
            'alpha': result.x[1],
            'n0': result.x[2],
            'gamma': result.x[3],
            'delta': result.x[4],
            'sigma': result.x[5]
        }
        
        return self.fitted_params
    
    def predict_performance(self, n: np.ndarray) -> np.ndarray:
        """
        Predict performance at given dataset sizes.
        
        Args:
            n: Dataset sizes for prediction
            
        Returns:
            Predicted loss values
        """
        if self.fitted_params is None:
            raise ValueError("Must fit BNSL parameters first")
        
        return self.scaling_law(n, **self.fitted_params)
    
    def optimal_dataset_size(self, cost_fn: Callable, target_performance: float) -> float:
        """
        Find optimal dataset size balancing performance and cost.
        
        Args:
            cost_fn: Function mapping dataset size to cost
            target_performance: Target performance level
            
        Returns:
            Optimal dataset size
        """
        if self.fitted_params is None:
            raise ValueError("Must fit BNSL parameters first")
        
        def objective(n):
            n = n[0]  # Unpack for scalar optimization
            predicted_loss = self.scaling_law(np.array([n]), **self.fitted_params)[0]
            performance_penalty = max(0, predicted_loss - target_performance) ** 2
            return cost_fn(n) + performance_penalty
        
        result = minimize(objective, [1000], method='BFGS')
        return result.x[0]


# Example usage and validation
def example_usage():
    """Demonstrate the hybrid accuracy functional with the provided numerical example."""
    
    # Initialize the functional
    hybrid_func = HybridAccuracyFunctional(lambda1=0.75, lambda2=0.25)
    
    # Single time step example from the specification
    S = np.array([0.65])
    N = np.array([0.85])
    alpha = np.array([0.3])
    Rcog = np.array([0.20])
    Reff = np.array([0.15])
    
    # Apply bias to probability: 0.75 * 1.3 = 0.975 (clipped to [0,1])
    P_base = np.array([0.75])
    P_corr = hybrid_func.calibrated_probability(P_base, bias=1.3)
    
    # Compute V(x)
    result = hybrid_func.compute_V(S, N, alpha, Rcog, Reff, P_corr)
    
    print(f"Single step example:")
    print(f"S = {S[0]:.2f}, N = {N[0]:.2f}, α = {alpha[0]:.2f}")
    print(f"Hybrid = α·S + (1-α)·N = {alpha[0]:.2f}·{S[0]:.2f} + {1-alpha[0]:.2f}·{N[0]:.2f} = {alpha[0]*S[0] + (1-alpha[0])*N[0]:.2f}")
    print(f"Rcog = {Rcog[0]:.2f}, Reff = {Reff[0]:.2f}")
    print(f"Penalty exp = exp(-({hybrid_func.lambda1}·{Rcog[0]:.2f} + {hybrid_func.lambda2}·{Reff[0]:.2f})) = {np.exp(-(hybrid_func.lambda1*Rcog[0] + hybrid_func.lambda2*Reff[0])):.4f}")
    print(f"P(H|E,β) = {P_corr[0]:.3f}")
    print(f"V(x) = {result:.3f}")
    print(f"Expected: ~0.638")
    
    return result


if __name__ == "__main__":
    # Run the example
    result = example_usage()