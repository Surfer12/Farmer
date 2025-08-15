"""
Hybrid Symbolic-Neural Accuracy Functional

This module implements the formalized hybrid accuracy metric that combines:
- S(x,t): RK4-based normalized accuracy
- N(x,t): ML/NN-based normalized accuracy  
- α(t): adaptive weight
- Rcog(t): cognitive/theoretical penalty
- Reff(t): efficiency penalty
- P(H|E,β,t): calibrated probability of correctness
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings


@dataclass
class HybridAccuracyConfig:
    """Configuration for the hybrid accuracy functional."""
    lambda1: float = 0.75  # Cognitive penalty weight
    lambda2: float = 0.25  # Efficiency penalty weight
    clip_probability: bool = True  # Whether to clip P(H|E,β,t) to [0,1]
    use_cross_modal: bool = False  # Whether to include cross-modal interaction
    w_cross: float = 0.1  # Weight for cross-modal term


class HybridAccuracyFunctional:
    """
    Implementation of the hybrid symbolic-neural accuracy functional.
    
    V(x) = (1/T) Σ_{k=1..T} [α(tk)S(x,tk) + (1-α(tk))N(x,tk)] 
           · exp(-[λ1*Rcog(tk) + λ2*Reff(tk)]) 
           · P(H|E,β,tk)
    """
    
    def __init__(self, config: Optional[HybridAccuracyConfig] = None):
        self.config = config or HybridAccuracyConfig()
        
    def compute_hybrid_term(self, S: np.ndarray, N: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute the hybrid accuracy term: α(t)S(x,t) + (1-α(t))N(x,t)"""
        return alpha * S + (1.0 - alpha) * N
    
    def compute_penalty_term(self, Rcog: np.ndarray, Reff: np.ndarray) -> np.ndarray:
        """Compute the penalty term: exp(-[λ1*Rcog + λ2*Reff])"""
        penalty = self.config.lambda1 * Rcog + self.config.lambda2 * Reff
        return np.exp(-penalty)
    
    def apply_bias_and_calibrate(self, P_base: np.ndarray, beta: float) -> np.ndarray:
        """
        Apply bias β and ensure probability stays in [0,1].
        
        Args:
            P_base: Base probability P(H|E,t)
            beta: Bias parameter (multiplicative boost)
            
        Returns:
            Calibrated probability P(H|E,β,t) ∈ [0,1]
        """
        P_calibrated = P_base * beta
        
        if self.config.clip_probability:
            P_calibrated = np.clip(P_calibrated, 0.0, 1.0)
            
        return P_calibrated
    
    def compute_cross_modal_commutator(self, S: np.ndarray, N: np.ndarray, 
                                     m1_idx: int, m2_idx: int) -> np.ndarray:
        """
        Compute the empirical commutator C(m1, m2) = S(m1)N(m2) - S(m2)N(m1).
        
        Args:
            S: Symbolic accuracy array of shape (M, T) where M is number of models
            N: Neural accuracy array of shape (M, T)
            m1_idx, m2_idx: Indices of the two models to compare
            
        Returns:
            Commutator array of shape (T,)
        """
        return S[m1_idx] * N[m2_idx] - S[m2_idx] * N[m1_idx]
    
    def compute_V(self, S: np.ndarray, N: np.ndarray, alpha: np.ndarray,
                  Rcog: np.ndarray, Reff: np.ndarray, P_base: np.ndarray,
                  beta: float = 1.0, cross_modal_indices: Optional[Tuple[int, int]] = None) -> float:
        """
        Compute the complete hybrid accuracy functional V(x).
        
        Args:
            S: Symbolic accuracy array of shape (T,) or (M, T)
            N: Neural accuracy array of shape (T,) or (M, T)  
            alpha: Adaptive weight array of shape (T,)
            Rcog: Cognitive penalty array of shape (T,)
            Reff: Efficiency penalty array of shape (T,)
            P_base: Base probability array of shape (T,)
            beta: Bias parameter for probability calibration
            cross_modal_indices: Optional tuple (m1, m2) for cross-modal interaction
            
        Returns:
            Hybrid accuracy score V(x)
        """
        # Handle single model vs multiple models
        if S.ndim == 1:
            S = S.reshape(1, -1)
            N = N.reshape(1, -1)
            single_model = True
        else:
            single_model = False
            
        T = S.shape[1]
        
        # Compute main hybrid term
        hybrid = self.compute_hybrid_term(S, N, alpha)
        
        # Compute penalty term
        penalty = self.compute_penalty_term(Rcog, Reff)
        
        # Apply bias and calibrate probability
        P_calibrated = self.apply_bias_and_calibrate(P_base, beta)
        
        # Main functional computation
        main_term = hybrid * penalty * P_calibrated
        
        # Add cross-modal interaction if requested
        if self.config.use_cross_modal and cross_modal_indices is not None:
            m1, m2 = cross_modal_indices
            if m1 < S.shape[0] and m2 < S.shape[0]:
                commutator = self.compute_cross_modal_commutator(S, N, m1, m2)
                cross_term = self.config.w_cross * np.tanh(commutator)  # Bounded cross-term
                main_term += cross_term.reshape(1, -1)
        
        # Average over time steps
        V = main_term.mean(axis=1)
        
        return V[0] if single_model else V


class AdaptiveWeightScheduler:
    """Scheduler for the adaptive weight α(t)."""
    
    @staticmethod
    def confidence_based(S_conf: np.ndarray, N_conf: np.ndarray, 
                         temperature: float = 1.0) -> np.ndarray:
        """
        Set α(t) based on model confidence comparison.
        
        Args:
            S_conf: Symbolic model confidence array
            N_conf: Neural model confidence array
            temperature: Softmax temperature parameter
            
        Returns:
            Adaptive weight array α(t) ∈ [0,1]
        """
        # Softmax over confidence scores
        conf_scores = np.stack([S_conf, N_conf], axis=0)
        alpha = np.exp(conf_scores[0] / temperature) / np.sum(np.exp(conf_scores / temperature), axis=0)
        return alpha
    
    @staticmethod
    def chaos_based(lyapunov_exponents: np.ndarray, kappa: float = 1.0) -> np.ndarray:
        """
        Set α(t) based on local chaos (Lyapunov exponents).
        Favor neural model in more chaotic regions.
        
        Args:
            lyapunov_exponents: Local Lyapunov exponent array
            kappa: Scaling parameter
            
        Returns:
            Adaptive weight array α(t) ∈ [0,1]
        """
        # Sigmoid function: favor symbolic (α→1) in stable regions, neural (α→0) in chaotic
        alpha = 1.0 / (1.0 + np.exp(kappa * lyapunov_exponents))
        return alpha


class PenaltyComputers:
    """Utilities for computing cognitive and efficiency penalties."""
    
    @staticmethod
    def energy_drift_penalty(energy_trajectory: np.ndarray, 
                           reference_energy: Optional[float] = None) -> np.ndarray:
        """
        Compute cognitive penalty based on energy drift.
        
        Args:
            energy_trajectory: Energy values over time
            reference_energy: Reference energy (if None, use initial value)
            
        Returns:
            Energy drift penalty array
        """
        if reference_energy is None:
            reference_energy = energy_trajectory[0]
            
        relative_drift = np.abs(energy_trajectory - reference_energy) / (abs(reference_energy) + 1e-8)
        return np.clip(relative_drift, 0.0, 1.0)
    
    @staticmethod
    def constraint_violation_penalty(constraint_residuals: np.ndarray) -> np.ndarray:
        """
        Compute cognitive penalty based on constraint violations.
        
        Args:
            constraint_residuals: Constraint residual values over time
            
        Returns:
            Constraint violation penalty array
        """
        # Normalize residuals to [0,1] range
        max_residual = np.max(np.abs(constraint_residuals)) + 1e-8
        normalized_residuals = np.abs(constraint_residuals) / max_residual
        return np.clip(normalized_residuals, 0.0, 1.0)
    
    @staticmethod
    def compute_budget_penalty(flops_per_step: np.ndarray, 
                             max_flops: float) -> np.ndarray:
        """
        Compute efficiency penalty based on computational budget.
        
        Args:
            flops_per_step: FLOPs per time step
            max_flops: Maximum allowed FLOPs per step
            
        Returns:
            Efficiency penalty array
        """
        return np.clip(flops_per_step / max_flops, 0.0, 1.0)


class BrokenNeuralScaling:
    """
    Implementation of Broken Neural Scaling Laws (BNSL).
    
    L(n) = A * n^(-α) * [1 + (n/n0)^γ]^(-δ) + σ
    """
    
    def __init__(self, A: float, alpha: float, n0: float, gamma: float, 
                 delta: float, sigma: float):
        self.A = A
        self.alpha = alpha
        self.n0 = n0
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
    
    def predict_error(self, n: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict error L(n) for given dataset size n."""
        n = np.asarray(n)
        term1 = self.A * (n ** (-self.alpha))
        term2 = (1 + (n / self.n0) ** self.gamma) ** (-self.delta)
        return term1 * term2 + self.sigma
    
    def predict_accuracy(self, n: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict accuracy (1 - error) for given dataset size n."""
        return 1.0 - self.predict_error(n)
    
    @classmethod
    def fit_from_data(cls, n_values: np.ndarray, errors: np.ndarray) -> 'BrokenNeuralScaling':
        """
        Fit BNSL parameters from historical data.
        
        Args:
            n_values: Dataset sizes
            errors: Corresponding error values
            
        Returns:
            Fitted BNSL instance
        """
        # This is a simplified fit - in practice you'd use proper optimization
        from scipy.optimize import curve_fit
        
        def bnsl_func(n, A, alpha, n0, gamma, delta, sigma):
            return A * (n ** (-alpha)) * (1 + (n / n0) ** gamma) ** (-delta) + sigma
        
        # Initial parameter guesses
        p0 = [1.0, 0.5, np.median(n_values), 1.0, 1.0, 0.1]
        
        try:
            popt, _ = curve_fit(bnsl_func, n_values, errors, p0=p0, maxfev=10000)
            return cls(*popt)
        except RuntimeError:
            warnings.warn("BNSL fitting failed, using default parameters")
            return cls(1.0, 0.5, np.median(n_values), 1.0, 1.0, 0.1)


def example_usage():
    """Example usage of the hybrid accuracy functional."""
    
    # Configuration
    config = HybridAccuracyConfig(
        lambda1=0.75,
        lambda2=0.25,
        use_cross_modal=True,
        w_cross=0.1
    )
    
    # Initialize functional
    haf = HybridAccuracyFunctional(config)
    
    # Single time step example (as in your formalization)
    S = np.array([0.65])
    N = np.array([0.85])
    alpha = np.array([0.3])
    Rcog = np.array([0.20])
    Reff = np.array([0.15])
    P_base = np.array([0.75])
    beta = 1.3
    
    # Compute V(x)
    V_single = haf.compute_V(S, N, alpha, Rcog, Reff, P_base, beta)
    print(f"Single step V(x): {V_single:.6f}")  # Should be ~0.638
    
    # Multi-time step example
    T = 10
    S_multi = np.random.uniform(0.6, 0.9, T)
    N_multi = np.random.uniform(0.7, 0.95, T)
    alpha_multi = np.random.uniform(0.2, 0.8, T)
    Rcog_multi = np.random.uniform(0.1, 0.3, T)
    Reff_multi = np.random.uniform(0.05, 0.25, T)
    P_base_multi = np.random.uniform(0.6, 0.9, T)
    
    V_multi = haf.compute_V(S_multi, N_multi, alpha_multi, 
                            Rcog_multi, Reff_multi, P_base_multi, beta)
    print(f"Multi-step V(x): {V_multi:.6f}")
    
    # Adaptive weight scheduling examples
    scheduler = AdaptiveWeightScheduler()
    
    # Confidence-based
    S_conf = np.random.uniform(0.5, 0.9, T)
    N_conf = np.random.uniform(0.6, 0.95, T)
    alpha_conf = scheduler.confidence_based(S_conf, N_conf, temperature=0.5)
    print(f"Confidence-based alpha range: [{alpha_conf.min():.3f}, {alpha_conf.max():.3f}]")
    
    # Chaos-based
    lyapunov = np.random.uniform(-2.0, 2.0, T)
    alpha_chaos = scheduler.chaos_based(lyapunov, kappa=1.0)
    print(f"Chaos-based alpha range: [{alpha_chaos.min():.3f}, {alpha_chaos.max():.3f}]")
    
    # Penalty computation examples
    penalty_comp = PenaltyComputers()
    
    # Energy drift penalty
    energy_traj = np.cumsum(np.random.normal(0, 0.1, T)) + 100
    energy_penalty = penalty_comp.energy_drift_penalty(energy_traj)
    print(f"Energy penalty range: [{energy_penalty.min():.3f}, {energy_penalty.max():.3f}]")
    
    # BNSL example
    n_values = np.logspace(3, 6, 20)
    errors = 0.1 * (n_values ** (-0.3)) + 0.05 * np.random.randn(20)
    
    bnsl = BrokenNeuralScaling.fit_from_data(n_values, errors)
    predicted_accuracy = bnsl.predict_accuracy(1e5)
    print(f"Predicted accuracy at n=1e5: {predicted_accuracy:.3f}")


if __name__ == "__main__":
    example_usage()