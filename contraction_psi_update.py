"""
Contraction-Guaranteed Ψ Update Implementation

This module implements the per-step Ψ update ψ_{t+1} = Φ(ψ_t) with rigorous
contraction guarantees based on the theoretical framework from contraction_spectral_theorems.tex.

Key theoretical results:
- Contraction modulus K = L_Φ/ω < 1 ensures convergence
- Lipschitz bound L_Φ ≤ B_max · (2κ + κL_C + Σ_m w_m L_m)
- Safe regime: κ ≤ 0.15, L_C ≤ 0.3, Σ w_m L_m ≤ 0.35 ⇒ K < 1

Connects to existing framework:
- Builds upon Hybrid Symbolic-Neural Accuracy Functional Ψ(x)
- Integrates with Fractal Ψ Framework self-interaction G(Ψ) = clamp(Ψ², g_max)
- Provides stability guarantees for cognitive-memory framework d_MC
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

@dataclass
class ContractionConfig:
    """Configuration for contraction-guaranteed Ψ update"""
    kappa: float = 0.15           # Self-interaction strength (≤ 0.15 for safety)
    g_max: float = 0.8            # Saturation limit for G(ψ) = min(ψ², g_max)
    L_C: float = 0.0              # Anchor Lipschitz constant (0 = independent)
    omega: float = 1.0            # Sequence space weighting (ω ∈ (0,1])
    B_max: float = 1.0            # Maximum penalty bound
    lambda1: float = 0.75         # Cognitive penalty weight
    lambda2: float = 0.25         # Efficiency penalty weight
    beta: float = 1.2             # Base penalty multiplier
    
    # Modality configuration
    modality_weights: List[float] = None
    modality_lipschitz: List[float] = None
    
    def __post_init__(self):
        if self.modality_weights is None:
            self.modality_weights = [0.2, 0.15]  # Default: visual, audio
        if self.modality_lipschitz is None:
            self.modality_lipschitz = [0.2, 0.15]  # Conservative bounds
    
    def validate_contraction(self) -> Tuple[bool, float, str]:
        """Validate contraction condition K < 1"""
        sum_modality = sum(w * L for w, L in zip(self.modality_weights, self.modality_lipschitz))
        L_phi_bound = self.B_max * (2 * self.kappa + self.kappa * self.L_C + sum_modality)
        K = L_phi_bound / self.omega
        
        is_contractive = K < 1.0
        margin = 1.0 - K if is_contractive else K - 1.0
        
        status = "CONTRACTIVE" if is_contractive else "NON-CONTRACTIVE"
        message = f"{status}: K = {K:.4f}, margin = {margin:.4f}"
        
        return is_contractive, K, message

class ContractionPsiUpdate:
    """
    Contraction-guaranteed Ψ update with theoretical stability bounds
    
    Implements ψ_{t+1} = Φ(ψ_t) where Φ(ψ) = B·core(ψ) with:
    - B = β·exp(-(λ₁R_cog + λ₂R_eff)) ∈ (0, β]
    - core(ψ) = αS + (1-α)N + κ·(G(ψ)+C(ψ)) + Σ_m w_m M_m(ψ)
    - G(ψ) = min(ψ², g_max) with |G'(ψ)| ≤ 2
    """
    
    def __init__(self, config: ContractionConfig):
        self.config = config
        self.history = []
        self.lipschitz_estimates = []
        
        # Validate contraction condition
        is_contractive, K, message = config.validate_contraction()
        if not is_contractive:
            warnings.warn(f"Configuration may not be contractive: {message}")
        else:
            print(f"Contraction validated: {message}")
    
    def penalty_function(self, R_cog: float, R_eff: float) -> float:
        """Compute penalty B = β·exp(-(λ₁R_cog + λ₂R_eff))"""
        exponent = -(self.config.lambda1 * R_cog + self.config.lambda2 * R_eff)
        return self.config.beta * np.exp(exponent)
    
    def fractal_interaction(self, psi: float) -> float:
        """Fractal self-interaction G(ψ) = min(ψ², g_max)"""
        return min(psi**2, self.config.g_max)
    
    def fractal_derivative(self, psi: float) -> float:
        """Derivative G'(ψ) with global bound |G'(ψ)| ≤ 2"""
        if psi**2 >= self.config.g_max:
            return 0.0  # Saturation region
        else:
            return 2 * psi  # |2ψ| ≤ 2 since ψ ∈ [0,1]
    
    def anchor_function(self, psi: float, anchor_type: str = "independent") -> float:
        """
        Anchor function C(ψ) with controlled Lipschitz constant
        
        Args:
            psi: Current state value
            anchor_type: "independent" (L_C=0) or "smooth" (bounded L_C)
        """
        if anchor_type == "independent":
            # Preferred: anchors independent of ψ (L_C = 0)
            return 0.5  # Fixed anchor value
        elif anchor_type == "smooth":
            # Alternative: smooth dependence with bounded L_C
            return 0.3 * np.tanh(2 * psi - 1)  # L_C ≤ 0.6, but we'll use L_C=0.3
        else:
            raise ValueError(f"Unknown anchor_type: {anchor_type}")
    
    def modality_maps(self, psi: float) -> List[float]:
        """
        Modality maps M_m(ψ) with Lipschitz constants L_m
        
        Returns list of modality contributions with bounded derivatives
        """
        maps = []
        
        # Visual modality (bounded slope)
        visual = 0.4 * np.sigmoid(4 * psi - 2)  # Smooth, bounded derivative
        maps.append(visual)
        
        # Audio modality (linear with cap)
        audio = min(0.3 * psi, 0.25)  # Linear with saturation
        maps.append(audio)
        
        return maps
    
    def core_function(self, psi: float, alpha: float, S: float, N: float, 
                     anchor_type: str = "independent") -> float:
        """
        Core function: core(ψ) = αS + (1-α)N + κ·(G(ψ)+C(ψ)) + Σ_m w_m M_m(ψ)
        """
        # Base hybrid term
        hybrid = alpha * S + (1 - alpha) * N
        
        # Fractal self-interaction
        fractal = self.fractal_interaction(psi)
        
        # Anchors
        anchor = self.anchor_function(psi, anchor_type)
        
        # Modality contributions
        modality_maps = self.modality_maps(psi)
        modality_sum = sum(w * m for w, m in zip(self.config.modality_weights, modality_maps))
        
        # Combine all terms
        core = hybrid + self.config.kappa * (fractal + anchor) + modality_sum
        
        return np.clip(core, 0, 1)  # Ensure core ∈ [0,1]
    
    def phi_function(self, psi: float, alpha: float, S: float, N: float,
                    R_cog: float, R_eff: float, anchor_type: str = "independent") -> float:
        """
        Complete update function Φ(ψ) = B·core(ψ)
        """
        penalty = self.penalty_function(R_cog, R_eff)
        core = self.core_function(psi, alpha, S, N, anchor_type)
        
        result = penalty * core
        return np.clip(result, 0, 1)  # Ensure ψ ∈ [0,1]
    
    def update_step(self, psi_t: float, alpha: float, S: float, N: float,
                   R_cog: float, R_eff: float, anchor_type: str = "independent") -> float:
        """Single update step: ψ_{t+1} = Φ(ψ_t)"""
        psi_next = self.phi_function(psi_t, alpha, S, N, R_cog, R_eff, anchor_type)
        
        # Store history for analysis
        self.history.append({
            'psi_t': psi_t,
            'psi_next': psi_next,
            'alpha': alpha,
            'S': S,
            'N': N,
            'R_cog': R_cog,
            'R_eff': R_eff,
            'penalty': self.penalty_function(R_cog, R_eff)
        })
        
        return psi_next
    
    def simulate_sequence(self, psi_0: float, steps: int, 
                         scenario_params: Dict, anchor_type: str = "independent") -> np.ndarray:
        """
        Simulate sequence evolution with contraction guarantees
        
        Args:
            psi_0: Initial state
            steps: Number of evolution steps
            scenario_params: Dictionary with 'alpha', 'S', 'N', 'R_cog', 'R_eff' arrays or constants
            anchor_type: Type of anchor function
        
        Returns:
            Array of ψ values over time
        """
        sequence = np.zeros(steps + 1)
        sequence[0] = psi_0
        
        for t in range(steps):
            # Extract parameters (can be time-varying or constant)
            alpha = self._get_param(scenario_params, 'alpha', t)
            S = self._get_param(scenario_params, 'S', t)
            N = self._get_param(scenario_params, 'N', t)
            R_cog = self._get_param(scenario_params, 'R_cog', t)
            R_eff = self._get_param(scenario_params, 'R_eff', t)
            
            sequence[t + 1] = self.update_step(sequence[t], alpha, S, N, R_cog, R_eff, anchor_type)
        
        return sequence
    
    def _get_param(self, params: Dict, key: str, t: int):
        """Extract parameter value (handle both arrays and constants)"""
        value = params[key]
        if isinstance(value, (list, np.ndarray)):
            return value[min(t, len(value) - 1)]
        else:
            return value
    
    def estimate_lipschitz_numerical(self, scenario_params: Dict, 
                                   psi_grid: np.ndarray = None, eps: float = 1e-5) -> float:
        """
        Numerical estimation of Lipschitz constant L_Φ
        
        Uses central difference approximation to estimate |∂Φ/∂ψ|
        """
        if psi_grid is None:
            psi_grid = np.linspace(0, 1, 1001)
        
        # Create closure for Φ with fixed exogenous parameters
        def phi_fn(psi):
            alpha = scenario_params.get('alpha', 0.5)
            S = scenario_params.get('S', 0.7)
            N = scenario_params.get('N', 0.8)
            R_cog = scenario_params.get('R_cog', 0.1)
            R_eff = scenario_params.get('R_eff', 0.1)
            return self.phi_function(psi, alpha, S, N, R_cog, R_eff)
        
        # Estimate derivatives
        derivs = []
        for psi in psi_grid:
            a = max(0.0, psi - eps)
            b = min(1.0, psi + eps)
            if b > a:
                d = (phi_fn(b) - phi_fn(a)) / (b - a)
                derivs.append(abs(d))
        
        L_hat = max(derivs) if derivs else 0.0
        K_hat = L_hat / self.config.omega
        
        self.lipschitz_estimates.append({
            'L_hat': L_hat,
            'K_hat': K_hat,
            'scenario': scenario_params.copy()
        })
        
        return L_hat
    
    def analyze_contraction_properties(self, scenarios: List[Dict]) -> Dict:
        """
        Comprehensive analysis of contraction properties across scenarios
        """
        results = {
            'theoretical_bound': {},
            'numerical_estimates': [],
            'convergence_analysis': []
        }
        
        # Theoretical bounds
        is_contractive, K_theory, message = self.config.validate_contraction()
        results['theoretical_bound'] = {
            'K_theory': K_theory,
            'is_contractive': is_contractive,
            'message': message
        }
        
        # Numerical estimates for each scenario
        for i, scenario in enumerate(scenarios):
            L_hat = self.estimate_lipschitz_numerical(scenario)
            K_hat = L_hat / self.config.omega
            
            results['numerical_estimates'].append({
                'scenario_id': i,
                'L_hat': L_hat,
                'K_hat': K_hat,
                'scenario': scenario
            })
            
            # Convergence test
            sequence = self.simulate_sequence(0.3, 100, scenario)
            convergence_rate = self._estimate_convergence_rate(sequence)
            
            results['convergence_analysis'].append({
                'scenario_id': i,
                'final_value': sequence[-1],
                'convergence_rate': convergence_rate,
                'theoretical_rate': -np.log(K_theory) if K_theory < 1 else None
            })
        
        return results
    
    def _estimate_convergence_rate(self, sequence: np.ndarray) -> float:
        """Estimate empirical convergence rate from sequence"""
        if len(sequence) < 10:
            return 0.0
        
        # Fit exponential decay to |ψ_t - ψ_∞|
        psi_inf = sequence[-1]  # Approximate fixed point
        deviations = np.abs(sequence[:-1] - psi_inf)
        
        # Avoid log(0)
        nonzero_mask = deviations > 1e-10
        if np.sum(nonzero_mask) < 5:
            return 0.0
        
        t_vals = np.arange(len(deviations))[nonzero_mask]
        log_devs = np.log(deviations[nonzero_mask])
        
        # Linear fit: log|ψ_t - ψ_∞| ≈ log(C) - λt
        if len(t_vals) >= 2:
            slope, _ = np.polyfit(t_vals, log_devs, 1)
            return -slope  # Convergence rate λ
        else:
            return 0.0
    
    def visualize_contraction_analysis(self, scenarios: List[Dict], 
                                     save_path: str = None) -> None:
        """Create comprehensive visualization of contraction analysis"""
        results = self.analyze_contraction_properties(scenarios)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Contraction Analysis for Ψ Update', fontsize=16)
        
        # 1. Theoretical vs Numerical Lipschitz constants
        ax1 = axes[0, 0]
        K_theory = results['theoretical_bound']['K_theory']
        K_numerical = [est['K_hat'] for est in results['numerical_estimates']]
        
        ax1.axhline(y=K_theory, color='red', linestyle='--', 
                   label=f'Theoretical K = {K_theory:.3f}')
        ax1.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='K = 1 (boundary)')
        ax1.plot(K_numerical, 'bo-', label='Numerical estimates')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Contraction Modulus K')
        ax1.set_title('Theoretical vs Numerical Contraction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sequence evolution for different scenarios
        ax2 = axes[0, 1]
        for i, scenario in enumerate(scenarios[:3]):  # Show first 3 scenarios
            sequence = self.simulate_sequence(0.3, 50, scenario)
            ax2.plot(sequence, label=f'Scenario {i+1}')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('ψ(t)')
        ax2.set_title('Sequence Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Convergence rates
        ax3 = axes[1, 0]
        empirical_rates = [conv['convergence_rate'] for conv in results['convergence_analysis']]
        theoretical_rate = results['convergence_analysis'][0]['theoretical_rate']
        
        ax3.bar(range(len(empirical_rates)), empirical_rates, alpha=0.7, label='Empirical')
        if theoretical_rate is not None:
            ax3.axhline(y=theoretical_rate, color='red', linestyle='--', 
                       label=f'Theoretical = {theoretical_rate:.3f}')
        ax3.set_xlabel('Scenario')
        ax3.set_ylabel('Convergence Rate λ')
        ax3.set_title('Convergence Rate Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Parameter sensitivity
        ax4 = axes[1, 1]
        kappa_values = np.linspace(0.05, 0.25, 20)
        K_values = []
        
        original_kappa = self.config.kappa
        for kappa in kappa_values:
            self.config.kappa = kappa
            _, K, _ = self.config.validate_contraction()
            K_values.append(K)
        self.config.kappa = original_kappa  # Restore
        
        ax4.plot(kappa_values, K_values, 'g-', linewidth=2)
        ax4.axhline(y=1.0, color='red', linestyle='--', label='K = 1')
        ax4.axvline(x=0.15, color='blue', linestyle=':', label='Safe κ = 0.15')
        ax4.set_xlabel('κ (Self-interaction strength)')
        ax4.set_ylabel('Contraction Modulus K')
        ax4.set_title('Parameter Sensitivity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Contraction analysis saved to {save_path}")
        
        plt.show()

def create_test_scenarios() -> List[Dict]:
    """Create test scenarios for contraction analysis"""
    scenarios = [
        # Scenario 1: Balanced collaboration
        {
            'alpha': 0.5,
            'S': 0.7,
            'N': 0.8,
            'R_cog': 0.1,
            'R_eff': 0.1
        },
        # Scenario 2: Neural-dominant with low penalties
        {
            'alpha': 0.2,
            'S': 0.6,
            'N': 0.9,
            'R_cog': 0.05,
            'R_eff': 0.08
        },
        # Scenario 3: Symbolic-dominant with higher penalties
        {
            'alpha': 0.8,
            'S': 0.85,
            'N': 0.7,
            'R_cog': 0.15,
            'R_eff': 0.12
        },
        # Scenario 4: High chaos (stress test)
        {
            'alpha': 0.3,
            'S': 0.5,
            'N': 0.6,
            'R_cog': 0.2,
            'R_eff': 0.18
        }
    ]
    return scenarios

if __name__ == "__main__":
    # Demonstration of contraction-guaranteed Ψ update
    print("=== Contraction-Guaranteed Ψ Update Analysis ===\n")
    
    # Create configuration with safe parameters
    config = ContractionConfig(
        kappa=0.15,
        g_max=0.8,
        L_C=0.0,  # Independent anchors
        omega=1.0,
        modality_weights=[0.2, 0.15],
        modality_lipschitz=[0.2, 0.15]
    )
    
    # Initialize update system
    psi_updater = ContractionPsiUpdate(config)
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Run comprehensive analysis
    print("Running contraction analysis...")
    results = psi_updater.analyze_contraction_properties(scenarios)
    
    # Display results
    print(f"\nTheoretical Analysis:")
    print(f"  {results['theoretical_bound']['message']}")
    
    print(f"\nNumerical Estimates:")
    for est in results['numerical_estimates']:
        print(f"  Scenario {est['scenario_id']}: K_hat = {est['K_hat']:.4f}")
    
    print(f"\nConvergence Analysis:")
    for conv in results['convergence_analysis']:
        print(f"  Scenario {conv['scenario_id']}: rate = {conv['convergence_rate']:.4f}, "
              f"final = {conv['final_value']:.4f}")
    
    # Create visualization
    psi_updater.visualize_contraction_analysis(scenarios, 
                                             save_path="/Users/ryan_david_oates/Farmer/contraction_analysis.png")
    
    print(f"\nContraction analysis complete!")
    print(f"History length: {len(psi_updater.history)}")
    print(f"Lipschitz estimates: {len(psi_updater.lipschitz_estimates)}")
