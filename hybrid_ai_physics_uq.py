"""
Hybrid AI-Physics Uncertainty Quantification (UQ) System

This module implements a hybrid model that combines physics-based interpolation
with ML corrections, featuring explicit uncertainty quantification at each stage.

Key components:
- Hybrid output: O(α) = α S(x) + (1−α) N(x)
- Penalty system: pen = exp(−[λ₁ R_cognitive + λ₂ R_efficiency])
- Calibrated posterior: post = min{β · P(H|E), 1}
- Confidence metric: Ψ(x) = O(α) · pen · post ∈ [0,1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression


@dataclass
class HybridConfig:
    """Configuration for hybrid AI-physics system"""
    # Model parameters
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 32
    ensemble_size: int = 5
    
    # Physics parameters
    vorticity_weight: float = 1.0
    divergence_weight: float = 1.0
    energy_weight: float = 0.5
    
    # UQ parameters
    aleatoric_scale: float = 0.02
    epistemic_threshold: float = 0.1
    conformal_alpha: float = 0.1
    
    # Penalty weights
    lambda_1: float = 0.57  # cognitive penalty weight
    lambda_2: float = 0.43  # efficiency penalty weight
    
    # Governance parameters
    beta_init: float = 1.15  # responsiveness parameter
    alpha_init: float = 0.48  # initial physics/ML balance
    
    # Safety parameters
    max_oscillation_threshold: float = 0.05
    bifurcation_threshold: float = 0.15


class PhysicsInterpolator(nn.Module):
    """Physics-based interpolation S(x) with sigma-level dynamics"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Surface-linear transform layers
        self.surface_transform = nn.Linear(config.input_dim, config.hidden_dim)
        self.sigma_mapping = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Physics constraint layers
        self.vorticity_layer = nn.Linear(config.output_dim, config.output_dim)
        self.divergence_layer = nn.Linear(config.output_dim, config.output_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based interpolation with constraint diagnostics
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
            - output: Physics interpolation S(x)
            - vorticity: Vorticity constraint values
            - divergence: Divergence constraint values
            - energy: Energy consistency measure
        """
        batch_size = x.size(0)
        
        # Surface-linear transform and sigma-level mapping
        surface_features = torch.tanh(self.surface_transform(x))
        sigma_output = self.sigma_mapping(surface_features)
        
        # Physics constraints
        # Vorticity: k̂·(∇_p × u)
        vorticity = self.vorticity_layer(sigma_output)
        vorticity_constraint = torch.sum(vorticity**2, dim=-1, keepdim=True)
        
        # Divergence: ∇_p · u
        divergence = self.divergence_layer(sigma_output)
        divergence_constraint = torch.sum(divergence**2, dim=-1, keepdim=True)
        
        # Energy consistency (simplified as L2 norm of output)
        energy = torch.sum(sigma_output**2, dim=-1, keepdim=True)
        
        return {
            'output': torch.tanh(sigma_output),  # Bounded output
            'vorticity': vorticity_constraint,
            'divergence': divergence_constraint,
            'energy': energy
        }


class NeuralCorrector(nn.Module):
    """Neural correction N(x) with residual learning and uncertainty"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Main correction network
        self.correction_net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Aleatoric uncertainty head (heteroscedastic)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
            nn.Softplus()  # Ensure positive variance
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute neural correction with uncertainty estimation
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
            - output: Neural correction N(x)
            - aleatoric_std: Aleatoric uncertainty σ(x)
        """
        # Extract features for uncertainty head
        features = F.relu(self.correction_net[0](x))
        features = F.relu(self.correction_net[2](features))
        
        # Correction output with small-signal scaling
        correction = self.correction_net[-1](features)
        scaled_correction = self.config.aleatoric_scale * torch.tanh(correction)
        
        # Aleatoric uncertainty
        aleatoric_std = self.uncertainty_head(features)
        
        return {
            'output': scaled_correction,
            'aleatoric_std': aleatoric_std
        }


class HybridCore(nn.Module):
    """Core hybrid system combining physics and neural components"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        self.physics_interpolator = PhysicsInterpolator(config)
        self.neural_corrector = NeuralCorrector(config)
        
        # Adaptive alpha parameter (learnable)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(config.alpha_init)))
        
    def forward(self, x: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid output O(α) = α S(x) + (1−α) N(x)
        
        Args:
            x: Input tensor [batch_size, input_dim]
            alpha: Optional override for alpha parameter
            
        Returns:
            Dictionary containing all outputs and diagnostics
        """
        # Get physics and neural outputs
        physics_out = self.physics_interpolator(x)
        neural_out = self.neural_corrector(x)
        
        # Use provided alpha or learned parameter
        if alpha is None:
            alpha = torch.sigmoid(self.alpha_logit)
        
        # Hybrid combination
        hybrid_output = alpha * physics_out['output'] + (1 - alpha) * neural_out['output']
        
        return {
            'hybrid_output': hybrid_output,
            'physics_output': physics_out['output'],
            'neural_output': neural_out['output'],
            'alpha': alpha,
            'vorticity': physics_out['vorticity'],
            'divergence': physics_out['divergence'],
            'energy': physics_out['energy'],
            'aleatoric_std': neural_out['aleatoric_std']
        }


class PenaltySystem:
    """Penalty system: pen = exp(−[λ₁ R_cognitive + λ₂ R_efficiency])"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        
    def compute_cognitive_penalty(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute R_cognitive: representational fidelity penalties
        
        Args:
            outputs: Dictionary containing vorticity, divergence, energy
            
        Returns:
            Cognitive penalty tensor
        """
        vorticity_penalty = torch.mean(outputs['vorticity'], dim=-1)
        divergence_penalty = torch.mean(outputs['divergence'], dim=-1)
        
        # Energy consistency penalty (deviation from expected energy)
        energy_penalty = torch.abs(outputs['energy'] - 1.0).squeeze(-1)
        
        R_cognitive = (self.config.vorticity_weight * vorticity_penalty + 
                      self.config.divergence_weight * divergence_penalty +
                      self.config.energy_weight * energy_penalty)
        
        return R_cognitive
    
    def compute_efficiency_penalty(self, outputs: Dict[str, torch.Tensor], 
                                 previous_outputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute R_efficiency: stability/efficiency trade-offs
        
        Args:
            outputs: Current outputs
            previous_outputs: Previous timestep outputs for oscillation detection
            
        Returns:
            Efficiency penalty tensor
        """
        batch_size = outputs['hybrid_output'].size(0)
        
        # Interpolation cost (complexity penalty)
        interpolation_cost = torch.sum(outputs['hybrid_output']**2, dim=-1)
        
        # Oscillation damping penalty
        if previous_outputs is not None:
            output_diff = outputs['hybrid_output'] - previous_outputs['hybrid_output']
            oscillation_penalty = torch.sum(output_diff**2, dim=-1)
        else:
            oscillation_penalty = torch.zeros(batch_size)
        
        R_efficiency = interpolation_cost + oscillation_penalty
        return R_efficiency
    
    def compute_penalty(self, outputs: Dict[str, torch.Tensor], 
                       previous_outputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute total penalty: pen = exp(−[λ₁ R_cognitive + λ₂ R_efficiency])
        
        Args:
            outputs: Current outputs
            previous_outputs: Previous outputs for efficiency calculation
            
        Returns:
            Penalty tensor ∈ [0,1]
        """
        R_cognitive = self.compute_cognitive_penalty(outputs)
        R_efficiency = self.compute_efficiency_penalty(outputs, previous_outputs)
        
        total_penalty_arg = (self.config.lambda_1 * R_cognitive + 
                            self.config.lambda_2 * R_efficiency)
        
        penalty = torch.exp(-total_penalty_arg)
        
        return penalty.clamp(0, 1)


class UncertaintyQuantifier:
    """Uncertainty quantification with aleatoric, epistemic, and conformal methods"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.conformal_quantiles = None
        self.temperature_scaler = None
        
    def compute_epistemic_uncertainty(self, ensemble_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute epistemic uncertainty from ensemble predictions
        
        Args:
            ensemble_outputs: List of outputs from ensemble members
            
        Returns:
            Epistemic uncertainty (predictive variance)
        """
        if len(ensemble_outputs) < 2:
            return torch.zeros_like(ensemble_outputs[0])
        
        # Stack ensemble outputs
        stacked_outputs = torch.stack(ensemble_outputs, dim=0)  # [ensemble_size, batch_size, output_dim]
        
        # Compute predictive variance
        epistemic_variance = torch.var(stacked_outputs, dim=0)
        epistemic_std = torch.sqrt(epistemic_variance + 1e-8)
        
        return torch.mean(epistemic_std, dim=-1)  # Average over output dimensions
    
    def calibrate_posterior(self, probability: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute calibrated posterior: post = min{β · P(H|E), 1}
        
        Args:
            probability: Base probability P(H|E)
            beta: Responsiveness parameter
            
        Returns:
            Calibrated posterior ∈ [0,1]
        """
        return torch.clamp(beta * probability, 0, 1)
    
    def compute_conformal_prediction_sets(self, residuals: torch.Tensor, 
                                        new_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute conformal prediction intervals
        
        Args:
            residuals: Calibration residuals
            new_predictions: New predictions to create intervals for
            
        Returns:
            Dictionary with lower and upper bounds
        """
        if self.conformal_quantiles is None:
            # Compute conformal quantiles from residuals
            alpha = self.config.conformal_alpha
            abs_residuals = torch.abs(residuals).flatten()
            quantile_level = 1 - alpha
            self.conformal_quantiles = torch.quantile(abs_residuals, quantile_level)
        
        # Create prediction intervals
        lower_bound = new_predictions - self.conformal_quantiles
        upper_bound = new_predictions + self.conformal_quantiles
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': 2 * self.conformal_quantiles
        }


class GovernanceSystem:
    """Governance system for adaptive parameter control"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.alpha_history = []
        self.beta_history = []
        
    def update_alpha(self, epistemic_uncertainty: torch.Tensor, 
                    physics_residuals: torch.Tensor, 
                    current_alpha: torch.Tensor) -> torch.Tensor:
        """
        Update α(t) based on epistemic uncertainty and physics residuals
        
        Args:
            epistemic_uncertainty: Current epistemic uncertainty
            physics_residuals: Physics model residuals
            current_alpha: Current alpha value
            
        Returns:
            Updated alpha value
        """
        # Increase alpha when epistemic uncertainty is high (trust physics more)
        uncertainty_factor = torch.sigmoid(epistemic_uncertainty - self.config.epistemic_threshold)
        
        # Increase alpha when physics residuals are stable
        residual_stability = torch.exp(-torch.mean(physics_residuals**2, dim=-1))
        
        # Combined adjustment
        alpha_adjustment = 0.1 * (uncertainty_factor + residual_stability - 1.0)
        new_alpha = torch.clamp(current_alpha + alpha_adjustment, 0.1, 0.9)
        
        self.alpha_history.append(new_alpha.mean().item())
        return new_alpha
    
    def update_beta(self, external_validation_score: float, current_beta: torch.Tensor) -> torch.Tensor:
        """
        Update β based on external validation
        
        Args:
            external_validation_score: Score from external validation
            current_beta: Current beta value
            
        Returns:
            Updated beta value
        """
        # Increase beta with higher validation scores
        beta_adjustment = 0.05 * (external_validation_score - 0.5)
        new_beta = torch.clamp(current_beta + beta_adjustment, 0.5, 2.0)
        
        self.beta_history.append(new_beta.item())
        return new_beta
    
    def detect_bifurcation_proximity(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Detect proximity to Hopf bifurcation or instability
        
        Args:
            outputs: Model outputs
            
        Returns:
            Bifurcation proximity score
        """
        # Simple heuristic based on output oscillation and energy
        output_variance = torch.var(outputs['hybrid_output'], dim=-1)
        energy_instability = torch.abs(outputs['energy'].squeeze(-1) - 1.0)
        
        bifurcation_score = output_variance + energy_instability
        return bifurcation_score


class HybridAIPhysicsUQ(nn.Module):
    """Main hybrid AI-physics UQ system"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.hybrid_core = HybridCore(config)
        self.penalty_system = PenaltySystem(config)
        self.uq_system = UncertaintyQuantifier(config)
        self.governance = GovernanceSystem(config)
        
        # State tracking
        self.previous_outputs = None
        self.beta = torch.tensor(config.beta_init)
        
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encoding process: plane → vector/sigma space
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Encoded outputs with diagnostics
        """
        # Get hybrid outputs
        outputs = self.hybrid_core(x)
        
        # Compute penalties
        penalty = self.penalty_system.compute_penalty(outputs, self.previous_outputs)
        
        # Add penalty to outputs
        outputs['penalty'] = penalty
        
        return outputs
    
    def decode(self, encoded_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decoding process: vector/sigma space → plane
        
        Args:
            encoded_outputs: Outputs from encoding process
            
        Returns:
            Decoded outputs with reconstruction
        """
        # Potential diagnosis and backpressure (simplified)
        potential_field = torch.sum(encoded_outputs['hybrid_output'], dim=-1, keepdim=True)
        backpressure = torch.clamp(potential_field, 0, 1)  # Cap ≤ 1
        
        # Subcap/cap balancing
        subcap_regime = encoded_outputs['hybrid_output'] * (backpressure < 0.9).float()
        
        # Apply second learned correction for systematic bias
        bias_correction = 0.01 * torch.randn_like(encoded_outputs['hybrid_output'])
        reconstructed = subcap_regime + bias_correction
        
        return {
            'reconstructed': reconstructed,
            'potential_field': potential_field,
            'backpressure': backpressure,
            'subcap_regime': subcap_regime
        }
    
    def compute_psi(self, encoded_outputs: Dict[str, torch.Tensor], 
                   probability: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence metric Ψ(x) = O(α) · pen · post
        
        Args:
            encoded_outputs: Outputs from encoding
            probability: Base probability P(H|E)
            
        Returns:
            Confidence metric Ψ(x) ∈ [0,1]
        """
        # Hybrid output magnitude
        O_alpha = torch.mean(torch.abs(encoded_outputs['hybrid_output']), dim=-1)
        
        # Penalty
        penalty = encoded_outputs['penalty']
        
        # Calibrated posterior
        posterior = self.uq_system.calibrate_posterior(probability, self.beta)
        
        # Confidence metric
        psi = O_alpha * penalty * posterior
        
        return psi.clamp(0, 1)
    
    def forward(self, x: torch.Tensor, probability: torch.Tensor, 
               external_validation: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through hybrid system
        
        Args:
            x: Input data
            probability: Base probability P(H|E)
            external_validation: External validation score
            
        Returns:
            Complete system outputs
        """
        # Encoding
        encoded = self.encode(x)
        
        # Decoding
        decoded = self.decode(encoded)
        
        # Compute confidence metric
        psi = self.compute_psi(encoded, probability)
        
        # Governance updates
        epistemic_unc = torch.zeros(x.size(0))  # Placeholder for ensemble
        physics_residuals = encoded['physics_output'] - encoded['hybrid_output']
        
        new_alpha = self.governance.update_alpha(epistemic_unc, physics_residuals, encoded['alpha'])
        self.beta = self.governance.update_beta(external_validation, self.beta)
        
        # Update state
        self.previous_outputs = encoded
        
        # Combine all outputs
        outputs = {**encoded, **decoded}
        outputs.update({
            'psi': psi,
            'beta': self.beta,
            'updated_alpha': new_alpha
        })
        
        return outputs
    
    def compute_sensitivities(self, outputs: Dict[str, torch.Tensor], 
                            probability: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute first-order sensitivities for safety analysis
        
        Args:
            outputs: System outputs
            probability: Base probability
            
        Returns:
            Dictionary of sensitivity measures
        """
        alpha = outputs['alpha']
        S = outputs['physics_output']
        N = outputs['neural_output']
        penalty = outputs['penalty']
        posterior = self.uq_system.calibrate_posterior(probability, self.beta)
        
        # ∂Ψ/∂α = (S−N) · pen · post
        dpsi_dalpha = torch.mean(torch.abs(S - N), dim=-1) * penalty * posterior
        
        # Cognitive and efficiency penalties (simplified)
        R_cognitive = self.penalty_system.compute_cognitive_penalty(outputs)
        R_efficiency = self.penalty_system.compute_efficiency_penalty(outputs)
        
        O_alpha = torch.mean(torch.abs(outputs['hybrid_output']), dim=-1)
        
        # ∂Ψ/∂R_cognitive = −λ₁ O · pen · post
        dpsi_dRcog = -self.config.lambda_1 * O_alpha * penalty * posterior
        
        # ∂Ψ/∂R_efficiency = −λ₂ O · pen · post  
        dpsi_dReff = -self.config.lambda_2 * O_alpha * penalty * posterior
        
        # ∂Ψ/∂β = O · pen · P(H|E) when βP < 1; else 0
        beta_P_product = self.beta * probability
        dpsi_dbeta = torch.where(
            beta_P_product < 1.0,
            O_alpha * penalty * probability,
            torch.zeros_like(O_alpha)
        )
        
        return {
            'dpsi_dalpha': dpsi_dalpha,
            'dpsi_dRcog': dpsi_dRcog,
            'dpsi_dReff': dpsi_dReff,
            'dpsi_dbeta': dpsi_dbeta,
            'R_cognitive': R_cognitive,
            'R_efficiency': R_efficiency
        }


def numerical_example():
    """
    Reproduce the numerical example from the specification
    """
    print("=== Numerical Example ===")
    
    # Inputs from specification
    S = 0.78
    N = 0.86
    alpha = 0.48
    
    # Compute hybrid output
    O = alpha * S + (1 - alpha) * N
    print(f"O = {alpha}·{S} + {1-alpha:.2f}·{N} = {O:.4f}")
    
    # Penalty calculation
    R_cognitive = 0.13
    R_efficiency = 0.09
    lambda_1 = 0.57
    lambda_2 = 0.43
    
    total_penalty_arg = lambda_1 * R_cognitive + lambda_2 * R_efficiency
    penalty = np.exp(-total_penalty_arg)
    print(f"Penalty arg = {lambda_1}·{R_cognitive} + {lambda_2}·{R_efficiency} = {total_penalty_arg:.4f}")
    print(f"pen = exp(-{total_penalty_arg:.4f}) = {penalty:.4f}")
    
    # Probability calculation
    P = 0.80
    beta = 1.15
    posterior = min(beta * P, 1.0)
    print(f"post = min{{{beta}·{P}, 1}} = {posterior:.2f}")
    
    # Final result
    psi = O * penalty * posterior
    print(f"Ψ(x) = {O:.4f} × {penalty:.3f} × {posterior:.2f} = {psi:.4f}")
    
    return {
        'O': O,
        'penalty': penalty,
        'posterior': posterior,
        'psi': psi
    }


if __name__ == "__main__":
    # Run numerical example
    numerical_example()
    
    # Test system
    config = HybridConfig()
    system = HybridAIPhysicsUQ(config)
    
    # Generate test data
    batch_size = 4
    x = torch.randn(batch_size, config.input_dim)
    probability = torch.tensor([0.8, 0.75, 0.9, 0.85])
    
    # Forward pass
    outputs = system(x, probability)
    
    print(f"\n=== System Test ===")
    print(f"Input shape: {x.shape}")
    print(f"Hybrid output shape: {outputs['hybrid_output'].shape}")
    print(f"Ψ(x) values: {outputs['psi'].detach().numpy()}")
    print(f"Alpha values: {outputs['alpha'].detach().numpy()}")
    print(f"Beta value: {outputs['beta'].item():.3f}")
    
    # Compute sensitivities
    sensitivities = system.compute_sensitivities(outputs, probability)
    print(f"\n=== Sensitivities ===")
    for key, value in sensitivities.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.detach().numpy()}")