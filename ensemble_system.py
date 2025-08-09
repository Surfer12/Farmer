"""
Ensemble System for Epistemic Uncertainty Quantification

This module provides ensemble-based epistemic uncertainty estimation
for the hybrid AI-physics system, supporting deep ensembles and 
SWAG/Laplace approximations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from copy import deepcopy
import warnings

from hybrid_ai_physics_uq import HybridAIPhysicsUQ, HybridConfig


class EnsembleMember(nn.Module):
    """Individual ensemble member with independent initialization"""
    
    def __init__(self, config: HybridConfig, member_id: int):
        super().__init__()
        self.config = config
        self.member_id = member_id
        
        # Create hybrid system with different initialization
        self.hybrid_system = HybridAIPhysicsUQ(config)
        
        # Apply different initialization for diversity
        self._diversify_initialization()
        
    def _diversify_initialization(self):
        """Apply different initialization to promote ensemble diversity"""
        torch.manual_seed(42 + self.member_id * 100)  # Different seed per member
        
        for module in self.hybrid_system.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with member-specific scaling
                scale = 1.0 + 0.1 * self.member_id
                nn.init.xavier_uniform_(module.weight, gain=scale)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor, probability: torch.Tensor, 
               external_validation: float = 0.5) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble member"""
        return self.hybrid_system(x, probability, external_validation)


class DeepEnsemble(nn.Module):
    """Deep ensemble for epistemic uncertainty quantification"""
    
    def __init__(self, config: HybridConfig, ensemble_size: int = 5):
        super().__init__()
        self.config = config
        self.ensemble_size = ensemble_size
        
        # Create ensemble members
        self.members = nn.ModuleList([
            EnsembleMember(config, i) for i in range(ensemble_size)
        ])
        
        # Ensemble aggregation weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
        
    def forward(self, x: torch.Tensor, probability: torch.Tensor,
               external_validation: float = 0.5, 
               return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor
            probability: Base probability
            external_validation: External validation score
            return_individual: Whether to return individual member outputs
            
        Returns:
            Dictionary with ensemble predictions and uncertainties
        """
        # Get predictions from all ensemble members
        member_outputs = []
        for member in self.members:
            with torch.no_grad() if not self.training else torch.enable_grad():
                output = member(x, probability, external_validation)
                member_outputs.append(output)
        
        # Extract key predictions for uncertainty computation
        hybrid_predictions = torch.stack([out['hybrid_output'] for out in member_outputs])
        psi_predictions = torch.stack([out['psi'] for out in member_outputs])
        alpha_predictions = torch.stack([out['alpha'] for out in member_outputs])
        
        # Compute ensemble statistics
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted ensemble mean
        ensemble_hybrid = torch.sum(weights.view(-1, 1, 1) * hybrid_predictions, dim=0)
        ensemble_psi = torch.sum(weights.view(-1, 1) * psi_predictions, dim=0)
        ensemble_alpha = torch.sum(weights.view(-1, 1) * alpha_predictions, dim=0)
        
        # Epistemic uncertainty (predictive variance)
        hybrid_variance = torch.var(hybrid_predictions, dim=0)
        psi_variance = torch.var(psi_predictions, dim=0)
        alpha_variance = torch.var(alpha_predictions, dim=0)
        
        # Mean epistemic uncertainty across dimensions
        epistemic_uncertainty = torch.mean(hybrid_variance, dim=-1)
        
        # Ensemble disagreement (max - min)
        hybrid_disagreement = torch.max(hybrid_predictions, dim=0)[0] - torch.min(hybrid_predictions, dim=0)[0]
        psi_disagreement = torch.max(psi_predictions, dim=0)[0] - torch.min(psi_predictions, dim=0)[0]
        
        # Aggregate other outputs (use first member as template)
        ensemble_output = deepcopy(member_outputs[0])
        ensemble_output.update({
            'hybrid_output': ensemble_hybrid,
            'psi': ensemble_psi,
            'alpha': ensemble_alpha,
            'epistemic_uncertainty': epistemic_uncertainty,
            'hybrid_variance': hybrid_variance,
            'psi_variance': psi_variance,
            'alpha_variance': alpha_variance,
            'hybrid_disagreement': torch.mean(hybrid_disagreement, dim=-1),
            'psi_disagreement': psi_disagreement,
            'ensemble_weights': weights
        })
        
        if return_individual:
            ensemble_output['individual_outputs'] = member_outputs
            
        return ensemble_output
    
    def compute_predictive_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy for epistemic uncertainty
        
        Args:
            predictions: Ensemble predictions [ensemble_size, batch_size, output_dim]
            
        Returns:
            Predictive entropy
        """
        # Convert to probabilities (assuming outputs are logits)
        probs = F.softmax(predictions, dim=-1)
        mean_probs = torch.mean(probs, dim=0)
        
        # Predictive entropy: H[p(y|x,D)] = -∑ p(y|x,D) log p(y|x,D)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        return entropy
    
    def compute_mutual_information(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information for epistemic uncertainty
        I[y,θ|x,D] = H[p(y|x,D)] - E_θ[H[p(y|x,θ)]]
        
        Args:
            predictions: Ensemble predictions
            
        Returns:
            Mutual information (epistemic uncertainty)
        """
        probs = F.softmax(predictions, dim=-1)
        
        # Predictive entropy
        mean_probs = torch.mean(probs, dim=0)
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # Expected entropy
        individual_entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        expected_entropy = torch.mean(individual_entropies, dim=0)
        
        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy
        
        return mutual_info


class SWAGApproximation:
    """
    Stochastic Weight Averaging Gaussian (SWAG) approximation
    for epistemic uncertainty in the last layer
    """
    
    def __init__(self, model: nn.Module, max_models: int = 20, var_clamp: float = 1e-6):
        self.model = model
        self.max_models = max_models
        self.var_clamp = var_clamp
        
        # Storage for weight statistics
        self.mean_weights = {}
        self.sq_weights = {}
        self.weight_deviations = []
        self.n_models = 0
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize weight storage"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.mean_weights[name] = torch.zeros_like(param.data)
                self.sq_weights[name] = torch.zeros_like(param.data)
    
    def collect_model(self):
        """Collect current model weights for SWAG approximation"""
        if self.n_models >= self.max_models:
            # Remove oldest deviation
            self.weight_deviations.pop(0)
        
        current_deviation = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Update running statistics
                self.n_models += 1
                self.mean_weights[name] = (
                    (self.n_models - 1) * self.mean_weights[name] + param.data
                ) / self.n_models
                
                self.sq_weights[name] = (
                    (self.n_models - 1) * self.sq_weights[name] + param.data ** 2
                ) / self.n_models
                
                # Store deviation from mean
                current_deviation[name] = param.data - self.mean_weights[name]
        
        self.weight_deviations.append(current_deviation)
        
        if len(self.weight_deviations) > self.max_models:
            self.weight_deviations = self.weight_deviations[-self.max_models:]
    
    def sample_parameters(self, n_samples: int = 10) -> List[Dict[str, torch.Tensor]]:
        """
        Sample parameters from SWAG approximation
        
        Args:
            n_samples: Number of parameter samples
            
        Returns:
            List of parameter dictionaries
        """
        if self.n_models < 2:
            warnings.warn("Not enough models collected for SWAG sampling")
            return [dict(self.model.named_parameters())]
        
        samples = []
        
        for _ in range(n_samples):
            sample_params = {}
            
            for name in self.mean_weights.keys():
                # Diagonal covariance approximation
                variance = torch.clamp(
                    self.sq_weights[name] - self.mean_weights[name] ** 2,
                    min=self.var_clamp
                )
                
                # Low-rank component from deviations
                if len(self.weight_deviations) > 1:
                    deviation_matrix = torch.stack([
                        dev[name] for dev in self.weight_deviations
                    ])  # [n_models, param_shape]
                    
                    # Random coefficients for low-rank component
                    coeffs = torch.randn(len(self.weight_deviations))
                    low_rank_component = torch.sum(
                        coeffs.view(-1, *([1] * deviation_matrix.dim()[1:])) * deviation_matrix,
                        dim=0
                    ) / np.sqrt(len(self.weight_deviations))
                else:
                    low_rank_component = 0
                
                # Sample parameter
                noise = torch.randn_like(self.mean_weights[name])
                sample_params[name] = (
                    self.mean_weights[name] +
                    torch.sqrt(variance) * noise +
                    low_rank_component
                )
            
            samples.append(sample_params)
        
        return samples
    
    def predict_with_uncertainty(self, x: torch.Tensor, probability: torch.Tensor,
                               n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Make predictions with SWAG uncertainty
        
        Args:
            x: Input tensor
            probability: Base probability
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            Predictions with uncertainty estimates
        """
        # Sample parameters
        param_samples = self.sample_parameters(n_samples)
        
        predictions = []
        original_params = dict(self.model.named_parameters())
        
        try:
            for sample_params in param_samples:
                # Load sampled parameters
                for name, param in self.model.named_parameters():
                    if name in sample_params:
                        param.data.copy_(sample_params[name])
                
                # Make prediction
                with torch.no_grad():
                    pred = self.model(x, probability)
                    predictions.append(pred['hybrid_output'])
        
        finally:
            # Restore original parameters
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name].data)
        
        # Compute statistics
        predictions_tensor = torch.stack(predictions)
        mean_pred = torch.mean(predictions_tensor, dim=0)
        var_pred = torch.var(predictions_tensor, dim=0)
        epistemic_uncertainty = torch.mean(var_pred, dim=-1)
        
        return {
            'mean_prediction': mean_pred,
            'predictive_variance': var_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'all_predictions': predictions_tensor
        }


class LaplaceBayesianLayer(nn.Module):
    """
    Laplace approximation for Bayesian last layer
    """
    
    def __init__(self, input_dim: int, output_dim: int, prior_precision: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_precision = prior_precision
        
        # Parameters
        self.weight_mean = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias_mean = nn.Parameter(torch.zeros(output_dim))
        
        # Precision matrix (inverse covariance) - diagonal approximation
        self.register_buffer('weight_precision', torch.ones(output_dim, input_dim) * prior_precision)
        self.register_buffer('bias_precision', torch.ones(output_dim) * prior_precision)
        
        self.fitted = False
    
    def fit_laplace(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Fit Laplace approximation using Hessian at MAP estimate
        
        Args:
            features: Input features [batch_size, input_dim]
            targets: Target values [batch_size, output_dim]
        """
        # Compute Hessian approximation (Gauss-Newton)
        batch_size = features.size(0)
        
        # Forward pass to get predictions
        predictions = F.linear(features, self.weight_mean, self.bias_mean)
        
        # Compute Jacobian w.r.t. parameters
        # For regression: Hessian ≈ J^T J where J is Jacobian of predictions w.r.t. parameters
        
        # Weight Hessian: X^T X (outer product of features)
        weight_hessian = torch.einsum('bi,bj->ij', features, features) / batch_size
        self.weight_precision = self.prior_precision + weight_hessian.unsqueeze(0).expand(self.output_dim, -1, -1).diagonal(dim1=-2, dim2=-1)
        
        # Bias Hessian: identity (since bias gradient is just the error)
        self.bias_precision = self.prior_precision + torch.ones_like(self.bias_mean)
        
        self.fitted = True
    
    def forward(self, x: torch.Tensor, n_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty sampling
        
        Args:
            x: Input features
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            Predictions with uncertainty
        """
        if n_samples == 1:
            # Deterministic forward pass
            output = F.linear(x, self.weight_mean, self.bias_mean)
            return {'mean': output, 'variance': torch.zeros_like(output)}
        
        if not self.fitted:
            warnings.warn("Laplace approximation not fitted, using prior")
        
        # Sample weights and biases
        weight_samples = []
        bias_samples = []
        
        for _ in range(n_samples):
            # Sample weights (diagonal covariance)
            weight_std = 1.0 / torch.sqrt(self.weight_precision + 1e-8)
            weight_noise = torch.randn_like(self.weight_mean)
            weight_sample = self.weight_mean + weight_std * weight_noise
            
            # Sample biases
            bias_std = 1.0 / torch.sqrt(self.bias_precision + 1e-8)
            bias_noise = torch.randn_like(self.bias_mean)
            bias_sample = self.bias_mean + bias_std * bias_noise
            
            weight_samples.append(weight_sample)
            bias_samples.append(bias_sample)
        
        # Make predictions with sampled parameters
        predictions = []
        for w, b in zip(weight_samples, bias_samples):
            pred = F.linear(x, w, b)
            predictions.append(pred)
        
        predictions_tensor = torch.stack(predictions)
        mean_pred = torch.mean(predictions_tensor, dim=0)
        var_pred = torch.var(predictions_tensor, dim=0)
        
        return {
            'mean': mean_pred,
            'variance': var_pred,
            'epistemic_uncertainty': torch.mean(var_pred, dim=-1)
        }


def test_ensemble_system():
    """Test the ensemble system"""
    print("=== Testing Ensemble System ===")
    
    # Configuration
    config = HybridConfig()
    config.ensemble_size = 3  # Small for testing
    
    # Create ensemble
    ensemble = DeepEnsemble(config, ensemble_size=3)
    
    # Test data
    batch_size = 2
    x = torch.randn(batch_size, config.input_dim)
    probability = torch.tensor([0.8, 0.75])
    
    # Forward pass
    ensemble_output = ensemble(x, probability, return_individual=True)
    
    print(f"Ensemble hybrid output shape: {ensemble_output['hybrid_output'].shape}")
    print(f"Epistemic uncertainty: {ensemble_output['epistemic_uncertainty'].detach().numpy()}")
    print(f"Ensemble weights: {ensemble_output['ensemble_weights'].detach().numpy()}")
    print(f"Number of individual outputs: {len(ensemble_output['individual_outputs'])}")
    
    # Test SWAG
    print("\n=== Testing SWAG ===")
    single_model = HybridAIPhysicsUQ(config)
    swag = SWAGApproximation(single_model, max_models=5)
    
    # Collect some models (simulate training)
    for i in range(3):
        # Simulate parameter updates
        for param in single_model.parameters():
            param.data += 0.01 * torch.randn_like(param.data)
        swag.collect_model()
    
    # Test prediction with uncertainty
    swag_output = swag.predict_with_uncertainty(x, probability, n_samples=5)
    print(f"SWAG mean prediction shape: {swag_output['mean_prediction'].shape}")
    print(f"SWAG epistemic uncertainty: {swag_output['epistemic_uncertainty'].detach().numpy()}")
    
    # Test Laplace layer
    print("\n=== Testing Laplace Layer ===")
    laplace_layer = LaplaceBayesianLayer(config.hidden_dim, config.output_dim)
    
    # Generate some training data
    features = torch.randn(50, config.hidden_dim)
    targets = torch.randn(50, config.output_dim)
    
    # Fit Laplace approximation
    laplace_layer.fit_laplace(features, targets)
    
    # Test prediction
    test_features = torch.randn(2, config.hidden_dim)
    laplace_output = laplace_layer(test_features, n_samples=10)
    print(f"Laplace mean shape: {laplace_output['mean'].shape}")
    print(f"Laplace epistemic uncertainty: {laplace_output['epistemic_uncertainty'].detach().numpy()}")


if __name__ == "__main__":
    test_ensemble_system()