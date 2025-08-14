"""
Lightweight Bayesian Methods for Uncertainty Quantification

Includes MC Dropout, SWAG (Stochastic Weight Averaging Gaussian),
and Laplace Approximation for efficient Bayesian inference.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import copy
from scipy import linalg


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses dropout at inference time to approximate Bayesian inference,
    treating dropout as approximate variational inference.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        dropout_rate: float = 0.1,
        n_samples: int = 100
    ):
        """
        Initialize MC Dropout wrapper.
        
        Args:
            base_model: PyTorch model with dropout layers
            dropout_rate: Dropout probability
            n_samples: Number of forward passes for uncertainty estimation
        """
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        
        # Enable dropout at inference
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                module.train()  # Keep dropout active
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.base_model(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty using MC Dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples (overrides default)
            
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        n_samples = n_samples or self.n_samples
        
        # Enable dropout for MC sampling
        self._enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # Shape: (n_samples, batch, classes)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Predictive entropy (total uncertainty)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
        
        # Expected entropy (aleatoric uncertainty)
        expected_entropy = torch.mean(
            -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1),
            dim=0
        )
        
        # Mutual information (epistemic uncertainty)
        mutual_info = entropy - expected_entropy
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'entropy': entropy,
            'aleatoric': expected_entropy,
            'epistemic': mutual_info,
            'samples': predictions
        }


class SWAG:
    """
    Stochastic Weight Averaging Gaussian (SWAG).
    
    Approximates posterior distribution over weights using first and second
    moments collected during training.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        no_cov_mat: bool = False,
        max_num_models: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SWAG.
        
        Args:
            base_model: PyTorch model
            no_cov_mat: If True, only diagonal covariance
            max_num_models: Maximum number of models for covariance
            device: Device to run on
        """
        self.base_model = base_model.to(device)
        self.device = device
        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        
        # Initialize mean and variance parameters
        self.mean = {}
        self.sq_mean = {}
        self.cov_mat_sqrt = {} if not no_cov_mat else None
        
        for name, param in self.base_model.named_parameters():
            self.mean[name] = param.data.clone()
            self.sq_mean[name] = param.data.clone() ** 2
            
            if not no_cov_mat:
                self.cov_mat_sqrt[name] = torch.zeros(
                    (max_num_models, *param.shape),
                    device=device
                )
        
        self.n_models = 0
    
    def collect_model(self, model: nn.Module):
        """
        Collect model parameters for SWAG averaging.
        
        Args:
            model: Model to collect parameters from
        """
        self.n_models += 1
        
        for name, param in model.named_parameters():
            # Update running average of mean
            self.mean[name] = (self.mean[name] * (self.n_models - 1) + 
                               param.data) / self.n_models
            
            # Update running average of second moment
            self.sq_mean[name] = (self.sq_mean[name] * (self.n_models - 1) + 
                                  param.data ** 2) / self.n_models
            
            # Update covariance factor
            if not self.no_cov_mat:
                dev = param.data - self.mean[name]
                
                if self.n_models <= self.max_num_models:
                    self.cov_mat_sqrt[name][self.n_models - 1] = dev
                else:
                    # Randomly replace old deviation
                    idx = np.random.randint(0, self.max_num_models)
                    self.cov_mat_sqrt[name][idx] = dev
    
    def sample(self, scale: float = 1.0) -> nn.Module:
        """
        Sample model from SWAG posterior.
        
        Args:
            scale: Scale factor for variance
            
        Returns:
            Sampled model
        """
        model = copy.deepcopy(self.base_model)
        
        for name, param in model.named_parameters():
            # Diagonal variance
            var = torch.clamp(self.sq_mean[name] - self.mean[name] ** 2, min=1e-6)
            
            # Sample from Gaussian
            eps = torch.randn_like(param.data)
            
            if self.no_cov_mat:
                # Only diagonal covariance
                param.data = self.mean[name] + torch.sqrt(var) * scale * eps
            else:
                # Full covariance
                cov_factor = self.cov_mat_sqrt[name][:min(self.n_models, self.max_num_models)]
                
                # Low-rank covariance: Σ = D + BB^T
                z1 = torch.randn(cov_factor.size(0), device=self.device)
                z2 = torch.randn_like(param.data)
                
                param.data = (self.mean[name] + 
                             scale * torch.sqrt(var) * z2 +
                             scale / np.sqrt(2 * (self.n_models - 1)) * 
                             torch.sum(cov_factor * z1.view(-1, *([1] * len(cov_factor.shape[1:]))), dim=0))
        
        return model
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
        scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty using SWAG samples.
        
        Args:
            x: Input tensor
            n_samples: Number of model samples
            scale: Scale factor for sampling
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                model = self.sample(scale=scale)
                model.eval()
                
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Uncertainty metrics
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
        expected_entropy = torch.mean(
            -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1),
            dim=0
        )
        mutual_info = entropy - expected_entropy
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'entropy': entropy,
            'aleatoric': expected_entropy,
            'epistemic': mutual_info,
            'samples': predictions
        }


class LaplaceApproximation:
    """
    Laplace Approximation for Bayesian Neural Networks.
    
    Approximates posterior with Gaussian centered at MAP estimate,
    with covariance given by inverse Hessian.
    """
    
    def __init__(
        self,
        model: nn.Module,
        subset_of_weights: str = 'last_layer',
        hessian_structure: str = 'diag',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Laplace Approximation.
        
        Args:
            model: Trained PyTorch model (MAP estimate)
            subset_of_weights: 'last_layer' or 'all'
            hessian_structure: 'diag', 'kron', or 'full'
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure
        
        # Get parameters to approximate
        self.params = self._get_params()
        self.n_params = sum(p.numel() for p in self.params)
        
        # Initialize Hessian approximation
        self.H = None  # Will be computed during fitting
        self.scale = 1.0  # Prior precision
    
    def _get_params(self) -> List[nn.Parameter]:
        """Get parameters for Laplace approximation."""
        if self.subset_of_weights == 'last_layer':
            # Find last linear layer
            last_layer = None
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    last_layer = module
            
            if last_layer is None:
                raise ValueError("No linear layer found")
            
            return list(last_layer.parameters())
        else:
            return list(self.model.parameters())
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None
    ):
        """
        Fit Laplace approximation by computing Hessian.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function (default: CrossEntropyLoss)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        
        # Compute Hessian approximation
        if self.hessian_structure == 'diag':
            self.H = self._compute_diag_hessian(train_loader, criterion)
        elif self.hessian_structure == 'kron':
            self.H = self._compute_kron_hessian(train_loader, criterion)
        else:
            self.H = self._compute_full_hessian(train_loader, criterion)
        
        # Add prior precision
        self.H = self.H + self.scale * torch.eye(self.n_params, device=self.device)
    
    def _compute_diag_hessian(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> torch.Tensor:
        """Compute diagonal Hessian approximation."""
        H_diag = torch.zeros(self.n_params, device=self.device)
        
        for batch_x, batch_y in tqdm(train_loader, desc="Computing Hessian"):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, self.params, create_graph=True)
            
            # Flatten gradients
            grads_flat = torch.cat([g.flatten() for g in grads])
            
            # Accumulate diagonal of Hessian
            for i, grad in enumerate(grads_flat):
                grad2 = torch.autograd.grad(grad, self.params, retain_graph=True)
                grad2_flat = torch.cat([g.flatten() for g in grad2])
                H_diag[i] += grad2_flat[i].item()
        
        H_diag /= len(train_loader)
        return torch.diag(H_diag)
    
    def _compute_kron_hessian(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict:
        """Compute Kronecker-factored Hessian approximation."""
        # Simplified implementation - would need KFAC for full version
        return self._compute_diag_hessian(train_loader, criterion)
    
    def _compute_full_hessian(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> torch.Tensor:
        """Compute full Hessian matrix."""
        H = torch.zeros(self.n_params, self.n_params, device=self.device)
        
        for batch_x, batch_y in tqdm(train_loader, desc="Computing Hessian"):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Compute Hessian
            grads = torch.autograd.grad(loss, self.params, create_graph=True)
            grads_flat = torch.cat([g.flatten() for g in grads])
            
            for i in range(self.n_params):
                grad2 = torch.autograd.grad(
                    grads_flat[i], self.params, 
                    retain_graph=True
                )
                grad2_flat = torch.cat([g.flatten() for g in grad2])
                H[i] += grad2_flat
        
        H /= len(train_loader)
        return H
    
    def sample_predictions(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Sample predictions from Laplace posterior.
        
        Args:
            x: Input tensor
            n_samples: Number of samples
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Get MAP prediction
        self.model.eval()
        with torch.no_grad():
            map_logits = self.model(x)
            map_probs = F.softmax(map_logits, dim=-1)
        
        # Sample from posterior
        predictions = [map_probs]
        
        # Compute covariance (inverse Hessian)
        try:
            if self.hessian_structure == 'diag':
                H_inv_diag = 1.0 / torch.diag(self.H)
                std = torch.sqrt(H_inv_diag)
            else:
                H_inv = torch.inverse(self.H)
                std = torch.sqrt(torch.diag(H_inv))
        except:
            # If Hessian is singular, use regularization
            H_reg = self.H + 1e-4 * torch.eye(self.n_params, device=self.device)
            H_inv = torch.inverse(H_reg)
            std = torch.sqrt(torch.diag(H_inv))
        
        # Sample parameters and predict
        for _ in range(n_samples - 1):
            # Sample from Gaussian posterior
            eps = torch.randn_like(std)
            
            # Perturb parameters
            param_samples = []
            idx = 0
            for param in self.params:
                n = param.numel()
                delta = (std[idx:idx+n] * eps[idx:idx+n]).reshape(param.shape)
                param_samples.append(param + delta)
                idx += n
            
            # Temporarily replace parameters
            old_params = []
            for i, param in enumerate(self.params):
                old_params.append(param.data.clone())
                param.data = param_samples[i]
            
            # Forward pass with sampled parameters
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
            
            # Restore original parameters
            for i, param in enumerate(self.params):
                param.data = old_params[i]
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Uncertainty metrics
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
        expected_entropy = torch.mean(
            -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1),
            dim=0
        )
        mutual_info = entropy - expected_entropy
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'entropy': entropy,
            'aleatoric': expected_entropy,
            'epistemic': mutual_info,
            'samples': predictions
        }