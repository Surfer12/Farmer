"""
Deep Ensemble for Epistemic Uncertainty Quantification

Strong baseline method that trains multiple neural networks with different
random initializations to capture model uncertainty.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import copy


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty quantification.
    
    Trains n_models independently with different random seeds to capture
    epistemic (model) uncertainty through disagreement between models.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        n_models: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Deep Ensemble.
        
        Args:
            base_model: PyTorch model to use as base architecture
            n_models: Number of models in ensemble (typically 5)
            device: Device to run models on
        """
        self.n_models = n_models
        self.device = device
        self.base_model = base_model
        
        # Create ensemble members
        self.models = []
        for i in range(n_models):
            model = copy.deepcopy(base_model)
            # Different initialization for each model
            self._reinitialize_model(model, seed=i)
            model.to(device)
            self.models.append(model)
    
    def _reinitialize_model(self, model: nn.Module, seed: int):
        """Reinitialize model weights with different random seed."""
        torch.manual_seed(seed)
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train ensemble members independently.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Whether to show progress
            
        Returns:
            Training history dictionary
        """
        history = {'train_loss': [], 'val_loss': []}
        
        for i, model in enumerate(self.models):
            if verbose:
                print(f"\nTraining model {i+1}/{self.n_models}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            model_history = self._train_single_model(
                model, train_loader, val_loader, 
                optimizer, criterion, epochs, verbose
            )
            
            history['train_loss'].append(model_history['train_loss'])
            if val_loader:
                history['val_loss'].append(model_history['val_loss'])
        
        return history
    
    def _train_single_model(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        verbose: bool
    ) -> Dict[str, List[float]]:
        """Train a single model in the ensemble."""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        return history
    
    def predict(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty metrics
            
        Returns:
            Dictionary containing:
                - 'mean': Mean prediction across ensemble
                - 'std': Standard deviation (epistemic uncertainty)
                - 'entropy': Predictive entropy
                - 'mutual_info': Mutual information (epistemic uncertainty)
                - 'predictions': Individual model predictions
        """
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions)  # Shape: (n_models, n_samples, n_classes)
        
        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)
        
        results = {'mean': mean_pred}
        
        if return_uncertainty:
            # Standard deviation (epistemic uncertainty)
            std_pred = np.std(predictions, axis=0)
            
            # Predictive entropy (total uncertainty)
            entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=-1)
            
            # Mutual information (epistemic uncertainty)
            # MI = H[E[p]] - E[H[p]]
            expected_entropy = np.mean(
                -np.sum(predictions * np.log(predictions + 1e-8), axis=-1),
                axis=0
            )
            mutual_info = entropy - expected_entropy
            
            results.update({
                'std': std_pred,
                'entropy': entropy,
                'mutual_info': mutual_info,
                'aleatoric': expected_entropy,
                'epistemic': mutual_info,
                'predictions': predictions
            })
        
        return results
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get probability predictions (mean across ensemble)."""
        return self.predict(x, return_uncertainty=False)['mean']
    
    def get_uncertainty_metrics(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Get comprehensive uncertainty metrics."""
        results = self.predict(x, return_uncertainty=True)
        
        # Additional metrics
        # Variation ratio: frequency of non-modal predictions
        mode_counts = np.apply_along_axis(
            lambda p: np.argmax(np.bincount(p)),
            axis=0,
            arr=np.argmax(results['predictions'], axis=-1)
        )
        variation_ratio = 1 - (mode_counts / self.n_models)
        
        return {
            'epistemic_uncertainty': results['epistemic'],
            'aleatoric_uncertainty': results['aleatoric'],
            'total_uncertainty': results['entropy'],
            'mutual_information': results['mutual_info'],
            'variation_ratio': variation_ratio,
            'prediction_std': np.mean(results['std'], axis=-1)
        }
    
    def save(self, path: str):
        """Save ensemble models."""
        torch.save({
            'n_models': self.n_models,
            'models_state_dict': [model.state_dict() for model in self.models],
            'base_model_class': self.base_model.__class__.__name__
        }, path)
    
    def load(self, path: str):
        """Load ensemble models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.n_models = checkpoint['n_models']
        
        for i, state_dict in enumerate(checkpoint['models_state_dict']):
            self.models[i].load_state_dict(state_dict)


class HeteroscedasticNN(nn.Module):
    """
    Neural network with heteroscedastic (input-dependent) uncertainty.
    
    Predicts both mean and variance for each output, capturing aleatoric uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu'
    ):
        """
        Initialize heteroscedastic neural network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (for classification or regression)
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        
        self.activation = self._get_activation(activation)
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Separate heads for mean and log-variance
        self.mean_head = nn.Linear(prev_dim, output_dim)
        self.log_var_head = nn.Linear(prev_dim, output_dim)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and variance.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        
        # Ensure positive variance
        variance = torch.exp(log_var)
        
        return mean, variance
    
    def sample(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Sample from predictive distribution.
        
        Args:
            x: Input tensor
            n_samples: Number of samples to draw
            
        Returns:
            Samples from predictive distribution
        """
        mean, variance = self.forward(x)
        std = torch.sqrt(variance)
        
        # Sample from Gaussian
        eps = torch.randn(n_samples, *mean.shape)
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        
        return samples
    
    def negative_log_likelihood(
        self,
        y_true: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for heteroscedastic regression.
        
        Args:
            y_true: True targets
            mean: Predicted mean
            variance: Predicted variance
            
        Returns:
            NLL loss
        """
        # Gaussian negative log-likelihood
        nll = 0.5 * (torch.log(2 * np.pi * variance) + 
                     (y_true - mean) ** 2 / variance)
        
        return torch.mean(nll)