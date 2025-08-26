#!/usr/bin/env python3
"""
Custom Loss Functions Based on Ψ(x) Framework
"""

import torch
import torch.nn as nn
from python.enhanced_psi_minimal import EnhancedPsiFramework

class PsiMaximizationLoss(nn.Module):
    """
    Loss function that maximizes Ψ(x) by minimizing -Ψ(x)
    """
    def __init__(self):
        super().__init__()
        self.framework = EnhancedPsiFramework()
    
    def forward(self, predictions, content_batch, time_batch):
        """
        predictions: Neural network outputs
        content_batch: List of content strings
        time_batch: Time values
        """
        total_loss = 0.0
        
        for i, (pred, content, t) in enumerate(zip(predictions, content_batch, time_batch)):
            # Compute Ψ(x) for this sample
            result = self.framework.compute_enhanced_psi(content, 'md', t.item())
            psi_value = result['psi_final']
            
            # Loss = -Ψ(x) (minimize negative to maximize Ψ(x))
            total_loss += -psi_value
        
        return total_loss / len(predictions)

class HybridSymbolicNeuralLoss(nn.Module):
    """
    Loss function that balances symbolic and neural components
    Based on your Ψ(x) formulation
    """
    def __init__(self, lambda1=0.75, lambda2=0.25):
        super().__init__()
        self.lambda1 = lambda1  # Weight for symbolic component
        self.lambda2 = lambda2  # Weight for neural component
        self.mse = nn.MSELoss()
    
    def forward(self, neural_pred, symbolic_target, neural_target, alpha_weights):
        """
        neural_pred: Neural network predictions
        symbolic_target: Target symbolic accuracy S(x,t)
        neural_target: Target neural accuracy N(x,t)  
        alpha_weights: Adaptive weights α(t)
        """
        # Weighted combination like in Ψ(x)
        hybrid_target = alpha_weights * symbolic_target + (1 - alpha_weights) * neural_target
        
        # MSE loss between prediction and hybrid target
        prediction_loss = self.mse(neural_pred, hybrid_target)
        
        # Regularization terms (like R_cognitive and R_efficiency in Ψ(x))
        cognitive_penalty = torch.mean(torch.abs(neural_pred - symbolic_target))
        efficiency_penalty = torch.mean(neural_pred ** 2)  # L2 regularization
        
        total_loss = (prediction_loss + 
                     self.lambda1 * cognitive_penalty + 
                     self.lambda2 * efficiency_penalty)
        
        return total_loss

class PhysicsInformedPsiLoss(nn.Module):
    """
    Physics-informed loss that incorporates PDE constraints from your framework
    """
    def __init__(self, physics_weight=1.0):
        super().__init__()
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, x_coords, t_coords):
        """
        predictions: Neural network outputs
        targets: Target Ψ(x,t) values
        x_coords, t_coords: Spatial and temporal coordinates
        """
        # Data loss
        data_loss = self.mse(predictions, targets)
        
        # Physics loss (PDE residual)
        # For your system: ∂Ψ/∂t = f(Ψ, ∂Ψ/∂x, ∂²Ψ/∂x², ...)
        physics_loss = self.compute_pde_residual(predictions, x_coords, t_coords)
        
        return data_loss + self.physics_weight * physics_loss
    
    def compute_pde_residual(self, psi, x, t):
        """Compute PDE residual for your specific system"""
        # Enable gradient computation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Compute derivatives
        psi_t = torch.autograd.grad(psi.sum(), t, create_graph=True)[0]
        psi_x = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x.sum(), x, create_graph=True)[0]
        
        # Your PDE (example: diffusion-like equation)
        # Modify this based on your actual system dynamics
        residual = psi_t - 0.1 * psi_xx + 0.01 * psi * psi_x
        
        return torch.mean(residual ** 2)

def demonstrate_psi_loss_functions():
    """Demonstrate how to use Ψ(x)-based loss functions"""
    print("🔥 Ψ(x)-Based Loss Functions Demo")
    print("=" * 50)
    
    # Sample data
    batch_size = 3
    predictions = torch.randn(batch_size, 1, requires_grad=True)
    
    # Test 1: Ψ(x) Maximization Loss
    print("\n1. Ψ(x) Maximization Loss:")
    psi_loss = PsiMaximizationLoss()
    content_batch = [
        "Mathematical analysis with equations",
        "Neural network optimization",
        "Physics-informed modeling"
    ]
    time_batch = torch.tensor([0.0, 1.0, 2.0])
    
    loss1 = psi_loss(predictions, content_batch, time_batch)
    print(f"   Loss (negative Ψ(x)): {loss1.item():.4f}")
    
    # Test 2: Hybrid Symbolic-Neural Loss
    print("\n2. Hybrid Symbolic-Neural Loss:")
    hybrid_loss = HybridSymbolicNeuralLoss()
    symbolic_target = torch.tensor([[0.3], [0.4], [0.5]])
    neural_target = torch.tensor([[0.6], [0.7], [0.8]])
    alpha_weights = torch.tensor([[0.4], [0.5], [0.6]])
    
    loss2 = hybrid_loss(predictions, symbolic_target, neural_target, alpha_weights)
    print(f"   Hybrid loss: {loss2.item():.4f}")
    
    # Test 3: Physics-Informed Loss
    print("\n3. Physics-Informed Ψ(x) Loss:")
    physics_loss = PhysicsInformedPsiLoss()
    targets = torch.randn(batch_size, 1)
    x_coords = torch.linspace(0, 1, batch_size, requires_grad=True)
    t_coords = torch.linspace(0, 1, batch_size, requires_grad=True)
    
    loss3 = physics_loss(predictions, targets, x_coords, t_coords)
    print(f"   Physics-informed loss: {loss3.item():.4f}")
    
    print("\n✅ All loss functions working!")
    print("\nKey insight: Ψ(x) can be used to CREATE loss functions")
    print("that guide neural network training toward better hybrid performance.")

if __name__ == "__main__":
    demonstrate_psi_loss_functions()
