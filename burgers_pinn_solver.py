"""
Viscous Burgers Equation Solver with PINN and RK4 Comparison

This module implements a Physics-Informed Neural Network (PINN) for solving
the viscous Burgers equation: u_t + u*u_x - ν*u_xx = 0

The implementation includes:
- PINN with automatic differentiation
- RK4 finite difference solver for comparison  
- Visualization and accuracy comparison
- Integration with the hybrid accuracy functional
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for the viscous Burgers equation.
    
    The network learns to satisfy both the PDE and boundary/initial conditions.
    """
    
    def __init__(self, layers: list = [2, 50, 50, 50, 1]):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Build network layers with Xavier initialization
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            # Xavier initialization for better convergence
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        inputs = torch.cat([x, t], dim=1)
        
        for i, layer in enumerate(self.layers[:-1]):
            inputs = torch.tanh(layer(inputs))  # Tanh activation
        
        # Linear output layer
        output = self.layers[-1](inputs)
        return output
    
    def physics_loss(self, x: torch.Tensor, t: torch.Tensor, nu: float = 0.01/np.pi) -> torch.Tensor:
        """
        Compute the physics loss for the viscous Burgers equation.
        
        PDE: u_t + u*u_x - ν*u_xx = 0
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        
        # Compute gradients using automatic differentiation
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                                 create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                 create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                  create_graph=True)[0]
        
        # Viscous Burgers equation residual
        pde_residual = u_t + u * u_x - nu * u_xx
        
        return torch.mean(pde_residual**2)
    
    def boundary_loss(self, x_bc: torch.Tensor, t_bc: torch.Tensor, 
                     u_bc: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss."""
        u_pred = self.forward(x_bc, t_bc)
        return torch.mean((u_pred - u_bc)**2)


class BurgersSolver:
    """
    Solver for the viscous Burgers equation using both PINN and traditional methods.
    """
    
    def __init__(self, nu: float = 0.01/np.pi):
        self.nu = nu  # Viscosity parameter
        self.pinn = PINN()
        
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """
        Initial condition: u(x, 0) = -sin(πx)
        
        This creates a smooth wave that will develop into a shock-like structure.
        """
        return -np.sin(np.pi * x)
    
    def analytical_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Analytical solution for small viscosity (Cole-Hopf transformation).
        
        For the viscous Burgers equation with u(x,0) = -sin(πx).
        """
        # This is an approximation for small viscosity
        # The exact solution involves infinite series
        phi = np.exp(-self.nu * np.pi**2 * t) * np.sin(np.pi * x)
        return -2 * self.nu * np.pi * phi / (1 + np.exp(-self.nu * np.pi**2 * t) * np.cos(np.pi * x))
    
    def solve_rk4(self, x_grid: np.ndarray, t_final: float, 
                  nt: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using RK4 finite difference method.
        
        Returns:
            t_grid: Time points
            u_solution: Solution array [nt, nx]
        """
        nx = len(x_grid)
        dx = x_grid[1] - x_grid[0]
        dt = t_final / nt
        t_grid = np.linspace(0, t_final, nt + 1)
        
        # Initialize solution array
        u_solution = np.zeros((nt + 1, nx))
        u_solution[0, :] = self.initial_condition(x_grid)
        
        def rhs(u):
            """Right-hand side of the Burgers equation."""
            # Compute spatial derivatives using finite differences
            u_x = np.gradient(u, dx)
            u_xx = np.gradient(u_x, dx)
            
            # Burgers equation: u_t = -u*u_x + ν*u_xx
            return -u * u_x + self.nu * u_xx
        
        # RK4 time integration
        for n in range(nt):
            u_n = u_solution[n, :]
            
            k1 = dt * rhs(u_n)
            k2 = dt * rhs(u_n + k1/2)
            k3 = dt * rhs(u_n + k2/2)
            k4 = dt * rhs(u_n + k3)
            
            u_solution[n + 1, :] = u_n + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return t_grid, u_solution
    
    def train_pinn(self, x_domain: np.ndarray, t_domain: np.ndarray, 
                   epochs: int = 1000, lr: float = 0.001) -> list:
        """
        Train the PINN to solve the Burgers equation.
        
        Args:
            x_domain: Spatial domain points
            t_domain: Temporal domain points  
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            loss_history: Training loss history
        """
        optimizer = optim.Adam(self.pinn.parameters(), lr=lr)
        loss_history = []
        
        # Create training data
        X, T = np.meshgrid(x_domain, t_domain)
        x_train = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32)
        t_train = torch.tensor(T.flatten().reshape(-1, 1), dtype=torch.float32)
        
        # Initial condition data
        x_ic = torch.tensor(x_domain.reshape(-1, 1), dtype=torch.float32)
        t_ic = torch.zeros_like(x_ic)
        u_ic = torch.tensor(self.initial_condition(x_domain).reshape(-1, 1), dtype=torch.float32)
        
        # Boundary conditions (periodic)
        x_bc = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
        t_bc = torch.tensor([[0.5], [0.5]], dtype=torch.float32)  # Sample time
        u_bc = torch.zeros_like(x_bc)  # Periodic boundary
        
        print("Training PINN...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Physics loss
            physics_loss = self.pinn.physics_loss(x_train, t_train, self.nu)
            
            # Initial condition loss
            ic_loss = self.pinn.boundary_loss(x_ic, t_ic, u_ic)
            
            # Boundary condition loss (simplified)
            bc_loss = self.pinn.boundary_loss(x_bc, t_bc, u_bc)
            
            # Total loss
            total_loss = physics_loss + 10 * ic_loss + bc_loss
            
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(total_loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
        
        print("PINN training completed.")
        return loss_history
    
    def evaluate_pinn(self, x_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Evaluate PINN solution on a grid."""
        self.pinn.eval()
        
        X, T = np.meshgrid(x_grid, t_grid)
        x_eval = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32)
        t_eval = torch.tensor(T.flatten().reshape(-1, 1), dtype=torch.float32)
        
        with torch.no_grad():
            u_pred = self.pinn(x_eval, t_eval).numpy()
        
        return u_pred.reshape(len(t_grid), len(x_grid))
    
    def compare_solutions(self, x_range: Tuple[float, float] = (-1, 1), 
                         t_final: float = 1.0, nx: int = 100, nt: int = 200):
        """
        Compare PINN and RK4 solutions and compute accuracy metrics.
        """
        # Setup grids
        x_grid = np.linspace(x_range[0], x_range[1], nx)
        t_domain = np.linspace(0, t_final, 50)  # Training time points
        
        # Train PINN
        loss_history = self.train_pinn(x_grid, t_domain, epochs=1000)
        
        # Solve with RK4
        t_grid, u_rk4 = self.solve_rk4(x_grid, t_final, nt)
        
        # Evaluate PINN
        u_pinn = self.evaluate_pinn(x_grid, t_grid)
        
        # Compute accuracies at final time
        u_rk4_final = u_rk4[-1, :]
        u_pinn_final = u_pinn[-1, :]
        
        # Compute metrics
        mse_pinn_rk4 = np.mean((u_pinn_final - u_rk4_final)**2)
        
        # Symbolic accuracy (RK4 fidelity) - higher for stable regions
        S_accuracy = 1.0 - np.clip(mse_pinn_rk4 / np.var(u_rk4_final), 0, 1)
        
        # Neural accuracy (PINN performance) - adaptive to complexity
        neural_loss = loss_history[-1] if loss_history else 0.1
        N_accuracy = np.exp(-neural_loss)  # Convert loss to accuracy
        
        return {
            'x_grid': x_grid,
            't_grid': t_grid,
            'u_rk4': u_rk4,
            'u_pinn': u_pinn,
            'S_accuracy': S_accuracy,
            'N_accuracy': N_accuracy,
            'mse': mse_pinn_rk4,
            'loss_history': loss_history
        }
    
    def plot_comparison(self, results: dict, t_plot: float = 1.0):
        """Plot comparison between PINN and RK4 solutions."""
        x_grid = results['x_grid']
        t_grid = results['t_grid']
        u_rk4 = results['u_rk4']
        u_pinn = results['u_pinn']
        
        # Find closest time index
        t_idx = np.argmin(np.abs(t_grid - t_plot))
        
        plt.figure(figsize=(12, 4))
        
        # Solution comparison
        plt.subplot(1, 3, 1)
        plt.plot(x_grid, u_rk4[t_idx, :], 'b-', label='RK4', linewidth=2)
        plt.plot(x_grid, u_pinn[t_idx, :], 'r--', label='PINN', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f'Solution at t = {t_plot:.1f}')
        plt.legend()
        plt.grid(True)
        
        # Error plot
        plt.subplot(1, 3, 2)
        error = np.abs(u_pinn[t_idx, :] - u_rk4[t_idx, :])
        plt.plot(x_grid, error, 'g-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('|u_PINN - u_RK4|')
        plt.title('Absolute Error')
        plt.grid(True)
        
        # Training loss
        plt.subplot(1, 3, 3)
        plt.semilogy(results['loss_history'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PINN Training Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/workspace/burgers_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print accuracy metrics
        print(f"\nAccuracy Metrics:")
        print(f"Symbolic Accuracy (S): {results['S_accuracy']:.3f}")
        print(f"Neural Accuracy (N): {results['N_accuracy']:.3f}")
        print(f"MSE between solutions: {results['mse']:.6f}")


def demonstrate_burgers_pinn():
    """Demonstrate the Burgers PINN solver and integration with hybrid functional."""
    print("Viscous Burgers Equation: PINN vs RK4 Comparison")
    print("=" * 50)
    
    # Initialize solver
    solver = BurgersSolver(nu=0.01/np.pi)
    
    # Compare solutions
    results = solver.compare_solutions(x_range=(-1, 1), t_final=1.0, nx=50, nt=100)
    
    # Plot results
    solver.plot_comparison(results, t_plot=1.0)
    
    return results


if __name__ == "__main__":
    # Demonstrate the Burgers PINN solver
    results = demonstrate_burgers_pinn()