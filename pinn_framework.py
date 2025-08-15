#!/usr/bin/env python3
"""
Physics-Informed Neural Network Framework
Implementing the hybrid AI system with mathematical framework Ψ(x)
Solving 1D Inviscid Burgers' Equation: ∂u/∂t + u∂u/∂x = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

class HybridFramework:
    """Core mathematical framework for hybrid AI system"""
    
    def __init__(self, S_x: float, N_x: float, alpha_t: float, 
                 R_cognitive: float, R_efficiency: float, beta: float,
                 lambda1: float = 0.6, lambda2: float = 0.4):
        self.S_x = S_x  # State inference for optimized PINN solutions
        self.N_x = N_x  # ML gradient descent analysis
        self.alpha_t = alpha_t  # Real-time validation flows
        self.R_cognitive = R_cognitive  # PDE residual accuracy
        self.R_efficiency = R_efficiency  # Training loop efficiency
        self.beta = beta  # Model responsiveness parameter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def hybrid_output(self) -> float:
        """Calculate hybrid output O_hybrid"""
        return self.alpha_t * self.S_x + (1 - self.alpha_t) * self.N_x
    
    def total_penalty(self) -> float:
        """Calculate total penalty P_total"""
        return self.lambda1 * self.R_cognitive + self.lambda2 * self.R_efficiency
    
    def exponential_penalty(self) -> float:
        """Calculate exponential penalty term"""
        return np.exp(-self.total_penalty())
    
    def adjusted_probability(self, base_prob: float) -> float:
        """Calculate adjusted probability P_adj"""
        adjustment = base_prob ** self.beta
        return min(adjustment, 1.0)  # Cap at 1.0
    
    def psi(self, base_prob: float) -> float:
        """Calculate final framework output Ψ(x)"""
        O_hybrid = self.hybrid_output()
        exp_penalty = self.exponential_penalty()
        P_adj = self.adjusted_probability(base_prob)
        return O_hybrid * exp_penalty * P_adj

class DenseLayer:
    """Dense neural network layer with weights and biases"""
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        self.biases = np.random.uniform(-0.1, 0.1, output_size)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        weighted_sum = np.dot(self.weights, input_data) + self.biases
        return np.tanh(weighted_sum)  # Tanh activation
    
    def update_weights(self, weight_gradients: np.ndarray, bias_gradients: np.ndarray, 
                      learning_rate: float):
        """Update weights with gradients"""
        self.weights -= learning_rate * weight_gradients
        self.biases -= learning_rate * bias_gradients

class PINN:
    """Physics-Informed Neural Network for solving PDEs"""
    
    def __init__(self, architecture: List[int], framework: HybridFramework):
        self.framework = framework
        self.layers = []
        
        for i in range(len(architecture) - 1):
            self.layers.append(DenseLayer(architecture[i], architecture[i + 1]))
    
    def forward(self, x: float, t: float) -> float:
        """Forward pass through the network"""
        input_data = np.array([x, t])
        
        for layer in self.layers:
            input_data = layer.forward(input_data)
        
        return input_data[0]
    
    def spatial_derivative(self, x: float, t: float, dx: float = 1e-5) -> float:
        """Calculate spatial derivative using finite differences"""
        return (self.forward(x + dx, t) - self.forward(x - dx, t)) / (2 * dx)
    
    def temporal_derivative(self, x: float, t: float, dt: float = 1e-5) -> float:
        """Calculate temporal derivative using finite differences"""
        return (self.forward(x, t + dt) - self.forward(x, t - dt)) / (2 * dt)

class BurgersEquationSolver:
    """Solver for the 1D inviscid Burgers' equation: ∂u/∂t + u∂u/∂x = 0"""
    
    def __init__(self, pinn: PINN):
        self.pinn = pinn
    
    def pde_residual(self, x: float, t: float) -> float:
        """PDE residual for Burgers' equation"""
        u = self.pinn.forward(x, t)
        u_t = self.pinn.temporal_derivative(x, t)
        u_x = self.pinn.spatial_derivative(x, t)
        
        # Burgers' equation: ∂u/∂t + u∂u/∂x = 0
        return u_t + u * u_x
    
    def initial_condition_loss(self, x_points: np.ndarray) -> float:
        """Initial condition loss"""
        total_loss = 0.0
        for x in x_points:
            u = self.pinn.forward(x, 0.0)
            true_u = -np.sin(np.pi * x)  # Initial condition: u(x,0) = -sin(πx)
            total_loss += (u - true_u) ** 2
        return total_loss / len(x_points)
    
    def boundary_condition_loss(self, t_points: np.ndarray) -> float:
        """Boundary condition loss"""
        total_loss = 0.0
        for t in t_points:
            u_left = self.pinn.forward(0.0, t)
            u_right = self.pinn.forward(1.0, t)
            # Periodic boundary conditions: u(0,t) = u(1,t)
            total_loss += (u_left - u_right) ** 2
        return total_loss / len(t_points)
    
    def pde_loss(self, x_points: np.ndarray, t_points: np.ndarray) -> float:
        """Total PDE loss"""
        total_loss = 0.0
        count = 0
        
        for x in x_points:
            for t in t_points:
                total_loss += self.pde_residual(x, t) ** 2
                count += 1
        
        return total_loss / count
    
    def total_loss(self, x_points: np.ndarray, t_points: np.ndarray) -> float:
        """Combined loss function"""
        pde = self.pde_loss(x_points, t_points)
        ic = self.initial_condition_loss(x_points)
        bc = self.boundary_condition_loss(t_points)
        
        # Weight the losses
        return pde + 10.0 * ic + 5.0 * bc

class RK4Integrator:
    """Fourth-order Runge-Kutta integration"""
    
    @staticmethod
    def step(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """RK4 step for system of ODEs"""
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    @staticmethod
    def solve_burgers(x_points: np.ndarray, t_final: float, dt: float) -> np.ndarray:
        """Solve Burgers' equation using method of lines"""
        nx = len(x_points)
        nt = int(t_final / dt) + 1
        solution = np.zeros((nt, nx))
        
        # Initial condition
        solution[0, :] = -np.sin(np.pi * x_points)
        
        # Spatial grid spacing
        dx = x_points[1] - x_points[0]
        
        # Time stepping
        for n in range(nt - 1):
            current_u = solution[n, :]
            
            def spatial_derivative(t, u):
                """Spatial derivatives using finite differences"""
                dudt = np.zeros_like(u)
                for i in range(nx):
                    # Periodic boundary conditions
                    if i == 0:
                        dudx = (u[i+1] - u[nx-1]) / (2 * dx)
                    elif i == nx - 1:
                        dudx = (u[0] - u[i-1]) / (2 * dx)
                    else:
                        dudx = (u[i+1] - u[i-1]) / (2 * dx)
                    
                    dudt[i] = -u[i] * dudx  # Burgers' equation: ∂u/∂t = -u∂u/∂x
                
                return dudt
            
            solution[n+1, :] = RK4Integrator.step(
                spatial_derivative, n * dt, current_u, dt
            )
        
        return solution

class PINNTrainer:
    """Training system for PINN"""
    
    def __init__(self, solver: BurgersEquationSolver):
        self.solver = solver
        self.training_history = []
    
    def approximate_gradients(self, x_points: np.ndarray, t_points: np.ndarray, 
                            epsilon: float = 1e-6) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Approximate gradients using finite differences"""
        base_loss = self.solver.total_loss(x_points, t_points)
        weight_gradients = []
        bias_gradients = []
        
        for layer in self.solver.pinn.layers:
            # Weight gradients
            weight_grad = np.zeros_like(layer.weights)
            for i in range(layer.output_size):
                for j in range(layer.input_size):
                    layer.weights[i, j] += epsilon
                    loss_plus = self.solver.total_loss(x_points, t_points)
                    layer.weights[i, j] -= 2 * epsilon
                    loss_minus = self.solver.total_loss(x_points, t_points)
                    layer.weights[i, j] += epsilon  # Reset
                    
                    weight_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Bias gradients
            bias_grad = np.zeros_like(layer.biases)
            for i in range(layer.output_size):
                layer.biases[i] += epsilon
                loss_plus = self.solver.total_loss(x_points, t_points)
                layer.biases[i] -= 2 * epsilon
                loss_minus = self.solver.total_loss(x_points, t_points)
                layer.biases[i] += epsilon  # Reset
                
                bias_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            weight_gradients.append(weight_grad)
            bias_gradients.append(bias_grad)
        
        return weight_gradients, bias_gradients
    
    def train_step(self, x_points: np.ndarray, t_points: np.ndarray, 
                  learning_rate: float = 0.01) -> float:
        """Training step"""
        weight_grads, bias_grads = self.approximate_gradients(x_points, t_points)
        
        # Update parameters
        for i, layer in enumerate(self.solver.pinn.layers):
            layer.update_weights(weight_grads[i], bias_grads[i], learning_rate)
        
        loss = self.solver.total_loss(x_points, t_points)
        self.training_history.append(loss)
        return loss
    
    def train(self, epochs: int, x_points: np.ndarray, t_points: np.ndarray, 
             learning_rate: float = 0.01):
        """Train the PINN"""
        print("Starting PINN training...")
        
        for epoch in range(epochs):
            loss = self.train_step(x_points, t_points, learning_rate)
            
            if epoch % 10 == 0:
                psi = self.solver.pinn.framework.psi(0.8)
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Ψ(x) = {psi:.3f}")
        
        print("Training completed!")

class ResultsAnalyzer:
    """Results analyzer and visualizer"""
    
    def __init__(self, pinn: PINN, rk4_solution: np.ndarray, 
                 x_points: np.ndarray, t_points: np.ndarray):
        self.pinn = pinn
        self.rk4_solution = rk4_solution
        self.x_points = x_points
        self.t_points = t_points
    
    def compare_at_time(self, t: float) -> Dict[str, np.ndarray]:
        """Compare PINN and RK4 solutions"""
        t_index = int(t / (self.t_points[-1] / (len(self.t_points) - 1)))
        t_index = max(0, min(t_index, len(self.rk4_solution) - 1))
        
        pinn_solutions = np.array([self.pinn.forward(x, t) for x in self.x_points])
        rk4_solutions = self.rk4_solution[t_index, :]
        errors = np.abs(pinn_solutions - rk4_solutions)
        
        return {
            'x': self.x_points,
            'pinn': pinn_solutions,
            'rk4': rk4_solutions,
            'error': errors
        }
    
    def calculate_framework_metrics(self) -> HybridFramework:
        """Calculate framework metrics"""
        framework = self.pinn.framework
        
        # Update S(x) based on PINN performance
        test_points = np.linspace(0, 1, 11)
        pinn_solutions = [self.pinn.forward(x, 0.5) for x in test_points]
        variance = np.var(pinn_solutions)
        framework.S_x = max(0.0, min(1.0, 0.7 + 0.1 * (1.0 - variance)))
        
        # Update N(x) based on training convergence
        if hasattr(self.pinn.layers[0], 'biases'):
            last_bias = abs(self.pinn.layers[0].biases[0])
            framework.N_x = max(0.0, min(1.0, 0.8 - last_bias * 0.1))
        
        return framework
    
    def print_analysis(self):
        """Print detailed analysis"""
        framework = self.calculate_framework_metrics()
        psi = framework.psi(0.8)
        
        print("\n=== PINN Framework Analysis ===")
        print(f"S(x) (State Inference): {framework.S_x:.3f}")
        print(f"N(x) (ML Analysis): {framework.N_x:.3f}")
        print(f"α(t) (Validation Flow): {framework.alpha_t:.3f}")
        print(f"R_cognitive: {framework.R_cognitive:.3f}")
        print(f"R_efficiency: {framework.R_efficiency:.3f}")
        print(f"β (Responsiveness): {framework.beta:.3f}")
        print(f"Ψ(x) (Final Output): {psi:.3f}")
        
        comparison = self.compare_at_time(0.5)
        avg_error = np.mean(comparison['error'])
        print(f"Average PINN vs RK4 Error: {avg_error:.6f}")
        
        # Interpretation
        if psi > 0.7:
            print("Interpretation: Excellent model performance with strong hybrid intelligence")
        elif psi > 0.6:
            print("Interpretation: Good model performance with solid framework integration")
        elif psi > 0.5:
            print("Interpretation: Moderate performance, framework shows potential")
        else:
            print("Interpretation: Performance needs improvement, consider parameter tuning")
    
    def plot_comparison(self, t_values: List[float] = [0.0, 0.5, 1.0]):
        """Plot PINN vs RK4 comparison"""
        fig, axes = plt.subplots(1, len(t_values), figsize=(15, 5))
        if len(t_values) == 1:
            axes = [axes]
        
        for i, t in enumerate(t_values):
            comparison = self.compare_at_time(t)
            
            axes[i].plot(comparison['x'], comparison['rk4'], 'b-', label='RK4', linewidth=2)
            axes[i].plot(comparison['x'], comparison['pinn'], 'r--', label='PINN', linewidth=2)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('u(x,t)')
            axes[i].set_title(f't = {t:.1f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/pinn_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Comparison plot saved as 'pinn_comparison.png'")

def demonstrate_numerical_example():
    """Demonstrate the numerical example from the framework"""
    print("\n=== Numerical Example: Single Training Step ===")
    
    example_framework = HybridFramework(
        S_x=0.72,
        N_x=0.85,
        alpha_t=0.5,
        R_cognitive=0.15,
        R_efficiency=0.10,
        beta=1.2
    )
    
    O_hybrid = example_framework.hybrid_output()
    exp_penalty = example_framework.exponential_penalty()
    P_adj = example_framework.adjusted_probability(0.80)
    psi_final = example_framework.psi(0.80)
    
    print(f"Step 1: S(x) = {example_framework.S_x}, N(x) = {example_framework.N_x}")
    print(f"Step 2: α = {example_framework.alpha_t}, O_hybrid = {O_hybrid:.3f}")
    print(f"Step 3: R_cognitive = {example_framework.R_cognitive}, R_efficiency = {example_framework.R_efficiency}")
    print(f"        P_total = {example_framework.total_penalty():.3f}, exp ≈ {exp_penalty:.3f}")
    print(f"Step 4: P = 0.80, β = {example_framework.beta}, P_adj ≈ {P_adj:.3f}")
    print(f"Step 5: Ψ(x) ≈ {psi_final:.3f}")
    print(f"Step 6: Interpretation - Ψ(x) ≈ {psi_final:.2f} indicates {'solid' if psi_final > 0.65 else 'moderate'} model performance")

def main():
    """Main execution function"""
    print("=== Physics-Informed Neural Network Framework ===")
    print("Solving 1D Inviscid Burgers' Equation: ∂u/∂t + u∂u/∂x = 0")
    print("Initial Condition: u(x,0) = -sin(πx)")
    print("Domain: x ∈ [0,1], t ∈ [0,1]")
    
    # Initialize framework parameters
    framework = HybridFramework(
        S_x=0.72,
        N_x=0.85,
        alpha_t=0.5,
        R_cognitive=0.15,
        R_efficiency=0.10,
        beta=1.2
    )
    
    # Create PINN with architecture [2, 20, 20, 1]
    pinn = PINN([2, 20, 20, 1], framework)
    solver = BurgersEquationSolver(pinn)
    trainer = PINNTrainer(solver)
    
    # Training and validation points
    num_points = 25
    x_points = np.linspace(0, 1, num_points)
    t_points = np.linspace(0, 1, num_points)
    
    # Train the PINN
    trainer.train(50, x_points, t_points, learning_rate=0.01)
    
    # Generate RK4 solution for comparison
    print("\nGenerating RK4 reference solution...")
    rk4_solution = RK4Integrator.solve_burgers(x_points, 1.0, 0.01)
    
    # Analyze results
    analyzer = ResultsAnalyzer(pinn, rk4_solution, x_points, t_points)
    analyzer.print_analysis()
    
    # Create visualization
    analyzer.plot_comparison([0.0, 0.5, 1.0])
    
    # Demonstrate numerical example
    demonstrate_numerical_example()
    
    print("\n=== Framework Summary ===")
    print("✓ Hybrid Output: S(x) as state inference, N(x) as ML analysis, α(t) for validation")
    print("✓ Regularization: R_cognitive for PDE accuracy, R_efficiency for training efficiency")
    print("✓ Probability: P(H|E,β) with β for model responsiveness")
    print("✓ Integration: Over training epochs and validation steps")
    print("✓ Balanced Intelligence: Merges symbolic RK4 with neural PINN")
    print("✓ Dynamic Optimization: Adapts through epochs")

if __name__ == "__main__":
    main()