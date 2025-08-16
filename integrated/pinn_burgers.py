import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore")


class SimplePINN:
    """
    Physics-Informed Neural Network for Viscous Burgers Equation:
    u_t + u*u_x - nu*u_xx = 0
    """

    def __init__(self, layers=[2, 20, 20, 1], nu=0.01 / np.pi):
        self.layers = layers
        self.nu = nu  # Viscosity parameter
        self.weights = []
        self.biases = []
        self.velocities_w = []  # For momentum
        self.velocities_b = []

        # Xavier initialization
        for i in range(len(layers) - 1):
            bound = np.sqrt(6.0 / (layers[i] + layers[i + 1]))
            w = np.random.uniform(-bound, bound, (layers[i], layers[i + 1]))
            b = np.zeros(layers[i + 1])

            self.weights.append(w)
            self.biases.append(b)
            self.velocities_w.append(np.zeros_like(w))
            self.velocities_b.append(np.zeros_like(b))

    def activation(self, x):
        """Activation function (tanh)"""
        return np.tanh(x)

    def activation_derivative(self, x):
        """Derivative of activation function"""
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        """Forward pass through the network"""
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = self.activation(z)
            else:
                a = z  # Linear output layer
        return a.flatten()

    def finite_diff_gradient(self, X, eps=1e-6):
        """Compute gradients using finite differences"""
        u = self.forward(X)
        x, t = X[:, 0], X[:, 1]

        # u_x
        X_plus_x = X.copy()
        X_plus_x[:, 0] += eps
        u_plus_x = self.forward(X_plus_x)

        X_minus_x = X.copy()
        X_minus_x[:, 0] -= eps
        u_minus_x = self.forward(X_minus_x)

        u_x = (u_plus_x - u_minus_x) / (2 * eps)

        # u_xx
        u_xx = (u_plus_x - 2 * u + u_minus_x) / (eps**2)

        # u_t
        X_plus_t = X.copy()
        X_plus_t[:, 1] += eps
        u_plus_t = self.forward(X_plus_t)

        X_minus_t = X.copy()
        X_minus_t[:, 1] -= eps
        u_minus_t = self.forward(X_minus_t)

        u_t = (u_plus_t - u_minus_t) / (2 * eps)

        return u, u_x, u_xx, u_t

    def pde_loss(self, X_pde):
        """Physics loss for Burgers equation"""
        u, u_x, u_xx, u_t = self.finite_diff_gradient(X_pde)

        # Burgers equation: u_t + u*u_x - nu*u_xx = 0
        pde_residual = u_t + u * u_x - self.nu * u_xx

        return np.mean(pde_residual**2)

    def initial_loss(self, X_ic, u_ic):
        """Initial condition loss"""
        u_pred = self.forward(X_ic)
        return np.mean((u_pred - u_ic) ** 2)

    def boundary_loss(self, X_bc, u_bc):
        """Boundary condition loss"""
        u_pred = self.forward(X_bc)
        return np.mean((u_pred - u_bc) ** 2)

    def total_loss(
        self,
        X_pde,
        X_ic,
        u_ic,
        X_bc,
        u_bc,
        lambda_pde=1.0,
        lambda_ic=10.0,
        lambda_bc=10.0,
    ):
        """Total loss function"""
        loss_pde = self.pde_loss(X_pde)
        loss_ic = self.initial_loss(X_ic, u_ic)
        loss_bc = self.boundary_loss(X_bc, u_bc)

        total = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc

        return total, loss_pde, loss_ic, loss_bc

    def compute_gradients(self, X_pde, X_ic, u_ic, X_bc, u_bc, eps=1e-6):
        """Compute gradients using finite differences"""
        base_loss, _, _, _ = self.total_loss(X_pde, X_ic, u_ic, X_bc, u_bc)

        grad_weights = []
        grad_biases = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Weight gradients
            grad_w = np.zeros_like(w)
            for j in range(w.shape[0]):
                for k in range(w.shape[1]):
                    self.weights[i][j, k] += eps
                    loss_plus, _, _, _ = self.total_loss(X_pde, X_ic, u_ic, X_bc, u_bc)
                    self.weights[i][j, k] -= 2 * eps
                    loss_minus, _, _, _ = self.total_loss(X_pde, X_ic, u_ic, X_bc, u_bc)
                    self.weights[i][j, k] += eps  # Reset

                    grad_w[j, k] = (loss_plus - loss_minus) / (2 * eps)

            # Bias gradients
            grad_b = np.zeros_like(b)
            for j in range(len(b)):
                self.biases[i][j] += eps
                loss_plus, _, _, _ = self.total_loss(X_pde, X_ic, u_ic, X_bc, u_bc)
                self.biases[i][j] -= 2 * eps
                loss_minus, _, _, _ = self.total_loss(X_pde, X_ic, u_ic, X_bc, u_bc)
                self.biases[i][j] += eps  # Reset

                grad_b[j] = (loss_plus - loss_minus) / (2 * eps)

            grad_weights.append(grad_w)
            grad_biases.append(grad_b)

        return grad_weights, grad_biases

    def train_step(
        self, X_pde, X_ic, u_ic, X_bc, u_bc, learning_rate=0.001, momentum=0.9
    ):
        """Single training step with momentum"""
        grad_w, grad_b = self.compute_gradients(X_pde, X_ic, u_ic, X_bc, u_bc)

        for i in range(len(self.weights)):
            # Update velocities (momentum)
            self.velocities_w[i] = (
                momentum * self.velocities_w[i] - learning_rate * grad_w[i]
            )
            self.velocities_b[i] = (
                momentum * self.velocities_b[i] - learning_rate * grad_b[i]
            )

            # Update parameters
            self.weights[i] += self.velocities_w[i]
            self.biases[i] += self.velocities_b[i]


def analytical_solution(x, t, nu=0.01 / np.pi):
    """
    Analytical solution for viscous Burgers equation with specific initial condition
    u(x,0) = -sin(π*x)
    """
    # This is an approximate solution for small viscosity
    # For exact solution, would need to solve the Cole-Hopf transformation
    return -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)


def rk4_burgers(x_grid, t_span, nu=0.01 / np.pi):
    """
    Solve viscous Burgers equation using RK4 method
    """
    dx = x_grid[1] - x_grid[0]

    def burgers_rhs(t, u):
        """Right-hand side of discretized Burgers equation"""
        dudt = np.zeros_like(u)

        # Interior points using central differences
        for i in range(1, len(u) - 1):
            # u_x using central difference
            u_x = (u[i + 1] - u[i - 1]) / (2 * dx)
            # u_xx using central difference
            u_xx = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx**2)

            # Burgers equation: u_t = -u*u_x + nu*u_xx
            dudt[i] = -u[i] * u_x + nu * u_xx

        # Boundary conditions (periodic or fixed)
        dudt[0] = dudt[-1] = 0  # Fixed boundaries

        return dudt

    # Initial condition
    u0 = -np.sin(np.pi * x_grid)

    # Solve using RK4
    sol = solve_ivp(
        burgers_rhs,
        t_span,
        u0,
        method="RK45",
        t_eval=np.linspace(t_span[0], t_span[1], 100),
        rtol=1e-8,
        atol=1e-10,
    )

    return sol.t, sol.y


def train_pinn_burgers():
    """Train PINN for viscous Burgers equation"""
    print("=== Training PINN for Viscous Burgers Equation ===")

    # Initialize PINN
    pinn = SimplePINN(layers=[2, 20, 20, 1], nu=0.01 / np.pi)

    # Domain setup
    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0

    # Training data
    N_pde = 1000
    N_ic = 100
    N_bc = 50

    # PDE collocation points
    x_pde = np.random.uniform(x_min, x_max, N_pde)
    t_pde = np.random.uniform(t_min, t_max, N_pde)
    X_pde = np.column_stack([x_pde, t_pde])

    # Initial condition points
    x_ic = np.linspace(x_min, x_max, N_ic)
    t_ic = np.zeros(N_ic)
    X_ic = np.column_stack([x_ic, t_ic])
    u_ic = -np.sin(np.pi * x_ic)  # Initial condition

    # Boundary condition points
    x_bc = np.concatenate([np.full(N_bc // 2, x_min), np.full(N_bc // 2, x_max)])
    t_bc = np.random.uniform(t_min, t_max, N_bc)
    X_bc = np.column_stack([x_bc, t_bc])
    u_bc = np.zeros(N_bc)  # Zero boundary conditions

    # Training loop
    epochs = 500
    losses = []

    print("Training PINN...")
    for epoch in range(epochs):
        pinn.train_step(X_pde, X_ic, u_ic, X_bc, u_bc)

        if epoch % 50 == 0:
            loss, loss_pde, loss_ic, loss_bc = pinn.total_loss(
                X_pde, X_ic, u_ic, X_bc, u_bc
            )
            losses.append(loss)
            print(
                f"Epoch {epoch}: Total Loss = {loss:.6f}, PDE = {loss_pde:.6f}, "
                f"IC = {loss_ic:.6f}, BC = {loss_bc:.6f}"
            )

    return pinn, losses


def compare_solutions():
    """Compare PINN, RK4, and analytical solutions"""
    print("\n=== Comparing Solutions ===")

    # Train PINN
    pinn, losses = train_pinn_burgers()

    # Test grid
    x_test = np.linspace(-1, 1, 100)
    t_test = 1.0
    X_test = np.column_stack([x_test, np.full(len(x_test), t_test)])

    # PINN solution
    u_pinn = pinn.forward(X_test)

    # Analytical solution
    u_analytical = analytical_solution(x_test, t_test)

    # RK4 solution
    t_rk4, u_rk4_grid = rk4_burgers(x_test, [0, t_test])
    u_rk4 = u_rk4_grid[:, -1]  # Solution at final time

    # Create comparison plot
    plt.figure(figsize=(15, 10))

    # Solution comparison
    plt.subplot(2, 2, 1)
    plt.plot(x_test, u_analytical, "k-", label="Analytical", linewidth=2)
    plt.plot(x_test, u_rk4, "b-", label="RK4", linewidth=1.5)
    plt.plot(x_test, u_pinn, "r--", label="PINN", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("u(x,t=1)")
    plt.title(f"Viscous Burgers Equation Solutions at t={t_test}")
    plt.legend()
    plt.grid(True)

    # Error analysis
    plt.subplot(2, 2, 2)
    error_rk4 = np.abs(u_rk4 - u_analytical)
    error_pinn = np.abs(u_pinn - u_analytical)
    plt.plot(x_test, error_rk4, "b-", label="RK4 Error")
    plt.plot(x_test, error_pinn, "r--", label="PINN Error")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title("Solution Errors")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")

    # Training loss
    plt.subplot(2, 2, 3)
    plt.plot(losses)
    plt.xlabel("Training Iterations (×50)")
    plt.ylabel("Total Loss")
    plt.title("PINN Training Loss")
    plt.grid(True)
    plt.yscale("log")

    # 2D solution visualization
    plt.subplot(2, 2, 4)
    x_2d = np.linspace(-1, 1, 50)
    t_2d = np.linspace(0, 1, 50)
    X_2d, T_2d = np.meshgrid(x_2d, t_2d)

    # PINN solution over space-time
    X_flat = np.column_stack([X_2d.flatten(), T_2d.flatten()])
    U_pinn_2d = pinn.forward(X_flat).reshape(X_2d.shape)

    im = plt.contourf(X_2d, T_2d, U_pinn_2d, levels=20, cmap="RdBu")
    plt.colorbar(im)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("PINN Solution u(x,t)")

    plt.tight_layout()
    plt.savefig("/workspace/pinn_burgers_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print statistics
    print(f"\nSolution Statistics at t={t_test}:")
    print(f"RK4 L2 Error: {np.sqrt(np.mean(error_rk4**2)):.6f}")
    print(f"PINN L2 Error: {np.sqrt(np.mean(error_pinn**2)):.6f}")
    print(f"RK4 Max Error: {np.max(error_rk4):.6f}")
    print(f"PINN Max Error: {np.max(error_pinn):.6f}")

    return {
        "x": x_test,
        "u_analytical": u_analytical,
        "u_rk4": u_rk4,
        "u_pinn": u_pinn,
        "losses": losses,
    }


if __name__ == "__main__":
    results = compare_solutions()
    print("Comparison completed and saved as 'pinn_burgers_comparison.png'")
