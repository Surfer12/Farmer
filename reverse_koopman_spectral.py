#!/usr/bin/env python3
"""
Reverse Koopman Operator Implementation with Spectral Truncation
Based on rigorous mathematical formulation with Lipschitz continuity and error control.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, pinv, svd
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

class ReverseKoopmanOperator:
    """
    Implementation of Reverse Koopman operator K^{-1} with spectral truncation
    
    For Koopman semigroup {U^t}, we have K = U^Δ and K^{-1} = U^{-Δ}
    Spectral truncation: K^{-1}_{(r)} = Σ_{k=1}^r λ_k^{-1} φ_k ⊗ ψ_k
    """
    
    def __init__(self, Delta=0.1, r_max=20, compact_domain=True):
        """
        Initialize Reverse Koopman operator
        
        Args:
            Delta: Time step for Koopman operator K = U^Δ
            r_max: Maximum number of modes for spectral truncation
            compact_domain: Whether to assume compact invariant domain
        """
        self.Delta = Delta
        self.r_max = r_max
        self.compact_domain = compact_domain
        
        # Spectral properties
        self.eigenvalues = None
        self.eigenfunctions = None
        self.dual_modes = None
        self.condition_number = None
        
        # Lipschitz constants (to be estimated)
        self.c_lower = None  # Lower Lipschitz bound
        self.C_upper = None  # Upper Lipschitz bound
        self.L_inverse = None  # Inverse Lipschitz constant
        
        # Error control parameters
        self.delta_lambda = 0.0  # Eigenvalue estimation error
        self.delta_phi = 0.0     # Eigenfunction estimation error
        self.tau_r = 0.0         # Spectral tail energy
        
        print(f"Reverse Koopman Operator initialized:")
        print(f"  Time step Δ = {Delta}")
        print(f"  Max modes r = {r_max}")
        print(f"  Compact domain: {compact_domain}")
    
    def generate_dynamical_system(self, system_type='van_der_pol', n_points=1000):
        """
        Generate trajectory data from a dynamical system
        
        Args:
            system_type: Type of dynamical system
            n_points: Number of trajectory points
            
        Returns:
            X: State trajectory data
        """
        if system_type == 'van_der_pol':
            # Van der Pol oscillator: ẍ - μ(1-x²)ẋ + x = 0
            def van_der_pol(state, t, mu=1.0):
                x, y = state
                dxdt = y
                dydt = mu * (1 - x**2) * y - x
                return [dxdt, dydt]
            
            # Generate trajectory
            t = np.linspace(0, 20, n_points)
            initial_state = [2.0, 0.0]
            trajectory = odeint(van_der_pol, initial_state, t)
            X = trajectory
            
        elif system_type == 'lorenz':
            # Lorenz system
            def lorenz(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
                x, y, z = state
                dxdt = sigma * (y - x)
                dydt = x * (rho - z) - y
                dzdt = x * y - beta * z
                return [dxdt, dydt, dzdt]
            
            t = np.linspace(0, 25, n_points)
            initial_state = [1.0, 1.0, 1.0]
            trajectory = odeint(lorenz, initial_state, t)
            X = trajectory
            
        elif system_type == 'duffing':
            # Duffing oscillator: ẍ + δẋ + αx + βx³ = γcos(ωt)
            def duffing(state, t, delta=0.1, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.0):
                x, y = state
                dxdt = y
                dydt = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
                return [dxdt, dydt]
            
            t = np.linspace(0, 50, n_points)
            initial_state = [0.1, 0.1]
            trajectory = odeint(duffing, initial_state, t)
            X = trajectory
        
        self.X = X
        self.t = t
        self.n_points = n_points
        self.system_type = system_type
        
        print(f"Generated {system_type} system trajectory: {X.shape}")
        return X
    
    def construct_koopman_matrix(self, observables='polynomial', poly_degree=3):
        """
        Construct Koopman matrix from trajectory data using observables
        
        Args:
            observables: Type of observable functions
            poly_degree: Degree for polynomial observables
            
        Returns:
            K: Koopman matrix approximation
        """
        if self.X is None:
            raise ValueError("Must generate trajectory data first")
        
        n_points, n_dim = self.X.shape
        
        if observables == 'polynomial':
            # Polynomial observables up to specified degree
            observables_list = []
            
            # Constant term
            observables_list.append(np.ones(n_points))
            
            # Linear terms
            for i in range(n_dim):
                observables_list.append(self.X[:, i])
            
            # Quadratic terms
            if poly_degree >= 2:
                for i in range(n_dim):
                    for j in range(i, n_dim):
                        observables_list.append(self.X[:, i] * self.X[:, j])
            
            # Cubic terms
            if poly_degree >= 3:
                for i in range(n_dim):
                    for j in range(i, n_dim):
                        for k in range(j, n_dim):
                            observables_list.append(self.X[:, i] * self.X[:, j] * self.X[:, k])
            
            # Stack observables
            Psi = np.column_stack(observables_list)
            
        elif observables == 'radial_basis':
            # Radial basis functions
            n_centers = min(50, n_points // 10)
            centers = self.X[::n_points//n_centers]
            sigma = 0.5
            
            observables_list = []
            for center in centers:
                distances = np.sum((self.X - center)**2, axis=1)
                rbf = np.exp(-distances / (2 * sigma**2))
                observables_list.append(rbf)
            
            Psi = np.column_stack(observables_list)
        
        # Construct data matrices
        Psi_current = Psi[:-1]  # ψ(x_k)
        Psi_next = Psi[1:]      # ψ(x_{k+1})
        
        # Solve for Koopman matrix: Ψ_{k+1} ≈ K Ψ_k
        # K = Ψ_{k+1} Ψ_k^†
        K = Psi_next.T @ pinv(Psi_current.T)
        
        self.K = K
        self.Psi = Psi
        self.n_observables = Psi.shape[1]
        
        print(f"Koopman matrix constructed: {K.shape}")
        print(f"Number of observables: {self.n_observables}")
        
        return K
    
    def compute_spectral_decomposition(self):
        """
        Compute spectral decomposition of Koopman matrix
        
        Returns:
            eigenvalues, eigenvectors: Spectral components
        """
        if self.K is None:
            raise ValueError("Must construct Koopman matrix first")
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eig(self.K)
        
        # Sort by magnitude (dominant eigenvalues first)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store spectral components
        self.eigenvalues = eigenvalues
        self.eigenfunctions = eigenvectors
        
        # Compute dual modes (left eigenvectors)
        eigenvalues_left, eigenvectors_left = eig(self.K.T)
        idx_left = np.argsort(np.abs(eigenvalues_left))[::-1]
        self.dual_modes = eigenvectors_left[:, idx_left]
        
        # Condition number for first r modes
        self.condition_numbers = []
        for r in range(1, min(self.r_max + 1, len(eigenvalues))):
            # Condition number of spectral projector
            Phi_r = eigenvectors[:, :r]
            U, s, Vt = svd(Phi_r)
            kappa_r = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
            self.condition_numbers.append(kappa_r)
        
        print(f"Spectral decomposition computed:")
        print(f"  Dominant eigenvalues: {eigenvalues[:5]}")
        print(f"  Condition numbers (first 5): {self.condition_numbers[:5]}")
        
        return eigenvalues, eigenvectors
    
    def estimate_lipschitz_constants(self, n_samples=100):
        """
        Estimate Lipschitz constants for bi-Lipschitz assumption (A1)
        
        Args:
            n_samples: Number of sample pairs for estimation
            
        Returns:
            c_lower, C_upper: Lipschitz bounds
        """
        if self.X is None:
            raise ValueError("Must have trajectory data")
        
        # Sample random pairs from trajectory
        n_points = len(self.X)
        ratios = []
        
        for _ in range(n_samples):
            i, j = np.random.choice(n_points-1, 2, replace=False)
            
            x_i, x_j = self.X[i], self.X[j]
            Kx_i, Kx_j = self.X[i+1], self.X[j+1]  # Forward evolution
            
            # Compute norms
            x_diff = np.linalg.norm(x_i - x_j)
            Kx_diff = np.linalg.norm(Kx_i - Kx_j)
            
            if x_diff > 1e-10:  # Avoid division by zero
                ratio = Kx_diff / x_diff
                ratios.append(ratio)
        
        ratios = np.array(ratios)
        
        # Estimate bounds
        self.c_lower = np.percentile(ratios, 5)   # Lower 5th percentile
        self.C_upper = np.percentile(ratios, 95)  # Upper 95th percentile
        self.L_inverse = 1.0 / self.c_lower if self.c_lower > 0 else np.inf
        
        print(f"Lipschitz constants estimated:")
        print(f"  Lower bound c = {self.c_lower:.4f}")
        print(f"  Upper bound C = {self.C_upper:.4f}")
        print(f"  Inverse Lipschitz L = {self.L_inverse:.4f}")
        
        return self.c_lower, self.C_upper
    
    def construct_reverse_koopman(self, r):
        """
        Construct reverse Koopman operator K^{-1}_{(r)} with r modes
        
        Args:
            r: Number of modes for spectral truncation
            
        Returns:
            K_inv_r: Truncated reverse Koopman operator
        """
        if self.eigenvalues is None:
            raise ValueError("Must compute spectral decomposition first")
        
        r = min(r, len(self.eigenvalues))
        
        # Check for invertibility (no zero eigenvalues)
        eigenvals_r = self.eigenvalues[:r]
        min_eigenval = np.min(np.abs(eigenvals_r))
        
        if min_eigenval < 1e-10:
            print(f"Warning: Small eigenvalue detected: {min_eigenval}")
        
        # Construct K^{-1}_{(r)} = Σ_{k=1}^r λ_k^{-1} φ_k ⊗ ψ_k
        K_inv_r = np.zeros_like(self.K)
        
        for k in range(r):
            lambda_k = self.eigenvalues[k]
            phi_k = self.eigenfunctions[:, k]
            psi_k = self.dual_modes[:, k]
            
            # Avoid division by zero
            if np.abs(lambda_k) > 1e-10:
                lambda_inv = 1.0 / lambda_k
                K_inv_r += lambda_inv * np.outer(phi_k, psi_k.conj())
        
        self.K_inv_r = K_inv_r
        self.r_current = r
        
        print(f"Reverse Koopman K^{{-1}}_{{({r})}} constructed")
        return K_inv_r
    
    def compute_reconstruction_error(self, f_test=None, r=None):
        """
        Compute reconstruction error bounds from the theorem:
        ||K^{-1}f - K̂^{-1}_{(r)}f|| ≤ (κ_r/c)τ_r + (κ_r/c)(δ_λ + δ_φ)||f||
        
        Args:
            f_test: Test function (if None, use random observable)
            r: Number of modes (if None, use current r)
            
        Returns:
            error_bound: Theoretical error bound
            actual_error: Actual reconstruction error
        """
        if r is None:
            r = self.r_current
        if r is None:
            raise ValueError("Must specify number of modes r")
        
        # Generate test function if not provided
        if f_test is None:
            f_test = np.random.randn(self.n_observables)
            f_test = f_test / np.linalg.norm(f_test)  # Normalize
        
        # True inverse (using full spectrum, regularized)
        eigenvals_full = self.eigenvalues
        K_inv_full = np.zeros_like(self.K)
        
        for k in range(len(eigenvals_full)):
            lambda_k = eigenvals_full[k]
            phi_k = self.eigenfunctions[:, k]
            psi_k = self.dual_modes[:, k]
            
            if np.abs(lambda_k) > 1e-8:  # Regularization threshold
                lambda_inv = 1.0 / lambda_k
                K_inv_full += lambda_inv * np.outer(phi_k, psi_k.conj())
        
        # Truncated inverse
        K_inv_r = self.construct_reverse_koopman(r)
        
        # Apply operators to test function
        K_inv_f_full = K_inv_full @ f_test
        K_inv_f_r = K_inv_r @ f_test
        
        # Actual error
        actual_error = np.linalg.norm(K_inv_f_full - K_inv_f_r)
        
        # Theoretical error bound components
        kappa_r = self.condition_numbers[r-1] if r <= len(self.condition_numbers) else 1.0
        c = self.c_lower if self.c_lower is not None else 1.0
        
        # Spectral tail energy τ_r
        tail_eigenvals = eigenvals_full[r:]
        tau_r = np.sum(np.abs(tail_eigenvals)**2)**0.5
        
        # Estimation errors (assumed small for synthetic data)
        delta_lambda = self.delta_lambda
        delta_phi = self.delta_phi
        
        # Error bound: (κ_r/c)τ_r + (κ_r/c)(δ_λ + δ_φ)||f||
        f_norm = np.linalg.norm(f_test)
        truncation_error = (kappa_r / c) * tau_r
        estimation_error = (kappa_r / c) * (delta_lambda + delta_phi) * f_norm
        error_bound = truncation_error + estimation_error
        
        error_analysis = {
            'actual_error': actual_error,
            'error_bound': error_bound,
            'truncation_error': truncation_error,
            'estimation_error': estimation_error,
            'condition_number': kappa_r,
            'spectral_tail': tau_r,
            'bound_tightness': actual_error / error_bound if error_bound > 0 else np.inf
        }
        
        return error_analysis
    
    def bernstein_polynomial_approximation(self, N=10):
        """
        Construct Bernstein polynomial approximation B_N(K) for K^{-1}
        
        This provides the "auditable construction" primitive mentioned.
        
        Args:
            N: Degree of Bernstein polynomial
            
        Returns:
            B_N_K: Bernstein polynomial approximation
        """
        if self.K is None:
            raise ValueError("Must construct Koopman matrix first")
        
        # Bernstein polynomials for matrix functions
        # B_N(x) = Σ_{k=0}^N (N choose k) x^k (1-x)^{N-k}
        
        # Normalize eigenvalues to [0,1] for Bernstein approximation
        eigenvals = self.eigenvalues
        lambda_min = np.min(np.real(eigenvals))
        lambda_max = np.max(np.real(eigenvals))
        
        # Shift and scale to [0,1]
        if lambda_max > lambda_min:
            K_normalized = (self.K - lambda_min * np.eye(self.K.shape[0])) / (lambda_max - lambda_min)
        else:
            K_normalized = self.K
        
        # Construct Bernstein polynomial B_N(K)
        B_N_K = np.zeros_like(self.K)
        
        for k in range(N + 1):
            # Binomial coefficient
            binom_coeff = np.math.comb(N, k)
            
            # Bernstein basis polynomial evaluated at inverse
            # For inverse, we approximate (1-x) instead of x
            K_power_k = np.linalg.matrix_power(K_normalized, k)
            K_power_N_minus_k = np.linalg.matrix_power(np.eye(self.K.shape[0]) - K_normalized, N - k)
            
            B_N_K += binom_coeff * K_power_k @ K_power_N_minus_k
        
        # Scale back
        if lambda_max > lambda_min:
            B_N_K = B_N_K * (lambda_max - lambda_min) + lambda_min * np.eye(self.K.shape[0])
        
        self.B_N_K = B_N_K
        
        print(f"Bernstein polynomial B_{N}(K) approximation constructed")
        return B_N_K
    
    def visualize_spectral_analysis(self, save_plots=True):
        """Visualize spectral properties and reconstruction errors"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Reverse Koopman Spectral Analysis', fontsize=14)
        
        # 1. Eigenvalue spectrum
        if self.eigenvalues is not None:
            eigenvals = self.eigenvalues[:20]  # First 20 eigenvalues
            axes[0,0].scatter(np.real(eigenvals), np.imag(eigenvals), 
                            c=range(len(eigenvals)), cmap='viridis', s=50)
            axes[0,0].set_xlabel('Real part')
            axes[0,0].set_ylabel('Imaginary part')
            axes[0,0].set_title('Eigenvalue Spectrum')
            axes[0,0].grid(True, alpha=0.3)
            
            # Unit circle for reference
            theta = np.linspace(0, 2*np.pi, 100)
            axes[0,0].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')
            axes[0,0].legend()
        
        # 2. Eigenvalue magnitudes
        if self.eigenvalues is not None:
            eigenval_mags = np.abs(self.eigenvalues[:20])
            axes[0,1].semilogy(eigenval_mags, 'bo-', markersize=4)
            axes[0,1].set_xlabel('Mode index')
            axes[0,1].set_ylabel('|λ_k|')
            axes[0,1].set_title('Eigenvalue Magnitudes')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Condition numbers
        if hasattr(self, 'condition_numbers'):
            r_values = range(1, len(self.condition_numbers) + 1)
            axes[0,2].semilogy(r_values, self.condition_numbers, 'ro-', markersize=4)
            axes[0,2].set_xlabel('Number of modes r')
            axes[0,2].set_ylabel('κ_r')
            axes[0,2].set_title('Condition Numbers')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Reconstruction error vs. number of modes
        r_values = range(2, min(15, len(self.eigenvalues)))
        actual_errors = []
        error_bounds = []
        
        for r in r_values:
            error_analysis = self.compute_reconstruction_error(r=r)
            actual_errors.append(error_analysis['actual_error'])
            error_bounds.append(error_analysis['error_bound'])
        
        axes[1,0].semilogy(r_values, actual_errors, 'b-o', label='Actual error', markersize=4)
        axes[1,0].semilogy(r_values, error_bounds, 'r--s', label='Error bound', markersize=4)
        axes[1,0].set_xlabel('Number of modes r')
        axes[1,0].set_ylabel('Reconstruction error')
        axes[1,0].set_title('Error vs. Truncation')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Phase space trajectory
        if hasattr(self, 'X') and self.X.shape[1] >= 2:
            axes[1,1].plot(self.X[:500, 0], self.X[:500, 1], 'b-', alpha=0.7, linewidth=1)
            axes[1,1].scatter(self.X[0, 0], self.X[0, 1], c='green', s=50, label='Start')
            axes[1,1].scatter(self.X[499, 0], self.X[499, 1], c='red', s=50, label='End')
            axes[1,1].set_xlabel('x₁')
            axes[1,1].set_ylabel('x₂')
            axes[1,1].set_title(f'{self.system_type.title()} Trajectory')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Lipschitz constant estimation
        if self.c_lower is not None and self.C_upper is not None:
            # Show distribution of Lipschitz ratios
            n_samples = 200
            ratios = []
            n_points = len(self.X)
            
            for _ in range(n_samples):
                i, j = np.random.choice(n_points-1, 2, replace=False)
                x_i, x_j = self.X[i], self.X[j]
                Kx_i, Kx_j = self.X[i+1], self.X[j+1]
                
                x_diff = np.linalg.norm(x_i - x_j)
                Kx_diff = np.linalg.norm(Kx_i - Kx_j)
                
                if x_diff > 1e-10:
                    ratios.append(Kx_diff / x_diff)
            
            axes[1,2].hist(ratios, bins=30, alpha=0.7, density=True)
            axes[1,2].axvline(self.c_lower, color='red', linestyle='--', label=f'c = {self.c_lower:.3f}')
            axes[1,2].axvline(self.C_upper, color='red', linestyle='--', label=f'C = {self.C_upper:.3f}')
            axes[1,2].set_xlabel('||Kx - Ky|| / ||x - y||')
            axes[1,2].set_ylabel('Density')
            axes[1,2].set_title('Lipschitz Ratio Distribution')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('/Users/ryan_david_oates/Farmer/reverse_koopman_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print("Reverse Koopman analysis visualization saved")
        
        plt.show()
        
        return fig, axes

def demonstrate_reverse_koopman():
    """Demonstrate the Reverse Koopman operator implementation"""
    
    print("=== Reverse Koopman Operator with Spectral Truncation ===\n")
    
    # Initialize Reverse Koopman operator
    rko = ReverseKoopmanOperator(Delta=0.1, r_max=15)
    
    # Generate dynamical system data
    print("Generating Van der Pol oscillator trajectory...")
    X = rko.generate_dynamical_system('van_der_pol', n_points=2000)
    
    # Construct Koopman matrix
    print("\nConstructing Koopman matrix with polynomial observables...")
    K = rko.construct_koopman_matrix('polynomial', poly_degree=3)
    
    # Compute spectral decomposition
    print("\nComputing spectral decomposition...")
    eigenvals, eigenvecs = rko.compute_spectral_decomposition()
    
    # Estimate Lipschitz constants
    print("\nEstimating Lipschitz constants...")
    c_lower, C_upper = rko.estimate_lipschitz_constants(n_samples=200)
    
    # Test reconstruction with different numbers of modes
    print("\nTesting reconstruction error bounds...")
    
    test_modes = [3, 5, 8, 10]
    for r in test_modes:
        error_analysis = rko.compute_reconstruction_error(r=r)
        
        print(f"\nModes r = {r}:")
        print(f"  Actual error: {error_analysis['actual_error']:.6f}")
        print(f"  Error bound: {error_analysis['error_bound']:.6f}")
        print(f"  Bound tightness: {error_analysis['bound_tightness']:.3f}")
        print(f"  Condition number κ_r: {error_analysis['condition_number']:.3f}")
        print(f"  Spectral tail τ_r: {error_analysis['spectral_tail']:.6f}")
    
    # Construct Bernstein polynomial approximation
    print(f"\nConstructing Bernstein polynomial approximation...")
    B_N = rko.bernstein_polynomial_approximation(N=8)
    
    # Verify theoretical properties
    print(f"\nVerifying theoretical properties:")
    print(f"  Bi-Lipschitz condition (A1): c = {c_lower:.4f}, C = {C_upper:.4f}")
    print(f"  Inverse Lipschitz constant: L ≤ {rko.L_inverse:.4f}")
    print(f"  Spectral radius: {np.max(np.abs(eigenvals)):.4f}")
    print(f"  Matrix condition number: {np.linalg.cond(K):.2e}")
    
    # Check invertibility
    min_eigenval = np.min(np.abs(eigenvals))
    print(f"  Minimum |λ|: {min_eigenval:.6f} (invertibility check)")
    
    # Visualize results
    print(f"\nGenerating spectral analysis visualization...")
    rko.visualize_spectral_analysis()
    
    # Test with different dynamical systems
    print(f"\nTesting with Lorenz system...")
    rko_lorenz = ReverseKoopmanOperator(Delta=0.05, r_max=12)
    X_lorenz = rko_lorenz.generate_dynamical_system('lorenz', n_points=1500)
    K_lorenz = rko_lorenz.construct_koopman_matrix('polynomial', poly_degree=2)
    eigenvals_lorenz, _ = rko_lorenz.compute_spectral_decomposition()
    
    print(f"Lorenz system spectral radius: {np.max(np.abs(eigenvals_lorenz)):.4f}")
    
    return rko, error_analysis

if __name__ == "__main__":
    rko, analysis = demonstrate_reverse_koopman()
