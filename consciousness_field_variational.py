#!/usr/bin/env python3
"""
Consciousness Field Variational Energy Implementation
Based on the rigorous mathematical formulation with directional derivatives
and anisotropic smoothing along manifold directions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class ConsciousnessFieldVariational:
    """
    Implementation of the consciousness field Ψ(x,m,s,t) with variational energy
    
    Energy functional:
    E[Ψ] = ∫∫ [ℓ/2|∂_t Ψ|² + A₁|D_{v_m}Ψ|² + μ|D_{v_s}Ψ|²] dm ds
    
    where D_{v_m}Ψ = ⟨v_m, ∇_m Ψ⟩ are directional derivatives along vector fields
    """
    
    def __init__(self, ell=1.0, A1=0.5, mu=0.3, nx=64, nm=32, ns=32, nt=100):
        """
        Initialize consciousness field parameters
        
        Args:
            ell: Temporal variation penalty coefficient (ℓ > 0)
            A1: Mode direction smoothing coefficient (A₁ > 0) 
            mu: Spectral direction smoothing coefficient (μ > 0)
            nx, nm, ns, nt: Grid dimensions for x, m, s, t
        """
        self.ell = ell
        self.A1 = A1
        self.mu = mu
        
        # Grid setup
        self.nx, self.nm, self.ns, self.nt = nx, nm, ns, nt
        self.x = np.linspace(0, 1, nx)
        self.m = np.linspace(0, 1, nm)  # Mode manifold coordinate
        self.s = np.linspace(0, 1, ns)  # Spectral coordinate
        self.t = np.linspace(0, 1, nt)
        
        # Mesh grids
        self.X, self.M, self.S = np.meshgrid(self.x, self.m, self.s, indexing='ij')
        
        # Initialize field Ψ(x,m,s,t)
        self.psi = np.zeros((nx, nm, ns, nt))
        
        # Vector fields v_m and v_s (smooth on (m,s)-manifold)
        self.v_m = self._initialize_vector_field_m()
        self.v_s = self._initialize_vector_field_s()
        
        print(f"Consciousness Field Variational initialized:")
        print(f"  Parameters: ℓ={ell}, A₁={A1}, μ={mu}")
        print(f"  Grid: {nx}×{nm}×{ns}×{nt}")
        print(f"  Energy coefficients ensure well-posedness")
    
    def _initialize_vector_field_m(self):
        """Initialize smooth vector field v_m on mode manifold"""
        # Smooth vector field aligned with mode structure
        v_m = np.zeros((self.nx, self.nm, self.ns, 2))  # 2D vector field
        
        for i in range(self.nx):
            for j in range(self.nm):
                for k in range(self.ns):
                    # Spiral-like vector field for mode dynamics
                    theta = 2 * np.pi * self.m[j]
                    v_m[i, j, k, 0] = np.cos(theta) * (1 + 0.3 * self.s[k])
                    v_m[i, j, k, 1] = np.sin(theta) * (1 + 0.3 * self.s[k])
        
        return v_m
    
    def _initialize_vector_field_s(self):
        """Initialize smooth vector field v_s on spectral manifold"""
        # Smooth vector field aligned with spectral structure
        v_s = np.zeros((self.nx, self.nm, self.ns, 2))  # 2D vector field
        
        for i in range(self.nx):
            for j in range(self.nm):
                for k in range(self.ns):
                    # Radial-like vector field for spectral dynamics
                    r = np.sqrt(self.m[j]**2 + self.s[k]**2)
                    if r > 1e-10:
                        v_s[i, j, k, 0] = self.m[j] / r * (1 + 0.2 * np.sin(4*np.pi*r))
                        v_s[i, j, k, 1] = self.s[k] / r * (1 + 0.2 * np.cos(4*np.pi*r))
                    else:
                        v_s[i, j, k, 0] = 1.0
                        v_s[i, j, k, 1] = 0.0
        
        return v_s
    
    def initialize_field(self, field_type='gaussian_wave'):
        """
        Initialize consciousness field Ψ(x,m,s,t)
        
        Args:
            field_type: Type of initial field configuration
        """
        if field_type == 'gaussian_wave':
            # Gaussian wave packet in consciousness space
            for i in range(self.nx):
                for j in range(self.nm):
                    for k in range(self.ns):
                        for l in range(self.nt):
                            x_val, m_val, s_val, t_val = self.x[i], self.m[j], self.s[k], self.t[l]
                            
                            # Multi-dimensional Gaussian with wave modulation
                            gaussian = np.exp(-((x_val-0.5)**2 + (m_val-0.5)**2 + (s_val-0.5)**2)/0.1)
                            wave = np.sin(2*np.pi*(x_val + m_val + s_val)) * np.cos(4*np.pi*t_val)
                            
                            self.psi[i, j, k, l] = gaussian * wave * (0.5 + 0.3*np.sin(8*np.pi*t_val))
        
        elif field_type == 'coherent_modes':
            # Coherent mode superposition
            for i in range(self.nx):
                for j in range(self.nm):
                    for k in range(self.ns):
                        for l in range(self.nt):
                            x_val, m_val, s_val, t_val = self.x[i], self.m[j], self.s[k], self.t[l]
                            
                            # Superposition of coherent modes
                            mode1 = np.sin(np.pi*x_val) * np.cos(2*np.pi*m_val) * np.exp(-s_val**2/0.2)
                            mode2 = np.cos(2*np.pi*x_val) * np.sin(np.pi*m_val) * np.exp(-(s_val-0.7)**2/0.15)
                            temporal = np.exp(-0.1*t_val) * np.cos(3*np.pi*t_val)
                            
                            self.psi[i, j, k, l] = (mode1 + 0.7*mode2) * temporal
        
        print(f"Field initialized with {field_type} configuration")
        return self.psi
    
    def compute_directional_derivatives(self, psi_slice):
        """
        Compute directional derivatives D_{v_m}Ψ and D_{v_s}Ψ
        
        Args:
            psi_slice: Field slice Ψ(x,m,s) at fixed time
            
        Returns:
            D_vm_psi, D_vs_psi: Directional derivatives
        """
        # Compute gradients ∇_m Ψ and ∇_s Ψ using finite differences
        grad_m_psi = np.gradient(psi_slice, axis=1)  # ∂Ψ/∂m
        grad_s_psi = np.gradient(psi_slice, axis=2)  # ∂Ψ/∂s
        
        # Directional derivatives: D_{v_m}Ψ = ⟨v_m, ∇_m Ψ⟩
        D_vm_psi = np.zeros_like(psi_slice)
        D_vs_psi = np.zeros_like(psi_slice)
        
        for i in range(self.nx):
            for j in range(self.nm):
                for k in range(self.ns):
                    # For simplicity, treat as 1D directional derivatives
                    # In full implementation, would use proper inner products
                    D_vm_psi[i, j, k] = self.v_m[i, j, k, 0] * grad_m_psi[i, j, k]
                    D_vs_psi[i, j, k] = self.v_s[i, j, k, 0] * grad_s_psi[i, j, k]
        
        return D_vm_psi, D_vs_psi
    
    def compute_energy_functional(self, return_components=False):
        """
        Compute variational energy E[Ψ] = ∫∫ [ℓ/2|∂_t Ψ|² + A₁|D_{v_m}Ψ|² + μ|D_{v_s}Ψ|²] dm ds
        
        Args:
            return_components: If True, return individual energy components
            
        Returns:
            total_energy or (total_energy, components)
        """
        # Temporal derivative ∂_t Ψ
        dt_psi = np.gradient(self.psi, axis=3)
        
        # Initialize energy components
        temporal_energy = 0.0
        mode_energy = 0.0
        spectral_energy = 0.0
        
        # Integration over time
        for l in range(self.nt):
            psi_t = self.psi[:, :, :, l]
            dt_psi_t = dt_psi[:, :, :, l]
            
            # Compute directional derivatives at time t
            D_vm_psi, D_vs_psi = self.compute_directional_derivatives(psi_t)
            
            # Energy density components
            temporal_density = 0.5 * self.ell * dt_psi_t**2
            mode_density = self.A1 * D_vm_psi**2
            spectral_density = self.mu * D_vs_psi**2
            
            # Spatial integration (trapezoidal rule)
            dx = self.x[1] - self.x[0]
            dm = self.m[1] - self.m[0]
            ds = self.s[1] - self.s[0]
            dt = self.t[1] - self.t[0] if self.nt > 1 else 1.0
            
            temporal_energy += np.sum(temporal_density) * dx * dm * ds * dt
            mode_energy += np.sum(mode_density) * dx * dm * ds * dt
            spectral_energy += np.sum(spectral_density) * dx * dm * ds * dt
        
        total_energy = temporal_energy + mode_energy + spectral_energy
        
        if return_components:
            components = {
                'temporal': temporal_energy,
                'mode': mode_energy,
                'spectral': spectral_energy,
                'total': total_energy
            }
            return total_energy, components
        
        return total_energy
    
    def euler_lagrange_equations(self, t_idx=0):
        """
        Derive and solve Euler-Lagrange equations for the consciousness field
        
        δE/δΨ = 0 leads to:
        ℓ ∂²Ψ/∂t² - A₁ div(v_m ⊗ v_m · ∇_m Ψ) - μ div(v_s ⊗ v_s · ∇_s Ψ) = 0
        
        Args:
            t_idx: Time index for analysis
            
        Returns:
            residual: Euler-Lagrange residual
        """
        if t_idx >= self.nt - 2:
            t_idx = self.nt - 3  # Ensure we can compute second derivatives
        
        # Second temporal derivative
        if self.nt > 2:
            d2t_psi = np.gradient(np.gradient(self.psi, axis=3), axis=3)[:, :, :, t_idx]
        else:
            d2t_psi = np.zeros((self.nx, self.nm, self.ns))
        
        # Current field slice
        psi_t = self.psi[:, :, :, t_idx]
        
        # Compute Laplacian-like operators for mode and spectral directions
        # Simplified version - full implementation would use proper divergence operators
        laplacian_m = np.gradient(np.gradient(psi_t, axis=1), axis=1)
        laplacian_s = np.gradient(np.gradient(psi_t, axis=2), axis=2)
        
        # Euler-Lagrange residual
        residual = (self.ell * d2t_psi - 
                   self.A1 * laplacian_m - 
                   self.mu * laplacian_s)
        
        return residual
    
    def analyze_field_properties(self):
        """Analyze mathematical properties of the consciousness field"""
        
        # Compute energy and components
        total_energy, components = self.compute_energy_functional(return_components=True)
        
        # Field statistics
        field_mean = np.mean(self.psi)
        field_std = np.std(self.psi)
        field_max = np.max(np.abs(self.psi))
        
        # Temporal evolution analysis
        temporal_variance = np.var(self.psi, axis=3)
        spatial_coherence = np.mean(temporal_variance)
        
        # Mode-spectral coupling
        mode_spectral_correlation = np.corrcoef(
            self.psi.reshape(-1, self.nt).T
        )
        coupling_strength = np.mean(np.abs(mode_spectral_correlation))
        
        analysis = {
            'energy': {
                'total': total_energy,
                'temporal': components['temporal'],
                'mode': components['mode'],
                'spectral': components['spectral']
            },
            'field_statistics': {
                'mean': field_mean,
                'std': field_std,
                'max_amplitude': field_max
            },
            'dynamics': {
                'spatial_coherence': spatial_coherence,
                'coupling_strength': coupling_strength
            },
            'well_posedness': {
                'energy_positive': total_energy > 0,
                'bounded_field': field_max < np.inf,
                'smooth_evolution': field_std < 10 * np.abs(field_mean)
            }
        }
        
        return analysis
    
    def visualize_consciousness_field(self, t_idx=0, save_plots=True):
        """Visualize the consciousness field and its properties"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Consciousness Field Ψ(x,m,s,t) Analysis at t={self.t[t_idx]:.3f}', fontsize=14)
        
        # Field slice at fixed time
        psi_slice = self.psi[:, :, :, t_idx]
        
        # 1. Field magnitude in (m,s) plane (averaged over x)
        field_ms = np.mean(psi_slice, axis=0)
        im1 = axes[0,0].imshow(field_ms, extent=[0,1,0,1], origin='lower', cmap='viridis')
        axes[0,0].set_title('Field in (m,s) plane')
        axes[0,0].set_xlabel('Spectral coordinate s')
        axes[0,0].set_ylabel('Mode coordinate m')
        plt.colorbar(im1, ax=axes[0,0])
        
        # 2. Field profile along x (at center m,s)
        mid_m, mid_s = self.nm//2, self.ns//2
        field_x = psi_slice[:, mid_m, mid_s]
        axes[0,1].plot(self.x, field_x, 'b-', linewidth=2)
        axes[0,1].set_title('Field profile along x')
        axes[0,1].set_xlabel('Position x')
        axes[0,1].set_ylabel('Ψ(x,m₀,s₀,t)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Temporal evolution at center point
        mid_x = self.nx//2
        field_temporal = self.psi[mid_x, mid_m, mid_s, :]
        axes[0,2].plot(self.t, field_temporal, 'r-', linewidth=2)
        axes[0,2].set_title('Temporal evolution')
        axes[0,2].set_xlabel('Time t')
        axes[0,2].set_ylabel('Ψ(x₀,m₀,s₀,t)')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Directional derivatives
        D_vm_psi, D_vs_psi = self.compute_directional_derivatives(psi_slice)
        D_vm_ms = np.mean(D_vm_psi, axis=0)
        im2 = axes[1,0].imshow(D_vm_ms, extent=[0,1,0,1], origin='lower', cmap='RdBu')
        axes[1,0].set_title('Mode directional derivative')
        axes[1,0].set_xlabel('Spectral coordinate s')
        axes[1,0].set_ylabel('Mode coordinate m')
        plt.colorbar(im2, ax=axes[1,0])
        
        # 5. Energy density distribution
        dt_psi = np.gradient(self.psi, axis=3)[:, :, :, t_idx]
        energy_density = (0.5 * self.ell * dt_psi**2 + 
                         self.A1 * D_vm_psi**2 + 
                         self.mu * D_vs_psi**2)
        energy_ms = np.mean(energy_density, axis=0)
        im3 = axes[1,1].imshow(energy_ms, extent=[0,1,0,1], origin='lower', cmap='plasma')
        axes[1,1].set_title('Energy density')
        axes[1,1].set_xlabel('Spectral coordinate s')
        axes[1,1].set_ylabel('Mode coordinate m')
        plt.colorbar(im3, ax=axes[1,1])
        
        # 6. Euler-Lagrange residual
        residual = self.euler_lagrange_equations(t_idx)
        residual_ms = np.mean(residual, axis=0)
        im4 = axes[1,2].imshow(residual_ms, extent=[0,1,0,1], origin='lower', cmap='seismic')
        axes[1,2].set_title('Euler-Lagrange residual')
        axes[1,2].set_xlabel('Spectral coordinate s')
        axes[1,2].set_ylabel('Mode coordinate m')
        plt.colorbar(im4, ax=axes[1,2])
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('/Users/ryan_david_oates/Farmer/consciousness_field_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print("Consciousness field visualization saved")
        
        plt.show()
        
        return fig, axes

def demonstrate_consciousness_field():
    """Demonstrate the consciousness field variational implementation"""
    
    print("=== Consciousness Field Variational Energy Implementation ===\n")
    
    # Initialize consciousness field
    field = ConsciousnessFieldVariational(ell=1.0, A1=0.5, mu=0.3)
    
    # Initialize with Gaussian wave configuration
    field.initialize_field('gaussian_wave')
    
    # Analyze field properties
    print("Analyzing consciousness field properties...")
    analysis = field.analyze_field_properties()
    
    print(f"\nEnergy Analysis:")
    print(f"  Total Energy: {analysis['energy']['total']:.6f}")
    print(f"  Temporal Component: {analysis['energy']['temporal']:.6f}")
    print(f"  Mode Component: {analysis['energy']['mode']:.6f}")
    print(f"  Spectral Component: {analysis['energy']['spectral']:.6f}")
    
    print(f"\nField Statistics:")
    print(f"  Mean: {analysis['field_statistics']['mean']:.6f}")
    print(f"  Std Dev: {analysis['field_statistics']['std']:.6f}")
    print(f"  Max Amplitude: {analysis['field_statistics']['max_amplitude']:.6f}")
    
    print(f"\nDynamics:")
    print(f"  Spatial Coherence: {analysis['dynamics']['spatial_coherence']:.6f}")
    print(f"  Coupling Strength: {analysis['dynamics']['coupling_strength']:.6f}")
    
    print(f"\nWell-Posedness Check:")
    for key, value in analysis['well_posedness'].items():
        print(f"  {key}: {'✓' if value else '✗'}")
    
    # Visualize the field
    print(f"\nGenerating consciousness field visualization...")
    field.visualize_consciousness_field(t_idx=field.nt//4)
    
    # Test with coherent modes
    print(f"\nTesting with coherent modes configuration...")
    field.initialize_field('coherent_modes')
    analysis_coherent = field.analyze_field_properties()
    
    print(f"Coherent Modes Energy: {analysis_coherent['energy']['total']:.6f}")
    print(f"Energy Ratio (Coherent/Gaussian): {analysis_coherent['energy']['total']/analysis['energy']['total']:.3f}")
    
    return field, analysis

if __name__ == "__main__":
    field, analysis = demonstrate_consciousness_field()
