"""
Comprehensive validation test suite for the CFD solver
Tests against analytical solutions and benchmark cases
"""

import numpy as np
import pytest
from typing import Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path

# Test data and analytical solutions
class AnalyticalSolutions:
    """Collection of analytical solutions for CFD validation"""
    
    @staticmethod
    def cylinder_flow_potential(x: np.ndarray, y: np.ndarray, 
                               U_inf: float = 1.0, R: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analytical solution for potential flow around a cylinder
        
        Args:
            x, y: Coordinate arrays
            U_inf: Freestream velocity
            R: Cylinder radius
            
        Returns:
            u_analytical, v_analytical, p_analytical
        """
        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Avoid singularity at cylinder surface
        r = np.maximum(r, R + 1e-10)
        
        # Velocity potential: φ = U_inf * r * (1 + R²/r²) * cos(θ)
        # Stream function: ψ = U_inf * r * (1 - R²/r²) * sin(θ)
        
        # Velocity components
        u_r = U_inf * (1 - R**2 / r**2) * np.cos(theta)
        u_theta = -U_inf * (1 + R**2 / r**2) * np.sin(theta)
        
        # Convert to Cartesian coordinates
        u_analytical = u_r * np.cos(theta) - u_theta * np.sin(theta)
        v_analytical = u_r * np.sin(theta) + u_theta * np.cos(theta)
        
        # Pressure from Bernoulli equation
        velocity_magnitude = np.sqrt(u_analytical**2 + v_analytical**2)
        p_analytical = 0.5 * (U_inf**2 - velocity_magnitude**2)  # ρ = 1 assumed
        
        return u_analytical, v_analytical, p_analytical
    
    @staticmethod
    def couette_flow(y: np.ndarray, U_wall: float = 1.0, h: float = 1.0) -> np.ndarray:
        """
        Analytical solution for Couette flow between parallel plates
        
        Args:
            y: y-coordinates
            U_wall: Wall velocity
            h: Channel height
            
        Returns:
            u_analytical: Velocity profile
        """
        return U_wall * y / h
    
    @staticmethod
    def poiseuille_flow(y: np.ndarray, dp_dx: float = -1.0, 
                       mu: float = 1.0, h: float = 1.0) -> np.ndarray:
        """
        Analytical solution for Poiseuille flow between parallel plates
        
        Args:
            y: y-coordinates  
            dp_dx: Pressure gradient
            mu: Dynamic viscosity
            h: Half channel height
            
        Returns:
            u_analytical: Velocity profile
        """
        return -dp_dx / (2 * mu) * (h**2 - y**2)

class CFDValidator:
    """Validation framework for CFD solver"""
    
    def __init__(self, tolerance: float = 0.05):
        """
        Initialize validator
        
        Args:
            tolerance: Acceptable relative error for validation
        """
        self.tolerance = tolerance
        self.test_results = []
        
    def validate_cylinder_flow(self, solver_results: dict, 
                             test_name: str = "Cylinder Flow") -> bool:
        """
        Validate solver against potential flow around cylinder
        
        Args:
            solver_results: Dictionary with 'u', 'v', 'p', 'x', 'y' arrays
            test_name: Name of the test
            
        Returns:
            bool: True if validation passes
        """
        print(f"\nValidating {test_name}...")
        
        # Extract solver results
        u_solver = solver_results['u']
        v_solver = solver_results['v'] 
        p_solver = solver_results['p']
        x_coords = solver_results['x']
        y_coords = solver_results['y']
        
        # Get analytical solution
        u_analytical, v_analytical, p_analytical = AnalyticalSolutions.cylinder_flow_potential(
            x_coords, y_coords
        )
        
        # Calculate errors (excluding near-wall region)
        mask = np.sqrt(x_coords**2 + y_coords**2) > 1.2  # Exclude near cylinder
        
        u_error = self._calculate_relative_error(u_solver[mask], u_analytical[mask])
        v_error = self._calculate_relative_error(v_solver[mask], v_analytical[mask])
        p_error = self._calculate_relative_error(p_solver[mask], p_analytical[mask])
        
        # Overall error
        total_error = np.sqrt(u_error**2 + v_error**2 + p_error**2) / 3.0
        
        # Check validation
        passed = total_error < self.tolerance
        
        # Store results
        result = {
            'test_name': test_name,
            'u_error': u_error,
            'v_error': v_error,
            'p_error': p_error,
            'total_error': total_error,
            'passed': passed,
            'tolerance': self.tolerance
        }
        self.test_results.append(result)
        
        # Print results
        print(f"  U-velocity error: {u_error:.3f}")
        print(f"  V-velocity error: {v_error:.3f}")
        print(f"  Pressure error: {p_error:.3f}")
        print(f"  Total error: {total_error:.3f}")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return passed
    
    def validate_couette_flow(self, solver_results: dict,
                            test_name: str = "Couette Flow") -> bool:
        """
        Validate solver against Couette flow analytical solution
        
        Args:
            solver_results: Dictionary with velocity profile data
            test_name: Name of the test
            
        Returns:
            bool: True if validation passes
        """
        print(f"\nValidating {test_name}...")
        
        # Extract middle section of domain for comparison
        u_solver = solver_results['u_profile']
        y_coords = solver_results['y_coords']
        
        # Analytical solution
        u_analytical = AnalyticalSolutions.couette_flow(y_coords)
        
        # Calculate error
        error = self._calculate_relative_error(u_solver, u_analytical)
        passed = error < self.tolerance
        
        # Store results
        result = {
            'test_name': test_name,
            'error': error,
            'passed': passed,
            'tolerance': self.tolerance
        }
        self.test_results.append(result)
        
        print(f"  Velocity profile error: {error:.3f}")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return passed
    
    def validate_mass_conservation(self, solver_results: dict,
                                 test_name: str = "Mass Conservation") -> bool:
        """
        Validate mass conservation (continuity equation)
        
        Args:
            solver_results: Dictionary with flow field data
            test_name: Name of the test
            
        Returns:
            bool: True if validation passes
        """
        print(f"\nValidating {test_name}...")
        
        u = solver_results['u']
        v = solver_results['v']
        dx = solver_results['dx']
        dy = solver_results['dy']
        
        # Calculate divergence using central differences
        du_dx = np.zeros_like(u)
        dv_dy = np.zeros_like(v)
        
        du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        dv_dy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dy)
        
        divergence = du_dx + dv_dy
        
        # Calculate RMS divergence (excluding boundaries)
        rms_divergence = np.sqrt(np.mean(divergence[1:-1, 1:-1]**2))
        
        # Mass conservation tolerance (should be near machine precision for incompressible flow)
        mass_tolerance = 1e-10
        passed = rms_divergence < mass_tolerance
        
        # Store results
        result = {
            'test_name': test_name,
            'rms_divergence': rms_divergence,
            'passed': passed,
            'tolerance': mass_tolerance
        }
        self.test_results.append(result)
        
        print(f"  RMS Divergence: {rms_divergence:.2e}")
        print(f"  Tolerance: {mass_tolerance:.2e}")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return passed
    
    def validate_fin_performance_bounds(self, lift_coeffs: List[float], 
                                      drag_coeffs: List[float],
                                      angles: List[float],
                                      test_name: str = "Fin Performance Bounds") -> bool:
        """
        Validate that fin performance coefficients are within realistic bounds
        
        Args:
            lift_coeffs: List of lift coefficients
            drag_coeffs: List of drag coefficients
            angles: Corresponding angles of attack
            test_name: Name of the test
            
        Returns:
            bool: True if validation passes
        """
        print(f"\nValidating {test_name}...")
        
        # Realistic bounds for surfboard fins
        max_cl = 1.5  # Maximum realistic lift coefficient
        max_cd = 0.3  # Maximum realistic drag coefficient at moderate AoA
        min_ld = 2.0  # Minimum acceptable L/D ratio
        
        # Check bounds
        cl_valid = all(0 <= cl <= max_cl for cl in lift_coeffs)
        cd_valid = all(0 <= cd <= max_cd for cd in drag_coeffs if cd > 0)
        
        # Check L/D ratios
        ld_ratios = [cl/cd for cl, cd in zip(lift_coeffs, drag_coeffs) if cd > 0]
        ld_valid = any(ld >= min_ld for ld in ld_ratios)
        
        # Check monotonic behavior (lift should increase with AoA initially)
        monotonic_valid = True
        for i in range(1, min(3, len(lift_coeffs))):  # Check first few points
            if lift_coeffs[i] <= lift_coeffs[i-1]:
                monotonic_valid = False
                break
        
        passed = cl_valid and cd_valid and ld_valid and monotonic_valid
        
        # Store results
        result = {
            'test_name': test_name,
            'cl_bounds_valid': cl_valid,
            'cd_bounds_valid': cd_valid,
            'ld_ratio_valid': ld_valid,
            'monotonic_valid': monotonic_valid,
            'passed': passed,
            'max_cl': max(lift_coeffs),
            'max_cd': max(drag_coeffs),
            'max_ld': max(ld_ratios) if ld_ratios else 0
        }
        self.test_results.append(result)
        
        print(f"  CL bounds (0-{max_cl}): {'✓' if cl_valid else '✗'}")
        print(f"  CD bounds (0-{max_cd}): {'✓' if cd_valid else '✗'}")
        print(f"  L/D ratio (>{min_ld}): {'✓' if ld_valid else '✗'}")
        print(f"  Monotonic lift: {'✓' if monotonic_valid else '✗'}")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return passed
    
    def _calculate_relative_error(self, numerical: np.ndarray, 
                                analytical: np.ndarray) -> float:
        """Calculate relative error between numerical and analytical solutions"""
        # Avoid division by zero
        analytical_safe = np.where(np.abs(analytical) < 1e-10, 1e-10, analytical)
        relative_errors = np.abs(numerical - analytical) / np.abs(analytical_safe)
        return np.mean(relative_errors)
    
    def generate_validation_report(self, save_path: str = "validation_report.txt") -> None:
        """Generate comprehensive validation report"""
        with open(save_path, 'w') as f:
            f.write("CFD SOLVER VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results if result['passed'])
            
            f.write(f"Summary: {passed_tests}/{total_tests} tests passed\n")
            f.write(f"Success Rate: {passed_tests/total_tests*100:.1f}%\n\n")
            
            for result in self.test_results:
                f.write(f"Test: {result['test_name']}\n")
                f.write(f"Status: {'PASSED' if result['passed'] else 'FAILED'}\n")
                
                if 'total_error' in result:
                    f.write(f"Total Error: {result['total_error']:.4f}\n")
                    f.write(f"Tolerance: {result['tolerance']:.4f}\n")
                elif 'error' in result:
                    f.write(f"Error: {result['error']:.4f}\n")
                    f.write(f"Tolerance: {result['tolerance']:.4f}\n")
                
                f.write("-" * 30 + "\n")
        
        print(f"\nValidation report saved to: {save_path}")
        print(f"Overall Success Rate: {passed_tests/total_tests*100:.1f}%")

def run_comprehensive_validation():
    """Run complete validation test suite"""
    print("COMPREHENSIVE CFD SOLVER VALIDATION")
    print("=" * 50)
    
    validator = CFDValidator(tolerance=0.1)  # 10% tolerance for complex cases
    
    # Test 1: Generate synthetic cylinder flow data for validation
    print("Generating test data...")
    nx, ny = 50, 30
    x = np.linspace(-3, 5, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)
    
    # Use analytical solution as "solver results" for validation framework test
    u_analytical, v_analytical, p_analytical = AnalyticalSolutions.cylinder_flow_potential(X, Y)
    
    # Add some noise to simulate solver results
    noise_level = 0.02
    u_solver = u_analytical + np.random.normal(0, noise_level, u_analytical.shape)
    v_solver = v_analytical + np.random.normal(0, noise_level, v_analytical.shape)
    p_solver = p_analytical + np.random.normal(0, noise_level, p_analytical.shape)
    
    solver_results = {
        'u': u_solver,
        'v': v_solver,
        'p': p_solver,
        'x': X,
        'y': Y,
        'dx': x[1] - x[0],
        'dy': y[1] - y[0]
    }
    
    # Run validations
    validator.validate_cylinder_flow(solver_results)
    validator.validate_mass_conservation(solver_results)
    
    # Test 2: Couette flow validation
    y_couette = np.linspace(0, 1, 21)
    u_analytical_couette = AnalyticalSolutions.couette_flow(y_couette)
    u_solver_couette = u_analytical_couette + np.random.normal(0, 0.01, len(y_couette))
    
    couette_results = {
        'u_profile': u_solver_couette,
        'y_coords': y_couette
    }
    
    validator.validate_couette_flow(couette_results)
    
    # Test 3: Fin performance bounds validation
    # Simulate realistic fin performance data
    angles = [0, 5, 10, 15, 20]
    lift_coeffs = [0.0, 0.3, 0.6, 0.8, 0.7]  # Realistic lift curve with stall
    drag_coeffs = [0.01, 0.02, 0.05, 0.08, 0.15]  # Increasing drag with AoA
    
    validator.validate_fin_performance_bounds(lift_coeffs, drag_coeffs, angles)
    
    # Generate report
    validator.generate_validation_report()
    
    return validator

# Benchmark test cases
class BenchmarkCases:
    """Standard CFD benchmark test cases"""
    
    @staticmethod
    def lid_driven_cavity_reference() -> dict:
        """Reference solution for lid-driven cavity at Re=100"""
        # Ghia et al. (1982) benchmark data
        return {
            'Re': 100,
            'u_centerline_y': [0.0000, 0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 
                              0.4375, 0.5000, 0.5625, 0.6250, 0.6875, 0.7500, 0.8125, 
                              0.8750, 0.9375, 1.0000],
            'u_centerline_values': [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, 
                                   -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 
                                   0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000]
        }
    
    @staticmethod
    def flow_over_backward_step_reference() -> dict:
        """Reference solution for flow over backward-facing step"""
        return {
            'Re': 100,
            'reattachment_length': 3.0,  # Approximate value
            'description': "Backward-facing step flow at Re=100"
        }

def create_test_visualization(validator: CFDValidator):
    """Create visualization of validation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CFD Solver Validation Results', fontsize=14)
    
    # Extract test data
    test_names = [result['test_name'] for result in validator.test_results]
    passed_status = [result['passed'] for result in validator.test_results]
    
    # Test 1: Pass/Fail summary
    ax1 = axes[0, 0]
    colors = ['green' if passed else 'red' for passed in passed_status]
    bars = ax1.bar(range(len(test_names)), [1]*len(test_names), color=colors, alpha=0.7)
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.set_ylabel('Test Status')
    ax1.set_title('Validation Test Results')
    ax1.set_ylim(0, 1.2)
    
    for i, (bar, passed) in enumerate(zip(bars, passed_status)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                '✓' if passed else '✗', ha='center', va='bottom', fontsize=12)
    
    # Test 2: Error analysis (for tests with error metrics)
    ax2 = axes[0, 1]
    errors = []
    error_labels = []
    for result in validator.test_results:
        if 'total_error' in result:
            errors.append(result['total_error'])
            error_labels.append(result['test_name'])
        elif 'error' in result:
            errors.append(result['error'])
            error_labels.append(result['test_name'])
    
    if errors:
        ax2.bar(range(len(errors)), errors, alpha=0.7)
        ax2.axhline(y=validator.tolerance, color='red', linestyle='--', 
                   label=f'Tolerance ({validator.tolerance})')
        ax2.set_xticks(range(len(errors)))
        ax2.set_xticklabels(error_labels, rotation=45, ha='right')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Error Analysis')
        ax2.legend()
        ax2.set_yscale('log')
    
    # Test 3: Success rate pie chart
    ax3 = axes[1, 0]
    passed_count = sum(passed_status)
    failed_count = len(passed_status) - passed_count
    
    if failed_count > 0:
        ax3.pie([passed_count, failed_count], labels=['Passed', 'Failed'], 
               colors=['green', 'red'], autopct='%1.1f%%', alpha=0.7)
    else:
        ax3.pie([passed_count], labels=['All Passed'], colors=['green'], 
               autopct='%1.1f%%', alpha=0.7)
    ax3.set_title('Overall Success Rate')
    
    # Test 4: Performance bounds validation (if available)
    ax4 = axes[1, 1]
    for result in validator.test_results:
        if result['test_name'] == 'Fin Performance Bounds':
            categories = ['CL Bounds', 'CD Bounds', 'L/D Ratio', 'Monotonic']
            values = [result['cl_bounds_valid'], result['cd_bounds_valid'],
                     result['ld_ratio_valid'], result['monotonic_valid']]
            colors = ['green' if v else 'red' for v in values]
            
            ax4.bar(categories, [1]*len(categories), color=colors, alpha=0.7)
            ax4.set_ylabel('Validation Status')
            ax4.set_title('Fin Performance Validation')
            ax4.set_xticklabels(categories, rotation=45, ha='right')
            
            for i, (cat, val) in enumerate(zip(categories, values)):
                ax4.text(i, 0.5, '✓' if val else '✗', ha='center', va='center', 
                        fontsize=12, color='white', weight='bold')
            break
    else:
        ax4.text(0.5, 0.5, 'No Performance\nValidation Data', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Fin Performance Validation')
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run comprehensive validation
    validator = run_comprehensive_validation()
    
    # Create visualization
    create_test_visualization(validator)
    
    print("\nValidation complete! Check 'validation_report.txt' and 'validation_results.png'")