#!/usr/bin/env python3
"""
Simplified Riemann Zeta Function Laurent Series Demo
Demonstrates the Laurent expansion ζ(s) = 1/(s-1) + γ + Σ(-1)^n γ_n (s-1)^n / n!
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.5772156649015329

# First few Stieltjes constants (computed values)
STIELTJES_CONSTANTS = [
    -0.07281584548367672,  # γ_1
    -0.009690363192872318, # γ_2
    0.002053834420303346,  # γ_3
    0.002325370065467300,  # γ_4
    0.000793323817301062,  # γ_5
    -0.000238769345430199, # γ_6
    -0.000527289567057751, # γ_7
    -0.000352123353803039, # γ_8
    -0.000343947744180880, # γ_9
    0.000205332814909065   # γ_10
]

class SimpleZetaLaurent:
    """Simplified Riemann Zeta Function Laurent Series"""
    
    def __init__(self, max_terms=10):
        self.max_terms = max_terms
        self.euler_mascheroni = EULER_MASCHERONI
        self.stieltjes = STIELTJES_CONSTANTS[:max_terms]
    
    def laurent_expansion(self, s, include_terms=None):
        """
        Compute ζ(s) using Laurent expansion around s = 1
        ζ(s) = 1/(s-1) + γ + Σ(-1)^n γ_n (s-1)^n / n!
        """
        if include_terms is None:
            include_terms = self.max_terms
        
        # Handle the pole
        if abs(s - 1) < 1e-15:
            return float('inf')
        
        # Principal part: 1/(s-1)
        principal_part = 1 / (s - 1)
        
        # Constant term: γ
        constant_term = self.euler_mascheroni
        
        # Higher-order terms: Σ(-1)^n γ_n (s-1)^n / n!
        higher_order = 0
        s_minus_1 = s - 1
        
        for n in range(1, min(include_terms + 1, len(self.stieltjes) + 1)):
            if n <= len(self.stieltjes):
                gamma_n = self.stieltjes[n-1]
                term = ((-1)**n * gamma_n * (s_minus_1**n)) / math.factorial(n)
                higher_order += term
        
        return principal_part + constant_term + higher_order
    
    def show_expansion_terms(self, s):
        """Show individual terms in the expansion"""
        print(f"\nLaurent Expansion Terms for s = {s}:")
        print("-" * 50)
        
        s_minus_1 = s - 1
        
        # Principal part
        principal = 1 / s_minus_1 if abs(s_minus_1) > 1e-15 else float('inf')
        print(f"Principal part: 1/(s-1) = 1/({s_minus_1:.6f}) = {principal:.6f}")
        
        # Constant term
        print(f"Constant term: γ = {self.euler_mascheroni:.6f}")
        
        # Higher-order terms
        running_sum = principal + self.euler_mascheroni
        print(f"Running sum after constant: {running_sum:.6f}")
        
        for n in range(1, min(8, len(self.stieltjes) + 1)):
            gamma_n = self.stieltjes[n-1]
            coeff = ((-1)**n * gamma_n) / math.factorial(n)
            term_value = coeff * (s_minus_1**n)
            running_sum += term_value
            
            sign = "+" if coeff >= 0 else "-"
            print(f"Term n={n}: {sign}{abs(coeff):.8f} × ({s_minus_1:.3f})^{n} = {term_value:.8f}")
            print(f"  Running sum: {running_sum:.6f}")
        
        return running_sum

def demonstrate_laurent_expansion():
    """Demonstrate the Laurent expansion with detailed analysis"""
    
    print("Riemann Zeta Function Laurent Series Demonstration")
    print("=" * 55)
    
    zeta = SimpleZetaLaurent(max_terms=10)
    
    # Test points near s = 1
    test_points = [1.1, 1.01, 1.001]
    
    print("\nLaurent Expansion Values:")
    print("-" * 30)
    
    for s_val in test_points:
        zeta_approx = zeta.laurent_expansion(s_val)
        print(f"s = {s_val:6.3f}: ζ(s) ≈ {zeta_approx:10.6f}")
    
    # Detailed breakdown for s = 1.1
    print("\n" + "="*55)
    print("DETAILED EXPANSION ANALYSIS")
    print("="*55)
    
    s_detail = 1.1
    final_sum = zeta.show_expansion_terms(s_detail)
    
    # Compare with direct computation
    direct_zeta = zeta.laurent_expansion(s_detail)
    print(f"\nFinal Laurent approximation: {final_sum:.6f}")
    print(f"Direct computation: {direct_zeta:.6f}")
    print(f"Difference: {abs(final_sum - direct_zeta):.2e}")
    
    # Show the mathematical formula
    print(f"\nMathematical Formula:")
    print(f"ζ(s) = 1/(s-1) + γ + Σ(-1)^n γ_n (s-1)^n / n!")
    print(f"where γ = {EULER_MASCHERONI:.6f} (Euler-Mascheroni constant)")
    
    # Show first few Stieltjes constants
    print(f"\nFirst few Stieltjes constants:")
    for i, gamma_n in enumerate(STIELTJES_CONSTANTS[:6], 1):
        print(f"γ_{i} = {gamma_n:12.9f}")
    
    # Convergence analysis
    print(f"\nConvergence Analysis:")
    print("-" * 25)
    
    distances = [0.1, 0.01, 0.001, 0.0001]
    
    for dist in distances:
        s_point = 1 + dist
        zeta_val = zeta.laurent_expansion(s_point)
        
        # Estimate convergence by looking at term ratios
        s_minus_1 = dist
        term_ratio = abs(s_minus_1)  # Simplified convergence indicator
        
        print(f"Distance {dist:6.4f}: ζ(s) = {zeta_val:8.4f}, convergence factor ≈ {term_ratio:.4f}")
    
    # Create visualization
    create_laurent_visualization(zeta)
    
    return zeta

def create_laurent_visualization(zeta):
    """Create visualization of the Laurent expansion"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Zeta function values near s = 1
    s_values = np.linspace(1.01, 2.0, 100)
    zeta_values = []
    
    for s in s_values:
        try:
            zeta_val = zeta.laurent_expansion(s)
            # Cap extreme values for visualization
            zeta_values.append(min(zeta_val, 50) if zeta_val > 0 else max(zeta_val, -50))
        except:
            zeta_values.append(0)
    
    ax1.plot(s_values, zeta_values, 'b-', linewidth=2)
    ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Pole at s=1')
    ax1.set_xlabel('s')
    ax1.set_ylabel('ζ(s)')
    ax1.set_title('Zeta Function Near s=1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-5, 20])
    
    # Plot 2: Term magnitudes for s = 1.1
    s_test = 1.1
    s_minus_1 = s_test - 1
    
    term_indices = list(range(-1, 8))  # -1 for pole, 0 for constant, 1+ for higher order
    term_magnitudes = []
    
    # Principal part
    term_magnitudes.append(abs(1 / s_minus_1))
    
    # Constant term
    term_magnitudes.append(abs(EULER_MASCHERONI))
    
    # Higher-order terms
    for n in range(1, 8):
        if n <= len(STIELTJES_CONSTANTS):
            gamma_n = STIELTJES_CONSTANTS[n-1]
            coeff = abs((-1)**n * gamma_n) / math.factorial(n)
            term_mag = coeff * (s_minus_1**n)
            term_magnitudes.append(term_mag)
        else:
            term_magnitudes.append(0)
    
    colors = ['red', 'blue'] + ['green'] * 6
    ax2.bar(term_indices, term_magnitudes, color=colors, alpha=0.7)
    ax2.set_xlabel('Term Index')
    ax2.set_ylabel('|Term Magnitude|')
    ax2.set_title('Laurent Series Term Magnitudes (s=1.1)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence behavior
    distances = np.logspace(-3, -0.5, 20)
    zeta_approx = []
    
    for dist in distances:
        s_point = 1 + dist
        try:
            zeta_val = zeta.laurent_expansion(s_point)
            zeta_approx.append(abs(zeta_val) if abs(zeta_val) < 100 else 100)
        except:
            zeta_approx.append(100)
    
    ax3.loglog(distances, zeta_approx, 'go-', linewidth=2, markersize=4)
    ax3.set_xlabel('Distance from s=1')
    ax3.set_ylabel('|ζ(s)|')
    ax3.set_title('Zeta Function Magnitude vs Distance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Partial sums convergence
    s_conv = 1.05
    s_minus_1_conv = s_conv - 1
    
    partial_sums = []
    n_terms = []
    
    # Start with principal part + constant
    current_sum = 1 / s_minus_1_conv + EULER_MASCHERONI
    partial_sums.append(current_sum)
    n_terms.append(0)
    
    # Add higher-order terms
    for n in range(1, min(10, len(STIELTJES_CONSTANTS) + 1)):
        gamma_n = STIELTJES_CONSTANTS[n-1]
        term = ((-1)**n * gamma_n * (s_minus_1_conv**n)) / math.factorial(n)
        current_sum += term
        partial_sums.append(current_sum)
        n_terms.append(n)
    
    ax4.plot(n_terms, partial_sums, 'mo-', linewidth=2, markersize=6)
    ax4.axhline(y=partial_sums[-1], color='r', linestyle='--', alpha=0.7, 
                label=f'Final: {partial_sums[-1]:.4f}')
    ax4.set_xlabel('Number of Higher-Order Terms')
    ax4.set_ylabel('Partial Sum')
    ax4.set_title('Laurent Series Convergence (s=1.05)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ryan_david_oates/Farmer/zeta_laurent_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as: zeta_laurent_analysis.png")

def analyze_cryptographic_implications():
    """Analyze implications for cryptography and prime distribution"""
    
    print("\n" + "="*60)
    print("CRYPTOGRAPHIC AND PRIME DISTRIBUTION IMPLICATIONS")
    print("="*60)
    
    zeta = SimpleZetaLaurent()
    
    print("\n1. Prime Number Theorem Connection:")
    print("-" * 40)
    print("The Prime Number Theorem states: π(x) ~ x/log(x)")
    print("The zeta function pole at s=1 with residue 1 is crucial for this result.")
    
    # Analyze behavior near critical points
    critical_points = [1.001, 1.01, 1.1, 1.5]
    
    print(f"\n2. Zeta Function Behavior Analysis:")
    print("-" * 40)
    
    for s in critical_points:
        zeta_val = zeta.laurent_expansion(s)
        distance = s - 1
        
        # Estimate "prime density influence" (simplified model)
        if abs(zeta_val) < 1000:
            density_factor = 1.0 / abs(zeta_val) if zeta_val != 0 else 0
        else:
            density_factor = 0.001
        
        print(f"s = {s:5.3f}: ζ(s) = {zeta_val:8.4f}, density factor ≈ {density_factor:.6f}")
    
    print(f"\n3. Cryptographic Security Implications:")
    print("-" * 45)
    print("• Current RSA security relies on unpredictable prime distribution")
    print("• The Laurent expansion shows the mathematical structure near s=1")
    print("• Any deviation from expected zeta behavior could affect:")
    print("  - Prime gap distributions")
    print("  - Factoring algorithm efficiency")
    print("  - Random prime generation")
    
    print(f"\n4. Stieltjes Constants Impact:")
    print("-" * 35)
    print("The higher-order terms (Stieltjes constants) provide fine structure:")
    
    for i, gamma_n in enumerate(STIELTJES_CONSTANTS[:5], 1):
        magnitude = abs(gamma_n)
        significance = "High" if magnitude > 0.01 else "Medium" if magnitude > 0.001 else "Low"
        print(f"γ_{i} = {gamma_n:10.6f} (magnitude: {magnitude:.6f}, significance: {significance})")
    
    print(f"\n5. Theoretical Bounds and Limits:")
    print("-" * 35)
    print(f"• Euler-Mascheroni constant: γ = {EULER_MASCHERONI:.10f}")
    print(f"• This constant appears in many number-theoretic estimates")
    print(f"• Its precise value affects error bounds in prime counting")
    
    return zeta

if __name__ == "__main__":
    print("Starting Riemann Zeta Function Laurent Series Analysis...")
    
    # Main demonstration
    zeta_system = demonstrate_laurent_expansion()
    
    # Cryptographic analysis
    analyze_cryptographic_implications()
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Key findings:")
    print("• Laurent expansion provides exact mathematical structure near s=1")
    print("• Stieltjes constants encode fine details of zeta function behavior")
    print("• Connection to prime distribution has cryptographic implications")
    print("• Framework ready for UOIF integration and enhancement")
    print("="*60)
