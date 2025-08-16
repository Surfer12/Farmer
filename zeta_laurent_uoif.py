#!/usr/bin/env python3
"""
Riemann Zeta Function Laurent Series with UOIF Integration
Implements the Laurent expansion ζ(s) = 1/(s-1) + γ + Σ(-1)^n γ_n (s-1)^n / n!
Integrated with Zeta Sub-Latent Manifold Reverse Theorem concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from scipy.special import gamma as gamma_func
from uoif_enhanced_psi import UOIFEnhancedPsi, SourceType, ClaimClass, ConfidenceMeasure

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

@dataclass
class ZetaExpansionTerm:
    """Individual term in the Laurent expansion"""
    coefficient: float
    power: int
    term_type: str  # 'pole', 'constant', 'regular'
    confidence: ConfidenceMeasure

class RiemannZetaLaurent:
    """
    Riemann Zeta Function Laurent Series Implementation
    [Primitive] Exact Laurent expansion around s = 1
    """
    
    def __init__(self, max_terms: int = 10):
        self.max_terms = max_terms
        self.euler_mascheroni = EULER_MASCHERONI
        self.stieltjes = STIELTJES_CONSTANTS[:max_terms]
        
        # UOIF confidence for zeta expansion
        self.confidence = ConfidenceMeasure(
            value=0.99,  # High confidence - well-established mathematical result
            source_type=SourceType.CANONICAL,
            claim_class=ClaimClass.PRIMITIVE,
            beta=1.10,
            reliability=0.99
        )
    
    def laurent_expansion(self, s: complex, include_terms: int = None) -> complex:
        """
        Compute ζ(s) using Laurent expansion around s = 1
        ζ(s) = 1/(s-1) + γ + Σ(-1)^n γ_n (s-1)^n / n!
        """
        if include_terms is None:
            include_terms = self.max_terms
        
        # Principal part: 1/(s-1)
        if abs(s - 1) < 1e-15:
            return complex(float('inf'), 0)
        
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
    
    def expansion_terms(self, s: complex) -> List[ZetaExpansionTerm]:
        """Get individual terms of the Laurent expansion"""
        terms = []
        s_minus_1 = s - 1
        
        # Principal part
        if abs(s_minus_1) > 1e-15:
            terms.append(ZetaExpansionTerm(
                coefficient=1.0,
                power=-1,
                term_type='pole',
                confidence=self.confidence
            ))
        
        # Constant term
        terms.append(ZetaExpansionTerm(
            coefficient=self.euler_mascheroni,
            power=0,
            term_type='constant',
            confidence=self.confidence
        ))
        
        # Higher-order terms
        for n in range(1, min(self.max_terms + 1, len(self.stieltjes) + 1)):
            gamma_n = self.stieltjes[n-1]
            coeff = ((-1)**n * gamma_n) / math.factorial(n)
            
            terms.append(ZetaExpansionTerm(
                coefficient=coeff,
                power=n,
                term_type='regular',
                confidence=self.confidence
            ))
        
        return terms
    
    def convergence_analysis(self, s: complex, max_terms: int = 20) -> Dict:
        """Analyze convergence of the Laurent series"""
        s_minus_1 = abs(s - 1)
        
        # Convergence radius analysis
        convergence_radius = 1.0  # Theoretical convergence radius
        
        # Term-by-term convergence
        terms = []
        partial_sums = []
        current_sum = 1 / (s - 1) + self.euler_mascheroni
        partial_sums.append(current_sum)
        
        for n in range(1, min(max_terms, len(self.stieltjes) + 1)):
            gamma_n = self.stieltjes[n-1]
            term = ((-1)**n * gamma_n * ((s - 1)**n)) / math.factorial(n)
            terms.append(abs(term))
            current_sum += term
            partial_sums.append(current_sum)
        
        return {
            'convergence_radius': convergence_radius,
            'distance_from_pole': s_minus_1,
            'converges': s_minus_1 < convergence_radius,
            'term_magnitudes': terms,
            'partial_sums': partial_sums,
            'final_approximation': current_sum
        }

class ZetaUOIFIntegration:
    """
    Integration of Riemann Zeta Laurent expansion with UOIF framework
    Connects zeta function behavior to Hybrid Symbolic-Neural Accuracy Functional
    """
    
    def __init__(self):
        self.zeta_laurent = RiemannZetaLaurent()
        self.uoif_psi = UOIFEnhancedPsi()
    
    def zeta_enhanced_psi(self, x: float, t: float, s_point: complex = 1.1) -> Dict:
        """
        Enhanced Ψ(x) computation incorporating zeta function behavior
        Uses zeta expansion coefficients to modulate the hybrid functional
        """
        # Standard UOIF computation
        symbolic_data = np.random.randn(10) * 0.1 + 0.7
        neural_data = np.random.randn(10) * 0.15 + 0.8
        
        base_result = self.uoif_psi.compute_enhanced_psi(x, t, symbolic_data, neural_data)
        
        # Zeta function modulation
        zeta_value = self.zeta_laurent.laurent_expansion(s_point)
        zeta_terms = self.zeta_laurent.expansion_terms(s_point)
        
        # Extract zeta coefficients for modulation
        principal_coeff = 1.0 / abs(s_point - 1) if abs(s_point - 1) > 1e-15 else 1.0
        constant_coeff = self.zeta_laurent.euler_mascheroni
        
        # Modulate Ψ components using zeta structure
        zeta_modulation = {
            'principal_influence': min(2.0, 1.0 + 0.1 * principal_coeff),
            'constant_influence': 1.0 + 0.05 * constant_coeff,
            'stieltjes_influence': 1.0 + 0.02 * sum(abs(term.coefficient) for term in zeta_terms[2:5])
        }
        
        # Enhanced Ψ with zeta modulation
        psi_zeta_enhanced = (base_result['psi_enhanced'] * 
                           zeta_modulation['principal_influence'] * 
                           zeta_modulation['constant_influence'] * 
                           zeta_modulation['stieltjes_influence'])
        
        return {
            'psi_zeta_enhanced': psi_zeta_enhanced,
            'psi_base_enhanced': base_result['psi_enhanced'],
            'psi_original': base_result['psi_original'],
            'zeta_value': zeta_value,
            'zeta_modulation': zeta_modulation,
            'zeta_terms': zeta_terms,
            'enhancement_factor': psi_zeta_enhanced / base_result['psi_enhanced'],
            'base_components': base_result['components'],
            'uoif_scores': base_result['uoif_scores']
        }
    
    def prime_distribution_analysis(self, x_range: np.ndarray) -> Dict:
        """
        Analyze prime distribution implications using zeta Laurent expansion
        Connects to cryptographic security assessment
        """
        results = []
        
        for x in x_range:
            s_point = complex(1 + x, 0)  # Points near the critical line
            
            # Zeta analysis
            zeta_val = self.zeta_laurent.laurent_expansion(s_point)
            convergence = self.zeta_laurent.convergence_analysis(s_point)
            
            # Prime counting function approximation influence
            # π(x) ~ x/log(x) is affected by zeta pole behavior
            prime_density_factor = abs(zeta_val) if abs(zeta_val) < 100 else 100
            
            # UOIF-enhanced assessment
            psi_result = self.zeta_enhanced_psi(x, 1.0, s_point)
            
            results.append({
                'x': x,
                's_point': s_point,
                'zeta_value': zeta_val,
                'prime_density_factor': prime_density_factor,
                'psi_enhancement': psi_result['enhancement_factor'],
                'convergence_quality': len(convergence['term_magnitudes']) if convergence['converges'] else 0
            })
        
        return {
            'analysis_points': results,
            'mean_enhancement': np.mean([r['psi_enhancement'] for r in results]),
            'prime_density_variation': np.std([r['prime_density_factor'] for r in results]),
            'convergence_stability': np.mean([r['convergence_quality'] for r in results])
        }

def demonstrate_zeta_laurent_integration():
    """Comprehensive demonstration of zeta Laurent series with UOIF integration"""
    
    print("Riemann Zeta Function Laurent Series with UOIF Integration")
    print("=" * 65)
    
    # Initialize components
    zeta_laurent = RiemannZetaLaurent(max_terms=10)
    zeta_uoif = ZetaUOIFIntegration()
    
    # Test points near s = 1
    test_points = [1.1, 1.01, 1.001, 1.0001]
    
    print("\nLaurent Expansion Analysis:")
    print("-" * 40)
    
    for s_val in test_points:
        s = complex(s_val, 0)
        zeta_approx = zeta_laurent.laurent_expansion(s)
        convergence = zeta_laurent.convergence_analysis(s)
        
        print(f"\ns = {s_val}:")
        print(f"  ζ(s) ≈ {zeta_approx:.6f}")
        print(f"  Distance from pole: {abs(s - 1):.6f}")
        print(f"  Converges: {convergence['converges']}")
        print(f"  Terms computed: {len(convergence['term_magnitudes'])}")
    
    # Detailed expansion for s = 1.1
    print(f"\nDetailed Expansion at s = 1.1:")
    print("-" * 35)
    
    s_detail = complex(1.1, 0)
    terms = zeta_laurent.expansion_terms(s_detail)
    
    print(f"Principal part: 1/(s-1) = 1/(0.1) = 10.0")
    print(f"Constant term: γ = {EULER_MASCHERONI:.6f}")
    
    for i, term in enumerate(terms[2:8]):  # Show first few higher-order terms
        s_minus_1 = 0.1
        term_value = term.coefficient * (s_minus_1 ** term.power)
        print(f"Term n={term.power}: {term.coefficient:.8f} × (0.1)^{term.power} = {term_value:.8f}")
    
    # UOIF Integration
    print(f"\nUOIF-Enhanced Analysis:")
    print("-" * 30)
    
    x_test, t_test = 0.5, 1.0
    s_test = complex(1.05, 0)
    
    result = zeta_uoif.zeta_enhanced_psi(x_test, t_test, s_test)
    
    print(f"Test point: x={x_test}, t={t_test}, s={s_test}")
    print(f"Zeta-enhanced Ψ(x): {result['psi_zeta_enhanced']:.6f}")
    print(f"Base enhanced Ψ(x): {result['psi_base_enhanced']:.6f}")
    print(f"Original Ψ(x): {result['psi_original']:.6f}")
    print(f"Zeta enhancement factor: {result['enhancement_factor']:.3f}")
    print(f"Zeta value at s: {result['zeta_value']:.6f}")
    
    print(f"\nZeta modulation factors:")
    for key, value in result['zeta_modulation'].items():
        print(f"  {key}: {value:.6f}")
    
    # Prime distribution analysis
    print(f"\nPrime Distribution Analysis:")
    print("-" * 35)
    
    x_range = np.linspace(0.001, 0.1, 10)
    prime_analysis = zeta_uoif.prime_distribution_analysis(x_range)
    
    print(f"Analysis points: {len(prime_analysis['analysis_points'])}")
    print(f"Mean Ψ enhancement: {prime_analysis['mean_enhancement']:.3f}")
    print(f"Prime density variation: {prime_analysis['prime_density_variation']:.3f}")
    print(f"Convergence stability: {prime_analysis['convergence_stability']:.1f}")
    
    # Visualization
    create_zeta_laurent_visualization(zeta_laurent, zeta_uoif)
    
    return result, prime_analysis

def create_zeta_laurent_visualization(zeta_laurent, zeta_uoif):
    """Create comprehensive visualization of zeta Laurent series and UOIF integration"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Laurent expansion convergence
    s_values = np.linspace(1.001, 2.0, 100)
    zeta_approx = []
    
    for s_val in s_values:
        try:
            zeta_val = zeta_laurent.laurent_expansion(complex(s_val, 0))
            zeta_approx.append(abs(zeta_val) if abs(zeta_val) < 50 else 50)
        except:
            zeta_approx.append(50)
    
    axes[0,0].plot(s_values, zeta_approx, 'b-', linewidth=2)
    axes[0,0].axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Pole at s=1')
    axes[0,0].set_xlabel('Re(s)')
    axes[0,0].set_ylabel('|ζ(s)|')
    axes[0,0].set_title('Zeta Function Near s=1')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0, 20])
    
    # Plot 2: Term magnitudes in Laurent expansion
    s_test = complex(1.1, 0)
    terms = zeta_laurent.expansion_terms(s_test)
    term_powers = [term.power for term in terms[:10]]
    term_magnitudes = [abs(term.coefficient * (0.1 ** term.power)) if term.power >= 0 
                      else abs(term.coefficient / 0.1) for term in terms[:10]]
    
    axes[0,1].bar(term_powers, term_magnitudes, color=['red' if p == -1 else 'blue' if p == 0 else 'green' 
                                                      for p in term_powers])
    axes[0,1].set_xlabel('Term Power')
    axes[0,1].set_ylabel('|Term Magnitude|')
    axes[0,1].set_title('Laurent Series Term Magnitudes (s=1.1)')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Convergence analysis
    distances = np.logspace(-3, -0.5, 20)
    convergence_quality = []
    
    for dist in distances:
        s_point = complex(1 + dist, 0)
        conv_analysis = zeta_laurent.convergence_analysis(s_point, max_terms=15)
        quality = len(conv_analysis['term_magnitudes']) if conv_analysis['converges'] else 0
        convergence_quality.append(quality)
    
    axes[0,2].semilogx(distances, convergence_quality, 'go-', linewidth=2, markersize=4)
    axes[0,2].set_xlabel('Distance from s=1')
    axes[0,2].set_ylabel('Convergent Terms')
    axes[0,2].set_title('Laurent Series Convergence Quality')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: UOIF enhancement factors
    x_range = np.linspace(0.01, 0.2, 15)
    enhancement_factors = []
    zeta_values = []
    
    for x in x_range:
        s_point = complex(1 + x, 0)
        result = zeta_uoif.zeta_enhanced_psi(0.5, 1.0, s_point)
        enhancement_factors.append(result['enhancement_factor'])
        zeta_values.append(abs(result['zeta_value']) if abs(result['zeta_value']) < 20 else 20)
    
    axes[1,0].plot(x_range, enhancement_factors, 'b-', linewidth=2, label='Enhancement Factor')
    axes[1,0].set_xlabel('Distance from s=1')
    axes[1,0].set_ylabel('Enhancement Factor')
    axes[1,0].set_title('UOIF-Zeta Enhancement')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Zeta values vs enhancement
    axes[1,1].scatter(zeta_values, enhancement_factors, c=x_range, cmap='viridis', s=50)
    axes[1,1].set_xlabel('|ζ(s)|')
    axes[1,1].set_ylabel('Enhancement Factor')
    axes[1,1].set_title('Zeta Value vs UOIF Enhancement')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Prime distribution implications
    prime_analysis = zeta_uoif.prime_distribution_analysis(np.linspace(0.01, 0.1, 10))
    analysis_points = prime_analysis['analysis_points']
    
    x_vals = [p['x'] for p in analysis_points]
    prime_factors = [p['prime_density_factor'] for p in analysis_points]
    psi_enhancements = [p['psi_enhancement'] for p in analysis_points]
    
    ax2 = axes[1,2].twinx()
    line1 = axes[1,2].plot(x_vals, prime_factors, 'r-', linewidth=2, label='Prime Density Factor')
    line2 = ax2.plot(x_vals, psi_enhancements, 'b--', linewidth=2, label='Ψ Enhancement')
    
    axes[1,2].set_xlabel('Distance from s=1')
    axes[1,2].set_ylabel('Prime Density Factor', color='r')
    ax2.set_ylabel('Ψ Enhancement Factor', color='b')
    axes[1,2].set_title('Prime Distribution & UOIF Analysis')
    
    # Combine legends
    lines1, labels1 = axes[1,2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1,2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ryan_david_oates/Farmer/zeta_laurent_uoif_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as: zeta_laurent_uoif_analysis.png")

if __name__ == "__main__":
    result, prime_analysis = demonstrate_zeta_laurent_integration()
