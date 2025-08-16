#!/usr/bin/env python3
"""
Test script for UOIF-Enhanced Hybrid Symbolic-Neural Accuracy Functional
Demonstrates integration with existing project components
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from uoif_enhanced_psi import UOIFEnhancedPsi, SourceType, ClaimClass
import json

def load_uoif_config():
    """Load UOIF configuration parameters"""
    try:
        with open('/Users/ryan_david_oates/Farmer/uoif_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: UOIF config file not found, using defaults")
        return {}

def test_reverse_koopman_integration():
    """Test Reverse Koopman Lipschitz continuity integration"""
    print("\n" + "="*50)
    print("Testing Reverse Koopman Integration")
    print("="*50)
    
    uoif_psi = UOIFEnhancedPsi()
    
    # Test Lipschitz continuity
    f = np.random.randn(10)
    g = f + np.random.randn(10) * 0.1
    
    lipschitz_bound = uoif_psi.reverse_koopman.apply_inverse(f, g)
    print(f"Lipschitz bound: {lipschitz_bound:.6f}")
    print(f"Confidence: {uoif_psi.reverse_koopman.confidence.value:.3f}")
    print(f"Source: {uoif_psi.reverse_koopman.confidence.source_type.value}")
    print(f"Classification: {uoif_psi.reverse_koopman.confidence.claim_class.value}")
    
    # Test spectral reconstruction
    modes = np.random.randn(5, 3)
    eigenvals = np.array([0.1, -0.05, 0.02, -0.01, 0.005])
    
    reconstruction = uoif_psi.reverse_koopman.spectral_reconstruction(modes, eigenvals)
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction norm: {np.linalg.norm(reconstruction):.6f}")

def test_rspo_dmd_integration():
    """Test RSPO convergence with DMD integration"""
    print("\n" + "="*50)
    print("Testing RSPO-DMD Integration")
    print("="*50)
    
    uoif_psi = UOIFEnhancedPsi()
    
    # Test reverse velocity update
    velocities = uoif_psi.rspo_optimizer.reverse_velocity_update(t=5)
    print(f"RSPO particles: {uoif_psi.rspo_optimizer.n_particles}")
    print(f"Velocity statistics: mean={np.mean(velocities):.6f}, std={np.std(velocities):.6f}")
    print(f"Confidence: {uoif_psi.rspo_optimizer.confidence.value:.3f}")
    
    # Test DMD mode selection
    data_matrix = np.random.randn(8, 20)  # 8 states, 20 time steps
    modes, eigenvals = uoif_psi.rspo_optimizer.dmd_mode_selection(data_matrix)
    
    print(f"DMD modes shape: {modes.shape}")
    print(f"Eigenvalues: {eigenvals[:5]}")  # Show first 5
    print(f"Dominant eigenvalue magnitude: {np.abs(eigenvals[0]):.6f}")

def test_euler_lagrange_confidence():
    """Test Oates Euler-Lagrange Confidence Theorem"""
    print("\n" + "="*50)
    print("Testing Euler-Lagrange Confidence")
    print("="*50)
    
    uoif_psi = UOIFEnhancedPsi()
    
    # Test consciousness field
    x = np.linspace(0, 1, 10)
    psi_field = uoif_psi.euler_lagrange.consciousness_field_psi(x, m=0.5, s=0.3)
    print(f"Consciousness field Ψ(x,m,s): {psi_field:.6f}")
    
    # Test hierarchical Bayesian confidence
    test_data = np.random.randn(100) * 0.1 + 0.8  # Simulated verified zero data
    confidence = uoif_psi.euler_lagrange.hierarchical_bayesian_confidence(test_data)
    print(f"Hierarchical Bayesian confidence: {confidence:.6f}")
    print(f"Constraint E[C] ≥ 1-ε: {confidence >= (1 - uoif_psi.euler_lagrange.epsilon)}")
    print(f"Framework confidence: {uoif_psi.euler_lagrange.confidence.value:.3f}")

def test_uoif_scoring_system():
    """Test UOIF scoring and allocation system"""
    print("\n" + "="*50)
    print("Testing UOIF Scoring System")
    print("="*50)
    
    uoif_psi = UOIFEnhancedPsi()
    
    # Test primitive scoring (Lipschitz example)
    primitive_score = uoif_psi.compute_uoif_score(
        ClaimClass.PRIMITIVE,
        auth=0.95, ver=0.97, depth=0.8, align=0.9, rec=0.85, noise=0.1
    )
    print(f"Primitive (Lipschitz) UOIF score: {primitive_score:.4f}")
    
    # Test interpretation scoring (convergence proof example)
    interpretation_score = uoif_psi.compute_uoif_score(
        ClaimClass.INTERPRETATION,
        auth=0.88, ver=0.89, depth=0.85, align=0.87, rec=0.82, noise=0.15
    )
    print(f"Interpretation (convergence) UOIF score: {interpretation_score:.4f}")
    
    # Test speculative scoring
    speculative_score = uoif_psi.compute_uoif_score(
        ClaimClass.INTERPRETATION,  # Using interpretation weights for speculative
        auth=0.6, ver=0.5, depth=0.7, align=0.6, rec=0.4, noise=0.4
    )
    print(f"Speculative (chaos analog) UOIF score: {speculative_score:.4f}")

def test_enhanced_psi_computation():
    """Test full enhanced Ψ(x) computation with UOIF components"""
    print("\n" + "="*50)
    print("Testing Enhanced Ψ(x) Computation")
    print("="*50)
    
    uoif_psi = UOIFEnhancedPsi()
    
    # Test parameters
    x_test = 0.5
    t_test = 1.0
    symbolic_data = np.random.randn(15) * 0.1 + 0.7
    neural_data = np.random.randn(15) * 0.15 + 0.8
    
    # Compute enhanced Ψ
    result = uoif_psi.compute_enhanced_psi(x_test, t_test, symbolic_data, neural_data)
    
    print(f"Test point: x={x_test}, t={t_test}")
    print(f"Enhanced Ψ(x): {result['psi_enhanced']:.6f}")
    print(f"Original Ψ(x):  {result['psi_original']:.6f}")
    print(f"Enhancement ratio: {result['psi_enhanced']/result['psi_original']:.3f}")
    
    print(f"\nComponent breakdown:")
    for key, value in result['components'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {np.mean(value):.6f} (mean)")
    
    print(f"\nUOIF component scores:")
    for key, value in result['uoif_scores'].items():
        print(f"  {key}: {value:.4f}")
    
    return result

def test_temporal_evolution():
    """Test temporal evolution of enhanced Ψ(x)"""
    print("\n" + "="*50)
    print("Testing Temporal Evolution")
    print("="*50)
    
    uoif_psi = UOIFEnhancedPsi()
    
    # Time range
    t_range = np.linspace(0, 5, 25)
    x_fixed = 0.5
    
    # Generate consistent test data
    np.random.seed(42)
    symbolic_data = np.random.randn(10) * 0.1 + 0.7
    neural_data = np.random.randn(10) * 0.15 + 0.8
    
    results = []
    for t in t_range:
        result = uoif_psi.compute_enhanced_psi(x_fixed, t, symbolic_data, neural_data)
        results.append({
            't': t,
            'psi_enhanced': result['psi_enhanced'],
            'psi_original': result['psi_original'],
            'enhancement_ratio': result['psi_enhanced'] / result['psi_original'],
            'koopman_stability': result['components']['koopman_stability'],
            'rspo_convergence': result['components']['rspo_convergence'],
            'el_confidence': result['components']['el_confidence']
        })
    
    # Analysis
    enhancement_ratios = [r['enhancement_ratio'] for r in results]
    print(f"Enhancement ratio statistics:")
    print(f"  Mean: {np.mean(enhancement_ratios):.3f}")
    print(f"  Std:  {np.std(enhancement_ratios):.3f}")
    print(f"  Min:  {np.min(enhancement_ratios):.3f}")
    print(f"  Max:  {np.max(enhancement_ratios):.3f}")
    
    return results

def create_comprehensive_visualization(temporal_results):
    """Create comprehensive visualization of UOIF integration"""
    print("\n" + "="*50)
    print("Creating Comprehensive Visualization")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    t_vals = [r['t'] for r in temporal_results]
    psi_enhanced = [r['psi_enhanced'] for r in temporal_results]
    psi_original = [r['psi_original'] for r in temporal_results]
    enhancement_ratios = [r['enhancement_ratio'] for r in temporal_results]
    koopman_stability = [r['koopman_stability'] for r in temporal_results]
    rspo_convergence = [r['rspo_convergence'] for r in temporal_results]
    el_confidence = [r['el_confidence'] for r in temporal_results]
    
    # Plot 1: Ψ(x) comparison
    axes[0,0].plot(t_vals, psi_enhanced, 'b-', linewidth=2, label='UOIF Enhanced')
    axes[0,0].plot(t_vals, psi_original, 'r--', linewidth=2, label='Original')
    axes[0,0].set_xlabel('Time t')
    axes[0,0].set_ylabel('Ψ(x)')
    axes[0,0].set_title('UOIF Enhanced vs Original Ψ(x)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Enhancement ratio
    axes[0,1].plot(t_vals, enhancement_ratios, 'g-', linewidth=2)
    axes[0,1].axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
    axes[0,1].set_xlabel('Time t')
    axes[0,1].set_ylabel('Enhancement Ratio')
    axes[0,1].set_title('UOIF Enhancement Factor')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Koopman stability
    axes[0,2].plot(t_vals, koopman_stability, 'c-', linewidth=2)
    axes[0,2].set_xlabel('Time t')
    axes[0,2].set_ylabel('Koopman Stability')
    axes[0,2].set_title('Reverse Koopman Stability')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: RSPO convergence
    axes[1,0].plot(t_vals, rspo_convergence, 'm-', linewidth=2)
    axes[1,0].set_xlabel('Time t')
    axes[1,0].set_ylabel('RSPO Convergence')
    axes[1,0].set_title('RSPO-DMD Convergence')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Euler-Lagrange confidence
    axes[1,1].plot(t_vals, el_confidence, 'orange', linewidth=2)
    axes[1,1].set_xlabel('Time t')
    axes[1,1].set_ylabel('E-L Confidence')
    axes[1,1].set_title('Euler-Lagrange Confidence')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Combined UOIF components
    axes[1,2].plot(t_vals, koopman_stability, 'c-', label='Koopman', alpha=0.7)
    axes[1,2].plot(t_vals, rspo_convergence, 'm-', label='RSPO', alpha=0.7)
    axes[1,2].plot(t_vals, el_confidence, 'orange', label='E-L Conf', alpha=0.7)
    axes[1,2].set_xlabel('Time t')
    axes[1,2].set_ylabel('Component Values')
    axes[1,2].set_title('UOIF Component Evolution')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ryan_david_oates/Farmer/uoif_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as: uoif_comprehensive_analysis.png")

def main():
    """Main test execution"""
    print("UOIF-Enhanced Hybrid Symbolic-Neural Accuracy Functional")
    print("Integration Test Suite")
    print("="*60)
    
    # Load configuration
    config = load_uoif_config()
    if config:
        print(f"Loaded UOIF config version: {config.get('uoif_ruleset', {}).get('version', 'unknown')}")
    
    # Run individual component tests
    test_reverse_koopman_integration()
    test_rspo_dmd_integration()
    test_euler_lagrange_confidence()
    test_uoif_scoring_system()
    
    # Run comprehensive test
    enhanced_result = test_enhanced_psi_computation()
    
    # Run temporal evolution test
    temporal_results = test_temporal_evolution()
    
    # Create visualization
    create_comprehensive_visualization(temporal_results)
    
    print("\n" + "="*60)
    print("UOIF Integration Test Suite Complete")
    print("="*60)
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"- Reverse Koopman confidence: {enhanced_result['confidence_measures']['koopman'].value:.3f}")
    print(f"- RSPO-DMD confidence: {enhanced_result['confidence_measures']['rspo'].value:.3f}")
    print(f"- Euler-Lagrange confidence: {enhanced_result['confidence_measures']['euler_lagrange'].value:.3f}")
    print(f"- Overall enhancement factor: {enhanced_result['psi_enhanced']/enhanced_result['psi_original']:.3f}")
    
    return enhanced_result, temporal_results

if __name__ == "__main__":
    enhanced_result, temporal_results = main()
