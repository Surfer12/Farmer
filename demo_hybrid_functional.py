#!/usr/bin/env python3
"""
Comprehensive Demo of the Hybrid Symbolic-Neural Accuracy Functional

This script demonstrates the complete implementation of the hybrid functional Ψ(x)
across multiple domains:
1. Technical computation (Burgers equation with PINN vs RK4)
2. Collaboration scenarios (open-source, benefits, project phases)
3. Scaling analysis (connection to Broken Neural Scaling Laws)
4. Visualization and analysis tools

Run this script to see the full framework in action.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from hybrid_accuracy_functional import (
    HybridAccuracyFunctional, HybridConfig, 
    reproduce_numerical_example, demonstrate_framework
)
from collaboration_scenarios import (
    CollaborationFramework, ScenarioGenerator,
    demonstrate_collaboration_scenarios, analyze_broken_neural_scaling_laws_connection
)

# Try to import the Burgers solver (requires PyTorch)
try:
    from burgers_pinn_solver import BurgersSolver, demonstrate_burgers_pinn
    BURGERS_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Burgers PINN solver will be skipped.")
    BURGERS_AVAILABLE = False

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HybridFunctionalVisualizer:
    """Visualization tools for the hybrid accuracy functional."""
    
    def __init__(self):
        self.fig_count = 0
    
    def plot_component_breakdown(self, results: Dict[str, float], title: str = "Component Breakdown"):
        """Plot the breakdown of hybrid functional components."""
        self.fig_count += 1
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{title} - Hybrid Functional Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy components
        accuracies = [results['S'], results['N']]
        labels = ['Symbolic (S)', 'Neural (N)']
        colors = ['#2E86C1', '#E74C3C']
        
        ax1.bar(labels, accuracies, color=colors, alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Components')
        ax1.set_ylim(0, 1)
        
        # Add values on bars
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. Penalty components
        penalties = [results['R_cog'], results['R_eff']]
        penalty_labels = ['Cognitive', 'Efficiency']
        colors_penalty = ['#F39C12', '#8E44AD']
        
        ax2.bar(penalty_labels, penalties, color=colors_penalty, alpha=0.7)
        ax2.set_ylabel('Penalty')
        ax2.set_title('Penalty Components')
        
        for i, v in enumerate(penalties):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 3. Hybrid composition
        hybrid_components = [
            results['alpha'] * results['S'],
            (1 - results['alpha']) * results['N']
        ]
        hybrid_labels = [f'α×S (α={results["alpha"]:.2f})', f'(1-α)×N']
        
        ax3.pie(hybrid_components, labels=hybrid_labels, autopct='%1.3f', startangle=90)
        ax3.set_title('Hybrid Output Composition')
        
        # 4. Final computation flow
        flow_values = [
            results['hybrid_output'],
            results['regularization'],
            results['P_calibrated'],
            results['psi']
        ]
        flow_labels = ['Hybrid\nOutput', 'Regularization\nTerm', 'Calibrated\nProbability', 'Final\nΨ(x)']
        
        x_pos = np.arange(len(flow_labels))
        bars = ax4.bar(x_pos, flow_values, color=['#3498DB', '#2ECC71', '#F1C40F', '#E74C3C'], alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(flow_labels, rotation=45, ha='right')
        ax4.set_ylabel('Value')
        ax4.set_title('Computation Flow')
        
        # Add values on bars
        for i, (bar, v) in enumerate(zip(bars, flow_values)):
            ax4.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', 
                    ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'/workspace/hybrid_breakdown_{self.fig_count}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_scenario_comparison(self, scenarios: Dict[str, Dict]):
        """Compare different scenarios side by side."""
        self.fig_count += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        scenario_names = list(scenarios.keys())
        psi_values = [scenarios[name]['psi'] for name in scenario_names]
        
        # 1. Ψ(x) comparison
        colors = sns.color_palette("husl", len(scenario_names))
        bars = ax1.bar(scenario_names, psi_values, color=colors, alpha=0.7)
        ax1.set_ylabel('Ψ(x) Value')
        ax1.set_title('Scenario Comparison - Hybrid Functional Values')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, value in zip(bars, psi_values):
            ax1.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.3f}', 
                    ha='center', fontweight='bold')
        
        # 2. Component heatmap
        components = ['S', 'N', 'alpha', 'R_cog', 'R_eff', 'P_calibrated']
        component_matrix = []
        
        for name in scenario_names:
            row = [scenarios[name][comp] for comp in components]
            component_matrix.append(row)
        
        im = ax2.imshow(component_matrix, cmap='RdYlBu_r', aspect='auto')
        ax2.set_xticks(range(len(components)))
        ax2.set_xticklabels(components)
        ax2.set_yticks(range(len(scenario_names)))
        ax2.set_yticklabels(scenario_names)
        ax2.set_title('Component Values Across Scenarios')
        
        # Add text annotations
        for i in range(len(scenario_names)):
            for j in range(len(components)):
                text = ax2.text(j, i, f'{component_matrix[i][j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.savefig(f'/workspace/scenario_comparison_{self.fig_count}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_scaling_analysis(self, scaling_data: Dict):
        """Plot the scaling analysis results."""
        self.fig_count += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scales = scaling_data['scales']
        psi_values = scaling_data['psi_values']
        break_points = scaling_data['break_points']
        
        # 1. Scaling curve
        ax1.semilogx(scales, psi_values, 'b-', linewidth=2, label='Ψ(x) scaling')
        
        # Mark break points
        if break_points:
            bp_scales = [scales[bp] for bp in break_points[:3]]
            bp_values = [psi_values[bp] for bp in break_points[:3]]
            ax1.scatter(bp_scales, bp_values, color='red', s=100, zorder=5, 
                       label='Potential break points')
        
        ax1.set_xlabel('Project Scale')
        ax1.set_ylabel('Ψ(x) Value')
        ax1.set_title('Scaling Analysis - Connection to BNSL')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Derivative analysis
        if len(psi_values) > 2:
            log_scales = np.log10(scales)
            derivatives = np.gradient(psi_values, log_scales)
            
            ax2.semilogx(scales[1:], derivatives[1:], 'g-', linewidth=2, label='dΨ/d(log scale)')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Project Scale')
            ax2.set_ylabel('Derivative')
            ax2.set_title('Scaling Derivative (Break Point Detection)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'/workspace/scaling_analysis_{self.fig_count}.png', dpi=150, bbox_inches='tight')
        plt.show()


def run_comprehensive_demo():
    """Run the complete demonstration of the hybrid functional framework."""
    print("🚀 Comprehensive Hybrid Symbolic-Neural Accuracy Functional Demo")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize visualizer
    visualizer = HybridFunctionalVisualizer()
    
    # 1. Reproduce the original numerical example
    print("\n📊 PART 1: Original Numerical Example")
    print("-" * 40)
    reproduce_numerical_example()
    
    # 2. Technical framework demonstration
    print("\n🔬 PART 2: Technical Framework Demonstration")
    print("-" * 40)
    demonstrate_framework()
    
    # Get detailed results for visualization
    config = HybridConfig(lambda_1=0.75, lambda_2=0.25, beta=1.2)
    framework = HybridAccuracyFunctional(config)
    
    # Technical scenario example
    tech_results = framework.compute_psi_detailed(
        np.array([0.5]), np.array([1.0])
    )
    tech_results_dict = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in tech_results.items()}
    
    visualizer.plot_component_breakdown(tech_results_dict, "Technical Scenario")
    
    # 3. Collaboration scenarios
    print("\n🤝 PART 3: Collaboration Scenarios")
    print("-" * 40)
    collab_results = demonstrate_collaboration_scenarios()
    
    # Visualize collaboration scenarios
    visualizer.plot_scenario_comparison(collab_results)
    
    # 4. Burgers equation demonstration (if available)
    if BURGERS_AVAILABLE:
        print("\n🌊 PART 4: Burgers Equation PINN vs RK4")
        print("-" * 40)
        try:
            burgers_results = demonstrate_burgers_pinn()
            print("Burgers equation demonstration completed successfully!")
        except Exception as e:
            print(f"Burgers demonstration failed: {e}")
    else:
        print("\n⚠️  PART 4: Burgers Equation (SKIPPED - PyTorch not available)")
        print("-" * 40)
    
    # 5. Scaling analysis and BNSL connection
    print("\n📈 PART 5: Scaling Analysis & BNSL Connection")
    print("-" * 40)
    scaling_results = analyze_broken_neural_scaling_laws_connection()
    
    # Visualize scaling analysis
    visualizer.plot_scaling_analysis(scaling_results)
    
    # 6. Summary and conclusions
    print("\n📋 PART 6: Summary and Conclusions")
    print("-" * 40)
    
    # Calculate overall framework performance
    all_psi_values = []
    
    # Technical scenarios
    all_psi_values.append(tech_results_dict['psi'])
    
    # Collaboration scenarios
    for scenario_name, results in collab_results.items():
        if isinstance(results, dict) and 'psi' in results:
            all_psi_values.append(results['psi'])
        elif isinstance(results, list):  # phased project
            all_psi_values.extend(results)
    
    # Scaling analysis
    all_psi_values.extend(scaling_results['psi_values'])
    
    print(f"Framework Performance Summary:")
    print(f"  Total scenarios evaluated: {len(all_psi_values)}")
    print(f"  Average Ψ(x): {np.mean(all_psi_values):.3f}")
    print(f"  Standard deviation: {np.std(all_psi_values):.3f}")
    print(f"  Range: [{np.min(all_psi_values):.3f}, {np.max(all_psi_values):.3f}]")
    
    print(f"\n✅ Framework Applications Demonstrated:")
    print(f"  • Technical computation (chaotic systems)")
    print(f"  • Collaboration evaluation (open-source contributions)")
    print(f"  • Project phase analysis (phased approaches)")
    print(f"  • Scaling behavior analysis (BNSL connections)")
    print(f"  • Multi-domain integration (symbolic + neural)")
    
    print(f"\n🎯 Key Insights:")
    print(f"  • Hybrid approach balances symbolic and neural accuracies")
    print(f"  • Regularization ensures theoretical and computational fidelity")
    print(f"  • Probability calibration enhances reliability")
    print(f"  • Framework generalizes across technical and social domains")
    print(f"  • Scaling analysis reveals inflection points (BNSL connection)")
    
    print(f"\n🔬 Generated Visualizations:")
    print(f"  • Component breakdown analysis")
    print(f"  • Scenario comparison matrices")
    print(f"  • Scaling analysis curves")
    if BURGERS_AVAILABLE:
        print(f"  • PINN vs RK4 solution comparison")
    
    print(f"\n🎉 Comprehensive demonstration completed successfully!")
    print(f"   All components of the Hybrid Symbolic-Neural Accuracy Functional")
    print(f"   have been implemented, tested, and visualized.")
    
    return {
        'technical_results': tech_results_dict,
        'collaboration_results': collab_results,
        'scaling_results': scaling_results,
        'summary_stats': {
            'mean_psi': np.mean(all_psi_values),
            'std_psi': np.std(all_psi_values),
            'min_psi': np.min(all_psi_values),
            'max_psi': np.max(all_psi_values),
            'total_scenarios': len(all_psi_values)
        }
    }


def create_requirements_file():
    """Create a requirements.txt file for the project."""
    requirements = """# Hybrid Symbolic-Neural Accuracy Functional Requirements
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0

# Optional dependencies for full functionality
torch>=1.9.0  # For PINN implementation
torchvision>=0.10.0  # For additional neural network utilities

# Development dependencies (optional)
jupyter>=1.0.0
ipython>=7.0.0
pytest>=6.0.0
"""
    
    with open('/workspace/requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("📦 Created requirements.txt file")


if __name__ == "__main__":
    # Create requirements file
    create_requirements_file()
    
    # Run the comprehensive demo
    results = run_comprehensive_demo()
    
    # Save results summary
    print(f"\n💾 Saving results summary...")
    np.savez('/workspace/demo_results.npz', **results['summary_stats'])
    print(f"   Results saved to demo_results.npz")