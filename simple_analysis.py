#!/usr/bin/env python3
"""
Simple Analysis and Visualization for Hybrid Symbolic-Neural Accuracy Functional
Creates ASCII-based plots and data tables without external dependencies
"""

import math
from minimal_hybrid_functional import MinimalHybridFunctional

def ascii_plot(x_values, y_values, title="Plot", width=60, height=20):
    """Create ASCII plot of data"""
    if not x_values or not y_values:
        return "No data to plot"
    
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Avoid division by zero
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    plot_lines = []
    
    # Title
    plot_lines.append(f" {title}")
    plot_lines.append(" " + "="*len(title))
    plot_lines.append("")
    
    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for x, y in zip(x_values, y_values):
        col = int((x - x_min) / x_range * (width - 1))
        row = int((y_max - y) / y_range * (height - 1))  # Flip y-axis
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = '*'
    
    # Add axes
    for row in range(height):
        for col in range(width):
            if col == 0:  # y-axis
                grid[row][col] = '|' if grid[row][col] == ' ' else grid[row][col]
            if row == height - 1:  # x-axis
                grid[row][col] = '-' if grid[row][col] == ' ' else grid[row][col]
    
    # Convert grid to strings
    for row in grid:
        plot_lines.append(''.join(row))
    
    # Add labels
    plot_lines.append(f" {x_min:.2f}" + " " * (width - 10) + f"{x_max:.2f}")
    plot_lines.append(f" Y: {y_min:.3f} to {y_max:.3f}")
    
    return '\n'.join(plot_lines)

def parameter_sensitivity_analysis():
    """Analyze sensitivity to different parameters"""
    print("=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    functional = MinimalHybridFunctional()
    base_result = functional.compute_psi_single(0.5, 1.0)
    base_psi = base_result['psi']
    
    print(f"Base case (x=0.5, t=1.0): Ψ(x) = {base_psi:.4f}")
    print()
    
    # Test different lambda1 values
    print("λ₁ Sensitivity (Cognitive penalty weight):")
    print("λ₁\tΨ(x)\tΔΨ\t% Change")
    print("-" * 40)
    
    lambda1_values = [0.25, 0.5, 0.75, 1.0, 1.25]
    for l1 in lambda1_values:
        func = MinimalHybridFunctional(lambda1=l1, lambda2=0.25, beta=1.2)
        result = func.compute_psi_single(0.5, 1.0)
        delta = result['psi'] - base_psi
        percent = (delta / base_psi) * 100
        print(f"{l1:.2f}\t{result['psi']:.4f}\t{delta:+.4f}\t{percent:+.1f}%")
    
    print()
    
    # Test different beta values
    print("β Sensitivity (Responsiveness bias):")
    print("β\tΨ(x)\tΔΨ\t% Change")
    print("-" * 40)
    
    beta_values = [0.8, 1.0, 1.2, 1.5, 2.0]
    for b in beta_values:
        func = MinimalHybridFunctional(lambda1=0.75, lambda2=0.25, beta=b)
        result = func.compute_psi_single(0.5, 1.0)
        delta = result['psi'] - base_psi
        percent = (delta / base_psi) * 100
        print(f"{b:.1f}\t{result['psi']:.4f}\t{delta:+.4f}\t{percent:+.1f}%")

def temporal_evolution_analysis():
    """Analyze how the functional evolves over time"""
    print("\n" + "=" * 70)
    print("TEMPORAL EVOLUTION ANALYSIS")
    print("=" * 70)
    
    functional = MinimalHybridFunctional()
    
    # Fixed spatial points
    x_points = [-0.8, -0.4, 0.0, 0.4, 0.8]
    t_values = [i * 0.2 for i in range(11)]  # 0.0 to 2.0
    
    print("Ψ(x,t) evolution over time:")
    print("t\\x", end="")
    for x in x_points:
        print(f"\tx={x:.1f}", end="")
    print()
    print("-" * 60)
    
    for t in t_values:
        print(f"{t:.1f}", end="")
        for x in x_points:
            result = functional.compute_psi_single(x, t)
            print(f"\t{result['psi']:.3f}", end="")
        print()
    
    # Create ASCII plot for center point
    print(f"\nTemporal evolution at x=0.0:")
    center_psi = [functional.compute_psi_single(0.0, t)['psi'] for t in t_values]
    print(ascii_plot(t_values, center_psi, "Ψ(x=0,t) vs Time", width=50, height=15))

def spatial_distribution_analysis():
    """Analyze spatial distribution at fixed times"""
    print("\n" + "=" * 70)
    print("SPATIAL DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    functional = MinimalHybridFunctional()
    
    x_values = [i * 0.2 - 1.0 for i in range(11)]  # -1.0 to 1.0
    time_points = [0.5, 1.0, 1.5, 2.0]
    
    print("Ψ(x,t) spatial distribution:")
    print("x\\t", end="")
    for t in time_points:
        print(f"\tt={t:.1f}", end="")
    print()
    print("-" * 50)
    
    for x in x_values:
        print(f"{x:.1f}", end="")
        for t in time_points:
            result = functional.compute_psi_single(x, t)
            print(f"\t{result['psi']:.3f}", end="")
        print()
    
    # Create ASCII plot for t=1.0
    print(f"\nSpatial distribution at t=1.0:")
    psi_spatial = [functional.compute_psi_single(x, 1.0)['psi'] for x in x_values]
    print(ascii_plot(x_values, psi_spatial, "Ψ(x,t=1) vs Position", width=50, height=15))

def component_analysis():
    """Detailed analysis of functional components"""
    print("\n" + "=" * 70)
    print("COMPONENT ANALYSIS")
    print("=" * 70)
    
    functional = MinimalHybridFunctional()
    
    # Test scenarios
    scenarios = [
        ("Early, low chaos", 0.2, 0.3),
        ("Early, high chaos", 0.8, 0.3),
        ("Late, low chaos", 0.2, 1.8),
        ("Late, high chaos", 0.8, 1.8),
        ("Balanced", 0.5, 1.0)
    ]
    
    print("Detailed component breakdown:")
    print("Scenario\t\tS(x,t)\tN(x,t)\tα(t)\tHybrid\tR_cog\tR_eff\tReg\tProb\tΨ(x)")
    print("-" * 100)
    
    for name, x, t in scenarios:
        result = functional.compute_psi_single(x, t)
        print(f"{name:<15}\t{result['S']:.3f}\t{result['N']:.3f}\t{result['alpha']:.3f}\t"
              f"{result['hybrid']:.3f}\t{result['R_cog']:.3f}\t{result['R_eff']:.3f}\t"
              f"{result['reg_term']:.3f}\t{result['prob_term']:.3f}\t{result['psi']:.3f}")
    
    print("\nComponent contributions at x=0.5, t=1.0:")
    result = functional.compute_psi_single(0.5, 1.0)
    
    components = [
        ("Symbolic accuracy", result['S']),
        ("Neural accuracy", result['N']),
        ("Adaptive weight", result['alpha']),
        ("Hybrid output", result['hybrid']),
        ("Cognitive penalty", result['R_cog']),
        ("Efficiency penalty", result['R_eff']),
        ("Regularization", result['reg_term']),
        ("Probability", result['prob_term']),
        ("Final Ψ(x)", result['psi'])
    ]
    
    max_name_len = max(len(name) for name, _ in components)
    for name, value in components:
        bar_length = int(value * 50) if value <= 1.0 else 50
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{name:<{max_name_len}} {value:.4f} |{bar}|")

def optimization_landscape():
    """Analyze the optimization landscape"""
    print("\n" + "=" * 70)
    print("OPTIMIZATION LANDSCAPE")
    print("=" * 70)
    
    # Find optimal parameters for different scenarios
    scenarios = [
        ("High accuracy", lambda r: r['S'] + r['N']),
        ("Low penalty", lambda r: -r['R_cog'] - r['R_eff']),
        ("Balanced hybrid", lambda r: min(r['S'], r['N'])),
        ("High probability", lambda r: r['prob_term'])
    ]
    
    x_test, t_test = 0.5, 1.0
    
    print("Parameter optimization for different objectives:")
    print("Objective\t\tλ₁\tλ₂\tβ\tΨ(x)\tScore")
    print("-" * 60)
    
    for obj_name, objective_func in scenarios:
        best_psi = 0
        best_params = None
        best_score = float('-inf')
        
        # Grid search over parameter space
        for l1 in [0.25, 0.5, 0.75, 1.0]:
            for l2 in [0.1, 0.25, 0.4, 0.5]:
                for beta in [0.8, 1.0, 1.2, 1.5]:
                    func = MinimalHybridFunctional(lambda1=l1, lambda2=l2, beta=beta)
                    result = func.compute_psi_single(x_test, t_test)
                    score = objective_func(result)
                    
                    if score > best_score:
                        best_score = score
                        best_psi = result['psi']
                        best_params = (l1, l2, beta)
        
        l1, l2, beta = best_params
        print(f"{obj_name:<15}\t{l1:.2f}\t{l2:.2f}\t{beta:.1f}\t{best_psi:.4f}\t{best_score:.3f}")

def collaboration_scenario_analysis():
    """Analyze different collaboration scenarios in detail"""
    print("\n" + "=" * 70)
    print("COLLABORATION SCENARIO ANALYSIS")
    print("=" * 70)
    
    scenarios = {
        "Academic Research": {
            "S": 0.85, "N": 0.75, "alpha": 0.6,
            "R_cog": 0.12, "R_eff": 0.08,
            "lambda1": 0.8, "lambda2": 0.2,
            "P_base": 0.82, "beta": 1.1
        },
        "Industry Application": {
            "S": 0.70, "N": 0.90, "alpha": 0.3,
            "R_cog": 0.18, "R_eff": 0.15,
            "lambda1": 0.4, "lambda2": 0.6,
            "P_base": 0.75, "beta": 1.4
        },
        "Open Source": {
            "S": 0.74, "N": 0.84, "alpha": 0.5,
            "R_cog": 0.14, "R_eff": 0.09,
            "lambda1": 0.55, "lambda2": 0.45,
            "P_base": 0.77, "beta": 1.3
        },
        "Healthcare": {
            "S": 0.95, "N": 0.80, "alpha": 0.7,
            "R_cog": 0.05, "R_eff": 0.20,
            "lambda1": 0.9, "lambda2": 0.1,
            "P_base": 0.90, "beta": 0.9
        }
    }
    
    print("Scenario\t\tS\tN\tα\tR_cog\tR_eff\tλ₁\tλ₂\tP\tβ\tΨ(x)")
    print("-" * 80)
    
    for name, params in scenarios.items():
        # Calculate Ψ(x) manually
        O_hybrid = params["alpha"] * params["S"] + (1 - params["alpha"]) * params["N"]
        P_total = params["lambda1"] * params["R_cog"] + params["lambda2"] * params["R_eff"]
        reg_term = math.exp(-P_total)
        
        logit_p = math.log(params["P_base"] / (1 - params["P_base"]))
        P_adj = 1 / (1 + math.exp(-(logit_p + math.log(params["beta"]))))
        P_adj = min(1.0, P_adj)
        
        psi = O_hybrid * reg_term * P_adj
        
        print(f"{name:<15}\t{params['S']:.2f}\t{params['N']:.2f}\t{params['alpha']:.1f}\t"
              f"{params['R_cog']:.2f}\t{params['R_eff']:.2f}\t{params['lambda1']:.1f}\t"
              f"{params['lambda2']:.1f}\t{params['P_base']:.2f}\t{params['beta']:.1f}\t{psi:.3f}")
    
    # Ranking
    print("\nCollaboration Potential Ranking:")
    results = []
    for name, params in scenarios.items():
        O_hybrid = params["alpha"] * params["S"] + (1 - params["alpha"]) * params["N"]
        P_total = params["lambda1"] * params["R_cog"] + params["lambda2"] * params["R_eff"]
        reg_term = math.exp(-P_total)
        logit_p = math.log(params["P_base"] / (1 - params["P_base"]))
        P_adj = 1 / (1 + math.exp(-(logit_p + math.log(params["beta"]))))
        P_adj = min(1.0, P_adj)
        psi = O_hybrid * reg_term * P_adj
        results.append((name, psi))
    
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (name, psi) in enumerate(results, 1):
        potential = "Excellent" if psi > 0.7 else "Good" if psi > 0.6 else "Moderate" if psi > 0.5 else "Limited"
        print(f"{i}. {name:<15} Ψ(x)={psi:.3f} ({potential})")

def main():
    """Run complete analysis suite"""
    print("HYBRID SYMBOLIC-NEURAL ACCURACY FUNCTIONAL")
    print("Comprehensive Analysis Suite")
    print("=" * 70)
    
    # Run all analyses
    parameter_sensitivity_analysis()
    temporal_evolution_analysis()
    spatial_distribution_analysis()
    component_analysis()
    optimization_landscape()
    collaboration_scenario_analysis()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("Key Insights:")
    print("• Functional shows strong sensitivity to λ₁ (cognitive penalty weight)")
    print("• Neural methods favored early, symbolic methods favored late")
    print("• Healthcare scenarios show highest collaboration potential")
    print("• Balanced parameters generally provide robust performance")
    print("• Regularization effectively prevents overfitting to penalties")

if __name__ == "__main__":
    main()