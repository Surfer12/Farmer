#!/usr/bin/env python3
"""
Complete Fin Validation Example
Demonstrates the full invisible fin testing workflow
"""

import numpy as np
import json
from pathlib import Path
import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from invisible_fin_imu_analysis import (
    InvisibleFinAnalyzer, 
    IMUReading,
    MicroCorrectionDetector,
    MovementSmoothnessAnalyzer
)
from blind_abx_testing_protocol import BlindTestProtocol

def simulate_imu_session(variant_characteristics: Dict, 
                         duration_seconds: float = 1200.0,
                         sample_rate: float = 100.0) -> List[IMUReading]:
    """
    Simulate IMU data for a surf session with specific fin characteristics
    
    Args:
        variant_characteristics: Dict with 'stiffness', 'damping', 'predictability'
        duration_seconds: Session duration
        sample_rate: IMU sample rate
    
    Returns:
        List of IMU readings
    """
    n_samples = int(duration_seconds * sample_rate)
    timestamps = np.linspace(0, duration_seconds, n_samples)
    
    # Extract characteristics
    stiffness = variant_characteristics.get('stiffness', 18.0)  # Nm/rad
    damping = variant_characteristics.get('damping', 0.7)  # ratio
    predictability = variant_characteristics.get('predictability', 0.9)  # 0-1
    
    # Generate base motion (carving turns)
    base_frequency = 0.1  # Hz (one turn every 10 seconds)
    base_yaw = np.sin(2 * np.pi * base_frequency * timestamps)
    
    # Add micro-corrections based on predictability
    # Less predictable = more corrections
    correction_amplitude = (1.0 - predictability) * 0.5
    correction_frequency = 3.0  # Hz
    corrections = correction_amplitude * np.sin(2 * np.pi * correction_frequency * timestamps)
    
    # Add noise inversely proportional to damping
    noise_level = (1.0 - damping) * 0.1
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Combine signals
    yaw_rate = base_yaw + corrections + noise
    
    # Generate roll and pitch (coupled to yaw)
    roll_rate = 0.3 * yaw_rate + np.random.normal(0, 0.05, n_samples)
    pitch_rate = 0.1 * np.abs(yaw_rate) + np.random.normal(0, 0.02, n_samples)
    
    # Generate accelerations (simplified physics)
    forward_accel = 2.0 + 0.5 * np.sin(2 * np.pi * 0.05 * timestamps)
    lateral_accel = np.gradient(yaw_rate) * 5.0
    vertical_accel = 9.81 + 0.2 * np.sin(2 * np.pi * 0.2 * timestamps)
    
    # Create IMU readings
    readings = []
    for i in range(n_samples):
        readings.append(IMUReading(
            timestamp=timestamps[i],
            accel=np.array([forward_accel[i], lateral_accel[i], vertical_accel[i]]),
            gyro=np.array([roll_rate[i], pitch_rate[i], yaw_rate[i]]),
            quat=np.array([1, 0, 0, 0])  # Simplified quaternion
        ))
    
    return readings

def run_complete_validation():
    """
    Run a complete fin validation test
    """
    print("=" * 60)
    print("INVISIBLE FIN VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Step 1: Define fin variants
    print("\n1. DEFINING FIN VARIANTS")
    print("-" * 40)
    
    variants = {
        'A': {
            'id': 'A',
            'specs': {
                'name': 'High Stiffness',
                'torsional_stiffness': 24.0,
                'damping_ratio': 0.6,
                'predictability': 0.85
            }
        },
        'B': {
            'id': 'B',
            'specs': {
                'name': 'Optimal Balance',
                'torsional_stiffness': 18.0,
                'damping_ratio': 0.75,
                'predictability': 0.95
            }
        },
        'X': {
            'id': 'X',
            'specs': {
                'name': 'High Damping',
                'torsional_stiffness': 15.0,
                'damping_ratio': 0.85,
                'predictability': 0.90
            }
        }
    }
    
    for vid, vdata in variants.items():
        print(f"Variant {vid}: {vdata['specs']['name']}")
        print(f"  - Stiffness: {vdata['specs']['torsional_stiffness']} Nm/rad")
        print(f"  - Damping: {vdata['specs']['damping_ratio']}")
        print(f"  - Predictability: {vdata['specs']['predictability']}")
    
    # Step 2: Initialize blind protocol
    print("\n2. INITIALIZING BLIND PROTOCOL")
    print("-" * 40)
    
    test_name = f"fin_test_{datetime.date.today().isoformat()}"
    protocol = BlindTestProtocol(test_name, seed=42)
    protocol.register_variants(list(variants.values()))
    
    # Step 3: Generate test sequences for riders
    print("\n3. GENERATING TEST SEQUENCES")
    print("-" * 40)
    
    riders = ['rider_001', 'rider_002', 'rider_003']
    sequences = {}
    
    for rider in riders:
        sequence = protocol.generate_test_sequence(rider, n_sessions=9)
        sequences[rider] = sequence
        print(f"{rider}: {' -> '.join(sequence[:3])}... ({len(sequence)} total)")
    
    # Step 4: Simulate test sessions
    print("\n4. SIMULATING TEST SESSIONS")
    print("-" * 40)
    
    analyzer = InvisibleFinAnalyzer(sample_rate=100.0)
    all_results = {'A': [], 'B': [], 'X': []}
    
    session_count = 0
    for rider in riders:
        for session_idx, blind_code in enumerate(sequences[rider][:3]):  # First 3 sessions
            session_count += 1
            
            # Get true variant (cheating for simulation purposes)
            true_variant = protocol.variants[blind_code].true_id
            variant_specs = variants[true_variant]['specs']
            
            # Simulate IMU data
            imu_data = simulate_imu_session(
                {
                    'stiffness': variant_specs['torsional_stiffness'],
                    'damping': variant_specs['damping_ratio'],
                    'predictability': variant_specs['predictability']
                },
                duration_seconds=600.0  # 10 minute session
            )
            
            # Analyze session
            results = analyzer.process_session(imu_data)
            
            # Simulate subjective ratings based on characteristics
            invisibility = 10 * (1 - variant_specs['predictability']) + np.random.normal(0, 0.5)
            invisibility = np.clip(invisibility, 0, 10)
            
            effortlessness = 10 * variant_specs['predictability'] + np.random.normal(0, 0.5)
            effortlessness = np.clip(effortlessness, 0, 10)
            
            disruptions = int(5 * (1 - variant_specs['predictability']) + np.random.poisson(1))
            disruptions = max(0, disruptions)
            
            # Record session
            session_data = {
                'session_id': f"session_{session_count:03d}",
                'date': datetime.date.today().isoformat(),
                'time_start': f"{9 + session_idx}:00",
                'time_end': f"{9 + session_idx}:30",
                'rider_id': rider,
                'blind_code': blind_code,
                'conditions': {
                    'wave_height': '3-4ft',
                    'wind': 'Light offshore',
                    'tide': 'Mid'
                },
                'subjective_ratings': {
                    'invisibility': float(invisibility),
                    'effortlessness': float(effortlessness),
                    'disruption_count': disruptions
                },
                'objective_metrics': {
                    'micro_correction_rate': results['micro_correction_rate'],
                    'smoothness': results['smoothness_metrics']['dimensionless_jerk'],
                    'consistency': results['path_consistency']['radius_cv']
                },
                'notes': f"Simulated session for {rider}"
            }
            
            protocol.record_session(session_data)
            all_results[true_variant].append(results)
            
            print(f"Session {session_count}: {rider} on {blind_code} "
                  f"(MCR: {results['micro_correction_rate']:.1f}/min)")
    
    # Step 5: Analyze blinded results
    print("\n5. ANALYZING BLINDED RESULTS")
    print("-" * 40)
    
    blinded_results = protocol.analyze_results()
    
    print(f"Total sessions analyzed: {blinded_results['n_sessions']}")
    print(f"Riders tested: {blinded_results['n_riders']}")
    
    for variant_code, data in blinded_results['variants'].items():
        print(f"\nBlind Code {variant_code}:")
        if 'invisibility' in data['subjective']:
            print(f"  Invisibility: {data['subjective']['invisibility']['mean']:.2f} ± "
                  f"{data['subjective']['invisibility']['std']:.2f}")
        if 'micro_correction_rate' in data['objective']:
            print(f"  MCR: {data['objective']['micro_correction_rate']['mean']:.2f} ± "
                  f"{data['objective']['micro_correction_rate']['std']:.2f}")
    
    # Step 6: Unblind and get final results
    print("\n6. UNBLINDING RESULTS")
    print("-" * 40)
    
    protocol.unblind()
    final_results = protocol.analyze_results()
    
    print("\nUNBLINDED RESULTS BY VARIANT:")
    for variant, data in final_results['variants'].items():
        print(f"\nVariant {variant} ({variants[variant]['specs']['name']}):")
        
        if 'invisibility' in data['subjective']:
            inv = data['subjective']['invisibility']
            print(f"  Equipment Invisibility: {inv['mean']:.2f} ± {inv['std']:.2f}")
        
        if 'effortlessness' in data['subjective']:
            eff = data['subjective']['effortlessness']
            print(f"  Effortlessness: {eff['mean']:.2f} ± {eff['std']:.2f}")
        
        if 'disruption_count' in data['subjective']:
            disr = data['subjective']['disruption_count']
            print(f"  Disruptions: {disr['mean']:.1f} ± {disr['std']:.1f}")
        
        if 'micro_correction_rate' in data['objective']:
            mcr = data['objective']['micro_correction_rate']
            print(f"  Micro-Correction Rate: {mcr['mean']:.2f} ± {mcr['std']:.2f} /min")
    
    # Step 7: Selection recommendation
    print("\n7. OPTIMAL VARIANT SELECTION")
    print("-" * 40)
    
    if 'recommendation' in final_results:
        rec = final_results['recommendation']
        selected = rec['selected_variant']
        print(f"\nRECOMMENDED: Variant {selected} ({variants[selected]['specs']['name']})")
        print(f"Score: {rec['score']:.3f}")
        print(f"Reasoning: {rec['reasoning']}")
        
        print("\nAll Scores:")
        for variant, score in rec['all_scores'].items():
            print(f"  Variant {variant}: {score:.3f}")
    
    # Step 8: Generate visualizations
    print("\n8. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    generate_validation_plots(final_results, variants)
    
    # Step 9: Save final report
    print("\n9. SAVING FINAL REPORT")
    print("-" * 40)
    
    report = protocol.generate_report()
    report_path = Path(f"data/blind_tests/{test_name}/final_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"- Optimal variant: {rec['selected_variant']}")
    print(f"- Best invisibility score: {final_results['variants'][rec['selected_variant']]['subjective']['invisibility']['mean']:.2f}")
    print(f"- Lowest MCR: {min(v['objective']['micro_correction_rate']['mean'] for v in final_results['variants'].values() if 'micro_correction_rate' in v['objective']):.2f} /min")
    print("\nThe selected fin achieves the goal of 'equipment that disappears'")

def generate_validation_plots(results: Dict, variants: Dict):
    """Generate visualization plots for validation results"""
    
    # Set up the plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data for plotting
    variant_names = []
    invisibility_means = []
    invisibility_stds = []
    mcr_means = []
    mcr_stds = []
    effortlessness_means = []
    disruption_means = []
    
    for variant, data in results['variants'].items():
        variant_names.append(f"{variant}\n{variants[variant]['specs']['name']}")
        
        if 'invisibility' in data['subjective']:
            invisibility_means.append(data['subjective']['invisibility']['mean'])
            invisibility_stds.append(data['subjective']['invisibility']['std'])
        else:
            invisibility_means.append(0)
            invisibility_stds.append(0)
        
        if 'micro_correction_rate' in data['objective']:
            mcr_means.append(data['objective']['micro_correction_rate']['mean'])
            mcr_stds.append(data['objective']['micro_correction_rate']['std'])
        else:
            mcr_means.append(0)
            mcr_stds.append(0)
        
        if 'effortlessness' in data['subjective']:
            effortlessness_means.append(data['subjective']['effortlessness']['mean'])
        else:
            effortlessness_means.append(0)
        
        if 'disruption_count' in data['subjective']:
            disruption_means.append(data['subjective']['disruption_count']['mean'])
        else:
            disruption_means.append(0)
    
    # Plot 1: Equipment Invisibility
    ax1 = axes[0, 0]
    x_pos = np.arange(len(variant_names))
    ax1.bar(x_pos, invisibility_means, yerr=invisibility_stds, 
            capsize=5, color=['coral', 'lightgreen', 'skyblue'])
    ax1.set_xlabel('Variant')
    ax1.set_ylabel('Equipment Invisibility (0-10)')
    ax1.set_title('Equipment Invisibility by Variant\n(Lower is Better)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(variant_names)
    ax1.axhline(y=3, color='g', linestyle='--', alpha=0.5, label='Target < 3')
    ax1.legend()
    
    # Plot 2: Micro-Correction Rate
    ax2 = axes[0, 1]
    ax2.bar(x_pos, mcr_means, yerr=mcr_stds, 
            capsize=5, color=['coral', 'lightgreen', 'skyblue'])
    ax2.set_xlabel('Variant')
    ax2.set_ylabel('Micro-Corrections per Minute')
    ax2.set_title('Micro-Correction Rate\n(Lower is Better)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(variant_names)
    
    # Plot 3: Effortlessness vs Disruptions
    ax3 = axes[1, 0]
    ax3.scatter(effortlessness_means, disruption_means, s=200, alpha=0.6)
    for i, txt in enumerate(['A', 'B', 'X']):
        ax3.annotate(txt, (effortlessness_means[i], disruption_means[i]),
                    ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Effortlessness (0-10)')
    ax3.set_ylabel('Disruption Count')
    ax3.set_title('Effortlessness vs Attention Disruptions')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, max(disruption_means) + 1)
    
    # Plot 4: Combined Score
    ax4 = axes[1, 1]
    if 'recommendation' in results and 'all_scores' in results['recommendation']:
        scores = results['recommendation']['all_scores']
        score_values = [scores.get(v, 0) for v in ['A', 'B', 'X']]
        colors = ['gold' if v == results['recommendation']['selected_variant'] 
                 else 'lightgray' for v in ['A', 'B', 'X']]
        ax4.bar(x_pos, score_values, color=colors)
        ax4.set_xlabel('Variant')
        ax4.set_ylabel('Combined Invisibility Score')
        ax4.set_title('Overall Invisibility Score\n(Higher is Better)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(variant_names)
        
        # Mark the winner
        winner_idx = ['A', 'B', 'X'].index(results['recommendation']['selected_variant'])
        ax4.scatter(winner_idx, score_values[winner_idx] + 0.02, 
                   s=200, marker='*', color='red', zorder=5)
    
    plt.suptitle('Invisible Fin Validation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path("data/validation_results.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    
    # Don't show in automated mode
    # plt.show()

if __name__ == "__main__":
    run_complete_validation()