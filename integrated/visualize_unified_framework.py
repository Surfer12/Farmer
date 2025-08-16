#!/usr/bin/env python3
"""
Comprehensive Visualization for Unified Theoretical Framework
Creates detailed multi-panel visualization showing all integrated components
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Any
from unified_theoretical_framework import (
    UnifiedTheoreticalFramework,
    demonstrate_unified_framework,
)


def create_unified_framework_visualization():
    """
    Create comprehensive 12-panel visualization of unified theoretical framework
    """
    # Run the framework to get fresh data
    print("Generating unified framework data for visualization...")
    results, states, analysis = demonstrate_unified_framework()

    # Extract time series data
    timestamps = [s.timestamp for s in states]

    # Hierarchical Bayesian data
    hb_psi_scores = [s.psi_probability for s in states]
    hb_penalties = [s.penalty_multiplicative for s in states]
    hb_eta_values = [s.eta_linear for s in states]

    # Swarm-Koopman data
    swarm_confidences = [s.swarm_confidence for s in states]
    error_bounds = [s.error_bound for s in states]
    koopman_energy = [np.sum(s.koopman_observables[:3] ** 2) for s in states]

    # LSTM data
    lstm_hidden_norms = [np.linalg.norm(s.lstm_hidden) for s in states]
    lstm_confidences = [s.lstm_confidence for s in states]
    lstm_errors = [s.lstm_error for s in states]

    # Contemplative data
    contemplative_scores = [s.contemplative_score for s in states]
    impermanence_levels = [s.impermanence_level for s in states]
    cognitive_distances = [0.0] + [s.cognitive_distance for s in states[1:]]

    # Create the comprehensive visualization
    fig = plt.figure(figsize=(20, 24))

    # Main title with framework summary
    hb_satisfied = analysis["hierarchical_bayesian"]["bounds_satisfied"]
    sk_satisfied = analysis["swarm_koopman"]["theorem_satisfied"]
    lstm_satisfied = analysis["lstm_convergence"]["theorem_satisfied"]
    overall_confidence = analysis["unified_performance"]["overall_confidence"]
    insight_quality = analysis["contemplative_integration"]["insight_quality"]

    quality_labels = {
        "primitive_direct": "Primitive Direct (Ψ > 0.85)",
        "empirically_grounded": "Empirically Grounded (Ψ > 0.70)",
        "interpretive_contextual": "Interpretive/Contextual (Ψ ≤ 0.70)",
    }
    quality_desc = quality_labels.get(insight_quality, insight_quality)

    fig.suptitle(
        "Unified Theoretical Framework Integration\n"
        + "HB Model + Swarm-Koopman + LSTM Convergence + Contemplative AI",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Subtitle with key metrics
    subtitle = (
        f"Overall Confidence: {overall_confidence:.3f} | "
        + f"Insight Quality: {quality_desc} | "
        + f"Theorems Satisfied: HB={hb_satisfied}, SK={sk_satisfied}, LSTM={lstm_satisfied}"
    )
    fig.text(0.5, 0.95, subtitle, ha="center", fontsize=12, style="italic")

    # Panel 1: Hierarchical Bayesian Ψ(x) Evolution
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(timestamps, hb_psi_scores, "b-", linewidth=2, label="Ψ(x) Probability")
    ax1.plot(timestamps, hb_penalties, "r--", linewidth=1.5, label="π(x) Penalty")
    ax1.axhline(
        y=0.85, color="g", linestyle=":", alpha=0.7, label="Primitive Threshold"
    )
    ax1.axhline(
        y=0.70, color="orange", linestyle=":", alpha=0.7, label="Empirical Threshold"
    )
    ax1.set_title(
        "Hierarchical Bayesian Model\nΨ(x) = sigmoid(η) × π(x)", fontweight="bold"
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Probability")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Swarm-Koopman Confidence vs Error
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(
        timestamps, swarm_confidences, "purple", linewidth=2, label="C(p) Confidence"
    )
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        timestamps, error_bounds, "red", linewidth=1.5, alpha=0.7, label="Error Bound"
    )

    # Add theorem validation line
    expected_conf = np.mean(swarm_confidences)
    expected_err = np.mean(error_bounds)
    theorem_line = 1.0 - expected_err
    ax2.axhline(
        y=theorem_line,
        color="green",
        linestyle="--",
        alpha=0.8,
        label=f"Theorem: 1-e = {theorem_line:.3f}",
    )

    ax2.set_title(
        "Swarm-Koopman Confidence Theorem\nE[C(p)] ≥ 1-e Validation", fontweight="bold"
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Confidence C(p)", color="purple")
    ax2_twin.set_ylabel("Error Bound", color="red")
    ax2.legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: LSTM Hidden State Convergence
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(
        timestamps, lstm_hidden_norms, "teal", linewidth=2, label="||h_t|| Hidden Norm"
    )
    ax3.plot(
        timestamps,
        [lstm_errors[0]] * len(timestamps),
        "orange",
        linestyle="--",
        linewidth=1.5,
        label=f"O(1/√T) = {lstm_errors[0]:.3f}",
    )
    ax3.set_title(
        "LSTM Hidden State Convergence\nh_t = o_t ⊙ tanh(c_t)", fontweight="bold"
    )
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Hidden State Norm")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Contemplative Score Evolution
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(
        timestamps,
        contemplative_scores,
        "darkgreen",
        linewidth=3,
        label="Ψ Contemplative Score",
    )
    ax4.fill_between(timestamps, contemplative_scores, alpha=0.3, color="green")
    ax4.axhline(
        y=0.85, color="gold", linestyle=":", linewidth=2, label="Primitive Direct"
    )
    ax4.axhline(
        y=0.70, color="orange", linestyle=":", linewidth=2, label="Empirically Grounded"
    )
    ax4.set_title(
        "Contemplative Integration\nStage-Four Insight (Udayabbaya Ñāṇa)",
        fontweight="bold",
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Contemplative Ψ Score")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Impermanence (Anicca) Detection
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(
        timestamps,
        impermanence_levels,
        "maroon",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Impermanence Level",
    )
    ax5.fill_between(timestamps, impermanence_levels, alpha=0.4, color="maroon")
    ax5.set_title(
        "Impermanence Detection\nAnicca Awareness from Error Dynamics",
        fontweight="bold",
    )
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Impermanence Level")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Cognitive-Memory Distance Metric
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(
        timestamps, cognitive_distances, "navy", linewidth=2, label="d_MC Distance"
    )
    ax6.fill_between(timestamps, cognitive_distances, alpha=0.2, color="navy")
    mean_dist = np.mean(cognitive_distances)
    ax6.axhline(
        y=mean_dist,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Mean: {mean_dist:.4f}",
    )
    ax6.set_title(
        "Cognitive-Memory Metric\nd_MC in Weighted Minkowski Space", fontweight="bold"
    )
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Cognitive Distance")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Panel 7: Framework Integration Heatmap
    ax7 = plt.subplot(4, 3, 7)

    # Create correlation matrix of all framework components
    data_matrix = np.array(
        [
            hb_psi_scores,
            swarm_confidences,
            lstm_confidences,
            contemplative_scores,
            impermanence_levels,
        ]
    )

    correlation_matrix = np.corrcoef(data_matrix)
    labels = ["HB Ψ(x)", "Swarm C(p)", "LSTM Conf", "Contemp Ψ", "Impermanence"]

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="RdYlBu_r",
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax7,
        cbar_kws={"shrink": 0.8},
    )
    ax7.set_title(
        "Framework Component Correlations\nIntegration Coherence Analysis",
        fontweight="bold",
    )

    # Panel 8: Koopman Observable Energy
    ax8 = plt.subplot(4, 3, 8)
    ax8.plot(
        timestamps, koopman_energy, "darkviolet", linewidth=2, label="||g(x)||² Energy"
    )
    ax8.fill_between(timestamps, koopman_energy, alpha=0.3, color="darkviolet")
    ax8.set_title(
        "Koopman Observable Energy\nDynamical System State Energy", fontweight="bold"
    )
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Observable Energy")
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # Panel 9: Theorem Validation Summary
    ax9 = plt.subplot(4, 3, 9)

    theorem_names = [
        "HB Bounds\n[0,1]",
        "Swarm-Koopman\nE[C(p)]≥1-e",
        "LSTM Conv\nO(1/√T)",
    ]
    theorem_satisfied = [hb_satisfied, sk_satisfied, lstm_satisfied]
    colors = ["green" if sat else "red" for sat in theorem_satisfied]

    bars = ax9.bar(
        theorem_names,
        [1 if sat else 0 for sat in theorem_satisfied],
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    # Add satisfaction percentages
    hb_score = analysis["hierarchical_bayesian"]["psi_stability"]
    sk_score = analysis["swarm_koopman"]["confidence_error_ratio"]
    lstm_score = analysis["lstm_convergence"]["confidence"]
    scores = [hb_score, sk_score, lstm_score]

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax9.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax9.set_title("Theorem Validation Status\nSatisfaction Scores", fontweight="bold")
    ax9.set_ylabel("Satisfied")
    ax9.set_ylim(0, 1.2)
    ax9.grid(True, alpha=0.3, axis="y")

    # Panel 10: Unified Confidence Timeline
    ax10 = plt.subplot(4, 3, 10)

    # Combine all confidence measures
    unified_confidence = [
        (hb + sk + lstm + cont) / 4
        for hb, sk, lstm, cont in zip(
            hb_psi_scores, swarm_confidences, lstm_confidences, contemplative_scores
        )
    ]

    ax10.plot(
        timestamps, unified_confidence, "black", linewidth=3, label="Unified Confidence"
    )
    ax10.plot(timestamps, hb_psi_scores, "blue", alpha=0.6, label="HB")
    ax10.plot(timestamps, swarm_confidences, "purple", alpha=0.6, label="Swarm")
    ax10.plot(timestamps, lstm_confidences, "teal", alpha=0.6, label="LSTM")
    ax10.plot(timestamps, contemplative_scores, "green", alpha=0.6, label="Contemp")

    ax10.axhline(
        y=overall_confidence,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Mean: {overall_confidence:.3f}",
    )

    ax10.set_title(
        "Unified Confidence Evolution\nIntegrated Framework Performance",
        fontweight="bold",
    )
    ax10.set_xlabel("Time")
    ax10.set_ylabel("Confidence")
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)

    # Panel 11: Error Analysis
    ax11 = plt.subplot(4, 3, 11)

    # Normalize errors for comparison
    normalized_errors = np.array(
        [
            error_bounds / np.max(error_bounds),
            [lstm_errors[0]] * len(timestamps),
            (1.0 - np.array(hb_psi_scores)),  # HB "error" as distance from certainty
        ]
    )

    ax11.plot(
        timestamps,
        normalized_errors[0],
        "red",
        linewidth=2,
        label="Swarm-Koopman Error",
    )
    ax11.plot(
        timestamps,
        normalized_errors[1],
        "orange",
        linewidth=2,
        label="LSTM Error O(1/√T)",
    )
    ax11.plot(
        timestamps, normalized_errors[2], "blue", linewidth=2, label="HB Uncertainty"
    )

    ax11.set_title(
        "Normalized Error Analysis\nComparative Framework Errors", fontweight="bold"
    )
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Normalized Error")
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)

    # Panel 12: Framework Performance Summary
    ax12 = plt.subplot(4, 3, 12)

    # Performance metrics
    metrics = [
        "Overall\nConfidence",
        "Framework\nCoherence",
        "Mean Cognitive\nDistance",
        "Final Contemp\nScore",
    ]
    values = [
        analysis["unified_performance"]["overall_confidence"],
        analysis["unified_performance"]["framework_coherence"],
        analysis["cognitive_memory"]["mean_distance"] * 100,  # Scale for visibility
        analysis["contemplative_integration"]["final_score"],
    ]

    bars = ax12.bar(
        metrics,
        values,
        color=["gold", "lightblue", "lightcoral", "lightgreen"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax12.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax12.set_title(
        "Unified Framework Summary\nKey Performance Indicators", fontweight="bold"
    )
    ax12.set_ylabel("Score/Value")
    ax12.grid(True, alpha=0.3, axis="y")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    # Add framework description
    framework_desc = (
        "Integrated Framework Components:\n"
        "• Hierarchical Bayesian: Multiplicative penalties preserve [0,1] bounds naturally\n"
        "• Swarm-Koopman: Confidence bounds E[C(p)] ≥ 1-ε with O(h⁴) + O(1/N) error\n"
        "• LSTM Convergence: Hidden states h_t = o_t ⊙ tanh(c_t) with O(1/√T) error bounds\n"
        "• Contemplative AI: Stage-four insight (udayabbaya ñāṇa) with impermanence detection"
    )

    fig.text(
        0.02,
        0.01,
        framework_desc,
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Save visualization
    output_path = "outputs/unified_theoretical_framework_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nComprehensive visualization saved to: {output_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    print("Creating Unified Theoretical Framework Visualization...")
    print("=" * 60)
    fig = create_unified_framework_visualization()
    print("\nVisualization complete! 12-panel comprehensive analysis generated.")
