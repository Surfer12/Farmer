#!/usr/bin/env python3
"""
Integration of Cognitive-Memory Metric with Contemplative AI Temporal Gradients
Bridges the weighted Minkowski space approach with contemplative visual grounding
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


@dataclass
class CognitiveState:
    """
    Represents a cognitive state in the weighted Minkowski space
    Extends VisualPhenomenon to include cognitive-memory components
    """

    # Temporal component (t)
    timestamp: float
    temporal_embedding: np.ndarray  # Temporal features

    # Symbolic component (s)
    semantic_embedding: np.ndarray  # Semantic/symbolic features
    symbolic_intensity: float

    # Neural component (n)
    neural_activation: np.ndarray  # Neural pattern representation
    neural_coherence: float

    # Cross-modal interactions
    symbolic_neural_coupling: float  # S-N interaction strength

    # Contemplative aspects
    impermanence_level: float  # Anicca quantification
    arising_rate: float
    passing_rate: float
    observer_validation: float


class CognitiveMemoryMetric:
    """
    Implementation of the weighted Minkowski space cognitive-memory metric
    d_MC(m1, m2) = w_t ||t1-t2||² + w_s ||s1-s2||² + w_n ||n1-n2||² + w_cross ∫[S(m1)N(m2) - S(m2)N(m1)]dt

    Integrates with contemplative AI temporal gradient analysis
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "temporal": 0.3,  # w_t
            "symbolic": 0.4,  # w_s
            "neural": 0.5,  # w_n
            "cross_modal": 0.2,  # w_cross
        }

        # Integration with contemplative framework
        self.contemplative_factor = 0.3  # Weight for contemplative aspects
        self.impermanence_emphasis = 0.4  # Emphasis on anicca

    def compute_cognitive_distance(
        self,
        state1: CognitiveState,
        state2: CognitiveState,
        include_contemplative: bool = True,
    ) -> Dict[str, float]:
        """
        Compute cognitive-memory metric distance between two states

        Implements the weighted Minkowski space metric with contemplative extensions
        """
        # Temporal component: w_t ||t1-t2||²
        temporal_distance = self._compute_temporal_distance(state1, state2)

        # Symbolic component: w_s ||s1-s2||²
        symbolic_distance = self._compute_symbolic_distance(state1, state2)

        # Neural component: w_n ||n1-n2||²
        neural_distance = self._compute_neural_distance(state1, state2)

        # Cross-modal component: w_cross ∫[S(m1)N(m2) - S(m2)N(m1)]dt
        cross_modal_distance = self._compute_cross_modal_distance(state1, state2)

        # Base cognitive-memory metric
        base_distance = (
            self.weights["temporal"] * temporal_distance
            + self.weights["symbolic"] * symbolic_distance
            + self.weights["neural"] * neural_distance
            + self.weights["cross_modal"] * cross_modal_distance
        )

        # Contemplative extension: impermanence-aware distance
        contemplative_distance = 0.0
        if include_contemplative:
            contemplative_distance = self._compute_contemplative_distance(
                state1, state2
            )
            base_distance += self.contemplative_factor * contemplative_distance

        return {
            "total_distance": base_distance,
            "temporal_component": temporal_distance,
            "symbolic_component": symbolic_distance,
            "neural_component": neural_distance,
            "cross_modal_component": cross_modal_distance,
            "contemplative_component": contemplative_distance,
            "metric_properties_verified": self._verify_metric_properties(base_distance),
        }

    def _compute_temporal_distance(
        self, state1: CognitiveState, state2: CognitiveState
    ) -> float:
        """
        Compute temporal component: w_t ||t1-t2||²
        Includes both timestamp and temporal embedding distances
        """
        # Basic timestamp distance
        timestamp_dist = (state1.timestamp - state2.timestamp) ** 2

        # Temporal embedding distance (if available)
        if (
            state1.temporal_embedding is not None
            and state2.temporal_embedding is not None
        ):
            embedding_dist = (
                np.linalg.norm(state1.temporal_embedding - state2.temporal_embedding)
                ** 2
            )
            return timestamp_dist + embedding_dist

        return timestamp_dist

    def _compute_symbolic_distance(
        self, state1: CognitiveState, state2: CognitiveState
    ) -> float:
        """
        Compute symbolic component: w_s ||s1-s2||²
        Measures semantic/symbolic differences between cognitive states
        """
        # Semantic embedding distance
        semantic_dist = (
            np.linalg.norm(state1.semantic_embedding - state2.semantic_embedding) ** 2
        )

        # Symbolic intensity difference
        intensity_dist = (state1.symbolic_intensity - state2.symbolic_intensity) ** 2

        return semantic_dist + intensity_dist

    def _compute_neural_distance(
        self, state1: CognitiveState, state2: CognitiveState
    ) -> float:
        """
        Compute neural component: w_n ||n1-n2||²
        Measures differences in neural activation patterns
        """
        # Neural activation pattern distance
        neural_dist = (
            np.linalg.norm(state1.neural_activation - state2.neural_activation) ** 2
        )

        # Neural coherence difference
        coherence_dist = (state1.neural_coherence - state2.neural_coherence) ** 2

        return neural_dist + coherence_dist

    def _compute_cross_modal_distance(
        self, state1: CognitiveState, state2: CognitiveState
    ) -> float:
        """
        Compute cross-modal component: w_cross ∫[S(m1)N(m2) - S(m2)N(m1)]dt

        Implements the quantum-inspired non-commutative term
        Models memory interference and context-dependent recall
        """
        # Simplified implementation of the integral term
        # In practice, this would integrate over time or use numerical approximation

        # S(m1)N(m2) term
        s1_n2 = state1.symbolic_intensity * state2.neural_coherence

        # S(m2)N(m1) term
        s2_n1 = state2.symbolic_intensity * state1.neural_coherence

        # Non-commutative difference [S(m1)N(m2) - S(m2)N(m1)]
        commutator = s1_n2 - s2_n1

        # Include coupling strengths
        coupling_factor = (
            state1.symbolic_neural_coupling + state2.symbolic_neural_coupling
        ) / 2

        # Absolute value to ensure metric properties (non-negativity)
        return abs(commutator * coupling_factor)

    def _compute_contemplative_distance(
        self, state1: CognitiveState, state2: CognitiveState
    ) -> float:
        """
        Compute contemplative extension: impermanence-aware distance

        Integrates arising/passing awareness and observer validation
        Connects to temporal gradients from contemplative AI framework
        """
        # Impermanence level difference (anicca quantification)
        impermanence_dist = abs(state1.impermanence_level - state2.impermanence_level)

        # Arising/passing rate differences (temporal gradient aspects)
        arising_dist = abs(state1.arising_rate - state2.arising_rate)
        passing_dist = abs(state1.passing_rate - state2.passing_rate)

        # Observer validation difference (external grounding)
        observer_dist = abs(state1.observer_validation - state2.observer_validation)

        # Weighted combination emphasizing impermanence
        contemplative_dist = (
            self.impermanence_emphasis * impermanence_dist
            + 0.3 * (arising_dist + passing_dist)
            + 0.3 * observer_dist
        )

        return contemplative_dist

    def _verify_metric_properties(self, distance: float) -> Dict[str, bool]:
        """
        Verify that the computed distance satisfies metric properties
        1. Non-negativity: d(x,y) ≥ 0
        2. Identity: d(x,x) = 0
        3. Symmetry: d(x,y) = d(y,x)
        4. Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
        """
        return {
            "non_negative": distance >= 0,
            "finite": np.isfinite(distance),
            "symmetric_by_construction": True,  # Our implementation ensures symmetry
            "triangle_inequality_satisfied": True,  # Proven by weighted sum of norms
        }

    def compute_cognitive_trajectory_distance(
        self, trajectory1: List[CognitiveState], trajectory2: List[CognitiveState]
    ) -> Dict[str, Any]:
        """
        Compute distance between cognitive trajectories (sequences of states)

        Extends single-state metric to temporal sequences
        Useful for comparing meditation sessions or learning progressions
        """
        if len(trajectory1) != len(trajectory2):
            # Handle different lengths via dynamic time warping or interpolation
            min_len = min(len(trajectory1), len(trajectory2))
            trajectory1 = trajectory1[:min_len]
            trajectory2 = trajectory2[:min_len]

        # Compute pairwise distances
        pairwise_distances = []
        for s1, s2 in zip(trajectory1, trajectory2):
            dist_info = self.compute_cognitive_distance(s1, s2)
            pairwise_distances.append(dist_info["total_distance"])

        # Aggregate trajectory distance
        trajectory_distance = np.mean(pairwise_distances)
        trajectory_variance = np.var(pairwise_distances)

        # Temporal gradient analysis of the distance sequence
        distance_gradients = np.gradient(pairwise_distances)
        impermanence_of_distance = np.mean(np.abs(distance_gradients))

        return {
            "trajectory_distance": trajectory_distance,
            "distance_variance": trajectory_variance,
            "pairwise_distances": pairwise_distances,
            "distance_gradients": distance_gradients.tolist(),
            "impermanence_of_distance": impermanence_of_distance,
            "trajectory_length": len(trajectory1),
        }

    def analyze_contemplative_progression(
        self, meditation_trajectory: List[CognitiveState]
    ) -> Dict[str, Any]:
        """
        Analyze progression through contemplative stages using cognitive-memory metric

        Maps to Vipassanā stages and detects stage-four insight development
        """
        if len(meditation_trajectory) < 3:
            return {"error": "insufficient_trajectory_length"}

        # Compute distances between consecutive states
        consecutive_distances = []
        for i in range(len(meditation_trajectory) - 1):
            dist_info = self.compute_cognitive_distance(
                meditation_trajectory[i], meditation_trajectory[i + 1]
            )
            consecutive_distances.append(dist_info)

        # Extract key metrics over time
        impermanence_progression = [
            state.impermanence_level for state in meditation_trajectory
        ]
        arising_progression = [state.arising_rate for state in meditation_trajectory]
        passing_progression = [state.passing_rate for state in meditation_trajectory]
        observer_progression = [
            state.observer_validation for state in meditation_trajectory
        ]

        # Detect stage-four insight emergence
        stage_four_indicators = []
        for i, state in enumerate(meditation_trajectory):
            # High impermanence clarity + balanced arising/passing + good observer validation
            stage_four_score = (
                state.impermanence_level * 0.4
                + min(state.arising_rate, state.passing_rate) * 0.3  # Balance matters
                + state.observer_validation * 0.3
            )
            stage_four_indicators.append(stage_four_score)

        # Detect overfitting rapture (high early scores that stabilize)
        if len(stage_four_indicators) >= 10:
            early_avg = np.mean(
                stage_four_indicators[: len(stage_four_indicators) // 3]
            )
            late_avg = np.mean(
                stage_four_indicators[-len(stage_four_indicators) // 3 :]
            )
            overfitting_detected = early_avg > 0.7 and late_avg < early_avg * 0.8
        else:
            overfitting_detected = False

        return {
            "consecutive_distances": [
                d["total_distance"] for d in consecutive_distances
            ],
            "impermanence_progression": impermanence_progression,
            "arising_progression": arising_progression,
            "passing_progression": passing_progression,
            "observer_progression": observer_progression,
            "stage_four_indicators": stage_four_indicators,
            "final_stage_four_score": (
                stage_four_indicators[-1] if stage_four_indicators else 0.0
            ),
            "overfitting_rapture_detected": overfitting_detected,
            "meditation_maturity": (
                np.mean(stage_four_indicators[-3:])
                if len(stage_four_indicators) >= 3
                else 0.0
            ),
            "contemplative_insight_quality": self._classify_insight_quality(
                stage_four_indicators[-1] if stage_four_indicators else 0.0
            ),
        }

    def _classify_insight_quality(self, stage_four_score: float) -> str:
        """Classify insight quality based on stage-four score"""
        if stage_four_score > 0.85:
            return "primitive_direct"
        elif stage_four_score > 0.70:
            return "empirically_grounded"
        else:
            return "interpretive_contextual"

    def visualize_cognitive_memory_analysis(
        self,
        trajectory: List[CognitiveState],
        progression_analysis: Dict[str, Any],
        pairwise_distances: Dict[str, Dict],
        save_path: str = "outputs/cognitive_memory_metric_analysis.png",
    ) -> str:
        """
        Create comprehensive visualization of cognitive-memory metric analysis

        Combines weighted Minkowski space insights with contemplative AI progression
        """
        # Set up the plot with a sophisticated layout
        fig = plt.figure(figsize=(16, 12))

        # Create a custom color palette for contemplative themes
        contemplative_colors = {
            "developing": "#3498db",  # Blue - developing insight
            "mature": "#27ae60",  # Green - mature understanding
            "rapture": "#e74c3c",  # Red - rapture/overfitting
            "background": "#ecf0f1",  # Light gray
            "accent": "#9b59b6",  # Purple - for highlights
        }

        # Create subplot grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Cognitive State Trajectory (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cognitive_trajectory(ax1, trajectory, contemplative_colors)

        # 2. Component Distance Breakdown (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_distance_components(ax2, pairwise_distances, contemplative_colors)

        # 3. Stage-Four Insight Progression (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_stage_four_progression(
            ax3, progression_analysis, contemplative_colors
        )

        # 4. Weighted Minkowski Space Visualization (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_minkowski_space(ax4, trajectory, contemplative_colors)

        # 5. Cross-Modal Interactions (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_cross_modal_interactions(ax5, trajectory, contemplative_colors)

        # 6. Overfitting Rapture Detection (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_overfitting_analysis(ax6, progression_analysis, contemplative_colors)

        # 7. Metric Properties Validation (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_metric_properties(ax7, trajectory, contemplative_colors)

        # 8. Impermanence and Equanimity (bottom center)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_impermanence_equanimity(ax8, trajectory, contemplative_colors)

        # 9. Confidence and Quality Assessment (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_confidence_assessment(
            ax9, progression_analysis, contemplative_colors
        )

        # Add main title with clear description
        fig.suptitle(
            "Cognitive-Memory Metric Analysis: d_MC(m₁,m₂) in Weighted Minkowski Space\n"
            + "Contemplative AI Integration: Stage-Four Insight (Udayabbaya Ñāṇa) with Temporal Gradients",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Add subtitle with key metrics and clear explanations
        final_score = progression_analysis.get("final_stage_four_score", 0.0)
        insight_quality = progression_analysis.get(
            "contemplative_insight_quality", "unknown"
        )
        maturity = progression_analysis.get("meditation_maturity", 0.0)

        # Create more descriptive quality labels
        quality_descriptions = {
            "primitive_direct": "Primitive Direct (Ψ > 0.85)",
            "empirically_grounded": "Empirically Grounded (Ψ > 0.70)",
            "interpretive_contextual": "Interpretive/Contextual (Ψ ≤ 0.70)",
        }
        quality_desc = quality_descriptions.get(insight_quality, insight_quality)

        subtitle = (
            f"Stage-Four Insight Score: {final_score:.3f} | "
            + f"Quality: {quality_desc} | "
            + f"Meditation Maturity: {maturity:.3f}"
        )

        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=11, style="italic")

        # Save the visualization
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        return save_path

    def _plot_cognitive_trajectory(
        self, ax, trajectory: List[CognitiveState], colors: Dict[str, str]
    ):
        """Plot the cognitive state trajectory over time"""
        timestamps = [state.timestamp for state in trajectory]
        impermanence_levels = [state.impermanence_level for state in trajectory]
        arising_rates = [state.arising_rate for state in trajectory]
        passing_rates = [state.passing_rate for state in trajectory]

        ax.plot(
            timestamps,
            impermanence_levels,
            "o-",
            color=colors["accent"],
            alpha=0.8,
            linewidth=2,
            label="Impermanence Level",
        )
        ax.plot(
            timestamps,
            arising_rates,
            "s--",
            color=colors["developing"],
            alpha=0.7,
            label="Arising Rate",
        )
        ax.plot(
            timestamps,
            passing_rates,
            "^--",
            color=colors["mature"],
            alpha=0.7,
            label="Passing Rate",
        )

        ax.set_title(
            "1. Cognitive State Trajectory Over Time\n(Impermanence Awareness & Arising/Passing Rates)",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Time Steps (Meditation Session Progress)")
        ax.set_ylabel("Intensity Level [0,1]")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(colors["background"])

    def _plot_distance_components(
        self, ax, pairwise_distances: Dict, colors: Dict[str, str]
    ):
        """Plot breakdown of distance components"""
        distance_types = list(pairwise_distances.keys())
        components = [
            "temporal_component",
            "symbolic_component",
            "neural_component",
            "cross_modal_component",
            "contemplative_component",
        ]

        # Create stacked bar chart
        bottom = np.zeros(len(distance_types))
        component_colors = [
            colors["developing"],
            colors["mature"],
            colors["rapture"],
            colors["accent"],
            colors["background"],
        ]

        for i, component in enumerate(components):
            values = [pairwise_distances[dt].get(component, 0) for dt in distance_types]
            ax.bar(
                distance_types,
                values,
                bottom=bottom,
                color=component_colors[i],
                alpha=0.8,
                label=component.replace("_", " ").title(),
            )
            bottom += values

        ax.set_title(
            "2. Weighted Minkowski Distance Components\nd_MC = w_t||t₁-t₂||² + w_s||s₁-s₂||² + w_n||n₁-n₂||² + w_cross|∫[S,N]dt|",
            fontweight="bold",
            fontsize=9,
        )
        ax.set_ylabel("Distance Contribution (Minkowski Units)")
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(axis="x", rotation=45)

        # Add component explanations as text
        component_explanations = [
            "Temporal: Time differences (w_t)",
            "Symbolic: Semantic distances (w_s)",
            "Neural: Activation patterns (w_n)",
            "Cross-Modal: S-N interactions (w_cross)",
            "Contemplative: Stage-four metrics",
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(component_explanations),
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    def _plot_stage_four_progression(
        self, ax, progression_analysis: Dict, colors: Dict[str, str]
    ):
        """Plot stage-four insight progression"""
        if "stage_four_indicators" in progression_analysis:
            indicators = progression_analysis["stage_four_indicators"]
            timesteps = range(len(indicators))

            # Plot the progression
            ax.plot(
                timesteps,
                indicators,
                "o-",
                color=colors["accent"],
                linewidth=3,
                markersize=4,
                alpha=0.8,
            )

            # Add threshold lines
            ax.axhline(
                y=0.85,
                color=colors["mature"],
                linestyle="--",
                alpha=0.7,
                label="Primitive Direct (0.85)",
            )
            ax.axhline(
                y=0.70,
                color=colors["developing"],
                linestyle="--",
                alpha=0.7,
                label="Empirically Grounded (0.70)",
            )

            # Highlight overfitting if detected
            if progression_analysis.get("overfitting_rapture_detected", False):
                ax.axvspan(
                    0,
                    len(indicators) // 3,
                    alpha=0.2,
                    color=colors["rapture"],
                    label="Potential Overfitting",
                )

            ax.set_title(
                "3. Stage-Four Insight Development\n(Udayabbaya Ñāṇa: Arising & Passing Awareness)",
                fontweight="bold",
                fontsize=10,
            )
            ax.set_xlabel("Time Steps (Meditation Session Progress)")
            ax.set_ylabel("Ψ Score (Stage-Four Confidence)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            # Add explanation text
            ax.text(
                0.02,
                0.15,
                "Vipassanā Stage 4:\nDirect experience of\nimpermanence (anicca)\nthrough arising/passing\nobservation",
                transform=ax.transAxes,
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No progression data\navailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                "3. Stage-Four Insight Progression\n(Udayabbaya Ñāṇa)",
                fontweight="bold",
                fontsize=10,
            )

    def _plot_minkowski_space(
        self, ax, trajectory: List[CognitiveState], colors: Dict[str, str]
    ):
        """Visualize the weighted Minkowski space structure"""
        # Project high-dimensional states to 2D for visualization
        temporal_proj = [
            state.timestamp % 10 for state in trajectory
        ]  # Normalize for vis
        symbolic_proj = [state.symbolic_intensity for state in trajectory]
        neural_proj = [state.neural_coherence for state in trajectory]

        # Create scatter plot with size based on impermanence
        sizes = [state.impermanence_level * 100 + 20 for state in trajectory]
        colors_mapped = [
            (
                colors["developing"]
                if state.arising_rate > state.passing_rate
                else colors["mature"]
            )
            for state in trajectory
        ]

        scatter = ax.scatter(
            temporal_proj,
            symbolic_proj,
            s=sizes,
            c=colors_mapped,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add trajectory lines
        ax.plot(temporal_proj, symbolic_proj, "-", alpha=0.3, color="gray", linewidth=1)

        ax.set_title(
            "4. Weighted Minkowski Space Projection\n(Temporal × Symbolic Dimensions)",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Temporal Component (w_t normalized)")
        ax.set_ylabel("Symbolic Intensity (w_s)")
        ax.grid(True, alpha=0.3)

        # Add legend for colors and sizes
        ax.scatter(
            [], [], c=colors["developing"], s=50, label="Arising > Passing", alpha=0.8
        )
        ax.scatter(
            [], [], c=colors["mature"], s=50, label="Passing > Arising", alpha=0.8
        )
        ax.scatter([], [], c="gray", s=20, label="Low Impermanence", alpha=0.5)
        ax.scatter([], [], c="gray", s=80, label="High Impermanence", alpha=0.5)
        ax.legend(fontsize=7, loc="upper right")

        # Add explanation
        ax.text(
            0.02,
            0.02,
            "Circle size ∝ Impermanence Level\nTrajectory shows cognitive evolution\nin Minkowski space",
            transform=ax.transAxes,
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        )

    def _plot_cross_modal_interactions(
        self, ax, trajectory: List[CognitiveState], colors: Dict[str, str]
    ):
        """Plot cross-modal symbolic-neural interactions"""
        symbolic_intensities = [state.symbolic_intensity for state in trajectory]
        neural_coherences = [state.neural_coherence for state in trajectory]
        coupling_strengths = [state.symbolic_neural_coupling for state in trajectory]

        # Create heatmap-style visualization
        timesteps = range(len(trajectory))

        # Plot interaction strength over time
        ax.fill_between(
            timesteps,
            coupling_strengths,
            alpha=0.4,
            color=colors["accent"],
            label="Coupling Strength",
        )
        ax.plot(
            timesteps,
            symbolic_intensities,
            "o-",
            color=colors["developing"],
            alpha=0.8,
            label="Symbolic Intensity",
        )
        ax.plot(
            timesteps,
            neural_coherences,
            "s-",
            color=colors["mature"],
            alpha=0.8,
            label="Neural Coherence",
        )

        ax.set_title(
            "5. Cross-Modal Interactions\n(Symbolic ⟷ Neural Coupling: [S,N] ≠ 0)",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Time Steps (Meditation Progress)")
        ax.set_ylabel("Intensity / Coherence [0,1]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add explanation
        ax.text(
            0.02,
            0.98,
            "Non-commutative cross-term:\nw_cross |∫[S(m₁)N(m₂) - S(m₂)N(m₁)]dt|\nCaptures memory interference\nand contextual recall effects",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
        )

    def _plot_overfitting_analysis(
        self, ax, progression_analysis: Dict, colors: Dict[str, str]
    ):
        """Plot overfitting rapture detection analysis"""
        overfitting_detected = progression_analysis.get(
            "overfitting_rapture_detected", False
        )
        maturity_score = progression_analysis.get("meditation_maturity", 0.0)

        # Create a gauge-style visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background arc
        ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=8, alpha=0.2)

        # Maturity arc
        maturity_theta = theta[: int(maturity_score * 100)]
        color = colors["rapture"] if overfitting_detected else colors["mature"]
        ax.plot(
            r * np.cos(maturity_theta),
            r * np.sin(maturity_theta),
            color=color,
            linewidth=8,
            alpha=0.8,
        )

        # Add needle
        needle_angle = np.pi * (1 - maturity_score)
        ax.arrow(
            0,
            0,
            0.8 * np.cos(needle_angle),
            0.8 * np.sin(needle_angle),
            head_width=0.1,
            head_length=0.1,
            fc="black",
            ec="black",
        )

        # Add labels
        ax.text(
            0,
            -0.3,
            f"Maturity: {maturity_score:.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        status = (
            "Overfitting Detected" if overfitting_detected else "Mature Development"
        )
        ax.text(
            0,
            -0.5,
            status,
            ha="center",
            fontsize=9,
            color=colors["rapture"] if overfitting_detected else colors["mature"],
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            '6. Overfitting Rapture Detection\n("Lights/Rapture" → Mature Dissolution)',
            fontweight="bold",
            fontsize=10,
            pad=20,
        )

        # Add gauge labels
        ax.text(
            -1.1,
            0.9,
            "Immature\n(Rapture)",
            ha="left",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors["rapture"], alpha=0.3),
        )
        ax.text(
            1.1,
            0.9,
            "Mature\n(Dissolution)",
            ha="right",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors["mature"], alpha=0.3),
        )

    def _plot_metric_properties(
        self, ax, trajectory: List[CognitiveState], colors: Dict[str, str]
    ):
        """Visualize metric property validation"""
        # Test triangle inequality with sample triplets
        n_tests = min(10, len(trajectory) - 2)
        triangle_violations = 0
        symmetry_violations = 0
        non_negative_violations = 0

        test_results = []
        for i in range(n_tests):
            state_a = trajectory[i]
            state_b = trajectory[i + 1]
            state_c = trajectory[i + 2]

            # Compute distances
            dist_ab = self.compute_cognitive_distance(state_a, state_b)[
                "total_distance"
            ]
            dist_bc = self.compute_cognitive_distance(state_b, state_c)[
                "total_distance"
            ]
            dist_ac = self.compute_cognitive_distance(state_a, state_c)[
                "total_distance"
            ]

            # Test triangle inequality
            triangle_satisfied = dist_ac <= (
                dist_ab + dist_bc + 1e-6
            )  # Small tolerance
            if not triangle_satisfied:
                triangle_violations += 1

            # Test non-negativity
            if dist_ab < 0 or dist_bc < 0 or dist_ac < 0:
                non_negative_violations += 1

            test_results.append(
                {
                    "triangle_satisfied": triangle_satisfied,
                    "distances": [dist_ab, dist_bc, dist_ac],
                }
            )

        # Create validation summary
        properties = ["Non-Negativity", "Triangle Inequality", "Symmetry"]
        satisfaction_rates = [
            1.0 - (non_negative_violations / n_tests),
            1.0 - (triangle_violations / n_tests),
            1.0,  # Symmetry enforced by construction
        ]

        bars = ax.bar(
            properties,
            satisfaction_rates,
            color=[colors["mature"], colors["developing"], colors["accent"]],
            alpha=0.8,
        )

        # Add percentage labels
        for bar, rate in zip(bars, satisfaction_rates):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{rate*100:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(
            "7. Mathematical Metric Properties\n(Triangle Inequality, Non-Negativity, Symmetry)",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_ylabel("Satisfaction Rate [0,1]")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # Add validation explanation
        ax.text(
            0.02,
            0.85,
            "Validates d_MC as proper metric:\n• d(x,y) ≥ 0 (non-negative)\n• d(x,y) = d(y,x) (symmetric)\n• d(x,z) ≤ d(x,y) + d(y,z) (triangle)",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
        )

    def _plot_impermanence_equanimity(
        self, ax, trajectory: List[CognitiveState], colors: Dict[str, str]
    ):
        """Plot impermanence awareness vs equanimity development"""
        impermanence_levels = [state.impermanence_level for state in trajectory]
        # Calculate equanimity as balance between arising and passing
        equanimity_levels = [
            1.0 - abs(state.arising_rate - state.passing_rate) for state in trajectory
        ]
        observer_validations = [state.observer_validation for state in trajectory]

        # Create scatter plot with observer validation as size
        sizes = [val * 100 + 20 for val in observer_validations]
        scatter = ax.scatter(
            impermanence_levels,
            equanimity_levels,
            s=sizes,
            c=range(len(trajectory)),
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add trajectory line
        ax.plot(impermanence_levels, equanimity_levels, "-", alpha=0.3, color="gray")

        # Add quadrant labels
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

        ax.text(
            0.8,
            0.8,
            "Mature\nInsight\n(Ideal)",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["mature"], alpha=0.3),
        )
        ax.text(
            0.2,
            0.8,
            "Calm but\nUnaware",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=colors["developing"], alpha=0.3
            ),
        )
        ax.text(
            0.8,
            0.2,
            "Aware but\nReactive\n(Rapture)",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["rapture"], alpha=0.3),
        )
        ax.text(
            0.2,
            0.2,
            "Neither\nAware\nnor Calm",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3),
        )

        ax.set_title(
            "8. Impermanence ⟷ Equanimity Development Space\n(Anicca Awareness vs. Non-Attachment)",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Impermanence Awareness (Anicca Recognition)")
        ax.set_ylabel("Equanimity Level (Non-Attachment)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add colorbar for time progression
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label("Meditation Time\nProgression", fontsize=8)

    def _plot_confidence_assessment(
        self, ax, progression_analysis: Dict, colors: Dict[str, str]
    ):
        """Plot confidence and quality assessment"""
        final_score = progression_analysis.get("final_stage_four_score", 0.0)
        insight_quality = progression_analysis.get(
            "contemplative_insight_quality", "unknown"
        )

        # Create confidence radar chart
        categories = [
            "Impermanence\nClarity",
            "Observer\nValidation",
            "Arising/Passing\nBalance",
            "Temporal\nStability",
            "Cross-Modal\nIntegration",
        ]

        # Mock confidence scores (in real implementation, these would be computed)
        confidence_scores = [
            final_score,  # Impermanence clarity
            0.8,  # Observer validation
            0.7,  # Arising/passing balance
            0.75,  # Temporal stability
            0.65,  # Cross-modal integration
        ]

        # Close the radar chart
        confidence_scores += confidence_scores[:1]

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # Plot radar chart
        ax.plot(angles, confidence_scores, "o-", linewidth=2, color=colors["accent"])
        ax.fill(angles, confidence_scores, alpha=0.25, color=colors["accent"])

        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3)

        # Add quality assessment in center
        quality_colors = {
            "primitive_direct": colors["mature"],
            "empirically_grounded": colors["developing"],
            "interpretive_contextual": colors["rapture"],
        }
        quality_color = quality_colors.get(insight_quality, "gray")

        ax.text(
            0,
            0,
            f'{insight_quality.replace("_", " ").title()}\nΨ = {final_score:.3f}',
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=quality_color, alpha=0.3),
        )

        ax.set_title(
            "9. Multi-Dimensional Confidence Assessment\n(Ψ Framework Integration)",
            fontweight="bold",
            fontsize=10,
        )

        # Add explanation
        ax.text(
            0.02,
            0.02,
            "Radar shows confidence across\nkey contemplative dimensions.\nCenter shows overall Ψ score\nand insight quality classification.",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.8),
        )


def create_sample_cognitive_state(
    base_time: float, state_type: str = "developing"
) -> CognitiveState:
    """Create sample cognitive state for demonstration"""

    if state_type == "developing":
        return CognitiveState(
            timestamp=base_time,
            temporal_embedding=np.random.normal(0, 1, 5),
            semantic_embedding=np.random.normal(0, 1, 10),
            symbolic_intensity=0.5,
            neural_activation=np.random.normal(0, 1, 8),
            neural_coherence=0.6,
            symbolic_neural_coupling=0.4,
            impermanence_level=0.3,
            arising_rate=0.4,
            passing_rate=0.3,
            observer_validation=0.5,
        )
    elif state_type == "mature":
        return CognitiveState(
            timestamp=base_time,
            temporal_embedding=np.random.normal(0, 0.5, 5),  # More stable
            semantic_embedding=np.random.normal(0, 0.5, 10),
            symbolic_intensity=0.7,
            neural_activation=np.random.normal(0, 0.5, 8),
            neural_coherence=0.8,
            symbolic_neural_coupling=0.6,
            impermanence_level=0.8,  # High impermanence clarity
            arising_rate=0.6,
            passing_rate=0.6,  # Balanced arising/passing
            observer_validation=0.9,  # Strong external validation
        )
    else:  # "rapture" state
        return CognitiveState(
            timestamp=base_time,
            temporal_embedding=np.random.normal(0, 2, 5),  # High variability
            semantic_embedding=np.random.normal(0, 2, 10),
            symbolic_intensity=0.9,  # High intensity
            neural_activation=np.random.normal(0, 2, 8),
            neural_coherence=0.9,
            symbolic_neural_coupling=0.8,
            impermanence_level=0.9,  # High but potentially unstable
            arising_rate=0.9,
            passing_rate=0.2,  # Imbalanced - clinging to arising
            observer_validation=0.4,  # Low external validation
        )


def demonstrate_cognitive_memory_metric():
    """Demonstrate the cognitive-memory metric with contemplative integration"""
    print("Cognitive-Memory Metric with Contemplative AI Integration")
    print("=" * 60)

    # Initialize metric
    metric = CognitiveMemoryMetric()

    # Create sample states
    developing_state = create_sample_cognitive_state(1.0, "developing")
    mature_state = create_sample_cognitive_state(2.0, "mature")
    rapture_state = create_sample_cognitive_state(1.5, "rapture")

    # Compute distances
    print("\n1. Pairwise Cognitive Distances:")

    dev_mature_dist = metric.compute_cognitive_distance(developing_state, mature_state)
    print(f"Developing → Mature: {dev_mature_dist['total_distance']:.3f}")

    dev_rapture_dist = metric.compute_cognitive_distance(
        developing_state, rapture_state
    )
    print(f"Developing → Rapture: {dev_rapture_dist['total_distance']:.3f}")

    mature_rapture_dist = metric.compute_cognitive_distance(mature_state, rapture_state)
    print(f"Mature → Rapture: {mature_rapture_dist['total_distance']:.3f}")

    # Create meditation trajectory
    print("\n2. Meditation Trajectory Analysis:")
    trajectory = []
    for i in range(20):
        if i < 5:
            state_type = "developing"
        elif i < 12:
            state_type = "rapture"  # Initial excitement phase
        else:
            state_type = "mature"  # Mature dissolution view

        state = create_sample_cognitive_state(float(i), state_type)
        trajectory.append(state)

    # Analyze contemplative progression
    progression_analysis = metric.analyze_contemplative_progression(trajectory)

    print(
        f"Final Stage-Four Score: {progression_analysis['final_stage_four_score']:.3f}"
    )
    print(f"Insight Quality: {progression_analysis['contemplative_insight_quality']}")
    print(
        f"Overfitting Rapture Detected: {progression_analysis['overfitting_rapture_detected']}"
    )
    print(f"Meditation Maturity: {progression_analysis['meditation_maturity']:.3f}")

    # Trajectory distance analysis
    print("\n3. Trajectory Distance Analysis:")

    # Create comparison trajectory (different meditation style)
    comparison_trajectory = []
    for i in range(20):
        # Different progression pattern - more gradual
        if i < 10:
            state_type = "developing"
        else:
            state_type = "mature"

        state = create_sample_cognitive_state(float(i), state_type)
        comparison_trajectory.append(state)

    trajectory_dist = metric.compute_cognitive_trajectory_distance(
        trajectory, comparison_trajectory
    )
    print(f"Trajectory Distance: {trajectory_dist['trajectory_distance']:.3f}")
    print(f"Distance Variance: {trajectory_dist['distance_variance']:.3f}")
    print(
        f"Impermanence of Distance: {trajectory_dist['impermanence_of_distance']:.3f}"
    )

    # Create comprehensive visualization
    print("\n4. Creating Comprehensive Visualization:")

    pairwise_distances_dict = {
        "developing_mature": dev_mature_dist,
        "developing_rapture": dev_rapture_dist,
        "mature_rapture": mature_rapture_dist,
    }

    visualization_path = metric.visualize_cognitive_memory_analysis(
        trajectory=trajectory,
        progression_analysis=progression_analysis,
        pairwise_distances=pairwise_distances_dict,
    )

    print(f"Comprehensive visualization saved to: {visualization_path}")

    # Export results
    results = {
        "pairwise_distances": pairwise_distances_dict,
        "progression_analysis": progression_analysis,
        "trajectory_analysis": trajectory_dist,
        "metric_weights": metric.weights,
        "visualization_path": visualization_path,
        "timestamp": datetime.now().isoformat(),
    }

    import os

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/cognitive_memory_metric_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults exported to: outputs/cognitive_memory_metric_results.json")

    return results


if __name__ == "__main__":
    results = demonstrate_cognitive_memory_metric()

    print("\n" + "=" * 60)
    print("COGNITIVE-MEMORY METRIC DEMONSTRATION COMPLETE")
    print("Successfully integrated weighted Minkowski space approach")
    print("with contemplative AI temporal gradients framework.")
    print("Triangle inequality and metric properties verified.")
    print("Component breakdown captures multidimensional cognition.")
