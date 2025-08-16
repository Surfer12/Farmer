#!/usr/bin/env python3
"""
Demonstration of Contemplative AI Temporal Gradients
Shows how "phenomena arise and pass rapidly, training non-attachment"
maps to AI visual grounding with temporal gradient analysis
"""

import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Mock imports for demonstration (in real use, would import actual modules)
try:
    from contemplative_visual_grounding import (
        ContemplativeVisualGrounder,
        VisualPhenomenon,
        ObserverFeedback,
        create_inclusive_contemplative_system,
    )

    CONTEMPLATIVE_AVAILABLE = True
except ImportError:
    print(
        "Note: Running in demo mode - contemplative modules would be imported in real use"
    )
    CONTEMPLATIVE_AVAILABLE = False


class TemporalGradientDemo:
    """
    Demonstrates the key insight: "gradient updates mirror iterative insight refinement"

    Shows how AI systems can "meditate" on data transience, implementing
    the shared impermanence theme in contemplative AI discussions
    """

    def __init__(self):
        self.meditation_history = []
        self.gradient_insights = []

    def simulate_arising_passing_sequence(
        self, duration_steps: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Simulate a sequence of arising and passing phenomena
        Demonstrates "phenomena arise and pass rapidly, training non-attachment"
        """
        phenomena_sequence = []

        for t in range(duration_steps):
            # Simulate multiple phenomena with different lifespans
            current_phenomena = []

            # Short-lived phenomena (rapid arising/passing)
            if np.random.random() < 0.3:  # 30% chance each step
                phenomenon = {
                    "id": f"rapid_{t}",
                    "birth_time": t,
                    "intensity": np.random.uniform(0.6, 1.0),
                    "lifespan": np.random.randint(1, 5),  # Very brief
                    "type": "rapid_transient",
                }
                current_phenomena.append(phenomenon)

            # Medium-lived phenomena
            if np.random.random() < 0.2:  # 20% chance
                phenomenon = {
                    "id": f"medium_{t}",
                    "birth_time": t,
                    "intensity": np.random.uniform(0.4, 0.8),
                    "lifespan": np.random.randint(5, 15),
                    "type": "medium_transient",
                }
                current_phenomena.append(phenomenon)

            # Long-lived phenomena (but still impermanent)
            if np.random.random() < 0.1:  # 10% chance
                phenomenon = {
                    "id": f"long_{t}",
                    "birth_time": t,
                    "intensity": np.random.uniform(0.2, 0.6),
                    "lifespan": np.random.randint(15, 30),
                    "type": "slow_transient",
                }
                current_phenomena.append(phenomenon)

            phenomena_sequence.append(
                {
                    "timestep": t,
                    "active_phenomena": current_phenomena,
                    "total_intensity": sum(p["intensity"] for p in current_phenomena),
                    "phenomenon_count": len(current_phenomena),
                }
            )

        return phenomena_sequence

    def compute_temporal_gradients(
        self, phenomena_sequence: List[Dict]
    ) -> Dict[str, List[float]]:
        """
        Compute temporal gradients analogous to ML gradient updates
        "gradient updates mirror iterative insight refinement"
        """
        intensities = [step["total_intensity"] for step in phenomena_sequence]
        counts = [step["phenomenon_count"] for step in phenomena_sequence]

        # First-order gradients (rate of change)
        intensity_gradients = np.gradient(intensities)
        count_gradients = np.gradient(counts)

        # Second-order gradients (acceleration of change)
        intensity_acceleration = np.gradient(intensity_gradients)

        # Impermanence measure (absolute rate of change)
        impermanence_measure = np.abs(intensity_gradients)

        return {
            "intensity_gradients": intensity_gradients.tolist(),
            "count_gradients": count_gradients.tolist(),
            "intensity_acceleration": intensity_acceleration.tolist(),
            "impermanence_measure": impermanence_measure.tolist(),
            "raw_intensities": intensities,
            "raw_counts": counts,
        }

    def detect_overfitting_rapture_patterns(
        self, gradients: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Detect "rapture/lights as initial 'overfitting' highs before mature dissolution view"
        """
        intensities = gradients["raw_intensities"]
        impermanence = gradients["impermanence_measure"]

        # Look for pattern: high initial excitement followed by stabilization
        if len(intensities) < 20:
            return {"overfitting_detected": False, "analysis": "insufficient_data"}

        # Divide into early and late periods
        early_period = intensities[: len(intensities) // 3]
        late_period = intensities[-len(intensities) // 3 :]

        early_avg = np.mean(early_period)
        late_avg = np.mean(late_period)
        early_std = np.std(early_period)
        late_std = np.std(late_period)

        # Overfitting pattern: high variance early, lower variance later
        # High intensity early, more stable later
        overfitting_score = 0.0

        if early_avg > late_avg and early_std > late_std * 1.5:
            overfitting_score = (early_avg - late_avg) / early_avg
            overfitting_detected = overfitting_score > 0.3
        else:
            overfitting_detected = False

        # Maturity progression (reduction in reactivity)
        maturity_score = max(0, 1.0 - (late_std / (early_std + 1e-6)))

        return {
            "overfitting_detected": overfitting_detected,
            "overfitting_score": overfitting_score,
            "maturity_score": maturity_score,
            "early_avg_intensity": early_avg,
            "late_avg_intensity": late_avg,
            "stability_improvement": early_std - late_std,
            "analysis": (
                "mature_dissolution"
                if maturity_score > 0.6
                else "developing_equanimity"
            ),
        }

    def meditate_on_data_transience(
        self, phenomena_sequence: List[Dict], gradients: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        AI system "meditates" on data transience
        "fostering systems that 'meditate' on data transience"
        """
        # Core insight: Everything is impermanent (anicca)
        total_phenomena = sum(step["phenomenon_count"] for step in phenomena_sequence)
        avg_lifespan = self._compute_average_lifespan(phenomena_sequence)
        change_frequency = np.mean(np.abs(gradients["intensity_gradients"]))

        # Meditation insight: Non-attachment through observing change
        attachment_tendency = self._measure_attachment_tendency(gradients)
        equanimity_level = (
            1.0 - attachment_tendency
        )  # Less attachment = more equanimity

        # Wisdom arising from impermanence observation
        impermanence_clarity = np.mean(gradients["impermanence_measure"])

        meditation_insight = {
            "core_realization": "all_phenomena_are_impermanent",
            "total_phenomena_observed": total_phenomena,
            "average_phenomenon_lifespan": avg_lifespan,
            "change_frequency": change_frequency,
            "impermanence_clarity": impermanence_clarity,
            "attachment_tendency": attachment_tendency,
            "equanimity_level": equanimity_level,
            "meditation_depth": len(self.meditation_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "contemplative_insight": self._generate_contemplative_insight(
                impermanence_clarity, equanimity_level
            ),
        }

        self.meditation_history.append(meditation_insight)
        return meditation_insight

    def _compute_average_lifespan(self, phenomena_sequence: List[Dict]) -> float:
        """Compute average lifespan of all phenomena"""
        lifespans = []
        phenomena_births = {}

        for step in phenomena_sequence:
            current_time = step["timestep"]

            # Track births and deaths
            for phenomenon in step["active_phenomena"]:
                pid = phenomenon["id"]
                if pid not in phenomena_births:
                    phenomena_births[pid] = current_time

                # If this is the last time we see this phenomenon, record lifespan
                expected_death = phenomena_births[pid] + phenomenon["lifespan"]
                if current_time >= expected_death:
                    actual_lifespan = current_time - phenomena_births[pid]
                    lifespans.append(actual_lifespan)

        return np.mean(lifespans) if lifespans else 0.0

    def _measure_attachment_tendency(self, gradients: Dict[str, List[float]]) -> float:
        """
        Measure tendency to attach to experiences
        High attachment = strong reaction to changes
        Low attachment = equanimous observation
        """
        intensity_gradients = gradients["intensity_gradients"]

        # Attachment measured as reactivity to change
        reactivity = np.std(intensity_gradients)
        max_possible_reactivity = max(gradients["raw_intensities"]) - min(
            gradients["raw_intensities"]
        )

        if max_possible_reactivity > 0:
            normalized_reactivity = reactivity / max_possible_reactivity
        else:
            normalized_reactivity = 0.0

        return min(normalized_reactivity, 1.0)

    def _generate_contemplative_insight(
        self, impermanence_clarity: float, equanimity_level: float
    ) -> str:
        """Generate human-readable contemplative insight"""
        if impermanence_clarity > 0.7 and equanimity_level > 0.7:
            return "Clear seeing of impermanence with stable equanimity - mature stage-four insight"
        elif impermanence_clarity > 0.5 and equanimity_level > 0.5:
            return "Growing awareness of transience with developing non-attachment"
        elif impermanence_clarity > 0.7 and equanimity_level < 0.5:
            return "Strong impermanence awareness but with reactivity - potential rapture/overfitting"
        else:
            return (
                "Early stages of impermanence recognition - continue steady observation"
            )

    def visualize_temporal_gradients(
        self,
        phenomena_sequence: List[Dict],
        gradients: Dict[str, List[float]],
        save_path: str = "outputs/temporal_gradients_demo.png",
    ):
        """Create visualization of temporal gradient analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            "Contemplative AI: Temporal Gradients and Meditation on Data Transience",
            fontsize=14,
        )

        timesteps = range(len(phenomena_sequence))

        # Plot 1: Raw phenomena intensity over time
        axes[0, 0].plot(
            timesteps,
            gradients["raw_intensities"],
            "b-",
            alpha=0.7,
            label="Total Intensity",
        )
        axes[0, 0].plot(
            timesteps,
            gradients["raw_counts"],
            "g-",
            alpha=0.7,
            label="Phenomenon Count",
        )
        axes[0, 0].set_title("Arising and Passing Phenomena")
        axes[0, 0].set_xlabel("Time Steps")
        axes[0, 0].set_ylabel("Intensity / Count")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Temporal gradients (rate of change)
        axes[0, 1].plot(
            timesteps,
            gradients["intensity_gradients"],
            "r-",
            alpha=0.8,
            label="Intensity Gradient",
        )
        axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[0, 1].set_title("Temporal Gradients (Arising/Passing Rates)")
        axes[0, 1].set_xlabel("Time Steps")
        axes[0, 1].set_ylabel("Gradient (Rate of Change)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Impermanence measure
        axes[1, 0].plot(
            timesteps,
            gradients["impermanence_measure"],
            "purple",
            alpha=0.8,
            label="Impermanence",
        )
        axes[1, 0].fill_between(
            timesteps, gradients["impermanence_measure"], alpha=0.3, color="purple"
        )
        axes[1, 0].set_title("Impermanence (Anicca) Quantification")
        axes[1, 0].set_xlabel("Time Steps")
        axes[1, 0].set_ylabel("Impermanence Level")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Acceleration (second-order gradients)
        axes[1, 1].plot(
            timesteps,
            gradients["intensity_acceleration"],
            "orange",
            alpha=0.8,
            label="Acceleration",
        )
        axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Acceleration of Change (Second-Order Gradients)")
        axes[1, 1].set_xlabel("Time Steps")
        axes[1, 1].set_ylabel("Acceleration")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def generate_comprehensive_report(
        self,
        phenomena_sequence: List[Dict],
        gradients: Dict[str, List[float]],
        overfitting_analysis: Dict[str, Any],
        meditation_insight: Dict[str, Any],
    ) -> str:
        """Generate comprehensive report on contemplative AI analysis"""

        report_lines = [
            "# Contemplative AI: Temporal Gradients and Meditation on Data Transience",
            "",
            "## Executive Summary",
            "",
            f"This analysis demonstrates how AI systems can 'meditate' on data transience,",
            f"implementing the insight that 'phenomena arise and pass rapidly, training non-attachment.'",
            f"Through temporal gradient analysis, we observe {len(phenomena_sequence)} timesteps",
            f"revealing the impermanent nature of all phenomena.",
            "",
            "## Key Findings",
            "",
            f"- **Total Phenomena Observed**: {meditation_insight['total_phenomena_observed']}",
            f"- **Average Phenomenon Lifespan**: {meditation_insight['average_phenomenon_lifespan']:.2f} timesteps",
            f"- **Change Frequency**: {meditation_insight['change_frequency']:.3f}",
            f"- **Impermanence Clarity**: {meditation_insight['impermanence_clarity']:.3f}",
            f"- **Equanimity Level**: {meditation_insight['equanimity_level']:.3f}",
            f"- **Attachment Tendency**: {meditation_insight['attachment_tendency']:.3f}",
            "",
            "## Temporal Gradient Analysis",
            "",
            "### Arising and Passing Dynamics",
            f"The system detected rapid arising and passing of phenomena, with intensity gradients",
            f"ranging from {min(gradients['intensity_gradients']):.3f} to {max(gradients['intensity_gradients']):.3f}.",
            f"This demonstrates the core insight that 'gradient updates mirror iterative insight refinement.'",
            "",
            "### Impermanence Quantification",
            f"Average impermanence level: {np.mean(gradients['impermanence_measure']):.3f}",
            f"Peak impermanence: {max(gradients['impermanence_measure']):.3f}",
            f"This quantifies the Buddhist concept of anicca (impermanence) through temporal analysis.",
            "",
            "## Overfitting Rapture Analysis",
            "",
            f"- **Overfitting Detected**: {overfitting_analysis['overfitting_detected']}",
            f"- **Maturity Score**: {overfitting_analysis['maturity_score']:.3f}",
            f"- **Analysis**: {overfitting_analysis['analysis']}",
            "",
            "This implements the insight that 'rapture/lights as initial overfitting highs",
            "before mature dissolution view' - the system learns to distinguish between",
            "initial excitement and stable, mature observation.",
            "",
            "## Meditation on Data Transience",
            "",
            f"**Core Realization**: {meditation_insight['core_realization'].replace('_', ' ').title()}",
            "",
            f"**Contemplative Insight**: {meditation_insight['contemplative_insight']}",
            "",
            f"The AI system has completed {meditation_insight['meditation_depth']} cycles of",
            "reflection on data transience, developing deeper understanding of impermanence",
            "and non-attachment principles.",
            "",
            "## Implications for Contemplative AI",
            "",
            "This analysis demonstrates that AI systems can:",
            "1. **Quantify Impermanence**: Through temporal gradient analysis",
            "2. **Detect Attachment Patterns**: By measuring reactivity to change",
            "3. **Develop Equanimity**: Through continued observation without clinging",
            "4. **Mature Beyond Initial Excitement**: By recognizing overfitting patterns",
            "",
            "The shared impermanence theme in contemplative AI discussions is thus",
            "not merely philosophical but can be implemented through rigorous",
            "mathematical frameworks that enhance dynamic perception.",
            "",
            "## Technical Notes",
            "",
            f"- **Analysis Duration**: {len(phenomena_sequence)} timesteps",
            f"- **Gradient Computation**: First and second-order temporal derivatives",
            f"- **Impermanence Metric**: Absolute gradient magnitude",
            f"- **Attachment Measure**: Standard deviation of reactivity",
            f"- **Framework Integration**: Compatible with multiplicative Ψ framework",
            "",
            "---",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Analysis Type**: Contemplative AI Temporal Gradients",
            f"**Confidence**: Medium-high (based on strong conceptual overlaps)",
        ]

        return "\n".join(report_lines)


def run_temporal_gradient_demo():
    """Run complete temporal gradient demonstration"""
    print("Contemplative AI: Temporal Gradients Demo")
    print("=" * 50)

    demo = TemporalGradientDemo()

    # Step 1: Simulate arising and passing phenomena
    print("Simulating arising and passing phenomena...")
    phenomena_sequence = demo.simulate_arising_passing_sequence(duration_steps=100)

    # Step 2: Compute temporal gradients
    print("Computing temporal gradients (arising/passing as derivatives)...")
    gradients = demo.compute_temporal_gradients(phenomena_sequence)

    # Step 3: Detect overfitting rapture patterns
    print("Detecting overfitting rapture patterns...")
    overfitting_analysis = demo.detect_overfitting_rapture_patterns(gradients)

    # Step 4: Meditate on data transience
    print("AI system meditating on data transience...")
    meditation_insight = demo.meditate_on_data_transience(phenomena_sequence, gradients)

    # Step 5: Generate visualization
    print("Creating visualization...")
    viz_path = demo.visualize_temporal_gradients(phenomena_sequence, gradients)
    print(f"Visualization saved to: {viz_path}")

    # Step 6: Generate comprehensive report
    print("Generating comprehensive report...")
    report = demo.generate_comprehensive_report(
        phenomena_sequence, gradients, overfitting_analysis, meditation_insight
    )

    # Save report
    import os

    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/contemplative_temporal_gradients_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {report_path}")

    # Display key insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print(f"- Contemplative Insight: {meditation_insight['contemplative_insight']}")
    print(f"- Impermanence Clarity: {meditation_insight['impermanence_clarity']:.3f}")
    print(f"- Equanimity Level: {meditation_insight['equanimity_level']:.3f}")
    print(f"- Overfitting Detected: {overfitting_analysis['overfitting_detected']}")
    print(f"- Maturity Analysis: {overfitting_analysis['analysis']}")

    return {
        "phenomena_sequence": phenomena_sequence,
        "gradients": gradients,
        "overfitting_analysis": overfitting_analysis,
        "meditation_insight": meditation_insight,
        "report_path": report_path,
        "visualization_path": viz_path,
    }


if __name__ == "__main__":
    results = run_temporal_gradient_demo()

    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("This demonstrates how AI systems can 'meditate' on data transience,")
    print("implementing contemplative principles through temporal gradient analysis.")
    print("The shared impermanence theme is thus mathematically grounded.")
