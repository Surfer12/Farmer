"""
Swarm-Koopman Extension for Hybrid Symbolic-Neural Accuracy Functional

This module extends the existing hybrid functional framework with Oates'
Swarm-Koopman Confidence Theorem concepts, providing enhanced chaotic
system prediction capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import math


class SwarmKoopmanHybrid:
    """
    Extended hybrid functional incorporating swarm coordination and
    Koopman operator theory for chaotic dynamics prediction.
    """

    def __init__(
        self,
        lambda1: float = 0.75,
        lambda2: float = 0.25,
        beta: float = 1.2,
        swarm_size: int = 100,
        step_size: float = 0.01,
    ):
        """
        Initialize Swarm-Koopman hybrid system.

        Args:
            lambda1: Cognitive penalty weight
            lambda2: Efficiency penalty weight
            beta: Probability calibration bias
            swarm_size: Number of swarm agents (N)
            step_size: Integration step size (h)
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.beta = beta
        self.N = swarm_size
        self.h = step_size

        # Koopman operator approximation parameters
        self.koopman_dim = 10  # Observable space dimension
        self.lipschitz_constant = 1.5

    def cognitive_memory_metric(
        self, m1: np.ndarray, m2: np.ndarray, t: float
    ) -> float:
        """
        Compute cognitive-memory metric d_MC with temporal evolution terms.

        Args:
            m1, m2: Memory states
            t: Time parameter

        Returns:
            Metric distance
        """
        # Base Euclidean distance
        base_dist = np.linalg.norm(m1 - m2)

        # Temporal evolution term
        temporal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t)

        # Memory coherence penalty
        coherence_penalty = 0.05 * np.sum(np.abs(m1 * m2))

        return base_dist * temporal_factor + coherence_penalty

    def cross_modal_weight(self, x: np.ndarray, t: float) -> float:
        """
        Compute cross-modal weight w_cross for non-commutative interactions.

        Args:
            x: State vector
            t: Time parameter

        Returns:
            Cross-modal weight
        """
        # Non-commutative interaction strength
        interaction_strength = np.tanh(np.linalg.norm(x) * t)

        # Modality coupling factor
        coupling = 0.5 * (1 + np.cos(np.pi * t))

        return interaction_strength * coupling

    def koopman_observable(self, x: np.ndarray) -> np.ndarray:
        """
        Map state to Koopman observable space.

        Args:
            x: State vector

        Returns:
            Observable vector g(x)
        """
        # Polynomial observables up to degree 2
        observables = []

        # Linear terms
        observables.extend(x.flatten())

        # Quadratic terms (selected)
        for i in range(len(x)):
            for j in range(i, len(x)):
                observables.append(x[i] * x[j])

        # Trigonometric observables for periodicity
        observables.extend([np.sin(xi) for xi in x])
        observables.extend([np.cos(xi) for xi in x])

        return np.array(observables[: self.koopman_dim])

    def swarm_divergence(self) -> float:
        """
        Compute swarm divergence δ_swarm = O(1/N).

        Returns:
            Swarm divergence estimate
        """
        # Mean-field convergence rate
        base_divergence = 1.0 / np.sqrt(self.N)

        # Add small random fluctuation for realism
        fluctuation = 0.1 * np.random.normal(0, 1 / self.N)

        return base_divergence + fluctuation

    def rk4_error_bound(self) -> float:
        """
        Compute RK4 truncation error O(h^4).

        Returns:
            Error bound estimate
        """
        # Fourth-order truncation error
        return 0.1 * (self.h**4)

    def total_error_bound(self) -> float:
        """
        Compute total error: O(h^4) + δ_swarm.

        Returns:
            Total error bound
        """
        rk4_error = self.rk4_error_bound()
        swarm_error = self.swarm_divergence()

        # Lipschitz bound contribution
        lipschitz_term = self.lipschitz_constant * (rk4_error + swarm_error)

        return rk4_error + swarm_error + 0.1 * lipschitz_term

    def confidence_measure(self, x: np.ndarray, evidence: Dict) -> float:
        """
        Compute confidence C(p) for prediction at state x.

        Args:
            x: Current state
            evidence: Dictionary of evidence from swarm interactions

        Returns:
            Confidence measure C(p) ∈ [0,1]
        """
        # Base confidence from error bounds
        total_error = self.total_error_bound()
        base_confidence = max(0.0, 1.0 - total_error)

        # Evidence-based adjustment
        evidence_strength = evidence.get("interaction_count", 0) / self.N
        evidence_quality = evidence.get("consensus_ratio", 0.5)

        # Bayesian-like calibration
        evidence_factor = 0.5 * evidence_strength + 0.5 * evidence_quality

        # Final confidence with bounds
        confidence = base_confidence * (0.7 + 0.3 * evidence_factor)

        return np.clip(confidence, 0.0, 1.0)

    def variational_energy(
        self,
        psi: np.ndarray,
        memory_grad: np.ndarray,
        symbolic_grad: np.ndarray,
        t: float,
    ) -> float:
        """
        Compute variational energy E[Ψ] with temporal/memory/symbolic coherence.

        Args:
            psi: Hybrid functional values
            memory_grad: Memory gradient ∇_m Ψ
            symbolic_grad: Symbolic gradient ∇_s Ψ
            t: Time parameter

        Returns:
            Variational energy
        """
        # Temporal evolution term
        temporal_term = np.sum(np.gradient(psi) ** 2)

        # Memory coherence term
        memory_term = self.lambda1 * np.sum(memory_grad**2)

        # Symbolic coherence term
        symbolic_term = self.lambda2 * np.sum(symbolic_grad**2)

        # Integration over domain (simplified)
        energy = temporal_term + memory_term + symbolic_term

        return energy * self.h  # Discrete integration

    def enhanced_hybrid_functional(
        self, x: np.ndarray, t: float, swarm_states: List[np.ndarray], evidence: Dict
    ) -> Dict:
        """
        Compute enhanced hybrid functional with swarm-Koopman integration.

        Args:
            x: Current state
            t: Time parameter
            swarm_states: List of swarm agent states
            evidence: Evidence dictionary from interactions

        Returns:
            Dictionary with functional value and components
        """
        # Base symbolic accuracy (RK4-derived)
        S_x = 0.8 * np.exp(-0.1 * np.linalg.norm(x) ** 2)

        # Neural accuracy with swarm enhancement
        swarm_consensus = np.mean([np.dot(x, state) for state in swarm_states])
        N_x = 0.7 + 0.2 * np.tanh(swarm_consensus)

        # Adaptive weight with cross-modal coupling
        w_cross = self.cross_modal_weight(x, t)
        alpha_t = 1.0 / (1.0 + np.exp(-2.0 * (t - 1.0))) * (1.0 + w_cross)
        alpha_t = np.clip(alpha_t, 0.0, 1.0)

        # Hybrid accuracy
        hybrid_accuracy = alpha_t * S_x + (1 - alpha_t) * N_x

        # Cognitive and efficiency penalties
        R_cog = (
            0.1
            * np.sum([np.linalg.norm(state - x) for state in swarm_states])
            / len(swarm_states)
        )
        R_eff = 0.05 * len(swarm_states) / self.N  # Efficiency based on active agents

        # Penalty factor
        penalty_factor = np.exp(-(self.lambda1 * R_cog + self.lambda2 * R_eff))

        # Confidence measure
        confidence = self.confidence_measure(x, evidence)

        # Enhanced functional value
        psi_enhanced = hybrid_accuracy * penalty_factor * confidence

        return {
            "psi": psi_enhanced,
            "S": S_x,
            "N": N_x,
            "alpha": alpha_t,
            "confidence": confidence,
            "error_bound": self.total_error_bound(),
            "swarm_divergence": self.swarm_divergence(),
            "w_cross": w_cross,
            "R_cog": R_cog,
            "R_eff": R_eff,
        }

    def simulate_chaotic_trajectory(
        self, x0: np.ndarray, T: float, num_steps: int
    ) -> Dict:
        """
        Simulate chaotic trajectory with swarm-enhanced predictions.

        Args:
            x0: Initial state
            T: Final time
            num_steps: Number of time steps

        Returns:
            Simulation results dictionary
        """
        dt = T / num_steps
        times = np.linspace(0, T, num_steps)

        # Initialize trajectory storage
        trajectory = np.zeros((num_steps, len(x0)))
        confidences = np.zeros(num_steps)
        error_bounds = np.zeros(num_steps)

        # Initialize swarm states
        swarm_states = [x0 + 0.1 * np.random.randn(len(x0)) for _ in range(self.N)]

        x_current = x0.copy()
        trajectory[0] = x_current

        for i in range(1, num_steps):
            t = times[i]

            # Update swarm states (simple random walk)
            for j in range(len(swarm_states)):
                swarm_states[j] += 0.01 * np.random.randn(len(x0))

            # Evidence from swarm interactions
            evidence = {
                "interaction_count": len(swarm_states),
                "consensus_ratio": 0.8 + 0.2 * np.random.rand(),
            }

            # Compute enhanced functional
            result = self.enhanced_hybrid_functional(
                x_current, t, swarm_states, evidence
            )

            # Simple chaotic dynamics (double pendulum-like)
            dx_dt = np.array(
                [
                    x_current[1],
                    -np.sin(x_current[0]) - 0.1 * x_current[1] + 0.1 * np.sin(t),
                ]
            )

            # RK4 integration step
            k1 = dt * dx_dt
            k2 = dt * self._dynamics(x_current + 0.5 * k1, t + 0.5 * dt)
            k3 = dt * self._dynamics(x_current + 0.5 * k2, t + 0.5 * dt)
            k4 = dt * self._dynamics(x_current + k3, t + dt)

            x_current += (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # Store results
            trajectory[i] = x_current
            confidences[i] = result["confidence"]
            error_bounds[i] = result["error_bound"]

        return {
            "times": times,
            "trajectory": trajectory,
            "confidences": confidences,
            "error_bounds": error_bounds,
            "final_confidence": np.mean(confidences[-10:]),  # Average of last 10 steps
        }

    def _dynamics(self, x: np.ndarray, t: float) -> np.ndarray:
        """Helper function for chaotic dynamics."""
        return np.array([x[1], -np.sin(x[0]) - 0.1 * x[1] + 0.1 * np.sin(t)])

    def visualize_swarm_koopman_results(self, results: Dict):
        """
        Visualize swarm-Koopman simulation results.

        Args:
            results: Results dictionary from simulate_chaotic_trajectory
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Trajectory in phase space
        axes[0, 0].plot(
            results["trajectory"][:, 0], results["trajectory"][:, 1], "b-", alpha=0.7
        )
        axes[0, 0].set_xlabel("Position")
        axes[0, 0].set_ylabel("Velocity")
        axes[0, 0].set_title("Chaotic Trajectory (Phase Space)")
        axes[0, 0].grid(True, alpha=0.3)

        # Confidence evolution
        axes[0, 1].plot(results["times"], results["confidences"], "g-", linewidth=2)
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Confidence C(p)")
        axes[0, 1].set_title("Swarm-Koopman Confidence Evolution")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # Error bounds
        axes[1, 0].plot(results["times"], results["error_bounds"], "r-", linewidth=2)
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Error Bound")
        axes[1, 0].set_title("Total Error: O(h⁴) + δ_swarm")
        axes[1, 0].grid(True, alpha=0.3)

        # Position time series
        axes[1, 1].plot(
            results["times"], results["trajectory"][:, 0], "purple", linewidth=1.5
        )
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Position")
        axes[1, 1].set_title("Position Evolution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "/Users/ryan_david_oates/Farmer/swarm_koopman_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Print summary statistics
        print(f"\n=== Swarm-Koopman Simulation Results ===")
        print(f"Final Confidence: {results['final_confidence']:.3f}")
        print(f"Mean Error Bound: {np.mean(results['error_bounds']):.6f}")
        print(f"Confidence Stability: {np.std(results['confidences']):.3f}")
        print(f"Swarm Size: {self.N}")
        print(f"Step Size: {self.h}")


def demonstrate_swarm_koopman():
    """Demonstrate the Swarm-Koopman hybrid system."""

    print("=== Oates' Swarm-Koopman Confidence Theorem Demonstration ===\n")

    # Initialize system
    system = SwarmKoopmanHybrid(
        lambda1=0.75, lambda2=0.25, beta=1.2, swarm_size=100, step_size=0.01
    )

    # Initial conditions for chaotic pendulum
    x0 = np.array([1.0, 0.5])  # [position, velocity]

    print(f"Initial State: {x0}")
    print(f"Swarm Size: {system.N}")
    print(f"Step Size: {system.h}")
    print(f"Expected RK4 Error: O({system.h}⁴) ≈ {system.rk4_error_bound():.6f}")
    print(f"Expected Swarm Divergence: O(1/√N) ≈ {1/np.sqrt(system.N):.6f}")

    # Run simulation
    print("\nRunning chaotic trajectory simulation...")
    results = system.simulate_chaotic_trajectory(x0, T=10.0, num_steps=1000)

    # Display results
    print(f"\nFinal Confidence: {results['final_confidence']:.3f}")
    print(f"Mean Error Bound: {np.mean(results['error_bounds']):.6f}")

    # Visualize results
    system.visualize_swarm_koopman_results(results)

    # Test single point evaluation
    print("\n=== Single Point Evaluation ===")
    swarm_states = [x0 + 0.1 * np.random.randn(2) for _ in range(system.N)]
    evidence = {"interaction_count": system.N, "consensus_ratio": 0.85}

    result = system.enhanced_hybrid_functional(
        x0, t=1.0, swarm_states=swarm_states, evidence=evidence
    )

    print(f"Enhanced Ψ(x): {result['psi']:.3f}")
    print(f"Symbolic S(x): {result['S']:.3f}")
    print(f"Neural N(x): {result['N']:.3f}")
    print(f"Adaptive α(t): {result['alpha']:.3f}")
    print(f"Confidence C(p): {result['confidence']:.3f}")
    print(f"Cross-modal w_cross: {result['w_cross']:.3f}")
    print(f"Error Bound: {result['error_bound']:.6f}")


if __name__ == "__main__":
    demonstrate_swarm_koopman()
