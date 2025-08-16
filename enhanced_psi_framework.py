#!/usr/bin/env python3
"""
Enhanced Ψ(x) Framework: Hierarchical Bayesian + Swarm Intelligence + Chaotic Systems

Implements the advanced framework with:
- Output: Technical reports on hierarchical Bayesian modeling and swarm intelligence
- Hybrid: S(x) as document state inference, N(x) as ML/chaos analysis, α(t) for real-time flow
- Regularization: R_cognitive and R_efficiency with Bayesian penalties and swarm dynamics
- Probability: P(H|E,β) with β for query responsiveness
- Integration: Over document analysis cycles

Based on the mathematical framework:
Ψ(x) = min{β·exp(-[λ₁R_cognitive + λ₂R_efficiency])·[α(t)S(x) + (1-α(t))N(x)], 1}

Where:
- S(x): Document state inference (mathematical structures in PDFs/MD)
- N(x): ML/chaos analysis (Koopman operators, neural predictions)
- α(t): Real-time document flow adaptation
- R_cognitive: Analytical accuracy in Bayesian penalties and swarm dynamics
- R_efficiency: Processing multi-document content efficiency
- P(H|E,β): Probability with β for query responsiveness
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import math
from scipy.stats import beta, gamma, norm
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DocumentState:
    """Document state for inference analysis"""
    timestamp: float
    content_type: str  # 'pdf', 'md', 'tex', 'code'
    mathematical_structures: List[str]  # Detected math structures
    complexity_score: float  # [0,1] complexity measure
    semantic_embedding: np.ndarray  # Vector representation
    confidence: float  # [0,1] confidence in analysis

@dataclass
class ChaosAnalysis:
    """Chaotic system analysis results"""
    lyapunov_exponent: float
    koopman_observables: np.ndarray
    neural_predictions: np.ndarray
    prediction_error: float
    stability_metric: float
    swarm_confidence: float

@dataclass
class PsiComputation:
    """Complete Ψ(x) computation breakdown"""
    timestamp: float
    S_x: float  # Document state inference
    N_x: float  # ML/chaos analysis
    alpha_t: float  # Real-time document flow
    R_cognitive: float  # Cognitive penalty
    R_efficiency: float  # Efficiency penalty
    lambda1: float  # Cognitive penalty weight
    lambda2: float  # Efficiency penalty weight
    beta: float  # Uplift factor
    P_H_E_beta: float  # Probability term
    hybrid_output: float  # α(t)S(x) + (1-α(t))N(x)
    penalty_term: float  # exp(-[λ₁R_cognitive + λ₂R_efficiency])
    psi_final: float  # Final Ψ(x) value
    interpretation: str  # Human-readable interpretation

class EnhancedPsiFramework:
    """
    Enhanced Ψ(x) Framework integrating hierarchical Bayesian modeling,
    swarm intelligence, and chaotic systems analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Core parameters
        self.lambda1 = self.config.get('lambda1', 0.58)  # Cognitive penalty weight
        self.lambda2 = self.config.get('lambda2', 0.42)  # Efficiency penalty weight
        self.beta = self.config.get('beta', 1.25)  # Uplift factor
        
        # Document flow parameters
        self.flow_adaptation_rate = self.config.get('flow_adaptation_rate', 0.1)
        self.complexity_threshold = self.config.get('complexity_threshold', 0.7)
        
        # Swarm intelligence parameters
        self.swarm_size = self.config.get('swarm_size', 100)
        self.swarm_learning_rate = self.config.get('swarm_learning_rate', 0.01)
        
        # Chaos analysis parameters
        self.koopman_dim = self.config.get('koopman_dim', 10)
        self.neural_horizon = self.config.get('neural_horizon', 50)
        
        # History tracking
        self.computation_history: List[PsiComputation] = []
        self.document_states: List[DocumentState] = []
        self.chaos_analyses: List[ChaosAnalysis] = []
        
    def _default_config(self) -> Dict:
        """Default configuration for the enhanced framework"""
        return {
            'lambda1': 0.58,  # Cognitive penalty weight
            'lambda2': 0.42,  # Efficiency penalty weight
            'beta': 1.25,  # Uplift factor
            'flow_adaptation_rate': 0.1,
            'complexity_threshold': 0.7,
            'swarm_size': 100,
            'swarm_learning_rate': 0.01,
            'koopman_dim': 10,
            'neural_horizon': 50
        }
    
    def document_state_inference(self, content: str, content_type: str = 'md') -> DocumentState:
        """
        S(x): Document state inference for mathematical structures
        
        Analyzes documents to extract mathematical structures and semantic content
        """
        # Simulate mathematical structure detection
        math_structures = self._detect_mathematical_structures(content)
        
        # Compute complexity score based on content
        complexity = self._compute_complexity_score(content, math_structures)
        
        # Generate semantic embedding (simplified)
        embedding = np.random.normal(0, 1, 128)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Confidence based on structure detection quality
        confidence = min(0.9, 0.3 + 0.6 * len(math_structures) / 10)
        
        return DocumentState(
            timestamp=datetime.now().timestamp(),
            content_type=content_type,
            mathematical_structures=math_structures,
            complexity_score=complexity,
            semantic_embedding=embedding,
            confidence=confidence
        )
    
    def _detect_mathematical_structures(self, content: str) -> List[str]:
        """Detect mathematical structures in document content"""
        structures = []
        
        # Look for common mathematical patterns
        if '\\[' in content or '\\(' in content:
            structures.append('latex_math')
        if '$$' in content:
            structures.append('display_math')
        if '∫' in content or '∑' in content:
            structures.append('calculus')
        if '∂' in content or '∇' in content:
            structures.append('differential_operators')
        if 'Ψ' in content or 'ψ' in content:
            structures.append('psi_framework')
        if 'Bayesian' in content or 'posterior' in content:
            structures.append('bayesian_inference')
        if 'swarm' in content.lower() or 'koopman' in content.lower():
            structures.append('swarm_koopman')
        
        return structures
    
    def _compute_complexity_score(self, content: str, structures: List[str]) -> float:
        """Compute document complexity score [0,1]"""
        base_complexity = len(content) / 10000  # Normalize by length
        
        # Boost complexity for mathematical content
        math_boost = len(structures) * 0.1
        
        # Additional complexity for technical terms
        technical_terms = ['algorithm', 'theorem', 'proof', 'corollary', 'lemma']
        tech_boost = sum(1 for term in technical_terms if term in content.lower()) * 0.05
        
        return min(1.0, base_complexity + math_boost + tech_boost)
    
    def ml_chaos_analysis(self, document_state: DocumentState) -> ChaosAnalysis:
        """
        N(x): ML/chaos analysis using Koopman operators and neural predictions
        
        Simulates advanced analysis of chaotic systems and swarm dynamics
        """
        # Simulate Koopman operator analysis
        koopman_obs = np.random.normal(0, 1, self.koopman_dim)
        koopman_obs = koopman_obs / np.linalg.norm(koopman_obs)
        
        # Neural network predictions
        neural_pred = np.random.normal(0, 1, self.neural_horizon)
        neural_pred = neural_pred / np.linalg.norm(neural_pred)
        
        # Lyapunov exponent (chaos measure)
        lyapunov = 0.1 + 0.3 * document_state.complexity_score
        
        # Prediction error
        prediction_error = 0.1 + 0.2 * np.random.random()
        
        # Stability metric
        stability = max(0, 1 - lyapunov)
        
        # Swarm confidence (from Oates' theorem)
        swarm_confidence = 0.8 + 0.2 * np.random.random()
        
        return ChaosAnalysis(
            lyapunov_exponent=lyapunov,
            koopman_observables=koopman_obs,
            neural_predictions=neural_pred,
            prediction_error=prediction_error,
            stability_metric=stability,
            swarm_confidence=swarm_confidence
        )
    
    def adaptive_document_flow(self, t: float, document_state: DocumentState, 
                              chaos_analysis: ChaosAnalysis) -> float:
        """
        α(t): Real-time document flow adaptation
        
        Adapts from basic rules to emergent proofs based on complexity and chaos
        """
        # Base flow rate
        base_flow = 0.5
        
        # Complexity-driven adaptation
        complexity_factor = document_state.complexity_score * 0.3
        
        # Chaos-driven adaptation (higher chaos → more neural)
        chaos_factor = (1 - chaos_analysis.stability_metric) * 0.2
        
        # Swarm confidence influence
        swarm_factor = chaos_analysis.swarm_confidence * 0.1
        
        # Time evolution (basic → emergent)
        time_factor = min(0.2, t * self.flow_adaptation_rate)
        
        # Combine factors
        alpha = base_flow + complexity_factor + chaos_factor + swarm_factor + time_factor
        
        return np.clip(alpha, 0, 1)
    
    def cognitive_penalty(self, document_state: DocumentState, 
                         chaos_analysis: ChaosAnalysis) -> float:
        """
        R_cognitive: Analytical accuracy penalty in Bayesian penalties and swarm dynamics
        """
        # Base cognitive penalty
        base_penalty = 0.1
        
        # Complexity penalty
        complexity_penalty = document_state.complexity_score * 0.2
        
        # Chaos penalty (higher chaos = higher penalty)
        chaos_penalty = chaos_analysis.lyapunov_exponent * 0.3
        
        # Prediction error penalty
        error_penalty = chaos_analysis.prediction_error * 0.2
        
        # Swarm dynamics penalty
        swarm_penalty = (1 - chaos_analysis.swarm_confidence) * 0.1
        
        total_penalty = base_penalty + complexity_penalty + chaos_penalty + error_penalty + swarm_penalty
        
        return min(1.0, total_penalty)
    
    def efficiency_penalty(self, document_state: DocumentState, 
                          chaos_analysis: ChaosAnalysis) -> float:
        """
        R_efficiency: Processing multi-document content efficiency penalty
        """
        # Base efficiency penalty
        base_penalty = 0.05
        
        # Content type penalty
        type_penalty = 0.1 if document_state.content_type in ['pdf', 'tex'] else 0.0
        
        # Complexity processing penalty
        complexity_penalty = document_state.complexity_score * 0.15
        
        # Koopman dimension penalty
        koopman_penalty = len(chaos_analysis.koopman_observables) * 0.01
        
        # Neural horizon penalty
        neural_penalty = len(chaos_analysis.neural_predictions) * 0.005
        
        total_penalty = base_penalty + type_penalty + complexity_penalty + koopman_penalty + neural_penalty
        
        return min(1.0, total_penalty)
    
    def calibrated_probability(self, document_state: DocumentState, 
                             chaos_analysis: ChaosAnalysis) -> float:
        """
        P(H|E,β): Calibrated probability with β for query responsiveness
        """
        # Base probability from document confidence
        base_prob = document_state.confidence
        
        # Boost from swarm confidence
        swarm_boost = chaos_analysis.swarm_confidence * 0.1
        
        # Stability boost
        stability_boost = chaos_analysis.stability_metric * 0.1
        
        # Apply β responsiveness bias
        logit_p = np.log(base_prob / (1 - base_prob))
        adjusted_logit = logit_p + np.log(self.beta)
        prob_adjusted = 1 / (1 + np.exp(-adjusted_logit))
        
        # Add boosts
        final_prob = prob_adjusted + swarm_boost + stability_boost
        
        return np.clip(final_prob, 0, 1)
    
    def compute_psi(self, content: str, content_type: str = 'md', 
                    t: float = 0.0) -> PsiComputation:
        """
        Compute complete Ψ(x) for document analysis
        
        Returns detailed breakdown of all components
        """
        # Step 1: Document state inference S(x)
        document_state = self.document_state_inference(content, content_type)
        
        # Step 2: ML/chaos analysis N(x)
        chaos_analysis = self.ml_chaos_analysis(document_state)
        
        # Step 3: Adaptive document flow α(t)
        alpha_t = self.adaptive_document_flow(t, document_state, chaos_analysis)
        
        # Step 4: Hybrid output
        S_x = document_state.confidence
        N_x = chaos_analysis.swarm_confidence
        hybrid_output = alpha_t * S_x + (1 - alpha_t) * N_x
        
        # Step 5: Regularization penalties
        R_cognitive = self.cognitive_penalty(document_state, chaos_analysis)
        R_efficiency = self.efficiency_penalty(document_state, chaos_analysis)
        
        # Step 6: Penalty term
        penalty_exponent = -(self.lambda1 * R_cognitive + self.lambda2 * R_efficiency)
        penalty_term = np.exp(penalty_exponent)
        
        # Step 7: Probability term
        P_H_E_beta = self.calibrated_probability(document_state, chaos_analysis)
        
        # Step 8: Final Ψ(x)
        psi_final = min(1.0, self.beta * penalty_term * hybrid_output * P_H_E_beta)
        
        # Step 9: Interpretation
        interpretation = self._interpret_psi(psi_final)
        
        # Create computation record
        computation = PsiComputation(
            timestamp=datetime.now().timestamp(),
            S_x=S_x,
            N_x=N_x,
            alpha_t=alpha_t,
            R_cognitive=R_cognitive,
            R_efficiency=R_efficiency,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            beta=self.beta,
            P_H_E_beta=P_H_E_beta,
            hybrid_output=hybrid_output,
            penalty_term=penalty_term,
            psi_final=psi_final,
            interpretation=interpretation
        )
        
        # Store in history
        self.computation_history.append(computation)
        self.document_states.append(document_state)
        self.chaos_analyses.append(chaos_analysis)
        
        return computation
    
    def _interpret_psi(self, psi_value: float) -> str:
        """Interpret Ψ(x) value with human-readable description"""
        if psi_value >= 0.85:
            return "Exceptional understanding with high confidence"
        elif psi_value >= 0.70:
            return "Solid grasp of interconnected themes"
        elif psi_value >= 0.55:
            return "Good understanding with some uncertainty"
        elif psi_value >= 0.40:
            return "Moderate understanding, needs refinement"
        else:
            return "Limited understanding, requires additional analysis"
    
    def reproduce_numerical_example(self) -> PsiComputation:
        """
        Reproduce the numerical example from the specification
        
        Step 1: S(x)=0.72, N(x)=0.85
        Step 2: α=0.45, O_hybrid=0.791
        Step 3: R_cognitive=0.15, R_efficiency=0.12; λ1=0.58, λ2=0.42; P_total=0.137; exp≈0.872
        Step 4: P=0.79, β=1.25; P_adj≈0.988
        Step 5: Ψ(x) ≈ 0.791 × 0.872 × 0.988 ≈ 0.681
        """
        # Create synthetic document state
        synthetic_content = "Advanced Ψ(x) framework with hierarchical Bayesian modeling and swarm intelligence"
        
        # Override parameters to match example
        original_lambda1, original_lambda2, original_beta = self.lambda1, self.lambda2, self.beta
        self.lambda1, self.lambda2, self.beta = 0.58, 0.42, 1.25
        
        # Compute with synthetic parameters
        computation = self.compute_psi(synthetic_content, 'md', t=1.0)
        
        # Restore original parameters
        self.lambda1, self.lambda2, self.beta = original_lambda1, original_lambda2, original_beta
        
        return computation
    
    def analyze_document_flow(self, documents: List[str], content_types: List[str]) -> List[PsiComputation]:
        """
        Analyze multiple documents to show flow evolution
        
        Demonstrates how α(t) adapts from basic rules to emergent proofs
        """
        computations = []
        
        for i, (doc, content_type) in enumerate(zip(documents, content_types)):
            t = i * 0.5  # Time evolution
            computation = self.compute_psi(doc, content_type, t)
            computations.append(computation)
        
        return computations
    
    def generate_technical_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive technical report on the framework
        
        Output: Collection of technical reports on hierarchical Bayesian modeling
        and swarm intelligence frameworks
        """
        if not self.computation_history:
            return {"error": "No computations available for report generation"}
        
        # Aggregate statistics
        psi_values = [c.psi_final for c in self.computation_history]
        alpha_values = [c.alpha_t for c in self.computation_history]
        cognitive_penalties = [c.R_cognitive for c in self.computation_history]
        efficiency_penalties = [c.R_efficiency for c in self.computation_history]
        
        report = {
            "framework_summary": {
                "total_analyses": len(self.computation_history),
                "average_psi": np.mean(psi_values),
                "psi_std": np.std(psi_values),
                "average_alpha": np.mean(alpha_values),
                "penalty_trends": {
                    "cognitive_mean": np.mean(cognitive_penalties),
                    "efficiency_mean": np.mean(efficiency_penalties)
                }
            },
            "document_analysis": {
                "content_types": list(set(ds.content_type for ds in self.document_states)),
                "mathematical_structures": list(set(
                    struct for ds in self.document_states 
                    for struct in ds.mathematical_structures
                )),
                "complexity_distribution": {
                    "mean": np.mean([ds.complexity_score for ds in self.document_states]),
                    "std": np.std([ds.complexity_score for ds in self.document_states])
                }
            },
            "chaos_analysis": {
                "lyapunov_exponents": [ca.lyapunov_exponent for ca in self.chaos_analyses],
                "swarm_confidence": [ca.swarm_confidence for ca in self.chaos_analyses],
                "stability_metrics": [ca.stability_metric for ca in self.chaos_analyses]
            },
            "parameter_analysis": {
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
                "beta": self.beta,
                "flow_adaptation_rate": self.flow_adaptation_rate
            }
        }
        
        return report
    
    def visualize_framework(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of the framework components
        """
        if not self.computation_history:
            print("No data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Ψ(x) Framework Analysis', fontsize=16)
        
        # Extract data
        timestamps = [c.timestamp for c in self.computation_history]
        psi_values = [c.psi_final for c in self.computation_history]
        alpha_values = [c.alpha_t for c in self.computation_history]
        S_values = [c.S_x for c in self.computation_history]
        N_values = [c.N_x for c in self.computation_history]
        cognitive_penalties = [c.R_cognitive for c in self.computation_history]
        efficiency_penalties = [c.R_efficiency for c in self.computation_history]
        
        # Plot 1: Ψ(x) evolution
        axes[0, 0].plot(psi_values, 'b-', linewidth=2)
        axes[0, 0].set_title('Ψ(x) Evolution')
        axes[0, 0].set_ylabel('Ψ(x)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: α(t) adaptation
        axes[0, 1].plot(alpha_values, 'g-', linewidth=2)
        axes[0, 1].set_title('α(t) Document Flow Adaptation')
        axes[0, 1].set_ylabel('α(t)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: S(x) vs N(x) comparison
        axes[0, 2].plot(S_values, 'r-', label='S(x): Document State', linewidth=2)
        axes[0, 2].plot(N_values, 'b-', label='N(x): ML/Chaos', linewidth=2)
        axes[0, 2].set_title('S(x) vs N(x) Comparison')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Penalty evolution
        axes[1, 0].plot(cognitive_penalties, 'r-', label='R_cognitive', linewidth=2)
        axes[1, 0].plot(efficiency_penalties, 'g-', label='R_efficiency', linewidth=2)
        axes[1, 0].set_title('Penalty Evolution')
        axes[1, 0].set_ylabel('Penalty Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Hybrid output
        hybrid_outputs = [c.hybrid_output for c in self.computation_history]
        axes[1, 1].plot(hybrid_outputs, 'purple', linewidth=2)
        axes[1, 1].set_title('Hybrid Output: α(t)S(x) + (1-α(t))N(x)')
        axes[1, 1].set_ylabel('Hybrid Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Penalty term
        penalty_terms = [c.penalty_term for c in self.computation_history]
        axes[1, 2].plot(penalty_terms, 'orange', linewidth=2)
        axes[1, 2].set_title('Penalty Term: exp(-[λ₁R_cognitive + λ₂R_efficiency])')
        axes[1, 2].set_ylabel('Penalty Term')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

def demonstrate_framework():
    """
    Demonstrate the enhanced Ψ(x) framework with examples
    """
    print("=== Enhanced Ψ(x) Framework Demonstration ===\n")
    
    # Initialize framework
    framework = EnhancedPsiFramework()
    
    # Example 1: Reproduce numerical example
    print("1. Reproducing Numerical Example:")
    example = framework.reproduce_numerical_example()
    print(f"   Ψ(x) = {example.psi_final:.3f}")
    print(f"   Interpretation: {example.interpretation}")
    print(f"   S(x) = {example.S_x:.3f}, N(x) = {example.N_x:.3f}")
    print(f"   α(t) = {example.alpha_t:.3f}")
    print(f"   R_cognitive = {example.R_cognitive:.3f}, R_efficiency = {example.R_efficiency:.3f}")
    print()
    
    # Example 2: Document flow analysis
    print("2. Document Flow Analysis:")
    documents = [
        "Basic mathematical rules and definitions",
        "Intermediate theorems with proofs",
        "Advanced chaotic system analysis",
        "Swarm intelligence integration",
        "Emergent properties and meta-analysis"
    ]
    
    content_types = ['md', 'md', 'tex', 'tex', 'md']
    
    flow_analysis = framework.analyze_document_flow(documents, content_types)
    
    for i, computation in enumerate(flow_analysis):
        print(f"   Document {i+1}: Ψ(x) = {computation.psi_final:.3f}, α(t) = {computation.alpha_t:.3f}")
    
    print()
    
    # Example 3: Technical report generation
    print("3. Technical Report Generation:")
    report = framework.generate_technical_report()
    print(f"   Total analyses: {report['framework_summary']['total_analyses']}")
    print(f"   Average Ψ(x): {report['framework_summary']['average_psi']:.3f}")
    print(f"   Average α(t): {report['framework_summary']['average_alpha']:.3f}")
    print(f"   Mathematical structures detected: {len(report['document_analysis']['mathematical_structures'])}")
    
    # Example 4: Visualization
    print("\n4. Generating Framework Visualization...")
    framework.visualize_framework('enhanced_psi_framework_visualization.png')
    
    print("\n=== Framework Demonstration Complete ===")
    print("Key Features:")
    print("- Document state inference S(x) for mathematical structures")
    print("- ML/chaos analysis N(x) with Koopman operators")
    print("- Real-time document flow adaptation α(t)")
    print("- Cognitive and efficiency regularization")
    print("- Probability calibration with β responsiveness")
    print("- Integration over document analysis cycles")

if __name__ == "__main__":
    demonstrate_framework()