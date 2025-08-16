#!/usr/bin/env python3
"""
Enhanced Ψ(x) Framework: Minimal Implementation

Extends the existing MinimalHybridFunctional with:
- Hierarchical Bayesian modeling
- Swarm intelligence for chaotic systems
- Document flow optimization
- Enhanced regularization and probability

Built on the working minimal_hybrid_functional.py foundation.
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentState:
    """Document state for inference analysis"""
    content_type: str
    mathematical_structures: List[str]
    complexity_score: float
    confidence: float

@dataclass
class SwarmState:
    """Swarm intelligence state"""
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float

class HierarchicalBayesianModel:
    """Hierarchical Bayesian model for probability estimation"""
    
    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels
        self.priors = {i: {'mu': 0.0, 'sigma': 1.0} for i in range(n_levels)}
        self.posteriors = {i: {'mu': 0.0, 'sigma': 1.0} for i in range(n_levels)}
        self.evidence = {i: [] for i in range(n_levels)}
    
    def update_posterior(self, level: int, data: List[float]):
        """Update posterior distribution at given level"""
        if level >= self.n_levels:
            return
        
        if not data:
            return
        
        prior = self.priors[level]
        n = len(data)
        data_mean = sum(data) / n
        data_var = sum((x - data_mean) ** 2 for x in data) / max(1, n - 1)
        
        # Simple conjugate update
        post_precision = 1/prior['sigma']**2 + n/max(data_var, 0.01)
        post_sigma = 1/math.sqrt(post_precision)
        post_mu = (prior['mu']/prior['sigma']**2 + n*data_mean/max(data_var, 0.01)) / post_precision
        
        self.posteriors[level] = {'mu': post_mu, 'sigma': post_sigma}
        self.evidence[level].extend(data)
    
    def get_level_confidence(self, level: int) -> float:
        """Get confidence level for a hierarchy level"""
        if level >= self.n_levels:
            return 0.0
        
        evidence_count = len(self.evidence[level])
        if evidence_count == 0:
            return 0.5  # Neutral prior
        
        # Confidence increases with evidence
        confidence = min(0.9, 0.3 + 0.6 * evidence_count / 20)
        return confidence

class SwarmIntelligence:
    """Swarm intelligence for chaotic systems analysis"""
    
    def __init__(self, n_agents: int = 50, dimension: int = 3):
        self.n_agents = n_agents
        self.dimension = dimension
        self.agents = []
        self.global_best = None
        self.global_best_fitness = float('inf')
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize swarm agents"""
        for _ in range(self.n_agents):
            agent = SwarmState(
                position=[random.uniform(-5, 5) for _ in range(self.dimension)],
                velocity=[random.uniform(-1, 1) for _ in range(self.dimension)],
                best_position=[],
                best_fitness=float('inf')
            )
            agent.best_position = agent.position.copy()
            self.agents.append(agent)
    
    def objective_function(self, position: List[float]) -> float:
        """Rastrigin function - multimodal, chaotic behavior"""
        A = 10
        n = len(position)
        return A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in position)
    
    def update_swarm(self, iteration: int):
        """Update swarm using PSO algorithm"""
        w, c1, c2 = 0.7, 1.5, 1.5  # Inertia, cognitive, social
        
        for agent in self.agents:
            # Update velocity
            for d in range(self.dimension):
                r1, r2 = random.random(), random.random()
                
                cognitive = c1 * r1 * (agent.best_position[d] - agent.position[d])
                social = c2 * r2 * (self.global_best[d] - agent.position[d]) if self.global_best else 0
                
                agent.velocity[d] = w * agent.velocity[d] + cognitive + social
            
            # Update position
            for d in range(self.dimension):
                agent.position[d] += agent.velocity[d]
            
            # Evaluate fitness
            fitness = self.objective_function(agent.position)
            
            # Update personal best
            if fitness < agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = agent.position.copy()
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = agent.position.copy()
    
    def get_swarm_confidence(self) -> float:
        """Get swarm confidence based on convergence"""
        if self.global_best is None:
            return 0.5
        
        # Confidence based on fitness and diversity
        fitness_confidence = 1 / (1 + abs(self.global_best_fitness))
        
        # Diversity measure
        positions = [agent.position for agent in self.agents]
        center = [sum(pos[i] for pos in positions) / len(positions) for i in range(self.dimension)]
        distances = [math.sqrt(sum((pos[i] - center[i])**2 for i in range(self.dimension))) for pos in positions]
        diversity = sum(distances) / len(distances)
        diversity_confidence = 1 / (1 + diversity)
        
        return (fitness_confidence + diversity_confidence) / 2

class DocumentFlowOptimizer:
    """Document flow optimization system"""
    
    def __init__(self):
        self.flow_history = []
    
    def optimize_flow(self, complexity: float, chaos_level: float, 
                     time_step: float, swarm_confidence: float) -> float:
        """Optimize document flow based on multiple factors"""
        # Base flow rate
        base_flow = 0.5
        
        # Complexity-driven adaptation
        complexity_factor = complexity * 0.3
        
        # Chaos-driven adaptation (higher chaos → more neural)
        chaos_factor = chaos_level * 0.2
        
        # Swarm intelligence influence
        swarm_factor = swarm_confidence * 0.1
        
        # Time evolution (basic → emergent)
        time_factor = min(0.2, time_step * 0.1)
        
        # Combine factors
        optimal_alpha = base_flow + complexity_factor + chaos_factor + swarm_factor + time_factor
        
        # Record flow decision
        self.flow_history.append({
            'time_step': time_step,
            'complexity': complexity,
            'chaos_level': chaos_level,
            'swarm_confidence': swarm_confidence,
            'optimal_alpha': optimal_alpha
        })
        
        return max(0, min(1, optimal_alpha))

class EnhancedPsiFramework:
    """
    Enhanced Ψ(x) Framework integrating all components
    
    Extends MinimalHybridFunctional with:
    - Hierarchical Bayesian modeling
    - Swarm intelligence
    - Document flow optimization
    - Enhanced regularization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Core parameters
        self.lambda1 = self.config.get('lambda1', 0.58)
        self.lambda2 = self.config.get('lambda2', 0.42)
        self.beta = self.config.get('beta', 1.25)
        
        # Component systems
        self.bayesian_model = HierarchicalBayesianModel(n_levels=3)
        self.swarm_intelligence = SwarmIntelligence(n_agents=50, dimension=3)
        self.flow_optimizer = DocumentFlowOptimizer()
        
        # History tracking
        self.computation_history = []
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'lambda1': 0.58,
            'lambda2': 0.42,
            'beta': 1.25,
            'swarm_iterations': 30
        }
    
    def document_state_inference(self, content: str, content_type: str = 'md') -> DocumentState:
        """S(x): Document state inference for mathematical structures"""
        # Detect mathematical structures
        structures = []
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
        
        # Compute complexity score
        base_complexity = len(content) / 10000
        math_boost = len(structures) * 0.1
        technical_terms = ['algorithm', 'theorem', 'proof', 'corollary', 'lemma']
        tech_boost = sum(1 for term in technical_terms if term in content.lower()) * 0.05
        
        complexity = min(1.0, base_complexity + math_boost + tech_boost)
        
        # Confidence based on structure detection
        confidence = min(0.9, 0.3 + 0.6 * len(structures) / 10)
        
        return DocumentState(
            content_type=content_type,
            mathematical_structures=structures,
            complexity_score=complexity,
            confidence=confidence
        )
    
    def ml_chaos_analysis(self, document_state: DocumentState) -> float:
        """N(x): ML/chaos analysis using swarm intelligence"""
        # Update swarm intelligence
        for iteration in range(self.config.get('swarm_iterations', 30)):
            self.swarm_intelligence.update_swarm(iteration)
        
        # Get swarm confidence
        swarm_confidence = self.swarm_intelligence.get_swarm_confidence()
        
        # Combine with document complexity
        chaos_factor = document_state.complexity_score * 0.3
        neural_confidence = swarm_confidence * (1 - chaos_factor) + chaos_factor * 0.8
        
        return max(0, min(1, neural_confidence))
    
    def compute_enhanced_psi(self, content: str, content_type: str = 'md', 
                            t: float = 0.0) -> Dict[str, float]:
        """
        Compute enhanced Ψ(x) with all components
        
        Returns complete breakdown of computation
        """
        # Step 1: Document state inference S(x)
        document_state = self.document_state_inference(content, content_type)
        S_x = document_state.confidence
        
        # Step 2: ML/chaos analysis N(x)
        N_x = self.ml_chaos_analysis(document_state)
        
        # Step 3: Document flow optimization α(t)
        complexity = document_state.complexity_score
        chaos_level = 1 - N_x  # Inverse of neural confidence
        swarm_confidence = N_x
        alpha_t = self.flow_optimizer.optimize_flow(complexity, chaos_level, t, swarm_confidence)
        
        # Step 4: Hybrid output
        hybrid_output = alpha_t * S_x + (1 - alpha_t) * N_x
        
        # Step 5: Enhanced regularization penalties
        R_cognitive = self._compute_cognitive_penalty(document_state, N_x)
        R_efficiency = self._compute_efficiency_penalty(document_state, N_x)
        
        # Step 6: Penalty term
        penalty_exponent = -(self.lambda1 * R_cognitive + self.lambda2 * R_efficiency)
        penalty_term = math.exp(penalty_exponent)
        
        # Step 7: Enhanced probability
        P_H_E_beta = self._compute_enhanced_probability(document_state, N_x)
        
        # Step 8: Final Ψ(x)
        psi_final = min(1.0, self.beta * penalty_term * hybrid_output * P_H_E_beta)
        
        # Store computation
        computation = {
            'timestamp': datetime.now().timestamp(),
            'S_x': S_x,
            'N_x': N_x,
            'alpha_t': alpha_t,
            'R_cognitive': R_cognitive,
            'R_efficiency': R_efficiency,
            'hybrid_output': hybrid_output,
            'penalty_term': penalty_term,
            'P_H_E_beta': P_H_E_beta,
            'psi_final': psi_final,
            'interpretation': self._interpret_psi(psi_final)
        }
        
        self.computation_history.append(computation)
        
        return computation
    
    def _compute_cognitive_penalty(self, document_state: DocumentState, neural_confidence: float) -> float:
        """R_cognitive: Enhanced cognitive penalty"""
        base_penalty = 0.1
        complexity_penalty = document_state.complexity_score * 0.2
        chaos_penalty = (1 - neural_confidence) * 0.3
        swarm_penalty = (1 - neural_confidence) * 0.1
        
        total_penalty = base_penalty + complexity_penalty + chaos_penalty + swarm_penalty
        return min(1.0, total_penalty)
    
    def _compute_efficiency_penalty(self, document_state: DocumentState, neural_confidence: float) -> float:
        """R_efficiency: Enhanced efficiency penalty"""
        base_penalty = 0.05
        type_penalty = 0.1 if document_state.content_type in ['pdf', 'tex'] else 0.0
        complexity_penalty = document_state.complexity_score * 0.15
        neural_penalty = (1 - neural_confidence) * 0.1
        
        total_penalty = base_penalty + type_penalty + complexity_penalty + neural_penalty
        return min(1.0, total_penalty)
    
    def _compute_enhanced_probability(self, document_state: DocumentState, neural_confidence: float) -> float:
        """P(H|E,β): Enhanced probability with Bayesian integration"""
        # Base probability from document confidence
        base_prob = document_state.confidence
        
        # Bayesian model confidence boost
        bayesian_boost = sum(
            self.bayesian_model.get_level_confidence(i) 
            for i in range(self.bayesian_model.n_levels)
        ) / self.bayesian_model.n_levels * 0.1
        
        # Swarm confidence boost
        swarm_boost = neural_confidence * 0.1
        
        # Apply β responsiveness bias
        logit_p = math.log(base_prob / (1 - base_prob))
        adjusted_logit = logit_p + math.log(self.beta)
        prob_adjusted = 1 / (1 + math.exp(-adjusted_logit))
        
        # Add boosts
        final_prob = prob_adjusted + bayesian_boost + swarm_boost
        
        return max(0, min(1, final_prob))
    
    def _interpret_psi(self, psi_value: float) -> str:
        """Interpret Ψ(x) value"""
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
    
    def reproduce_numerical_example(self) -> Dict[str, float]:
        """Reproduce the numerical example from specification"""
        # Create synthetic document
        synthetic_content = "Advanced Ψ(x) framework with hierarchical Bayesian modeling and swarm intelligence"
        
        # Override parameters to match example
        original_lambda1, original_lambda2, original_beta = self.lambda1, self.lambda2, self.beta
        self.lambda1, self.lambda2, self.beta = 0.58, 0.42, 1.25
        
        # Compute with synthetic parameters
        result = self.compute_enhanced_psi(synthetic_content, 'md', t=1.0)
        
        # Restore original parameters
        self.lambda1, self.lambda2, self.beta = original_lambda1, original_lambda2, original_beta
        
        return result

def demonstrate_enhanced_framework():
    """Demonstrate the enhanced Ψ(x) framework"""
    print("=== Enhanced Ψ(x) Framework Demonstration ===\n")
    
    # Initialize framework
    framework = EnhancedPsiFramework()
    
    # Example 1: Reproduce numerical example
    print("1. Reproducing Numerical Example:")
    example = framework.reproduce_numerical_example()
    print(f"   Ψ(x) = {example['psi_final']:.3f}")
    print(f"   Interpretation: {example['interpretation']}")
    print(f"   S(x) = {example['S_x']:.3f}, N(x) = {example['N_x']:.3f}")
    print(f"   α(t) = {example['alpha_t']:.3f}")
    print(f"   R_cognitive = {example['R_cognitive']:.3f}, R_efficiency = {example['R_efficiency']:.3f}")
    print()
    
    # Example 2: Document analysis
    print("2. Document Analysis:")
    documents = [
        "Basic mathematical rules and definitions",
        "Intermediate theorems with proofs",
        "Advanced chaotic system analysis using Koopman operators",
        "Swarm intelligence integration with hierarchical Bayesian models"
    ]
    
    content_types = ['md', 'md', 'tex', 'md']
    
    for i, (doc, content_type) in enumerate(zip(documents, content_types)):
        t = i * 0.5
        result = framework.compute_enhanced_psi(doc, content_type, t)
        print(f"   Document {i+1}: Ψ(x) = {result['psi_final']:.3f}, α(t) = {result['alpha_t']:.3f}")
    
    # Example 3: Framework summary
    print("\n3. Framework Summary:")
    print(f"   Total analyses: {len(framework.computation_history)}")
    avg_psi = sum(r['psi_final'] for r in framework.computation_history) / len(framework.computation_history)
    print(f"   Average Ψ(x): {avg_psi:.3f}")
    
    print("\n=== Enhanced Framework Demonstration Complete ===")
    print("Key Features:")
    print("- Document state inference S(x) for mathematical structures")
    print("- ML/chaos analysis N(x) with swarm intelligence")
    print("- Real-time document flow adaptation α(t)")
    print("- Enhanced cognitive and efficiency regularization")
    print("- Hierarchical Bayesian integration")
    print("- Probability calibration with β responsiveness")

if __name__ == "__main__":
    demonstrate_enhanced_framework()
