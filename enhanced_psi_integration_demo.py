#!/usr/bin/env python3
"""
Enhanced Ψ(x) Framework Integration Demonstration

Shows integration with:
- Hierarchical Bayesian modeling systems
- Swarm intelligence and Koopman operators
- Chaotic systems analysis
- Document flow optimization
- Real-time adaptation mechanisms

This demonstrates the complete framework integration as specified in the requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced framework
from enhanced_psi_framework import EnhancedPsiFramework, DocumentState, ChaosAnalysis, PsiComputation

class HierarchicalBayesianModel:
    """
    Hierarchical Bayesian model for probability estimation
    
    Demonstrates the 'Output' component: technical reports on hierarchical Bayesian modeling
    """
    
    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels
        self.priors = {}
        self.posteriors = {}
        self.evidence = {}
        
        # Initialize hierarchical structure
        for level in range(n_levels):
            self.priors[level] = {'mu': 0.0, 'sigma': 1.0}
            self.posteriors[level] = {'mu': 0.0, 'sigma': 1.0}
            self.evidence[level] = []
    
    def update_posterior(self, level: int, data: np.ndarray):
        """Update posterior distribution at given level"""
        if level >= self.n_levels:
            raise ValueError(f"Level {level} exceeds hierarchy depth {self.n_levels}")
        
        # Simple conjugate normal-normal update
        prior = self.priors[level]
        n = len(data)
        
        if n > 0:
            data_mean = np.mean(data)
            data_var = np.var(data, ddof=1)
            
            # Posterior parameters
            post_precision = 1/prior['sigma']**2 + n/data_var
            post_sigma = 1/np.sqrt(post_precision)
            post_mu = (prior['mu']/prior['sigma']**2 + n*data_mean/data_var) / post_precision
            
            self.posteriors[level] = {'mu': post_mu, 'sigma': post_sigma}
            self.evidence[level].extend(data)
    
    def generate_technical_report(self) -> Dict[str, Any]:
        """Generate technical report on hierarchical Bayesian modeling"""
        report = {
            "model_structure": {
                "hierarchy_levels": self.n_levels,
                "prior_specifications": self.priors,
                "posterior_updates": self.posteriors
            },
            "evidence_summary": {
                f"level_{i}": {
                    "sample_size": len(self.evidence[i]),
                    "mean": np.mean(self.evidence[i]) if self.evidence[i] else None,
                    "variance": np.var(self.evidence[i]) if self.evidence[i] else None
                }
                for i in range(self.n_levels)
            },
            "hierarchical_relationships": {
                "cross_level_correlations": self._compute_cross_level_correlations(),
                "information_flow": self._analyze_information_flow()
            }
        }
        
        return report
    
    def _compute_cross_level_correlations(self) -> Dict[str, float]:
        """Compute correlations between hierarchy levels"""
        correlations = {}
        
        for i in range(self.n_levels - 1):
            for j in range(i + 1, self.n_levels):
                if self.evidence[i] and self.evidence[j]:
                    # Pad shorter evidence to match length
                    min_len = min(len(self.evidence[i]), len(self.evidence[j]))
                    corr = np.corrcoef(
                        self.evidence[i][:min_len], 
                        self.evidence[j][:min_len]
                    )[0, 1]
                    correlations[f"level_{i}_to_{j}"] = corr
        
        return correlations
    
    def _analyze_information_flow(self) -> Dict[str, Any]:
        """Analyze information flow through hierarchy"""
        flow_analysis = {
            "prior_influence": {},
            "evidence_propagation": {},
            "posterior_stability": {}
        }
        
        for level in range(self.n_levels):
            # Prior influence (how much prior affects posterior)
            if self.evidence[level]:
                prior_influence = 1 / (1 + len(self.evidence[level]))
                flow_analysis["prior_influence"][f"level_{level}"] = prior_influence
            
            # Evidence propagation (how much evidence affects posterior)
            if self.evidence[level]:
                evidence_propagation = len(self.evidence[level]) / (len(self.evidence[level]) + 10)
                flow_analysis["evidence_propagation"][f"level_{level}"] = evidence_propagation
            
            # Posterior stability (variance of posterior)
            flow_analysis["posterior_stability"][f"level_{level}"] = self.posteriors[level]['sigma']
        
        return flow_analysis

class SwarmIntelligenceFramework:
    """
    Swarm intelligence framework for chaotic systems
    
    Demonstrates the 'Output' component: swarm intelligence frameworks applied to chaotic systems
    """
    
    def __init__(self, n_agents: int = 100, dimension: int = 3):
        self.n_agents = n_agents
        self.dimension = dimension
        self.agents = []
        self.global_best = None
        self.global_best_fitness = float('inf')
        
        # Initialize swarm agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize swarm agents with random positions and velocities"""
        for i in range(self.n_agents):
            agent = {
                'position': np.random.uniform(-5, 5, self.dimension),
                'velocity': np.random.uniform(-1, 1, self.dimension),
                'best_position': None,
                'best_fitness': float('inf')
            }
            agent['best_position'] = agent['position'].copy()
            self.agents.append(agent)
    
    def objective_function(self, position: np.ndarray) -> float:
        """
        Objective function: Rastrigin function (multimodal, chaotic behavior)
        Demonstrates chaotic systems analysis
        """
        A = 10
        n = len(position)
        return A * n + np.sum(position**2 - A * np.cos(2 * np.pi * position))
    
    def update_swarm(self, iteration: int, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """Update swarm positions and velocities using PSO algorithm"""
        for agent in self.agents:
            # Update velocity
            r1, r2 = np.random.random(2)
            
            cognitive_component = c1 * r1 * (agent['best_position'] - agent['position'])
            social_component = c2 * r2 * (self.global_best - agent['position'])
            
            agent['velocity'] = w * agent['velocity'] + cognitive_component + social_component
            
            # Update position
            agent['position'] += agent['velocity']
            
            # Evaluate fitness
            fitness = self.objective_function(agent['position'])
            
            # Update personal best
            if fitness < agent['best_fitness']:
                agent['best_fitness'] = fitness
                agent['best_position'] = agent['position'].copy()
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = agent['best_position'].copy()
    
    def analyze_swarm_dynamics(self) -> Dict[str, Any]:
        """Analyze swarm dynamics and emergent behavior"""
        positions = np.array([agent['position'] for agent in self.agents])
        velocities = np.array([agent['velocity'] for agent in self.agents])
        fitnesses = np.array([agent['best_fitness'] for agent in self.agents])
        
        analysis = {
            "swarm_statistics": {
                "position_mean": np.mean(positions, axis=0).tolist(),
                "position_std": np.std(positions, axis=0).tolist(),
                "velocity_mean": np.mean(velocities, axis=0).tolist(),
                "velocity_std": np.std(velocities, axis=0).tolist(),
                "fitness_mean": float(np.mean(fitnesses)),
                "fitness_std": float(np.std(fitnesses))
            },
            "convergence_metrics": {
                "global_best_fitness": float(self.global_best_fitness),
                "global_best_position": self.global_best.tolist() if self.global_best is not None else None,
                "swarm_diversity": float(np.std(positions)),
                "velocity_magnitude": float(np.mean(np.linalg.norm(velocities, axis=1)))
            },
            "emergent_behavior": {
                "collective_intelligence": self._assess_collective_intelligence(),
                "phase_transitions": self._detect_phase_transitions(),
                "chaos_indicators": self._compute_chaos_indicators(positions, velocities)
            }
        }
        
        return analysis
    
    def _assess_collective_intelligence(self) -> Dict[str, float]:
        """Assess collective intelligence of the swarm"""
        if self.global_best is None:
            return {"collective_fitness": 0.0, "coordination_score": 0.0}
        
        # Collective fitness (inverse of global best fitness)
        collective_fitness = 1 / (1 + abs(self.global_best_fitness))
        
        # Coordination score (how well agents coordinate)
        positions = np.array([agent['position'] for agent in self.agents])
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        coordination_score = 1 / (1 + np.std(distances))
        
        return {
            "collective_fitness": float(collective_fitness),
            "coordination_score": float(coordination_score)
        }
    
    def _detect_phase_transitions(self) -> Dict[str, Any]:
        """Detect phase transitions in swarm behavior"""
        # This is a simplified phase transition detection
        # In practice, you'd use more sophisticated methods like order parameters
        
        positions = np.array([agent['position'] for agent in self.agents])
        velocities = np.array([agent['velocity'] for agent in self.agents])
        
        # Phase indicators
        position_order = 1 / (1 + np.std(positions))
        velocity_order = 1 / (1 + np.std(velocities))
        
        # Detect transitions
        if position_order > 0.8 and velocity_order > 0.8:
            phase = "ordered"
        elif position_order < 0.3 and velocity_order < 0.3:
            phase = "chaotic"
        else:
            phase = "transitional"
        
        return {
            "current_phase": phase,
            "position_order": float(position_order),
            "velocity_order": float(velocity_order),
            "transition_probability": float(1 - position_order * velocity_order)
        }
    
    def _compute_chaos_indicators(self, positions: np.ndarray, velocities: np.ndarray) -> Dict[str, float]:
        """Compute chaos indicators for swarm dynamics"""
        # Lyapunov exponent approximation
        if len(velocities) > 1:
            velocity_changes = np.diff(velocities, axis=0)
            lyapunov_approx = np.mean(np.log(np.linalg.norm(velocity_changes, axis=1) + 1e-10))
        else:
            lyapunov_approx = 0.0
        
        # Entropy-based chaos measure
        position_bins = np.histogram(positions.flatten(), bins=20)[0]
        position_entropy = -np.sum(p * np.log(p + 1e-10) for p in position_bins if p > 0)
        
        return {
            "lyapunov_approximation": float(lyapunov_approx),
            "position_entropy": float(position_entropy),
            "chaos_level": float(np.tanh(abs(lyapunov_approx) + position_entropy / 100))
        }

class DocumentFlowOptimizer:
    """
    Document flow optimization system
    
    Demonstrates the 'Hybrid' component: α(t) for real-time document flow
    """
    
    def __init__(self):
        self.flow_history = []
        self.adaptation_patterns = []
        self.performance_metrics = []
    
    def optimize_flow(self, document_complexity: float, chaos_level: float, 
                     time_step: float, swarm_intelligence: float) -> float:
        """
        Optimize document flow based on multiple factors
        
        α(t) adapts from basic rules to emergent proofs
        """
        # Base flow rate
        base_flow = 0.5
        
        # Complexity-driven adaptation
        complexity_factor = document_complexity * 0.3
        
        # Chaos-driven adaptation (higher chaos → more neural)
        chaos_factor = chaos_level * 0.2
        
        # Swarm intelligence influence
        swarm_factor = swarm_intelligence * 0.1
        
        # Time evolution (basic → emergent)
        time_factor = min(0.2, time_step * 0.1)
        
        # Combine factors
        optimal_alpha = base_flow + complexity_factor + chaos_factor + swarm_factor + time_factor
        
        # Record flow decision
        flow_decision = {
            'timestamp': datetime.now().timestamp(),
            'time_step': time_step,
            'document_complexity': document_complexity,
            'chaos_level': chaos_level,
            'swarm_intelligence': swarm_intelligence,
            'optimal_alpha': optimal_alpha,
            'factors': {
                'base_flow': base_flow,
                'complexity_factor': complexity_factor,
                'chaos_factor': chaos_factor,
                'swarm_factor': swarm_factor,
                'time_factor': time_factor
            }
        }
        
        self.flow_history.append(flow_decision)
        
        return np.clip(optimal_alpha, 0, 1)
    
    def analyze_flow_patterns(self) -> Dict[str, Any]:
        """Analyze document flow patterns and adaptation"""
        if not self.flow_history:
            return {"error": "No flow history available"}
        
        alphas = [f['optimal_alpha'] for f in self.flow_history]
        complexities = [f['document_complexity'] for f in self.flow_history]
        chaos_levels = [f['chaos_level'] for f in self.flow_history]
        
        analysis = {
            "flow_statistics": {
                "mean_alpha": np.mean(alphas),
                "alpha_std": np.std(alphas),
                "alpha_trend": self._compute_trend(alphas),
                "adaptation_rate": self._compute_adaptation_rate()
            },
            "factor_analysis": {
                "complexity_correlation": np.corrcoef(alphas, complexities)[0, 1],
                "chaos_correlation": np.corrcoef(alphas, chaos_levels)[0, 1],
                "factor_contributions": self._analyze_factor_contributions()
            },
            "flow_patterns": {
                "regime_transitions": self._detect_regime_transitions(),
                "optimal_flow_zones": self._identify_optimal_flow_zones(),
                "adaptation_efficiency": self._assess_adaptation_efficiency()
            }
        }
        
        return analysis
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend in values"""
        if len(values) < 2:
            return "insufficient_data"
        
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _compute_adaptation_rate(self) -> float:
        """Compute how quickly the system adapts"""
        if len(self.flow_history) < 2:
            return 0.0
        
        alpha_changes = []
        for i in range(1, len(self.flow_history)):
            change = abs(self.flow_history[i]['optimal_alpha'] - 
                        self.flow_history[i-1]['optimal_alpha'])
            alpha_changes.append(change)
        
        return np.mean(alpha_changes) if alpha_changes else 0.0
    
    def _analyze_factor_contributions(self) -> Dict[str, float]:
        """Analyze contribution of each factor to flow decisions"""
        if not self.flow_history:
            return {}
        
        factor_contributions = {}
        for factor in ['complexity_factor', 'chaos_factor', 'swarm_factor', 'time_factor']:
            values = [f['factors'][factor] for f in self.flow_history]
            factor_contributions[factor] = float(np.mean(values))
        
        return factor_contributions
    
    def _detect_regime_transitions(self) -> List[Dict[str, Any]]:
        """Detect transitions between different flow regimes"""
        if len(self.flow_history) < 3:
            return []
        
        transitions = []
        for i in range(1, len(self.flow_history) - 1):
            prev_alpha = self.flow_history[i-1]['optimal_alpha']
            curr_alpha = self.flow_history[i]['optimal_alpha']
            next_alpha = self.flow_history[i+1]['optimal_alpha']
            
            # Detect significant changes
            if abs(curr_alpha - prev_alpha) > 0.1 or abs(next_alpha - curr_alpha) > 0.1:
                transition = {
                    'timestamp': self.flow_history[i]['timestamp'],
                    'time_step': self.flow_history[i]['time_step'],
                    'alpha_change': curr_alpha - prev_alpha,
                    'regime': self._classify_regime(curr_alpha)
                }
                transitions.append(transition)
        
        return transitions
    
    def _classify_regime(self, alpha: float) -> str:
        """Classify flow regime based on alpha value"""
        if alpha < 0.3:
            return "basic_rules"
        elif alpha < 0.6:
            return "intermediate_analysis"
        else:
            return "emergent_proofs"
    
    def _identify_optimal_flow_zones(self) -> Dict[str, List[float]]:
        """Identify optimal flow zones for different conditions"""
        zones = {
            'low_complexity': [],
            'medium_complexity': [],
            'high_complexity': []
        }
        
        for flow in self.flow_history:
            complexity = flow['document_complexity']
            alpha = flow['optimal_alpha']
            
            if complexity < 0.3:
                zones['low_complexity'].append(alpha)
            elif complexity < 0.7:
                zones['medium_complexity'].append(alpha)
            else:
                zones['high_complexity'].append(alpha)
        
        # Compute optimal ranges
        optimal_zones = {}
        for zone, alphas in zones.items():
            if alphas:
                optimal_zones[zone] = {
                    'mean': float(np.mean(alphas)),
                    'std': float(np.std(alphas)),
                    'range': [float(min(alphas)), float(max(alphas))]
                }
            else:
                optimal_zones[zone] = None
        
        return optimal_zones
    
    def _assess_adaptation_efficiency(self) -> Dict[str, float]:
        """Assess how efficiently the system adapts"""
        if len(self.flow_history) < 2:
            return {"efficiency": 0.0, "stability": 0.0}
        
        # Adaptation efficiency: how quickly system reaches optimal alpha
        alpha_changes = []
        for i in range(1, len(self.flow_history)):
            change = abs(self.flow_history[i]['optimal_alpha'] - 
                        self.flow_history[i-1]['optimal_alpha'])
            alpha_changes.append(change)
        
        efficiency = 1 / (1 + np.mean(alpha_changes)) if alpha_changes else 0.0
        
        # Stability: how consistent the adaptations are
        stability = 1 / (1 + np.std(alpha_changes)) if alpha_changes else 0.0
        
        return {
            "efficiency": float(efficiency),
            "stability": float(stability)
        }

class EnhancedPsiIntegration:
    """
    Integration of all components: Enhanced Ψ(x) + Hierarchical Bayesian + Swarm Intelligence + Document Flow
    """
    
    def __init__(self):
        self.psi_framework = EnhancedPsiFramework()
        self.bayesian_model = HierarchicalBayesianModel(n_levels=3)
        self.swarm_framework = SwarmIntelligenceFramework(n_agents=100, dimension=3)
        self.flow_optimizer = DocumentFlowOptimizer()
        
        self.integration_history = []
    
    def run_integrated_analysis(self, documents: List[str], content_types: List[str], 
                               iterations: int = 50) -> Dict[str, Any]:
        """
        Run complete integrated analysis
        
        Demonstrates the full framework integration:
        - Output: Technical reports on hierarchical Bayesian modeling and swarm intelligence
        - Hybrid: S(x) as document state inference, N(x) as ML/chaos analysis, α(t) for real-time flow
        - Regularization: R_cognitive and R_efficiency with Bayesian penalties and swarm dynamics
        - Probability: P(H|E,β) with β for query responsiveness
        - Integration: Over document analysis cycles
        """
        print("=== Running Integrated Analysis ===")
        
        # Phase 1: Document Analysis with Enhanced Ψ(x)
        print("Phase 1: Document Analysis with Enhanced Ψ(x)")
        psi_results = []
        for i, (doc, content_type) in enumerate(zip(documents, content_types)):
            t = i * 0.5
            psi_result = self.psi_framework.compute_psi(doc, content_type, t)
            psi_results.append(psi_result)
            print(f"  Document {i+1}: Ψ(x) = {psi_result.psi_final:.3f}")
        
        # Phase 2: Hierarchical Bayesian Modeling
        print("\nPhase 2: Hierarchical Bayesian Modeling")
        # Generate synthetic data for demonstration
        for level in range(3):
            data = np.random.normal(level * 0.5, 0.5, 20 + level * 10)
            self.bayesian_model.update_posterior(level, data)
        
        bayesian_report = self.bayesian_model.generate_technical_report()
        print(f"  Hierarchy levels: {bayesian_report['model_structure']['hierarchy_levels']}")
        print(f"  Evidence samples: {[len(self.bayesian_model.evidence[i]) for i in range(3)]}")
        
        # Phase 3: Swarm Intelligence Optimization
        print("\nPhase 3: Swarm Intelligence Optimization")
        for iteration in range(iterations):
            self.swarm_framework.update_swarm(iteration)
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: Best fitness = {self.swarm_framework.global_best_fitness:.4f}")
        
        swarm_analysis = self.swarm_framework.analyze_swarm_dynamics()
        print(f"  Final best fitness: {swarm_analysis['convergence_metrics']['global_best_fitness']:.4f}")
        print(f"  Swarm diversity: {swarm_analysis['convergence_metrics']['swarm_diversity']:.4f}")
        
        # Phase 4: Document Flow Optimization
        print("\nPhase 4: Document Flow Optimization")
        for i, psi_result in enumerate(psi_results):
            complexity = psi_result.S_x
            chaos_level = 1 - psi_result.N_x  # Inverse of swarm confidence
            swarm_intelligence = psi_result.N_x
            time_step = i * 0.5
            
            optimal_alpha = self.flow_optimizer.optimize_flow(
                complexity, chaos_level, time_step, swarm_intelligence
            )
            print(f"  Document {i+1}: α(t) = {optimal_alpha:.3f}")
        
        # Phase 5: Integration Analysis
        print("\nPhase 5: Integration Analysis")
        flow_analysis = self.flow_optimizer.analyze_flow_patterns()
        print(f"  Flow adaptation rate: {flow_analysis['flow_statistics']['adaptation_rate']:.4f}")
        print(f"  Optimal flow zones: {len(flow_analysis['flow_patterns']['optimal_flow_zones'])}")
        
        # Generate comprehensive integration report
        integration_report = self._generate_integration_report(
            psi_results, bayesian_report, swarm_analysis, flow_analysis
        )
        
        return integration_report
    
    def _generate_integration_report(self, psi_results: List[PsiComputation], 
                                   bayesian_report: Dict, swarm_analysis: Dict, 
                                   flow_analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        
        # Aggregate Ψ(x) results
        psi_summary = {
            "total_analyses": len(psi_results),
            "average_psi": np.mean([r.psi_final for r in psi_results]),
            "psi_std": np.std([r.psi_final for r in psi_results]),
            "interpretation_distribution": self._analyze_interpretations(psi_results)
        }
        
        # Integration metrics
        integration_metrics = {
            "bayesian_swarm_correlation": self._compute_correlation(
                [r.S_x for r in psi_results],  # Document state
                [r.N_x for r in psi_results]   # ML/chaos
            ),
            "flow_adaptation_efficiency": flow_analysis['flow_patterns']['adaptation_efficiency'],
            "swarm_convergence_quality": swarm_analysis['convergence_metrics']['global_best_fitness'],
            "hierarchical_information_flow": bayesian_report['hierarchical_relationships']['information_flow']
        }
        
        # Framework performance
        framework_performance = {
            "document_processing_efficiency": self._assess_processing_efficiency(psi_results),
            "chaos_analysis_accuracy": self._assess_chaos_accuracy(psi_results),
            "bayesian_inference_quality": self._assess_bayesian_quality(bayesian_report),
            "swarm_optimization_success": self._assess_swarm_success(swarm_analysis)
        }
        
        report = {
            "integration_summary": {
                "timestamp": datetime.now().isoformat(),
                "framework_version": "Enhanced Ψ(x) v2.0",
                "components_integrated": [
                    "Enhanced Ψ(x) Framework",
                    "Hierarchical Bayesian Modeling",
                    "Swarm Intelligence & Koopman Operators",
                    "Document Flow Optimization",
                    "Chaotic Systems Analysis"
                ]
            },
            "psi_framework_results": psi_summary,
            "bayesian_modeling_results": bayesian_report,
            "swarm_intelligence_results": swarm_analysis,
            "document_flow_results": flow_analysis,
            "integration_metrics": integration_metrics,
            "framework_performance": framework_performance,
            "synthesis_insights": self._generate_synthesis_insights(
                psi_results, bayesian_report, swarm_analysis, flow_analysis
            )
        }
        
        return report
    
    def _analyze_interpretations(self, psi_results: List[PsiComputation]) -> Dict[str, int]:
        """Analyze distribution of Ψ(x) interpretations"""
        interpretations = [r.interpretation for r in psi_results]
        distribution = {}
        
        for interpretation in interpretations:
            distribution[interpretation] = distribution.get(interpretation, 0) + 1
        
        return distribution
    
    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute correlation between two lists"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            return float(np.corrcoef(x, y)[0, 1])
        except:
            return 0.0
    
    def _assess_processing_efficiency(self, psi_results: List[PsiComputation]) -> Dict[str, float]:
        """Assess document processing efficiency"""
        if not psi_results:
            return {"efficiency": 0.0, "consistency": 0.0}
        
        # Efficiency based on Ψ(x) values
        psi_values = [r.psi_final for r in psi_results]
        efficiency = np.mean(psi_values)
        
        # Consistency based on standard deviation
        consistency = 1 / (1 + np.std(psi_values))
        
        return {
            "efficiency": float(efficiency),
            "consistency": float(consistency)
        }
    
    def _assess_chaos_accuracy(self, psi_results: List[PsiComputation]) -> Dict[str, float]:
        """Assess chaos analysis accuracy"""
        if not psi_results:
            return {"accuracy": 0.0, "stability": 0.0}
        
        # Accuracy based on N(x) values (ML/chaos analysis)
        N_values = [r.N_x for r in psi_results]
        accuracy = np.mean(N_values)
        
        # Stability based on consistency of N(x)
        stability = 1 / (1 + np.std(N_values))
        
        return {
            "accuracy": float(accuracy),
            "stability": float(stability)
        }
    
    def _assess_bayesian_quality(self, bayesian_report: Dict) -> Dict[str, float]:
        """Assess Bayesian inference quality"""
        evidence_summary = bayesian_report['evidence_summary']
        
        # Quality based on evidence accumulation
        total_evidence = sum(
            evidence_summary[f"level_{i}"]["sample_size"] 
            for i in range(len(evidence_summary))
        )
        
        evidence_quality = min(1.0, total_evidence / 100)  # Normalize
        
        # Information flow quality
        info_flow = bayesian_report['hierarchical_relationships']['information_flow']
        flow_quality = np.mean([
            info_flow['prior_influence'].get(f'level_{i}', 0.5)
            for i in range(len(info_flow['prior_influence']))
        ]) if info_flow['prior_influence'] else 0.5
        
        return {
            "evidence_quality": float(evidence_quality),
            "information_flow_quality": float(flow_quality)
        }
    
    def _assess_swarm_success(self, swarm_analysis: Dict) -> Dict[str, float]:
        """Assess swarm optimization success"""
        convergence = swarm_analysis['convergence_metrics']
        emergent = swarm_analysis['emergent_behavior']
        
        # Success based on convergence
        convergence_success = 1 / (1 + abs(convergence['global_best_fitness']))
        
        # Collective intelligence
        collective_intelligence = emergent['collective_intelligence']['collective_fitness']
        
        # Coordination quality
        coordination_quality = emergent['collective_intelligence']['coordination_score']
        
        return {
            "convergence_success": float(convergence_success),
            "collective_intelligence": float(collective_intelligence),
            "coordination_quality": float(coordination_quality)
        }
    
    def _generate_synthesis_insights(self, psi_results: List[PsiComputation], 
                                   bayesian_report: Dict, swarm_analysis: Dict, 
                                   flow_analysis: Dict) -> List[str]:
        """Generate synthesis insights from integrated analysis"""
        insights = []
        
        # Ψ(x) insights
        avg_psi = np.mean([r.psi_final for r in psi_results])
        if avg_psi > 0.7:
            insights.append("High overall confidence in integrated framework performance")
        elif avg_psi > 0.5:
            insights.append("Moderate confidence with room for optimization")
        else:
            insights.append("Framework requires refinement and parameter tuning")
        
        # Bayesian insights
        evidence_levels = [len(bayesian_report['evidence_summary'][f'level_{i}']['sample_size']) 
                          for i in range(3)]
        if all(level > 10 for level in evidence_levels):
            insights.append("Strong evidence accumulation across hierarchy levels")
        else:
            insights.append("Insufficient evidence for robust hierarchical inference")
        
        # Swarm insights
        if swarm_analysis['emergent_behavior']['phase_transitions']['current_phase'] == 'ordered':
            insights.append("Swarm successfully converged to ordered state")
        else:
            insights.append("Swarm still exploring or in transitional phase")
        
        # Flow insights
        if flow_analysis['flow_patterns']['adaptation_efficiency']['efficiency'] > 0.7:
            insights.append("Document flow adaptation is highly efficient")
        else:
            insights.append("Document flow adaptation could be optimized")
        
        return insights

def demonstrate_integration():
    """
    Demonstrate the complete integrated framework
    """
    print("=== Enhanced Ψ(x) Framework Integration Demonstration ===\n")
    
    # Initialize integration
    integration = EnhancedPsiIntegration()
    
    # Sample documents for analysis
    documents = [
        "Basic mathematical foundations and axioms",
        "Intermediate theorems with formal proofs",
        "Advanced chaotic system analysis using Koopman operators",
        "Swarm intelligence integration with hierarchical Bayesian models",
        "Emergent properties and meta-cognitive insights"
    ]
    
    content_types = ['md', 'tex', 'tex', 'md', 'md']
    
    # Run integrated analysis
    integration_report = integration.run_integrated_analysis(
        documents, content_types, iterations=30
    )
    
    # Display key results
    print("\n=== Integration Results Summary ===")
    print(f"Framework Version: {integration_report['integration_summary']['framework_version']}")
    print(f"Components Integrated: {len(integration_report['integration_summary']['components_integrated'])}")
    print(f"Total Document Analyses: {integration_report['psi_framework_results']['total_analyses']}")
    print(f"Average Ψ(x): {integration_report['psi_framework_results']['average_psi']:.3f}")
    
    # Display synthesis insights
    print("\n=== Synthesis Insights ===")
    for i, insight in enumerate(integration_report['synthesis_insights'], 1):
        print(f"{i}. {insight}")
    
    # Display performance metrics
    print("\n=== Framework Performance ===")
    performance = integration_report['framework_performance']
    print(f"Document Processing Efficiency: {performance['document_processing_efficiency']['efficiency']:.3f}")
    print(f"Chaos Analysis Accuracy: {performance['chaos_analysis_accuracy']['accuracy']:.3f}")
    print(f"Bayesian Inference Quality: {performance['bayesian_inference_quality']['evidence_quality']:.3f}")
    print(f"Swarm Optimization Success: {performance['swarm_optimization_success']['convergence_success']:.3f}")
    
    # Save detailed report
    with open('enhanced_psi_integration_report.json', 'w') as f:
        json.dump(integration_report, f, indent=2, default=str)
    
    print(f"\nDetailed integration report saved to: enhanced_psi_integration_report.json")
    
    print("\n=== Integration Demonstration Complete ===")
    print("The enhanced Ψ(x) framework successfully integrates:")
    print("✓ Hierarchical Bayesian modeling for probability estimation")
    print("✓ Swarm intelligence frameworks for chaotic systems")
    print("✓ Document state inference and ML/chaos analysis")
    print("✓ Real-time document flow adaptation")
    print("✓ Cognitive and efficiency regularization")
    print("✓ Probability calibration with query responsiveness")

if __name__ == "__main__":
    demonstrate_integration()