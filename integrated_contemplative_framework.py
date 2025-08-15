#!/usr/bin/env python3
"""
Integrated Contemplative Framework
Bridges contemplative AI with existing Integrated Research Conform structure
"""

import sys
import os
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add paths for existing framework imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integrated', 'integrated_research_conform', 'core'))

try:
    from contemplative_visual_grounding import ContemplativeVisualGrounder, ContemplativeStage, ObserverFeedback
    # Try to import existing framework components
    from minimal_contraction_psi import MinimalContractionConfig, MinimalContractionPsi
    from minimal_hybrid_functional import MinimalHybridFunctional
except ImportError as e:
    print(f"Warning: Could not import some framework components: {e}")
    print("This is expected if running outside the full framework directory")

class IntegratedContemplativeFramework:
    """
    Unified framework integrating contemplative AI with proven Ψ mathematics
    
    Combines:
    - Your validated multiplicative Ψ framework (K=0.3625 contraction)
    - Contemplative stage-four insight principles
    - Inclusive observer networks
    - Cultural adaptivity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_integrated_config()
        
        # Initialize core components
        try:
            self.contraction_config = MinimalContractionConfig()
            self.contraction_psi = MinimalContractionPsi(self.contraction_config)
            self.hybrid_functional = MinimalHybridFunctional()
            self.framework_available = True
        except NameError:
            print("Running in standalone mode - core framework components not available")
            self.framework_available = False
        
        # Initialize contemplative components
        self.contemplative_grounder = ContemplativeVisualGrounder(config)
        
        # Observer network for inclusive participation
        self.observer_network = {}
        self.cultural_contexts = {}
        self.accessibility_profiles = {}
        
        # Integration metrics
        self.integration_history = []
        
    def _default_integrated_config(self) -> Dict:
        """Default configuration for integrated framework"""
        return {
            # Core Ψ parameters (from your proven framework)
            'alpha': 0.6,
            'lambda_1': 0.8,
            'lambda_2': 0.7,
            'beta': 1.2,
            
            # Contemplative parameters
            'stage_four_threshold': 0.70,
            'observer_validation_weight': 0.3,
            'cultural_adaptation_enabled': True,
            'accessibility_modes': ['visual', 'auditory', 'tactile', 'symbolic'],
            
            # Integration parameters
            'multiplicative_integration': True,
            'bounds_enforcement': True,
            'contraction_validation': True,
            
            # Inclusive participation
            'peer_observation_enabled': True,
            'expert_rotation_enabled': True,
            'feedback_loop_active': True
        }
    
    def register_observer(self, 
                         observer_id: str, 
                         expertise_level: float,
                         cultural_context: str,
                         accessibility_needs: List[str] = None) -> bool:
        """
        Register an observer in the inclusive network
        Implements your collaborative development framework principles
        """
        observer_profile = {
            'id': observer_id,
            'expertise_level': expertise_level,
            'cultural_context': cultural_context,
            'accessibility_needs': accessibility_needs or [],
            'registration_time': datetime.now().isoformat(),
            'feedback_count': 0,
            'validation_accuracy': 1.0  # Will be updated based on performance
        }
        
        self.observer_network[observer_id] = observer_profile
        return True
    
    def process_contemplative_session(self, 
                                    session_data: Dict[str, Any],
                                    observer_feedbacks: List[ObserverFeedback] = None) -> Dict[str, Any]:
        """
        Process a complete contemplative session using integrated framework
        
        Combines:
        1. Visual grounding for arising/passing detection
        2. Multiplicative Ψ computation for bounded confidence
        3. Observer validation for stage-four insight
        4. Cultural and accessibility adaptations
        """
        session_results = {
            'session_id': session_data.get('session_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'framework_version': 'integrated_contemplative_v1.0'
        }
        
        # Extract visual phenomena if available
        phenomena = session_data.get('visual_phenomena', [])
        
        # Process observer feedbacks
        if observer_feedbacks:
            for feedback in observer_feedbacks:
                self.contemplative_grounder.integrate_observer_feedback(feedback)
        
        # Compute stage-four insight using contemplative framework
        contemplative_insight = self.contemplative_grounder.compute_stage_four_insight(
            phenomena, 
            session_data.get('cultural_context', 'secular')
        )
        
        # Integrate with core Ψ framework if available
        if self.framework_available:
            try:
                # Map contemplative metrics to core framework parameters
                S = contemplative_insight.get('arising_awareness', 0.5)
                N = 1.0 - session_data.get('uncertainty', 0.2)
                
                # Compute using proven multiplicative framework
                core_result = self.hybrid_functional.compute_psi_single(S, N)
                
                # Validate contraction properties
                is_contractive, K, message = self.contraction_config.validate_contraction()
                
                session_results['core_framework'] = {
                    'psi_core': core_result.get('psi', 0.0),
                    'contraction_valid': is_contractive,
                    'contraction_modulus': K,
                    'integration_message': message
                }
            except Exception as e:
                session_results['core_framework'] = {
                    'error': str(e),
                    'fallback_mode': True
                }
        
        # Add contemplative results
        session_results['contemplative_insight'] = contemplative_insight
        
        # Generate accessibility adaptations
        accessibility_adaptations = {}
        for modality in self.config['accessibility_modes']:
            if phenomena:  # Only if we have visual phenomena
                adaptations = self.contemplative_grounder.adapt_for_accessibility(
                    modality, phenomena
                )
                accessibility_adaptations[modality] = adaptations
        
        session_results['accessibility_adaptations'] = accessibility_adaptations
        
        # Compute inclusive participation metrics
        participation_metrics = self._compute_participation_metrics(observer_feedbacks or [])
        session_results['participation_metrics'] = participation_metrics
        
        # Store in integration history
        self.integration_history.append(session_results)
        
        return session_results
    
    def _compute_participation_metrics(self, feedbacks: List[ObserverFeedback]) -> Dict[str, Any]:
        """
        Compute metrics for inclusive participation
        Ensures universal access and values diverse expertise
        """
        if not feedbacks:
            return {'participation_level': 0.0, 'diversity_index': 0.0}
        
        # Diversity metrics
        unique_observers = len(set(f.observer_id for f in feedbacks))
        cultural_diversity = len(set(f.cultural_context for f in feedbacks))
        expertise_range = max(f.expertise_level for f in feedbacks) - min(f.expertise_level for f in feedbacks)
        
        # Participation quality
        avg_validation = sum(f.validation_score for f in feedbacks) / len(feedbacks)
        feedback_consistency = 1.0 - np.std([f.validation_score for f in feedbacks]) if len(feedbacks) > 1 else 1.0
        
        return {
            'participation_level': min(unique_observers / 5.0, 1.0),  # Normalize to max 5 observers
            'diversity_index': (cultural_diversity * expertise_range) / 4.0,  # Rough diversity measure
            'validation_quality': avg_validation,
            'feedback_consistency': feedback_consistency,
            'total_observers': unique_observers,
            'total_feedbacks': len(feedbacks)
        }
    
    def generate_contemplative_report(self, session_results: Dict[str, Any]) -> str:
        """
        Generate human-readable report for contemplative session
        Following your documentation writing guidelines
        """
        report_lines = [
            "# Contemplative AI Session Report",
            f"**Session ID**: {session_results['session_id']}",
            f"**Timestamp**: {session_results['timestamp']}",
            f"**Framework Version**: {session_results['framework_version']}",
            "",
            "## Stage-Four Insight Analysis",
        ]
        
        insight = session_results.get('contemplative_insight', {})
        report_lines.extend([
            f"- **Ψ Score**: {insight.get('stage_four_psi', 0.0):.3f}",
            f"- **Insight Quality**: {insight.get('insight_quality', 'unknown')}",
            f"- **Arising Awareness**: {insight.get('arising_awareness', 0.0):.3f}",
            f"- **Passing Awareness**: {insight.get('passing_awareness', 0.0):.3f}",
            f"- **Impermanence Clarity**: {insight.get('impermanence_clarity', 0.0):.3f}",
            f"- **Observer Validation**: {insight.get('observer_validation', 0.0):.3f}",
            ""
        ])
        
        # Core framework integration
        if 'core_framework' in session_results:
            core = session_results['core_framework']
            if 'error' not in core:
                report_lines.extend([
                    "## Core Framework Integration",
                    f"- **Core Ψ**: {core.get('psi_core', 0.0):.3f}",
                    f"- **Contraction Valid**: {core.get('contraction_valid', False)}",
                    f"- **Contraction Modulus**: {core.get('contraction_modulus', 'N/A')}",
                    f"- **Status**: {core.get('integration_message', 'Unknown')}",
                    ""
                ])
        
        # Participation metrics
        participation = session_results.get('participation_metrics', {})
        report_lines.extend([
            "## Inclusive Participation",
            f"- **Participation Level**: {participation.get('participation_level', 0.0):.3f}",
            f"- **Diversity Index**: {participation.get('diversity_index', 0.0):.3f}",
            f"- **Validation Quality**: {participation.get('validation_quality', 0.0):.3f}",
            f"- **Total Observers**: {participation.get('total_observers', 0)}",
            ""
        ])
        
        # Accessibility summary
        adaptations = session_results.get('accessibility_adaptations', {})
        if adaptations:
            report_lines.extend([
                "## Accessibility Adaptations",
                f"- **Available Modalities**: {', '.join(adaptations.keys())}",
                f"- **Adaptation Count**: {sum(len(v) for v in adaptations.values() if isinstance(v, (list, dict)))}",
                ""
            ])
        
        report_lines.extend([
            "## Recommendations",
            "Based on the analysis above:",
            ""
        ])
        
        # Generate recommendations based on results
        psi_score = insight.get('stage_four_psi', 0.0)
        if psi_score > 0.85:
            report_lines.append("- **Excellent**: Stage-four insight shows primitive direct awareness")
        elif psi_score > 0.70:
            report_lines.append("- **Good**: Empirically grounded insight with solid foundation")
        else:
            report_lines.append("- **Developing**: Continue practice with increased observer support")
        
        participation_level = participation.get('participation_level', 0.0)
        if participation_level < 0.5:
            report_lines.append("- **Participation**: Consider expanding observer network for better validation")
        
        return "\n".join(report_lines)
    
    def export_session_data(self, output_path: str = "outputs/contemplative_sessions.jsonl") -> bool:
        """
        Export session data in JSONL format
        Compatible with your existing output patterns
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                for session in self.integration_history:
                    f.write(json.dumps(session) + "\n")
            
            return True
        except Exception as e:
            print(f"Error exporting session data: {e}")
            return False

# Import numpy for std calculation
import numpy as np

def create_integrated_contemplative_framework(config: Optional[Dict] = None) -> IntegratedContemplativeFramework:
    """
    Factory function to create integrated contemplative framework
    Following your collaborative development principles
    """
    return IntegratedContemplativeFramework(config)

if __name__ == "__main__":
    print("Integrated Contemplative Framework")
    print("=" * 50)
    
    # Initialize framework
    framework = create_integrated_contemplative_framework()
    
    # Register sample observers (inclusive network)
    framework.register_observer("expert_teacher", 0.9, "theravada", ["visual"])
    framework.register_observer("peer_practitioner", 0.6, "secular", ["auditory"])
    framework.register_observer("accessibility_user", 0.4, "zen", ["tactile", "symbolic"])
    
    print(f"Registered {len(framework.observer_network)} observers")
    
    # Sample session processing
    sample_session = {
        'session_id': 'demo_001',
        'cultural_context': 'secular',
        'uncertainty': 0.2,
        'visual_phenomena': []  # Would contain actual VisualPhenomenon objects
    }
    
    sample_feedbacks = [
        ObserverFeedback(
            observer_id="expert_teacher",
            timestamp=datetime.now().timestamp(),
            phenomenon_id="demo",
            validation_score=0.8,
            cultural_context="theravada",
            expertise_level=0.9
        )
    ]
    
    # Process session
    results = framework.process_contemplative_session(sample_session, sample_feedbacks)
    
    # Generate report
    report = framework.generate_contemplative_report(results)
    print("\nSample Report:")
    print(report)
    
    # Export data
    success = framework.export_session_data()
    print(f"\nData export successful: {success}")
    
    print("\nIntegrated framework demonstration complete!")
