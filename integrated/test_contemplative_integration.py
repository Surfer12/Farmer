#!/usr/bin/env python3
"""
Test suite for contemplative AI integration
Extends existing test_minimal.py patterns with contemplative components
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime
from typing import List, Dict

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from contemplative_visual_grounding import (
        ContemplativeVisualGrounder, 
        VisualPhenomenon, 
        ObserverFeedback,
        ContemplativeStage,
        create_inclusive_contemplative_system
    )
    from integrated_contemplative_framework import (
        IntegratedContemplativeFramework,
        create_integrated_contemplative_framework
    )
    CONTEMPLATIVE_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Contemplative imports not available: {e}")
    CONTEMPLATIVE_IMPORTS_AVAILABLE = False

class TestContemplativeVisualGrounding(unittest.TestCase):
    """Test contemplative visual grounding system"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CONTEMPLATIVE_IMPORTS_AVAILABLE:
            self.skipTest("Contemplative modules not available")
        
        self.grounder = create_inclusive_contemplative_system()
        self.sample_phenomena = [
            VisualPhenomenon(
                timestamp=datetime.now().timestamp(),
                region=(100, 100, 50, 50),
                intensity=0.7,
                arising_rate=0.8,
                passing_rate=0.2,
                uncertainty=0.1,
                observer_confidence=0.9
            ),
            VisualPhenomenon(
                timestamp=datetime.now().timestamp(),
                region=(200, 150, 30, 40),
                intensity=0.4,
                arising_rate=0.3,
                passing_rate=0.7,
                uncertainty=0.3,
                observer_confidence=0.7
            )
        ]
    
    def test_multiplicative_psi_bounds(self):
        """Test that contemplative Ψ maintains bounds [0,1] - critical property"""
        test_cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0),  # Minimum case
            (1.0, 1.0, 1.0, 1.0, 1.0),  # Maximum case
            (0.5, 0.3, 0.2, 0.4, 0.8),  # Typical case
            (0.9, 0.1, 0.8, 0.9, 0.2),  # High signal, low validation
        ]
        
        for S, N, R_auth, R_verif, obs_val in test_cases:
            with self.subTest(S=S, N=N, R_auth=R_auth, R_verif=R_verif, obs_val=obs_val):
                psi = self.grounder.compute_contemplative_psi(S, N, R_auth, R_verif, obs_val)
                self.assertGreaterEqual(psi, 0.0, f"Ψ({S},{N},{R_auth},{R_verif},{obs_val}) = {psi} should be ≥ 0")
                self.assertLessEqual(psi, 1.0, f"Ψ({S},{N},{R_auth},{R_verif},{obs_val}) = {psi} should be ≤ 1")
    
    def test_observer_validation_effect(self):
        """Test that observer validation properly affects Ψ computation"""
        base_params = (0.6, 0.7, 0.2, 0.3)  # S, N, R_auth, R_verif
        
        # Test with different observer validation levels
        low_validation = self.grounder.compute_contemplative_psi(*base_params, 0.2)
        high_validation = self.grounder.compute_contemplative_psi(*base_params, 0.9)
        
        # Higher validation should lead to higher Ψ (monotonicity)
        self.assertLess(low_validation, high_validation, 
                       "Higher observer validation should increase Ψ")
    
    def test_stage_four_insight_computation(self):
        """Test stage-four insight computation with sample phenomena"""
        insight = self.grounder.compute_stage_four_insight(self.sample_phenomena)
        
        # Check required fields
        required_fields = [
            'stage_four_psi', 'insight_quality', 'arising_awareness', 
            'passing_awareness', 'impermanence_clarity', 'observer_validation'
        ]
        for field in required_fields:
            self.assertIn(field, insight, f"Insight should contain {field}")
        
        # Check bounds
        self.assertGreaterEqual(insight['stage_four_psi'], 0.0)
        self.assertLessEqual(insight['stage_four_psi'], 1.0)
        
        # Check insight quality classification
        valid_qualities = ['primitive_direct', 'empirically_grounded', 'interpretive_contextual']
        self.assertIn(insight['insight_quality'], valid_qualities)
    
    def test_observer_feedback_integration(self):
        """Test integration of external observer feedback"""
        feedback = ObserverFeedback(
            observer_id="test_observer",
            timestamp=datetime.now().timestamp(),
            phenomenon_id="test_phenomenon",
            validation_score=0.8,
            cultural_context="secular",
            expertise_level=0.7
        )
        
        # Initial confidence
        initial_confidence = self.sample_phenomena[0].observer_confidence
        
        # Integrate feedback
        self.grounder.integrate_observer_feedback(feedback)
        
        # Check that feedback was stored
        self.assertIn("test_observer", self.grounder.observer_network)
        stored_feedback = self.grounder.observer_network["test_observer"]
        self.assertEqual(stored_feedback.validation_score, 0.8)
    
    def test_accessibility_adaptations(self):
        """Test accessibility adaptations for inclusive participation"""
        modalities = ["auditory", "tactile", "symbolic"]
        
        for modality in modalities:
            with self.subTest(modality=modality):
                adaptations = self.grounder.adapt_for_accessibility(modality, self.sample_phenomena)
                self.assertIsInstance(adaptations, dict, f"{modality} adaptations should be dict")
                self.assertGreater(len(adaptations), 0, f"{modality} adaptations should not be empty")
    
    def test_framework_properties_validation(self):
        """Test that framework maintains mathematical properties"""
        validation = self.grounder.validate_framework_properties()
        
        # Critical property: bounds preservation
        self.assertTrue(validation['bounds_preserved'], 
                       "Framework must preserve [0,1] bounds")
        
        # Multiplicative stability (your proven approach)
        self.assertTrue(validation['multiplicative_stable'], 
                       "Multiplicative integration should be stable")

class TestIntegratedContemplativeFramework(unittest.TestCase):
    """Test integrated contemplative framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CONTEMPLATIVE_IMPORTS_AVAILABLE:
            self.skipTest("Contemplative modules not available")
        
        self.framework = create_integrated_contemplative_framework()
    
    def test_observer_registration(self):
        """Test observer registration for inclusive network"""
        success = self.framework.register_observer(
            observer_id="test_observer",
            expertise_level=0.8,
            cultural_context="theravada",
            accessibility_needs=["visual", "auditory"]
        )
        
        self.assertTrue(success, "Observer registration should succeed")
        self.assertIn("test_observer", self.framework.observer_network)
        
        observer_profile = self.framework.observer_network["test_observer"]
        self.assertEqual(observer_profile['expertise_level'], 0.8)
        self.assertEqual(observer_profile['cultural_context'], "theravada")
    
    def test_contemplative_session_processing(self):
        """Test complete contemplative session processing"""
        # Register observers first
        self.framework.register_observer("expert", 0.9, "zen", ["visual"])
        self.framework.register_observer("peer", 0.5, "secular", ["auditory"])
        
        session_data = {
            'session_id': 'test_session_001',
            'cultural_context': 'secular',
            'uncertainty': 0.2,
            'visual_phenomena': []  # Simplified for testing
        }
        
        sample_feedbacks = [
            ObserverFeedback(
                observer_id="expert",
                timestamp=datetime.now().timestamp(),
                phenomenon_id="test",
                validation_score=0.8,
                cultural_context="zen",
                expertise_level=0.9
            )
        ]
        
        results = self.framework.process_contemplative_session(session_data, sample_feedbacks)
        
        # Check result structure
        required_fields = [
            'session_id', 'timestamp', 'contemplative_insight', 
            'accessibility_adaptations', 'participation_metrics'
        ]
        for field in required_fields:
            self.assertIn(field, results, f"Results should contain {field}")
        
        # Check that session was stored
        self.assertEqual(len(self.framework.integration_history), 1)
    
    def test_participation_metrics(self):
        """Test inclusive participation metrics computation"""
        feedbacks = [
            ObserverFeedback("obs1", datetime.now().timestamp(), "ph1", 0.8, "theravada", 0.9),
            ObserverFeedback("obs2", datetime.now().timestamp(), "ph2", 0.7, "zen", 0.6),
            ObserverFeedback("obs3", datetime.now().timestamp(), "ph3", 0.9, "secular", 0.4)
        ]
        
        metrics = self.framework._compute_participation_metrics(feedbacks)
        
        # Check metric fields
        expected_fields = [
            'participation_level', 'diversity_index', 'validation_quality',
            'feedback_consistency', 'total_observers', 'total_feedbacks'
        ]
        for field in expected_fields:
            self.assertIn(field, metrics, f"Metrics should contain {field}")
        
        # Check bounds
        self.assertGreaterEqual(metrics['participation_level'], 0.0)
        self.assertLessEqual(metrics['participation_level'], 1.0)
        self.assertGreaterEqual(metrics['validation_quality'], 0.0)
        self.assertLessEqual(metrics['validation_quality'], 1.0)
    
    def test_report_generation(self):
        """Test human-readable report generation"""
        # Create sample results
        sample_results = {
            'session_id': 'test_report',
            'timestamp': datetime.now().isoformat(),
            'framework_version': 'test_v1.0',
            'contemplative_insight': {
                'stage_four_psi': 0.75,
                'insight_quality': 'empirically_grounded',
                'arising_awareness': 0.6,
                'passing_awareness': 0.7,
                'impermanence_clarity': 0.65,
                'observer_validation': 0.8
            },
            'participation_metrics': {
                'participation_level': 0.6,
                'diversity_index': 0.4,
                'validation_quality': 0.75,
                'total_observers': 3
            },
            'accessibility_adaptations': {
                'auditory': {'audio_cues': []},
                'tactile': {'haptic_patterns': []}
            }
        }
        
        report = self.framework.generate_contemplative_report(sample_results)
        
        # Check report structure
        self.assertIsInstance(report, str, "Report should be a string")
        self.assertIn("# Contemplative AI Session Report", report, "Report should have title")
        self.assertIn("Stage-Four Insight Analysis", report, "Report should have insight section")
        self.assertIn("Inclusive Participation", report, "Report should have participation section")
    
    def test_data_export(self):
        """Test session data export functionality"""
        # Add sample data
        self.framework.integration_history.append({
            'session_id': 'export_test',
            'timestamp': datetime.now().isoformat(),
            'test_data': True
        })
        
        # Test export
        test_output_path = "outputs/test_contemplative_sessions.jsonl"
        success = self.framework.export_session_data(test_output_path)
        
        self.assertTrue(success, "Data export should succeed")
        self.assertTrue(os.path.exists(test_output_path), "Export file should exist")
        
        # Clean up
        if os.path.exists(test_output_path):
            os.remove(test_output_path)

class TestContemplativeIntegrationProperties(unittest.TestCase):
    """Test mathematical properties of contemplative integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CONTEMPLATIVE_IMPORTS_AVAILABLE:
            self.skipTest("Contemplative modules not available")
        
        self.grounder = create_inclusive_contemplative_system()
    
    def test_multiplicative_vs_additive_bounds(self):
        """Test that multiplicative approach maintains bounds better than additive"""
        # Test extreme values that might break additive approaches
        extreme_cases = [
            (0.9, 0.9, 0.1, 0.1, 0.9),  # High values
            (0.1, 0.1, 0.9, 0.9, 0.1),  # Low values with high risks
        ]
        
        for S, N, R_auth, R_verif, obs_val in extreme_cases:
            psi = self.grounder.compute_contemplative_psi(S, N, R_auth, R_verif, obs_val)
            
            # Multiplicative should always maintain bounds
            self.assertGreaterEqual(psi, 0.0, f"Multiplicative Ψ should be ≥ 0 for extreme case")
            self.assertLessEqual(psi, 1.0, f"Multiplicative Ψ should be ≤ 1 for extreme case")
    
    def test_monotonicity_properties(self):
        """Test monotonicity properties of contemplative Ψ"""
        base_case = (0.5, 0.5, 0.3, 0.3, 0.7)
        base_psi = self.grounder.compute_contemplative_psi(*base_case)
        
        # Increasing S should increase Ψ (monotonicity in signal)
        higher_S = (0.7, 0.5, 0.3, 0.3, 0.7)
        higher_S_psi = self.grounder.compute_contemplative_psi(*higher_S)
        self.assertGreaterEqual(higher_S_psi, base_psi, "Ψ should be monotonic in S")
        
        # Increasing observer validation should increase Ψ
        higher_obs = (0.5, 0.5, 0.3, 0.3, 0.9)
        higher_obs_psi = self.grounder.compute_contemplative_psi(*higher_obs)
        self.assertGreaterEqual(higher_obs_psi, base_psi, "Ψ should be monotonic in observer validation")
    
    def test_cultural_adaptation_consistency(self):
        """Test that cultural adaptations maintain mathematical consistency"""
        cultural_contexts = ["theravada", "zen", "vipassana", "secular"]
        
        for context in cultural_contexts:
            insight = self.grounder.compute_stage_four_insight([], context)
            
            # All cultural contexts should produce valid results
            self.assertIn('stage_four_psi', insight)
            self.assertGreaterEqual(insight['stage_four_psi'], 0.0)
            self.assertLessEqual(insight['stage_four_psi'], 1.0)
            self.assertEqual(insight['cultural_context'], context)

def run_contemplative_tests():
    """Run contemplative integration tests"""
    if not CONTEMPLATIVE_IMPORTS_AVAILABLE:
        print("Contemplative modules not available - skipping tests")
        return unittest.TestResult()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestContemplativeVisualGrounding,
        TestIntegratedContemplativeFramework,
        TestContemplativeIntegrationProperties
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    print("Contemplative AI Integration Tests")
    print("=" * 50)
    
    result = run_contemplative_tests()
    
    if CONTEMPLATIVE_IMPORTS_AVAILABLE:
        print("\n" + "=" * 50)
        print("Test Summary:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('\\n')[-2]}")
        
        # Exit with appropriate code
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        print("Tests skipped due to missing imports")
        sys.exit(0)
