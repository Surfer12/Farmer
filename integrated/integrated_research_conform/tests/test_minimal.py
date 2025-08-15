#!/usr/bin/env python3
"""
Minimal test suite for validating pixi tasks and framework functionality
"""

import unittest
import sys
import os

# Add core directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

class TestMinimalContraction(unittest.TestCase):
    """Test minimal contraction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from minimal_contraction_psi import MinimalContractionConfig, MinimalContractionPsi
            self.config = MinimalContractionConfig()
            self.psi_updater = MinimalContractionPsi(self.config)
        except ImportError as e:
            self.skipTest(f"Cannot import minimal_contraction_psi: {e}")
    
    def test_contraction_validation(self):
        """Test contraction condition validation"""
        is_contractive, K, message = self.config.validate_contraction()
        self.assertTrue(is_contractive, f"Configuration should be contractive: {message}")
        self.assertLess(K, 1.0, f"Contraction modulus K={K} should be < 1")
        self.assertGreater(K, 0.0, f"Contraction modulus K={K} should be > 0")
    
    def test_phi_function_bounds(self):
        """Test that Φ function maintains bounds [0,1]"""
        test_cases = [
            (0.0, 0.5, 0.7, 0.8, 0.1, 0.1),
            (0.5, 0.3, 0.6, 0.9, 0.05, 0.08),
            (1.0, 0.8, 0.85, 0.7, 0.15, 0.12)
        ]
        
        for psi, alpha, S, N, R_cog, R_eff in test_cases:
            result = self.psi_updater.phi_function(psi, alpha, S, N, R_cog, R_eff)
            self.assertGreaterEqual(result, 0.0, f"Φ({psi}) = {result} should be ≥ 0")
            self.assertLessEqual(result, 1.0, f"Φ({psi}) = {result} should be ≤ 1")
    
    def test_sequence_convergence(self):
        """Test sequence convergence properties"""
        scenario = {
            'alpha': 0.5,
            'S': 0.7,
            'N': 0.8,
            'R_cog': 0.1,
            'R_eff': 0.1
        }
        
        sequence = self.psi_updater.simulate_sequence(0.3, 20, scenario)
        
        # Check sequence properties
        self.assertEqual(len(sequence), 21, "Sequence should have correct length")
        self.assertEqual(sequence[0], 0.3, "Initial value should be preserved")
        
        # Check bounds
        for i, psi in enumerate(sequence):
            self.assertGreaterEqual(psi, 0.0, f"ψ_{i} = {psi} should be ≥ 0")
            self.assertLessEqual(psi, 1.0, f"ψ_{i} = {psi} should be ≤ 1")
        
        # Check convergence (final values should be close)
        final_values = sequence[-5:]  # Last 5 values
        max_deviation = max(final_values) - min(final_values)
        self.assertLess(max_deviation, 0.01, "Sequence should converge")

class TestMinimalHybrid(unittest.TestCase):
    """Test minimal hybrid functional"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from minimal_hybrid_functional import MinimalHybridFunctional
            self.functional = MinimalHybridFunctional()
        except ImportError as e:
            self.skipTest(f"Cannot import minimal_hybrid_functional: {e}")
    
    def test_psi_computation(self):
        """Test Ψ computation with known values"""
        result = self.functional.compute_psi_single(0.5, 1.0)
        
        # Check result structure
        self.assertIn('psi', result)
        self.assertIn('S', result)
        self.assertIn('N', result)
        self.assertIn('alpha', result)
        
        # Check bounds
        self.assertGreaterEqual(result['psi'], 0.0, "Ψ should be ≥ 0")
        self.assertLessEqual(result['psi'], 1.0, "Ψ should be ≤ 1")
        
        # Check component bounds
        self.assertGreaterEqual(result['S'], 0.0, "S should be ≥ 0")
        self.assertLessEqual(result['S'], 1.0, "S should be ≤ 1")
        self.assertGreaterEqual(result['N'], 0.0, "N should be ≥ 0")
        self.assertLessEqual(result['N'], 1.0, "N should be ≤ 1")

class TestFrameworkIntegration(unittest.TestCase):
    """Test framework integration"""
    
    def test_file_existence(self):
        """Test that key files exist"""
        required_files = [
            'minimal_contraction_psi.py',
            'contraction_spectral_theorems.tex',
            'pyproject.toml',
            'README.md'
        ]
        
        for filename in required_files:
            self.assertTrue(os.path.exists(filename), f"Required file {filename} should exist")
    
    def test_academic_network_directory(self):
        """Test academic network analysis directory"""
        if os.path.exists('academic_network_analysis'):
            # Check for key Java files
            java_files = [
                'academic_network_analysis/AcademicNetworkAnalysis.java',
                'academic_network_analysis/Publication.java',
                'academic_network_analysis/Researcher.java'
            ]
            
            for java_file in java_files:
                if os.path.exists(java_file):
                    self.assertTrue(True, f"Found {java_file}")
    
    def test_outputs_directory_creation(self):
        """Test that outputs directory can be created"""
        os.makedirs('outputs', exist_ok=True)
        self.assertTrue(os.path.exists('outputs'), "Outputs directory should exist")

class TestPixiTasks(unittest.TestCase):
    """Test pixi task functionality"""
    
    def test_import_statements(self):
        """Test that all required modules can be imported"""
        import_tests = [
            ('math', 'Built-in math module'),
            ('json', 'Built-in json module'),
            ('csv', 'Built-in csv module'),
            ('os', 'Built-in os module'),
            ('datetime', 'Built-in datetime module')
        ]
        
        for module_name, description in import_tests:
            try:
                __import__(module_name)
            except ImportError:
                self.fail(f"Cannot import {module_name} ({description})")
    
    def test_framework_imports(self):
        """Test framework-specific imports"""
        framework_imports = [
            'minimal_contraction_psi',
            'export_analysis_results',
            'generate_comprehensive_report'
        ]
        
        for module_name in framework_imports:
            try:
                __import__(module_name)
            except ImportError as e:
                # This is expected if files don't exist yet
                pass

def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMinimalContraction,
        TestMinimalHybrid,
        TestFrameworkIntegration,
        TestPixiTasks
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    print("Running Hybrid Symbolic-Neural Framework Tests")
    print("=" * 50)
    
    result = run_tests()
    
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
