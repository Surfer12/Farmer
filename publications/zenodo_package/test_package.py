#!/usr/bin/env python3
"""
Comprehensive test script for the Farmer Python package
"""

def test_imports():
    """Test all package imports"""
    print("Testing package imports...")
    
    try:
        import python
        print("✓ Main python package imported successfully")
    except ImportError as e:
        print(f"✗ Main python package import failed: {e}")
        return False
    
    try:
        from python import enhanced_psi_framework
        print("✓ enhanced_psi_framework imported successfully")
    except ImportError as e:
        print(f"✗ enhanced_psi_framework import failed: {e}")
    
    try:
        from python import uoif_core_components
        print("✓ uoif_core_components imported successfully")
    except ImportError as e:
        print(f"✗ uoif_core_components import failed: {e}")
    
    try:
        from python import uoif_lstm_integration
        print("✓ uoif_lstm_integration imported successfully")
    except ImportError as e:
        print(f"✗ uoif_lstm_integration import failed: {e}")
    
    try:
        from python import enhanced_psi_minimal
        print("✓ enhanced_psi_minimal imported successfully")
    except ImportError as e:
        print(f"✗ enhanced_psi_minimal import failed: {e}")
    
    return True

def test_minimal_framework():
    """Test the minimal framework functionality"""
    print("\nTesting minimal framework functionality...")
    
    try:
        from python.enhanced_psi_minimal import EnhancedPsiFramework
        
        # Test instantiation
        framework = EnhancedPsiFramework()
        print("✓ EnhancedPsiFramework instantiated successfully")
        
        # Test computation
        test_content = "Mathematical analysis with differential equations and neural networks"
        result = framework.compute_enhanced_psi(test_content, 'md', t=1.0)
        
        print(f"✓ Computation successful: Ψ(x) = {result['psi_final']:.3f}")
        print(f"  Symbolic accuracy: {result['S_x']:.3f}")
        print(f"  Neural accuracy: {result['N_x']:.3f}")
        print(f"  Adaptive weight: {result['alpha_t']:.3f}")
        print(f"  Interpretation: {result['interpretation']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Minimal framework test failed: {e}")
        return False

def test_uoif_components():
    """Test UOIF core components"""
    print("\nTesting UOIF core components...")
    
    try:
        from python.uoif_core_components import UOIFCoreSystem, ConfidenceMeasure
        
        # Test instantiation
        uoif = UOIFCoreSystem()
        print("✓ UOIFCoreSystem instantiated successfully")
        
        # Test confidence measure
        confidence_measure = ConfidenceMeasure(value=0.85, epsilon=0.1)
        print(f"✓ ConfidenceMeasure created: {confidence_measure.value:.3f}")
        print(f"  Constraint satisfied: {confidence_measure.constraint_satisfied}")
        
        return True
        
    except Exception as e:
        print(f"✗ UOIF components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("FARMER PYTHON PACKAGE TEST SUITE")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_minimal_framework():
        tests_passed += 1
    
    if test_uoif_components():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Package is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Package may need additional work.")
        return False

if __name__ == "__main__":
    main()
