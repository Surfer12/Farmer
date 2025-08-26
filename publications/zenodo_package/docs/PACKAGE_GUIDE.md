# Farmer Python Package Installation and Usage Guide

## Package Structure

The Farmer project now includes a properly structured Python package alongside the Swift components:

```
Farmer/
├── setup.py                    # Package configuration
├── requirements.txt            # Dependencies
├── test_package.py            # Comprehensive test suite
├── python/                    # Python package directory
│   ├── __init__.py           # Package initialization
│   ├── enhanced_psi_framework.py      # Full-featured implementation
│   ├── enhanced_psi_minimal.py        # Minimal implementation
│   ├── uoif_core_components.py        # UOIF core system
│   ├── uoif_lstm_integration.py       # LSTM integration
│   ├── uoif_enhanced_psi.py          # Enhanced Ψ implementation
│   └── uoif_complete_system.py       # Complete system integration
├── Farmer/                    # Swift project directory
│   ├── (Swift source files)
│   └── ...
└── (other project files)
```

## Installation

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Package in Development Mode
```bash
pip install -e .
```

### 3. Verify Installation
```bash
python3 test_package.py
```

## Usage Examples

### Basic Hybrid Functional Computation

```python
from python.enhanced_psi_minimal import EnhancedPsiFramework

# Initialize framework
framework = EnhancedPsiFramework()

# Compute hybrid functional for mathematical content
content = "Mathematical analysis with differential equations and neural networks"
result = framework.compute_enhanced_psi(content, 'md', t=1.0)

print(f"Ψ(x) = {result['psi_final']:.3f}")
print(f"Symbolic accuracy: {result['S_x']:.3f}")
print(f"Neural accuracy: {result['N_x']:.3f}")
print(f"Interpretation: {result['interpretation']}")
```

### UOIF Core Components

```python
from python.uoif_core_components import UOIFCoreSystem, ConfidenceMeasure

# Initialize UOIF system
uoif = UOIFCoreSystem()

# Create confidence measure
confidence = ConfidenceMeasure(value=0.85, epsilon=0.1)
print(f"Confidence: {confidence.value:.3f}")
print(f"Constraint satisfied: {confidence.constraint_satisfied}")
```

### Advanced Features

```python
from python.uoif_core_components import (
    ReverseKoopmanOperator,
    RSPOOptimizer,
    ConsciousnessField,
    OatesEulerLagrangeConfidence
)

# Reverse Koopman operator for spectral analysis
koopman = ReverseKoopmanOperator(n_modes=10)

# Swarm optimization
optimizer = RSPOOptimizer(n_particles=20, dimensions=3)

# Consciousness field modeling
field = ConsciousnessField(A1=1.0, mu=0.5)

# Euler-Lagrange confidence theorem
confidence_theorem = OatesEulerLagrangeConfidence()
```

## Key Features

### 1. Hybrid Symbolic-Neural Framework
- Combines symbolic reasoning (RK4-derived) with neural predictions
- Adaptive weighting based on system characteristics
- Physics-informed regularization

### 2. UOIF Integration
- Unified Oversight Integration Framework
- Hierarchical Bayesian confidence estimation
- Risk assessment and reliability metrics

### 3. Advanced Mathematical Components
- Reverse Koopman operators for spectral analysis
- Dynamic Mode Decomposition (DMD)
- Consciousness field modeling
- Swarm intelligence optimization

### 4. Testing Infrastructure
- Comprehensive test suite (`test_package.py`)
- Import verification
- Functionality testing
- Component integration tests

## Dependencies

All dependencies are automatically installed via `requirements.txt`:
- torch>=1.9.0
- numpy>=1.21.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- pandas>=1.3.0
- scipy>=1.7.0
- jupyter>=1.0.0
- notebook>=6.4.0
- tqdm>=4.62.0

## Test Results

The package passes all tests:
- ✅ Package imports
- ✅ Minimal framework functionality
- ✅ UOIF core components
- ✅ Mathematical computations
- ✅ Component integration

## Next Steps for Testing Infrastructure

Now that the package compiles and basic functionality works, we can implement:

1. **Unit Tests**: Individual component testing with pytest
2. **Integration Tests**: Cross-component functionality
3. **Performance Tests**: Benchmarking and optimization
4. **Documentation Tests**: Docstring and example validation
5. **Continuous Integration**: Automated testing pipeline

The foundation is solid and ready for comprehensive testing infrastructure development.
