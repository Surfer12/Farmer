# Pixi Usage Guide: Hybrid Symbolic-Neural Framework

This guide provides comprehensive instructions for using pixi to run analysis tools for the Hybrid Symbolic-Neural Accuracy Functional framework with contraction guarantees.

## Quick Start

### Installation
```bash
# Install pixi if not already installed
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies
pixi install
```

### Quick Demos
```bash
# Quick contraction analysis demo (2 scenarios)
pixi run demo-contraction

# Quick hybrid functional demo
pixi run demo-hybrid
```

## Core Analysis Commands

### Individual Components
```bash
# Minimal contraction analysis (pure Python, no dependencies)
pixi run contraction-minimal

# Full contraction analysis (with NumPy/SciPy if available)
pixi run contraction-full

# Minimal hybrid functional analysis
pixi run hybrid-minimal

# Full hybrid functional analysis
pixi run hybrid-full

# Physics-Informed Neural Network (Burgers equation)
pixi run pinn-burgers
```

### Academic Network Analysis
```bash
# Basic academic network analysis (Java)
pixi run academic-basic

# Enhanced research matching analysis
pixi run academic-enhanced

# Nature article validation analysis
pixi run academic-nature
```

## Comprehensive Analysis Suite

### Full Analysis Pipeline
```bash
# Run all analyses and capture outputs
pixi run analyze-all

# Export results to JSON/CSV files
pixi run export-results

# Generate comprehensive report
pixi run generate-report
```

### Individual Analysis Components
```bash
# Run contraction analysis with output capture
pixi run contraction-analysis

# Run hybrid functional analysis with output capture
pixi run hybrid-analysis

# Run academic network analysis with output capture
pixi run academic-analysis
```

## Utility Commands

### File Management
```bash
# Create outputs directory
pixi run setup-outputs

# Clean output files
pixi run clean-outputs

# Clean Java class files
pixi run clean-java

# Clean all generated files
pixi run clean-all
```

### Development Tools
```bash
# Format Python code with black
pixi run format

# Lint Python code with flake8
pixi run lint

# Run minimal test suite
pixi run test-minimal

# Run all tests
pixi run test-all
```

### Documentation and Validation
```bash
# Compile LaTeX documents (requires pdflatex)
pixi run compile-latex

# Validate theoretical properties
pixi run validate-theory
```

## Interactive Analysis

### Jupyter and IPython
```bash
# Launch Jupyter Lab
pixi run jupyter-lab

# Launch IPython shell
pixi run ipython-shell
```

## Performance and Benchmarking

### Benchmarking
```bash
# Benchmark contraction analysis performance
pixi run benchmark-contraction
```

## Advanced Features

### Visualization Environment
```bash
# Install visualization dependencies
pixi install --environment viz

# Run visualization tasks (requires viz environment)
pixi run --environment viz plot-contraction
pixi run --environment viz plot-hybrid
pixi run --environment viz interactive-demo
```

### GPU Acceleration
```bash
# Install GPU dependencies
pixi install --environment gpu

# Run GPU-accelerated analysis
pixi run --environment gpu gpu-benchmark
```

### Full Environment (Visualization + GPU)
```bash
# Install all features
pixi install --environment full

# Access all advanced features
pixi run --environment full interactive-demo
pixi run --environment full gpu-benchmark
```

## Output Files

The analysis commands generate various output files:

### Analysis Results
- `outputs/contraction_analysis.txt` - Contraction analysis output
- `outputs/hybrid_analysis.txt` - Hybrid functional analysis output
- `outputs/academic_analysis.txt` - Academic network analysis output

### Exported Data
- `outputs/contraction_analysis_results.json` - Detailed contraction results
- `outputs/contraction_summary.csv` - Summary data for spreadsheets
- `outputs/hybrid_functional_results.json` - Hybrid functional test results

### Reports
- `outputs/comprehensive_analysis_report.md` - Full detailed report
- `outputs/executive_summary.md` - Quick summary report
- `outputs/analysis_summary_report.md` - Analysis summary

### Academic Network Results
- `academic_network_analysis/output/` - Academic analysis results
- `academic_network_analysis/enhanced_output/` - Enhanced analysis results
- `academic_network_analysis/nature_output/` - Nature article validation

## Example Workflows

### 1. Quick Analysis
```bash
# Get quick overview of framework capabilities
pixi run demo-contraction
pixi run demo-hybrid
```

### 2. Comprehensive Research Analysis
```bash
# Full analysis pipeline
pixi run analyze-all
pixi run export-results
pixi run generate-report

# View results
cat outputs/executive_summary.md
```

### 3. Development Workflow
```bash
# Format and test code
pixi run format
pixi run lint
pixi run test-minimal

# Run specific analysis
pixi run contraction-minimal
```

### 4. Academic Research Validation
```bash
# Run academic network analysis
pixi run academic-basic
pixi run academic-enhanced
pixi run academic-nature

# Generate comprehensive report
pixi run generate-report
```

### 5. Performance Analysis
```bash
# Benchmark performance
pixi run benchmark-contraction

# Validate theoretical properties
pixi run validate-theory
```

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Reinstall dependencies
pixi install

# Check environment
pixi info
```

#### Java Compilation Issues
```bash
# Clean Java files and retry
pixi run clean-java
pixi run academic-basic
```

#### Output Directory Issues
```bash
# Ensure outputs directory exists
pixi run setup-outputs
```

#### LaTeX Compilation
```bash
# Install LaTeX if needed (macOS)
brew install --cask mactex

# Or use minimal LaTeX
brew install basictex
```

### Environment-Specific Commands

#### Check Available Environments
```bash
pixi info
```

#### List Available Tasks
```bash
pixi task list
```

#### Check Dependencies
```bash
pixi list
```

## Integration with Existing Framework

The pixi commands integrate seamlessly with your existing mathematical frameworks:

- **Contraction Theory**: Validates theoretical bounds and convergence
- **Hybrid Functional**: Tests symbolic-neural integration
- **Academic Networks**: Analyzes research collaboration dynamics
- **LSTM Convergence**: Validates O(1/√T) error bounds
- **Swarm-Koopman**: Tests nonlinear system stability
- **Fractal Dynamics**: Validates bounded self-interaction

## Performance Characteristics

### Execution Times (Approximate)
- `demo-contraction`: ~2-3 seconds
- `demo-hybrid`: ~1-2 seconds
- `contraction-minimal`: ~5-10 seconds
- `academic-basic`: ~10-30 seconds (depends on Java compilation)
- `analyze-all`: ~30-60 seconds
- `generate-report`: ~10-20 seconds

### Resource Usage
- **Memory**: Minimal versions use <50MB, full versions may use 100-500MB
- **CPU**: Single-threaded for most analyses, multi-core for academic networks
- **Storage**: Output files typically 1-10MB total

## Next Steps

After running the analyses:

1. **Review Results**: Check `outputs/executive_summary.md`
2. **Detailed Analysis**: Read `outputs/comprehensive_analysis_report.md`
3. **Data Analysis**: Import CSV files into spreadsheet software
4. **Visualization**: Use viz environment for plotting
5. **Extension**: Modify parameters and re-run analyses

---

*This guide covers all available pixi commands for the Hybrid Symbolic-Neural Framework. For theoretical background, see the comprehensive analysis report generated by `pixi run generate-report`.*
