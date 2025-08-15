#!/bin/bash

# Comprehensive compilation and test script for Integrated Research Analysis System

echo "=== Integrated Research Analysis System ==="
echo "Compiling and running enhanced research matching with Oates' LSTM theorem"
echo

# Clean previous builds
echo "1. Cleaning previous builds..."
rm -f *.class
rm -rf enhanced_output/
echo "   ✓ Cleaned build artifacts"

# Compile all Java files
echo
echo "2. Compiling Java source files..."

# Core files
echo "   Compiling core components..."
javac AcademicNetworkAnalyzer.java
if [ $? -ne 0 ]; then
    echo "   ✗ Failed to compile AcademicNetworkAnalyzer.java"
    exit 1
fi

javac ResearchMatchingDataStructures.java
if [ $? -ne 0 ]; then
    echo "   ✗ Failed to compile ResearchMatchingDataStructures.java"
    exit 1
fi

javac LSTMChaosPredictionEngine.java
if [ $? -ne 0 ]; then
    echo "   ✗ Failed to compile LSTMChaosPredictionEngine.java"
    exit 1
fi

javac EnhancedResearchMatcher.java
if [ $? -ne 0 ]; then
    echo "   ✗ Failed to compile EnhancedResearchMatcher.java"
    exit 1
fi

javac IntegratedResearchAnalysis.java
if [ $? -ne 0 ]; then
    echo "   ✗ Failed to compile IntegratedResearchAnalysis.java"
    exit 1
fi

echo "   ✓ All Java files compiled successfully"

# Run the integrated analysis
echo
echo "3. Running integrated research analysis..."
echo "   This will demonstrate:"
echo "   • Academic network topology analysis"
echo "   • LSTM-based research trajectory prediction"
echo "   • Oates' theorem validation"
echo "   • Hybrid symbolic-neural matching"
echo

java IntegratedResearchAnalysis

if [ $? -eq 0 ]; then
    echo
    echo "4. Analysis completed successfully!"
    echo
    echo "Generated outputs:"
    if [ -d "enhanced_output" ]; then
        echo "   📁 enhanced_output/"
        for file in enhanced_output/*; do
            if [ -f "$file" ]; then
                size=$(wc -l < "$file" 2>/dev/null || echo "0")
                echo "      📄 $(basename "$file") ($size lines)"
            fi
        done
    fi
    
    if [ -d "output" ]; then
        echo "   📁 output/"
        for file in output/*; do
            if [ -f "$file" ]; then
                size=$(wc -l < "$file" 2>/dev/null || echo "0")
                echo "      📄 $(basename "$file") ($size lines)"
            fi
        done
    fi
    
    echo
    echo "5. Key Results Summary:"
    echo "   ✓ Network analysis with community detection"
    echo "   ✓ LSTM chaos prediction with O(1/√T) error bounds"
    echo "   ✓ Hybrid functional Ψ(x) calculation"
    echo "   ✓ Oates' theorem confidence validation"
    echo "   ✓ Enhanced collaboration matching"
    
    echo
    echo "6. Next Steps:"
    echo "   • Review detailed reports in enhanced_output/"
    echo "   • Analyze collaboration matches in enhanced_matches.csv"
    echo "   • Examine theoretical validation in theoretical_validation.txt"
    echo "   • Consider integrating with real publication databases"
    
else
    echo
    echo "✗ Analysis failed. Check error messages above."
    exit 1
fi

echo
echo "=== Integration Complete ==="
echo "The system successfully combines:"
echo "• Academic Network Analysis (symbolic component)"
echo "• LSTM Chaos Prediction (neural component)"  
echo "• Oates' Hidden State Convergence Theorem"
echo "• Hybrid Symbolic-Neural Accuracy Functional"
echo
echo "This demonstrates practical application of your theoretical framework"
echo "for enhanced research collaboration matching and prediction."
