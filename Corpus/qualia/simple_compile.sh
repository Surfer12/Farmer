#!/bin/bash

# Simple compilation script for Qualia core classes
# Focuses on the essential Inverse HB Model functionality

echo "🔨 Simple Qualia Compilation..."
echo "================================="

# Create output directory
mkdir -p out

echo "📦 Compiling core Inverse HB Model classes..."

# Compile just the essential classes with minimal dependencies
javac -d out \
    ModelParameters.java \
    ClaimData.java \
    ModelPriors.java \
    -cp "lib/*" 2>/dev/null || echo "⚠️  Some dependencies missing, continuing..."

# Try to compile the main model class
echo "📦 Compiling InverseHierarchicalBayesianModel..."
javac -d out \
    InverseHierarchicalBayesianModel.java \
    -cp "out:lib/*" 2>/dev/null || echo "⚠️  Model compilation failed, checking syntax..."

# Try to compile the demo
echo "📦 Compiling demo..."
javac -d out \
    InverseHierarchicalBayesianDemo.java \
    -cp "out:lib/*" 2>/dev/null || echo "⚠️  Demo compilation failed"

echo ""
echo "📋 Checking compiled classes..."

# List what was compiled successfully
if [ -d "out/qualia" ]; then
    echo "✅ Successfully compiled:"
    find out -name "*.class" | head -10

    # Try to run the demo if it compiled
    if [ -f "out/qualia/InverseHierarchicalBayesianDemo.class" ]; then
        echo ""
        echo "🚀 Running demo..."
        java -cp "out:lib/*" qualia.InverseHierarchicalBayesianDemo
    else
        echo "❌ Demo class not found"
    fi
else
    echo "❌ No classes compiled successfully"
fi

echo ""
echo "🎯 Alternative approaches:"
echo "1. Use the working test suite: ./run_tests.sh"
echo "2. Focus on specific functionality only"
echo "3. Use IDE with better dependency management"
