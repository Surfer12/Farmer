#!/bin/bash

# Compile and run the Academic Network Analyzer

echo "Compiling AcademicNetworkAnalyzer..."
javac AcademicNetworkAnalyzer.java

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running analyzer..."
    java AcademicNetworkAnalyzer
else
    echo "Compilation failed!"
    exit 1
fi

echo "Analysis complete. Check the output directory for results."
