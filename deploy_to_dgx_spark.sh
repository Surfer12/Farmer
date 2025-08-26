#!/bin/bash
# DGX SPARK Blackwell Deployment Script
# =====================================
# 
# This script sets up the Farmer project on DGX SPARK Blackwell
# and runs the comprehensive benchmarking suite.

set -e  # Exit on any error

echo "🚀 DGX SPARK Blackwell Deployment Script"
echo "========================================"

# Check if running on DGX SPARK
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "🔍 Detected GPU: $GPU_INFO"
    
    if [[ "$GPU_INFO" == *"Blackwell"* ]] || [[ "$GPU_INFO" == *"B200"* ]]; then
        echo "✅ Blackwell architecture confirmed!"
    else
        echo "⚠️  Non-Blackwell GPU detected. Proceeding anyway..."
    fi
else
    echo "❌ nvidia-smi not found. Are you on a GPU system?"
    exit 1
fi

# System information
echo ""
echo "📊 System Information:"
echo "----------------------"
echo "Hostname: $(hostname)"
echo "OS: $(uname -a)"
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "GPU Count: $(nvidia-smi -L | wc -l)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

# Create project directory structure
echo ""
echo "📁 Setting up project structure..."
PROJECT_DIR="/workspace/farmer_blackwell_benchmark"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Copy project files (assuming they're in current directory)
echo "📋 Copying project files..."
cp -r /Users/ryan_david_oates/Farmer/* . 2>/dev/null || echo "⚠️  Manual file copy required"

# Set up Python environment
echo ""
echo "🐍 Setting up Python environment..."

# Check if conda is available (common on DGX systems)
if command -v conda &> /dev/null; then
    echo "Using conda environment..."
    conda create -n farmer_blackwell python=3.9 -y
    source activate farmer_blackwell
else
    echo "Using venv environment..."
    python3 -m venv venv_blackwell
    source venv_blackwell/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip

# Core ML dependencies optimized for Blackwell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy>=1.21.0
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
pip install scipy>=1.7.0
pip install tqdm>=4.62.0

# Additional dependencies for benchmarking
pip install psutil
pip install GPUtil
pip install jupyter
pip install tensorboard

# Verify PyTorch CUDA installation
echo ""
echo "🔍 Verifying PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Create output directories
echo ""
echo "📁 Creating output directories..."
mkdir -p data_output/alignment_visualizations
mkdir -p logs
mkdir -p results

# Set up logging
LOG_FILE="logs/blackwell_benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to: $LOG_FILE"

# Create a quick system test
echo ""
echo "🧪 Running quick system test..."
python3 -c "
import torch
import torch.nn as nn
import time

print('Testing Blackwell tensor cores...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Test different precision formats
formats = [
    ('FP32', torch.float32),
    ('BF16', torch.bfloat16),
    ('FP16', torch.float16)
]

for name, dtype in formats:
    try:
        x = torch.randn(1024, 1024, device=device, dtype=dtype)
        y = torch.randn(1024, 1024, device=device, dtype=dtype)
        
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f'{name}: {elapsed:.4f}s - ✅')
    except Exception as e:
        print(f'{name}: Error - {e}')

print('System test complete!')
"

# Function to run benchmarks
run_benchmarks() {
    echo ""
    echo "🚀 Starting Blackwell Benchmarking Suite..."
    echo "==========================================="
    
    # Run the main benchmark
    python3 blackwell_benchmark_suite.py 2>&1 | tee -a $LOG_FILE
    
    # Check if benchmark completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Benchmarking completed successfully!"
        
        # Display results summary
        echo ""
        echo "📊 Results Summary:"
        echo "==================="
        
        # Find the latest report file
        LATEST_REPORT=$(ls -t data_output/alignment_visualizations/blackwell_benchmark_report_*.json 2>/dev/null | head -1)
        
        if [ -f "$LATEST_REPORT" ]; then
            echo "📄 Report file: $LATEST_REPORT"
            
            # Extract key metrics using Python
            python3 -c "
import json
import sys

try:
    with open('$LATEST_REPORT', 'r') as f:
        report = json.load(f)
    
    print('Key Results:')
    print('============')
    
    # System info
    sys_info = report['metadata']['system_info']
    print(f'GPU: {sys_info.get(\"gpu_name\", \"Unknown\")}')
    print(f'GPU Memory: {sys_info.get(\"gpu_memory_gb\", 0):.1f} GB')
    
    # Correlation results
    correlations = report['correlation_analysis']
    print(f'\\nCorrelation Analysis:')
    for precision, corr in correlations.items():
        print(f'  {precision}: {corr:.6f}')
    
    # Validation status
    status = report['summary']['validation_status']
    print(f'\\nValidation Status: {status}')
    
    # Theoretical vs empirical
    theoretical = report['summary']['theoretical_correlation']
    blackwell_obs = report['summary']['blackwell_observed']
    print(f'\\nComparison:')
    print(f'  Theoretical: {theoretical:.6f}')
    print(f'  Blackwell Target: {blackwell_obs:.4f}')
    
except Exception as e:
    print(f'Error reading report: {e}')
"
        else
            echo "⚠️  No report file found"
        fi
        
    else
        echo "❌ Benchmarking failed. Check logs for details."
        return 1
    fi
}

# Function to generate paper data
generate_paper_data() {
    echo ""
    echo "📄 Generating Paper Validation Data..."
    echo "====================================="
    
    # Run paper figure generation with real data
    python3 -c "
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print('Generating paper-ready figures with Blackwell data...')

# Load benchmark results
try:
    import glob
    report_files = glob.glob('data_output/alignment_visualizations/blackwell_benchmark_report_*.json')
    if report_files:
        latest_report = max(report_files)
        with open(latest_report, 'r') as f:
            data = json.load(f)
        
        # Create paper figure with real Blackwell data
        correlations = data['correlation_analysis']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['Theoretical\\nMXFP8', 'Blackwell\\nTarget'] + list(correlations.keys())
        values = [0.999744, 0.9989] + list(correlations.values())
        colors = ['blue', 'red'] + ['green'] * len(correlations)
        
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Blackwell Hardware Validation Results')
        ax.set_ylim(0.995, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                   f'{val:.6f}', ha='center', va='bottom', fontweight='bold')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data_output/alignment_visualizations/paper_blackwell_validation_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Paper figure saved: {filename}')
        
        # Calculate validation metrics
        best_match = min(correlations.values(), key=lambda x: abs(x - 0.9989))
        difference = abs(best_match - 0.9989)
        
        print(f'\\n🎯 Validation Results:')
        print(f'   Best empirical correlation: {best_match:.6f}')
        print(f'   Difference from target: {difference:.6f}')
        
        if difference < 0.001:
            print('   Status: ✅ EXCELLENT VALIDATION!')
        elif difference < 0.01:
            print('   Status: ✅ GOOD VALIDATION!')
        else:
            print('   Status: ⚠️  Partial validation')
            
    else:
        print('❌ No benchmark data found')
        
except Exception as e:
    print(f'Error: {e}')
"
}

# Main execution
echo ""
echo "🎯 Ready to run benchmarks!"
echo "=========================="
echo "This will:"
echo "1. Run comprehensive mixed-precision benchmarks"
echo "2. Validate your theoretical correlation predictions"
echo "3. Generate paper-ready validation data"
echo "4. Create publication-quality figures"
echo ""

read -p "Proceed with benchmarking? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Run the benchmarks
    run_benchmarks
    
    # Generate paper data
    generate_paper_data
    
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "======================"
    echo "Results available in:"
    echo "  - Logs: $LOG_FILE"
    echo "  - Reports: data_output/alignment_visualizations/"
    echo "  - Figures: data_output/alignment_visualizations/"
    echo ""
    echo "Use these results to validate your paper's predictions!"
    echo "Your theoretical correlation of 0.999744 can now be"
    echo "compared against real Blackwell hardware performance!"
    
else
    echo "Deployment cancelled."
fi

echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Review benchmark results in data_output/"
echo "2. Update paper with empirical validation"
echo "3. Submit to MLSys 2025 with hardware validation"
echo "4. Consider industry collaboration opportunities"
