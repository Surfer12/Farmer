#!/bin/bash

# UOIF Epistemic Model Documentation Compilation Script

echo "Compiling UOIF Epistemic Model Documentation..."

# Compile the main comprehensive document
echo "Building comprehensive rationale document..."
pdflatex epistemic-model-rationale.tex
bibtex epistemic-model-rationale
pdflatex epistemic-model-rationale.tex
pdflatex epistemic-model-rationale.tex

# Compile the existing UOIF analysis document
echo "Building UOIF analysis document..."
pdflatex uoif-analysis-2025.tex
pdflatex uoif-analysis-2025.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls

echo "Compilation complete!"
echo "Generated documents:"
echo "  - epistemic-model-rationale.pdf (Comprehensive rationale with proofs)"
echo "  - uoif-analysis-2025.pdf (Computational analysis)"
echo "  - model-rationale-summary.md (Executive summary)"
echo "  - epistemic-references.bib (Bibliography)"