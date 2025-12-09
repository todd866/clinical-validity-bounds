#!/bin/bash
# Clean build script for LaTeX papers
# Compiles PDF and automatically cleans up auxiliary files
# Usage: ./build_clean.sh [paper_name_without_extension]

# Auto-detect paper name if not provided
if [ -z "$1" ]; then
    # Find first .tex file in directory
    PAPER=$(ls *.tex 2>/dev/null | head -1 | sed 's/\.tex$//')
    if [ -z "$PAPER" ]; then
        echo "ERROR: No .tex file found in current directory"
        exit 1
    fi
else
    PAPER="$1"
fi

echo "Building ${PAPER}.pdf..."

# First pass
pdflatex -interaction=nonstopmode ${PAPER}.tex > /tmp/${PAPER}_build.log 2>&1 || true

# BibTeX pass
bibtex ${PAPER} >> /tmp/${PAPER}_build.log 2>&1 || true

# Second pass for references
pdflatex -interaction=nonstopmode ${PAPER}.tex >> /tmp/${PAPER}_build.log 2>&1 || true

# Third pass for cross-references
pdflatex -interaction=nonstopmode ${PAPER}.tex >> /tmp/${PAPER}_build.log 2>&1 || true

# Check PDF was created (primary indicator of success)
if [ ! -f ${PAPER}.pdf ]; then
    echo "ERROR: PDF was not generated. See /tmp/${PAPER}_build.log"
    tail -30 /tmp/${PAPER}_build.log
    exit 1
fi

# Check for fatal errors (excluding missing figure warnings)
if grep "^!" /tmp/${PAPER}_build.log | grep -v "File.*not found" | grep -q "^!"; then
    echo "WARNING: Non-fatal LaTeX errors detected:"
    grep "^!" /tmp/${PAPER}_build.log | grep -v "File.*not found" | head -5
    echo "See /tmp/${PAPER}_build.log for details"
fi

# Get page count
PAGES=$(pdfinfo ${PAPER}.pdf 2>/dev/null | grep Pages | awk '{print $2}')

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f ${PAPER}.aux ${PAPER}.log ${PAPER}.out ${PAPER}.bbl ${PAPER}.blg ${PAPER}.toc ${PAPER}.lof ${PAPER}.lot

echo "✓ Build complete: ${PAPER}.pdf (${PAGES} pages)"
echo "  Build log: /tmp/${PAPER}_build.log"
