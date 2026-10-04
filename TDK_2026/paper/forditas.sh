#!/usr/bin/env bash
# Build the TDK paper.  Windows: forditas.ps1
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
rm -f paper.aux paper.blg paper.log paper.out
echo "done: paper.pdf"
