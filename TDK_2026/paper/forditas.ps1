# Build the TDK paper (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
Remove-Item -ErrorAction SilentlyContinue paper.aux, paper.blg, paper.log, paper.out
Write-Host "done: paper.pdf"
