# Full reproduction run (Windows PowerShell).
# Runs the model on data/budapest_gtfs.zip, regenerates every table in the
# paper (tdk_rerun.py -> paper_dynamics.py -> paper_tables.py), then builds
# the PDF.  Takes roughly 30-40 minutes, mostly the node-removal sweep.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pip install --quiet -e ".[all]"
python scripts/tdk_rerun.py --zip data/budapest_gtfs.zip
Set-Location paper
.\forditas.ps1
