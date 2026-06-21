# Script de lancement du projet Deep Learning
# Usage: .\run_project.ps1
# Durée estimée: ~15-20 min sur CPU

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Python = Join-Path $Root ".." ".venv" "Scripts" "python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
    Write-Host "venv non trouvé, utilisation de python système"
}

Set-Location $Root

Write-Host "`n=== 1/4 Installation des dépendances ===" -ForegroundColor Cyan
& $Python -m pip install -r requirements.txt -q

Write-Host "`n=== 2/4 Exécution des 3 parties (MLP, CNN, RNN) ===" -ForegroundColor Cyan
Write-Host "Patience: ~15 min sur CPU..." -ForegroundColor Yellow
$env:PYTHONUNBUFFERED = "1"
& $Python deliverables\run_all_and_collect.py
if ($LASTEXITCODE -ne 0) { throw "Echec run_all_and_collect.py" }

Write-Host "`n=== 3/4 Génération rapport HTML + PDF ===" -ForegroundColor Cyan
& $Python deliverables\generate_report_html.py
& $Python deliverables\make_pdf_from_report_and_figures.py

Write-Host "`n=== 4/4 Création archive ZIP ===" -ForegroundColor Cyan
& $Python package_submission.py --skip-run

Write-Host "`n=== TERMINÉ ===" -ForegroundColor Green
Write-Host "Rapport PDF : deliverables\report_generated.pdf"
Write-Host "Archive     : deliverables\submission_final_new.zip"
Write-Host "Log         : deliverables\last_run.log"
