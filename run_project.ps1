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

Write-Host "`n=== 1/3 Installation des dépendances ===" -ForegroundColor Cyan
& $Python -m pip install -r requirements.txt -q

Write-Host "`n=== 2/3 Exécution des 3 parties + collecte ===" -ForegroundColor Cyan
Write-Host "Patience: ~15 min sur CPU..." -ForegroundColor Yellow
$env:PYTHONUNBUFFERED = "1"
& $Python package_submission.py

Write-Host "`n=== TERMINÉ ===" -ForegroundColor Green
Write-Host "Archive à remettre : deliverables\submission_final_new.zip"
