# AgriVault – App Launcher
# Run this script to start the Flask dashboard + API.
# Press Ctrl+C in this window to stop the server.

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "AgriVault - http://127.0.0.1:5000"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  AgriVault Dashboard" -ForegroundColor Green
Write-Host "  http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting server... (Press Ctrl+C to stop)" -ForegroundColor Yellow
Write-Host ""

$env:PYTHONPATH = "."
py -m src.api.app
