# AgriVault – Nightly Serving Snapshot Refresh
# ==============================================
# Rebuilds the live-prediction feature snapshot from the Gold price_features
# table.  Schedule this to run daily AFTER the Spark standardization + Gold
# build finishes (e.g. 03:00 AM).
#
# S3 input : s3://agrivault-lake-pawan/features/price_features/
# S3 output: s3://agrivault-lake-pawan/features/serving_snapshot/latest.parquet
#
# Usage (manual):
#   .\scripts\refresh_serving_snapshot.ps1
#
# Task Scheduler setup:
#   Action  : PowerShell.exe -ExecutionPolicy Bypass -File "D:\agri-vault\scripts\refresh_serving_snapshot.ps1"
#   Trigger : Daily at 03:00 AM
#   Start in: D:\agri-vault
#
# Requires: Python 3.10+, AWS CLI profile "agrivault" configured.

$ErrorActionPreference = "Stop"

# ── Python executable (same as run_spark.ps1) ────────────────────────────
$python = "C:\Users\LENOVO\AppData\Local\Programs\Python\Python314\python.exe"

# ── Project root (parent of scripts/) ────────────────────────────────────
$ProjectRoot = $PSScriptRoot | Split-Path -Parent

# ── Log file (append daily) ──────────────────────────────────────────────
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$LogFile = Join-Path $LogDir ("snapshot_refresh_{0:yyyy-MM-dd}.log" -f (Get-Date))

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts  $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

# ── Banner ────────────────────────────────────────────────────────────────
Write-Log "============================================================"
Write-Log "AgriVault – Serving Snapshot Refresh"
Write-Log "  Project root : $ProjectRoot"
Write-Log "  Python       : $python"
Write-Log "  Log file     : $LogFile"
Write-Log "============================================================"

# ── Run snapshot build ────────────────────────────────────────────────────
$env:PYTHONPATH = $ProjectRoot

Write-Log "Starting snapshot build..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()

try {
    & $python -m src.features.build_serving_snapshot 2>&1 | ForEach-Object {
        Write-Log "  $_"
    }
    $exitCode = $LASTEXITCODE
}
catch {
    $exitCode = 1
    Write-Log "ERROR: $($_.Exception.Message)"
}

$sw.Stop()
$duration = $sw.Elapsed

# ── Result ────────────────────────────────────────────────────────────────
if ($exitCode -eq 0) {
    Write-Log "Snapshot refresh completed successfully in $($duration.ToString('hh\:mm\:ss'))"
} else {
    Write-Log "Snapshot refresh FAILED (exit code $exitCode) after $($duration.ToString('hh\:mm\:ss'))"
    Write-Log "Check S3 credentials and network connectivity."
}

Write-Log "Done."
Write-Log ""
