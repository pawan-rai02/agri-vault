# =============================================================================
# AgriVault – Fetch Multi-Year Data (MODIS NDVI + APMC History)
# =============================================================================
# Runs both data pipelines in sequence:
#   1. MODIS NDVI from NASA/GEE (2021–2025) — baseline + anomalies
#   2. APMC historical prices from Agmarknet (2021–2025)
#
# Prerequisites:
#   - earthengine authenticate (for GEE access)
#   - Active internet connection
#   - Python environment with earthengine-api, pandas, requests
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/fetch_multiyear_data.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/fetch_multiyear_data.ps1 -Limit 50
# =============================================================================

param(
    [int]$Limit = $null,
    [int]$BatchSize = 250,
    [string]$ApmcStart = "2021-01-01",
    [string]$ApmcEnd = "2025-12-31"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogFile = "$ProjectRoot\logs\fetch_multiyear_$(Get-Date -Format 'yyyy-MM-dd_HHmmss').log"

# Ensure logs directory exists
New-Item -ItemType Directory -Path "$ProjectRoot\logs" -Force | Out-Null

function Write-Step {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LogFile -Value $line
}

function Write-Success {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] ✓ $Message"
    Write-Host $line -ForegroundColor Green
    Add-Content -Path $LogFile -Value $line
}

function Write-Error {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] ✗ ERROR: $Message"
    Write-Host $line -ForegroundColor Red
    Add-Content -Path $LogFile -Value $line
}

# Start
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host "  AgriVault Multi-Year Data Fetch" -ForegroundColor Yellow
Write-Host "  MODIS NDVI (NASA) + APMC History (Agmarknet)" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host ""

$StartTime = Get-Date

# ---------------------------------------------------------------------------
# Step 1: MODIS NDVI
# ---------------------------------------------------------------------------
Write-Step "PHASE 1/2 — Fetching MODIS NDVI from NASA (via Google Earth Engine)"
Write-Step "  Baseline: 2021-01-01 → 2024-12-31 (4 years)"
Write-Step "  Current:  2025-01-01 → 2025-12-31"
Write-Host ""

$geeArgs = @("-m", "src.ingestion.fetch_modis_ndvi")
if ($Limit) { $geeArgs += "--limit"; $geeArgs += $Limit.ToString() }
$geeArgs += "--batch-size"; $geeArgs += $BatchSize.ToString()

try {
    & python @geeArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) { throw "MODIS NDVI fetch failed with exit code $LASTEXITCODE" }
    Write-Success "MODIS NDVI data fetched successfully"
} catch {
    Write-Error "MODIS NDVI fetch failed: $_"
    Write-Host ""
    Write-Host "Common fixes:" -ForegroundColor Yellow
    Write-Host "  1. Run: earthengine authenticate" -ForegroundColor Yellow
    Write-Host "  2. Check GEE project ID in configs/gee_config.yaml" -ForegroundColor Yellow
    Write-Host "  3. Try with --limit 10 first to test" -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------------------
# Step 2: APMC Historical Prices
# ---------------------------------------------------------------------------
Write-Host ""
Write-Step "PHASE 2/2 — Fetching APMC historical prices from Agmarknet"
Write-Step "  Date range: $ApmcStart → $ApmcEnd"
Write-Host ""

$apmcArgs = @("-m", "src.ingestion.fetch_apmc_history",
              "--start", $ApmcStart,
              "--end", $ApmcEnd)
if ($Limit) { $apmcArgs += "--limit"; $apmcArgs += $Limit.ToString() }

try {
    & python @apmcArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) { throw "APMC fetch failed with exit code $LASTEXITCODE" }
    Write-Success "APMC historical prices fetched successfully"
} catch {
    Write-Error "APMC fetch failed: $_"
    Write-Host ""
    Write-Host "Fallback: manually download CSVs from https://agmarknet.gov.in" -ForegroundColor Yellow
    Write-Host "  Place files in data/raw/apmc/" -ForegroundColor Yellow
    # Don't exit — APMC failure is non-fatal for MODIS data
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
$Duration = (Get-Date) - $StartTime

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host "  FETCH COMPLETE" -ForegroundColor Green
Write-Host "  Duration: $($Duration.Minutes)m $($Duration.Seconds)s" -ForegroundColor Green
Write-Host "  Log: $LogFile" -ForegroundColor Gray
Write-Host ""

# List output files
$ndviDir = "$ProjectRoot\data\raw\ndvi_modis"
$apmcDir = "$ProjectRoot\data\raw\apmc"

if (Test-Path $ndviDir) {
    Write-Host "MODIS NDVI outputs:" -ForegroundColor Cyan
    Get-ChildItem $ndviDir -Filter "*.csv" | ForEach-Object {
        Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" -ForegroundColor White
    }
}

if (Test-Path $apmcDir) {
    Write-Host "APMC outputs:" -ForegroundColor Cyan
    Get-ChildItem $apmcDir -Filter "*.csv" | ForEach-Object {
        Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. python -m src.standardization.join_ndvi_anomaly" -ForegroundColor White
Write-Host "  2. python -m src.features.build_price_features" -ForegroundColor White
Write-Host ""

# Write tail of log
Write-Host "Last 10 log lines:" -ForegroundColor Gray
Get-Content $LogFile -Tail 10
