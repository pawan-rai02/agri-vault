# =============================================================================
# AgriVault – Join NDVI Anomaly + APMC and Upload to S3
# =============================================================================
# Runs the joiner script that matches MODIS NDVI anomalies with APMC data,
# then uploads the combined dataset to S3.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/join_and_upload_ndvi_anomaly.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host "  AgriVault NDVI Anomaly × APMC Join + Upload" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Yellow

# Step 1: Run the joiner
Write-Host ""
Write-Host "[1/2] Joining NDVI anomalies with APMC prices..." -ForegroundColor Cyan

& python -m src.standardization.join_ndvi_anomaly
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Join failed" -ForegroundColor Red
    exit 1
}

# Step 2: Upload to S3
Write-Host ""
Write-Host "[2/2] Uploading to S3..." -ForegroundColor Cyan

& python scripts/s3_upload_standardized.py --dataset joined
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: S3 upload failed (non-fatal)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Next: python -m src.features.build_price_features" -ForegroundColor White
