# AgriVault – PySpark Runner
# ===========================
# Sets all required environment variables and runs a Python module.
#
# Usage:
#   .\scripts\run_spark.ps1 src.standardization.clean_apmc
#   .\scripts\run_spark.ps1 src.standardization.clean_weather
#   .\scripts\run_spark.ps1 src.standardization.clean_ndvi
#   .\scripts\run_spark.ps1 src.features.build_price_features
#
# All env vars are set session-locally (don't pollute system env).

param(
    [Parameter(Mandatory=$true)]
    [string]$Module
)

$ErrorActionPreference = "Stop"

# ── Java ──────────────────────────────────────────────────────────────────
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.20.8-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

# ── Hadoop winutils (required on Windows for PySpark) ────────────────────
$env:HADOOP_HOME = "C:\hadoop"

# ── Python path (makes `src.*` imports work) ─────────────────────────────
$env:PYTHONPATH = $PSScriptRoot | Split-Path -Parent

# ── Python executable ─────────────────────────────────────────────────────
$python = "C:\Users\LENOVO\AppData\Local\Programs\Python\Python314\python.exe"

Write-Host ""
Write-Host "=" * 60
Write-Host "AgriVault PySpark Runner"
Write-Host "  Module     : $Module"
Write-Host "  JAVA_HOME  : $env:JAVA_HOME"
Write-Host "  HADOOP_HOME: $env:HADOOP_HOME"
Write-Host "  PYTHONPATH : $env:PYTHONPATH"
Write-Host "=" * 60
Write-Host ""

& $python -m $Module
