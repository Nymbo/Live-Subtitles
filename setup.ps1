<#
.SYNOPSIS
    Create or update Live Subtitles' Python environment (.venv) with uv.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Recreate
    .\setup.ps1 -CacheDir D:\uv-cache    # share one uv cache across projects on that drive

.NOTES
    Needs uv (https://docs.astral.sh/uv/) and Python 3.13 (the "py" launcher, or pass -Python).
    With -CacheDir on the same drive as this folder, uv hardlinks packages from the cache
    instead of copying them, so the ~2 GB of CUDA libraries are stored once.
#>
param(
    [string]$Python = "py -3.13",
    [string]$CacheDir = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$venvDir = Join-Path $PSScriptRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $PSScriptRoot "requirements.txt"

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
}
if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
    throw "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
}
if ($CacheDir) {
    $env:UV_CACHE_DIR = $CacheDir
    $env:UV_LINK_MODE = "hardlink"
}

if ($Recreate -and (Test-Path -LiteralPath $venvDir)) {
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    $parts = $Python -split " "
    $basePython = & $parts[0] @($parts | Select-Object -Skip 1) -c "import sys; print(sys.executable)"
    & $uv venv $venvDir --python $basePython --seed --no-python-downloads
}

& $uv pip install --python $pythonExe -r $requirements
& $uv pip check --python $pythonExe

Write-Host "Ready. Start the tray app with: .\.venv\Scripts\pythonw.exe run.pyw"
