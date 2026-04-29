param(
    [string]$Config = "configs/barcelona.current.yaml",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path (Join-Path $repoRoot $VenvPath) "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found at $pythonExe. Run ./scripts/setup_env.ps1 first."
}

Push-Location $repoRoot
try {
    $env:PYTHONPATH = "src"
    & $pythonExe -m opensr_pipeline.run_inference --config $Config
}
finally {
    Pop-Location
}
