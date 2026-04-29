param(
    [string]$VenvPath = ".venv",
    [string]$PythonVersion = "3.12",
    [switch]$CpuOnly,
    [switch]$SkipOpenSRModel
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvFullPath = Join-Path $repoRoot $VenvPath

function Get-VenvBootstrapCommand {
    param(
        [string]$PreferredPythonVersion
    )

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyCommand) {
        $available = & py -0p 2>$null
        if ($LASTEXITCODE -eq 0 -and $available -match [regex]::Escape("-V:$PreferredPythonVersion")) {
            return @("py", "-$PreferredPythonVersion", "-m", "venv")
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        return @($pythonCommand.Source, "-m", "venv")
    }

    throw "No Python interpreter was found. Install Python $PreferredPythonVersion or newer first."
}

$venvBootstrap = Get-VenvBootstrapCommand -PreferredPythonVersion $PythonVersion

Write-Host "Creating virtual environment at $venvFullPath"
& $venvBootstrap[0] $venvBootstrap[1] $venvBootstrap[2] $venvBootstrap[3] $venvFullPath

$pythonExe = Join-Path $venvFullPath "Scripts\python.exe"

& $pythonExe -m pip install --upgrade pip setuptools wheel

function Install-Torch {
    param(
        [switch]$UseCpuOnly
    )

    & $pythonExe -m pip uninstall -y torch torchvision | Out-Null

    if ($UseCpuOnly) {
        & $pythonExe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
        return
    }

    & $pythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

if ($CpuOnly) {
    Install-Torch -UseCpuOnly
} else {
    Install-Torch

    $torchImportCheck = & $pythonExe -c "import torch; print(torch.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CUDA Torch import failed. Falling back to CPU wheels."
        Install-Torch -UseCpuOnly
    }
}

& $pythonExe -m pip install -r (Join-Path $repoRoot "requirements\base.txt")

if (-not $SkipOpenSRModel) {
    & $pythonExe -m pip install -r (Join-Path $repoRoot "requirements\opensr-model.txt")
}

Write-Host "Environment ready."
Write-Host "Inspect current input with: ./scripts/inspect_input.ps1 -Config configs/barcelona.current.yaml"
Write-Host "Run inference with: ./scripts/run_inference.ps1 -Config configs/barcelona.current.yaml"
