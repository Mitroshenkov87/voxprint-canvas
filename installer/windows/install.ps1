# Install Voxprint AI Canvas into a local environment.
# The setup does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $App "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Shared = Join-Path $env:LOCALAPPDATA "Voxprint\shared"
$Wheels = Join-Path $Shared "wheels"
$WindowsPythonUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"

function Invoke-Py([string[]]$ArgList) {
    & $Py @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "Setup stopped."
    }
}

Write-Host "Checking Python..."
$Base = $null
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    foreach ($spec in @("-3.12", "-3")) {
        & py $spec -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) {
            $Base = $spec
            break
        }
    }
}
if (-not $Base) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        & $pythonCmd.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) {
            $Base = $pythonCmd.Source
        }
    }
}
if (-not $Base) {
    $uv = Join-Path $App "uv.exe"
    if (-not (Test-Path $uv)) {
        throw "Python 3.12 is not installed, and uv.exe is not bundled next to install.ps1. Install Python 3.12 from $WindowsPythonUrl and run this setup again."
    }
    Write-Host "Installing Python with uv..."
    & $uv python install 3.12
    if ($LASTEXITCODE -ne 0) {
        throw "uv could not install Python 3.12. Install Python 3.12 from $WindowsPythonUrl and run this setup again."
    }
    $found = (& $uv python find 3.12 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $found) {
        throw "uv installed Python 3.12 but the interpreter could not be found. Install Python 3.12 from $WindowsPythonUrl and run this setup again."
    }
    $Base = (($found -split '\r?\n') | Where-Object { $_.Trim() } | Select-Object -Last 1).Trim()
    if (-not (Test-Path $Base)) {
        throw "uv reported Python at '$Base', but that file does not exist. Install Python 3.12 from $WindowsPythonUrl and run this setup again."
    }
}

if (-not (Test-Path $Py)) {
    if ($Base -eq "-3.12" -or $Base -eq "-3") {
        & py $Base -m venv $Venv
    } else {
        & $Base -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Python could not create the environment." }
}

$Constraints = Join-Path $App "constraints.txt"
if (-not (Test-Path $Constraints)) {
    throw "constraints.txt is missing from the setup. The install cannot continue."
}
Invoke-Py @("-m", "pip", "install", "--upgrade", "pip")
Write-Host "Installing PyTorch..."
New-Item -ItemType Directory -Force $Wheels | Out-Null
Invoke-Py @("-m", "pip", "download", "-c", $Constraints, "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128", "-d", $Wheels)
Invoke-Py @("-m", "pip", "install", "-c", $Constraints, "--upgrade-strategy", "only-if-needed", "--find-links", $Wheels, "torch", "torchvision")
Invoke-Py @("-m", "pip", "install", "-c", $Constraints, "--upgrade-strategy", "only-if-needed", "${App}[gen]")

Write-Host "Installing the default image pack..."
Invoke-Py @("-m", "canvas.app.provision")
$tierDir = Join-Path $env:LOCALAPPDATA "Voxprint\Canvas"
New-Item -ItemType Directory -Force $tierDir | Out-Null
& $Py -m canvas.models.tiers | Set-Content -Encoding ascii (Join-Path $tierDir "tier.txt")
if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
Invoke-Py @("-m", "canvas.app.main", "--register-models")
Write-Host "Installed."
