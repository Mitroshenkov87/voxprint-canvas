# Install Voxprint AI Canvas into a local environment.
# The setup does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $App "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Shared = Join-Path $env:LOCALAPPDATA "Voxprint\shared"
$Wheels = Join-Path $Shared "wheels"
$RuntimeRel = "runtime/py314-torch2.11-cu130"
$Runtime = Join-Path $Shared ($RuntimeRel -replace '/', '\')
$RuntimePy = Join-Path $Runtime "Scripts\python.exe"
$Trt = Join-Path $Shared "tensorrt-10"
$WindowsPythonUrl = "https://www.python.org/ftp/python/3.14.8/python-3.14.8-amd64.exe"
$TorchIndex = "https://download.pytorch.org/whl/cu130"

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
    foreach ($spec in @("-3.14", "-3")) {
        & py $spec -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"
        if ($LASTEXITCODE -eq 0) {
            $Base = $spec
            break
        }
    }
}
if (-not $Base) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        & $pythonCmd.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"
        if ($LASTEXITCODE -eq 0) {
            $Base = $pythonCmd.Source
        }
    }
}
if (-not $Base) {
    $uv = Join-Path $App "uv.exe"
    if (-not (Test-Path $uv)) {
        throw "Python 3.14 is not installed, and uv.exe is not bundled next to install.ps1. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    Write-Host "Installing Python with uv..."
    & $uv python install 3.14
    if ($LASTEXITCODE -ne 0) {
        throw "uv could not install Python 3.14. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    $found = (& $uv python find 3.14 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $found) {
        throw "uv installed Python 3.14 but the interpreter could not be found. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    $Base = (($found -split '\r?\n') | Where-Object { $_.Trim() } | Select-Object -Last 1).Trim()
    if (-not (Test-Path $Base)) {
        throw "uv reported Python at '$Base', but that file does not exist. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
}

$RootPy = Join-Path $Runtime "python.exe"
if (Test-Path $RootPy) {
    $RuntimePy = $RootPy
}
$Fresh = -not (Test-Path $RuntimePy)
if ($Fresh) {
    $occupied = Get-ChildItem -Force $Runtime -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($occupied) {
        throw "The suite runtime is already installed. Canvas will not replace it."
    }
    New-Item -ItemType Directory -Force $Shared | Out-Null
    if ($Base -eq "-3.14" -or $Base -eq "-3") {
        & py $Base -m venv $Runtime
    } else {
        & $Base -m venv $Runtime
    }
    if ($LASTEXITCODE -ne 0) { throw "Python could not create the suite runtime." }
}

if (-not (Test-Path $Py)) {
    & $RuntimePy -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Python could not create the environment." }
}

$Constraints = Join-Path $App "constraints.txt"
if (-not (Test-Path $Constraints)) {
    throw "constraints.txt is missing from the setup. The install cannot continue."
}
Invoke-Py @("-m", "pip", "install", "--upgrade", "pip")
# PyPI is the main index. uv pins torch and torchvision to the CUDA 13.0 index.
$Uv = Join-Path $App "uv.exe"
if (-not (Test-Path $Uv)) {
    Invoke-Py @("-m", "pip", "install", "--index-url", "https://pypi.org/simple", "uv==0.13.0")
    $Uv = Join-Path $Venv "Scripts\uv.exe"
}
if (-not (Test-Path $Uv)) {
    throw "uv 0.13.0 is required to install PyTorch."
}
$UvConfig = Join-Path $App "canvas\infra\uv.cu130.toml"
if (-not (Test-Path $UvConfig)) {
    throw "The CUDA index configuration is missing from the setup. The install cannot continue."
}
if ($Fresh) {
    & $RuntimePy -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    & $Uv pip install --python $RuntimePy --config-file $UvConfig -c $Constraints torch torchvision onnxruntime-gpu
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
}
Write-Host "Installing PyTorch..."
New-Item -ItemType Directory -Force $Wheels | Out-Null
Invoke-Py @(
    "-m", "pip", "download", "-c", $Constraints,
    "torch==2.11.0+cu130", "torchvision==0.26.0+cu130", "onnxruntime-gpu",
    "--index-url", "https://pypi.org/simple", "--extra-index-url", $TorchIndex, "-d", $Wheels
)
& $Uv pip install --python $Py --config-file $UvConfig -c $Constraints "${App}[gen]"
if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
if (-not (Get-ChildItem -Force $Trt -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    New-Item -ItemType Directory -Force $Trt | Out-Null
    Invoke-Py @("-m", "pip", "install", "-c", $Constraints, "--target", $Trt, "--upgrade-strategy", "only-if-needed", "tensorrt-cu13==10.16.1.11")
}

Write-Host "Installing the default image pack..."
Invoke-Py @("-m", "canvas.app.provision")
$tierDir = Join-Path $env:LOCALAPPDATA "Voxprint\Canvas"
New-Item -ItemType Directory -Force $tierDir | Out-Null
& $Py -m canvas.models.tiers | Set-Content -Encoding ascii (Join-Path $tierDir "tier.txt")
if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
Invoke-Py @("-m", "canvas.app.main", "--register-models")
Write-Host "Installed."
