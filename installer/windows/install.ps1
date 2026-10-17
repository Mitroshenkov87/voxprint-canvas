# Install VOXPrint AI Canvas. Program files live beside this script.
# Per-user data stays in %LOCALAPPDATA%\Voxprint.
# The setup does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$App = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $App "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Shared = Join-Path $env:LOCALAPPDATA "Voxprint\shared"
$RuntimeRel = "runtime/py314-torch2.11-cu130"
$Runtime = Join-Path $Shared ($RuntimeRel -replace '/', '\')
$RuntimePy = Join-Path $Runtime "Scripts\python.exe"
$RuntimeCfg = Join-Path $Runtime "pyvenv.cfg"
$RuntimeSite = Join-Path $Runtime "Lib\site-packages"
$Fetch = $null
$Trt = Join-Path $Shared "tensorrt-10"
$WindowsPythonUrl = "https://www.python.org/ftp/python/3.14.8/python-3.14.8-amd64.exe"
$TorchIndex = "https://download.pytorch.org/whl/cu130"
$LogDir = Join-Path $env:LOCALAPPDATA "Voxprint\logs"
$Transcript = Join-Path $LogDir "canvas-install.log"
$TranscriptOn = $false

function Test-StorePythonAlias([string]$CommandPath) {
    if ([string]::IsNullOrWhiteSpace($CommandPath)) {
        return $false
    }
    $root = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
    $full = [System.IO.Path]::GetFullPath($CommandPath)
    return $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-SuiteGpuPackages([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python)) {
        return $false
    }
    $ort = "onnx" + "runtime"
    $code = @"
import importlib
import importlib.metadata as meta
import sys
try:
    import torch
except Exception:
    sys.exit(1)
cuda = getattr(getattr(torch, "version", None), "cuda", None) or ""
if not str(cuda).startswith("13.0"):
    sys.exit(1)
name = "$ort"
gpu = name + "-gpu"
try:
    importlib.import_module(name)
except Exception:
    sys.exit(1)
found = {dist.metadata["Name"].lower() for dist in meta.distributions()}
if gpu not in found or name in found:
    sys.exit(1)
sys.exit(0)
"@
    $probe = Join-Path $env:TEMP ("voxprint-gpu-probe-" + [guid]::NewGuid().ToString("n") + ".py")
    Set-Content -LiteralPath $probe -Value $code -Encoding ascii
    try {
        & $Python $probe | Out-Null
        return $LASTEXITCODE -eq 0
    } finally {
        Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
    }
}

function Test-MetadataVersion([string]$SitePackages, [string]$NamePrefix, [string]$Version) {
    if (-not (Test-Path -LiteralPath $SitePackages)) {
        return $false
    }
    $dirs = @(Get-ChildItem -LiteralPath $SitePackages -Directory -ErrorAction SilentlyContinue)
    foreach ($dir in $dirs) {
        $name = $dir.Name
        if (-not $name.StartsWith($NamePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if (-not $name.EndsWith(".dist-info", [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        $meta = Join-Path $dir.FullName "METADATA"
        if (-not (Test-Path -LiteralPath $meta)) {
            continue
        }
        $raw = Get-Content -LiteralPath $meta -Raw -ErrorAction SilentlyContinue
        if ($raw -and $raw.Contains("Version: $Version")) {
            return $true
        }
    }
    return $false
}

function Test-SuiteGpuFiles([string]$SitePackages) {
    if (-not (Test-Path -LiteralPath $SitePackages)) {
        return $false
    }
    $versionPy = Join-Path $SitePackages "torch\version.py"
    if (-not (Test-Path -LiteralPath $versionPy)) {
        return $false
    }
    $text = Get-Content -LiteralPath $versionPy -Raw -ErrorAction SilentlyContinue
    if (-not $text -or -not $text.Contains("13.0")) {
        return $false
    }
    $gpuName = "onnx" + "runtime" + "_gpu-"
    $cpuName = "onnx" + "runtime" + "-"
    $cpu = $false
    $dirs = @(Get-ChildItem -LiteralPath $SitePackages -Directory -ErrorAction SilentlyContinue)
    foreach ($dir in $dirs) {
        $name = $dir.Name
        if ($name.StartsWith($cpuName, [System.StringComparison]::OrdinalIgnoreCase) -and $name.EndsWith(".dist-info", [System.StringComparison]::OrdinalIgnoreCase)) {
            $cpu = $true
        }
    }
    return ((Test-MetadataVersion $SitePackages "torch-" "2.11.0+cu130") -and (Test-MetadataVersion $SitePackages "torchvision-" "0.26.0+cu130") -and (Test-MetadataVersion $SitePackages $gpuName "1.31.0") -and -not $cpu)
}

function Install-SuiteGpuPackages([string]$Python, [string]$Target) {
    New-Item -ItemType Directory -Force $Fetch | Out-Null
    $into = @("--python", $Python, "--find-links", $Fetch, "-c", $Constraints)
    if (-not [string]::IsNullOrWhiteSpace($Target)) {
        New-Item -ItemType Directory -Force $Target | Out-Null
        $into += @("--target", $Target)
    }
    & $Uv pip install @into --index $TorchIndex --index-strategy unsafe-best-match "torch==2.11.0+cu130" "torchvision==0.26.0+cu130"
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    & $Uv pip install @into --index-url "https://pypi.org/simple" "onnxruntime-gpu==1.31.0"
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    $cpu = "onnx" + "runtime"
    if (-not [string]::IsNullOrWhiteSpace($Target)) {
        if (Test-MetadataVersion $Target ($cpu + "-") "") {
            & $Uv pip uninstall --python $Python --target $Target $cpu
            if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
        }
    } else {
        & $Uv pip uninstall --python $Python $cpu
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    }
}

function Clear-WheelFetch {
    if ($Fetch -and (Test-Path -LiteralPath $Fetch)) {
        Remove-Item -LiteralPath $Fetch -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-Py([string[]]$ArgList) {
    & $Py @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "Setup stopped."
    }
}

New-Item -ItemType Directory -Force $LogDir | Out-Null
try {
    Start-Transcript -Path $Transcript -Force | Out-Null
    $TranscriptOn = $true
} catch {
    Write-Host "The install log could not be opened: $Transcript"
}

try {
    Write-Host "Checking Python..."
    $uv = Join-Path $App "uv.exe"
    if (-not (Test-Path $uv)) {
        throw "uv.exe is not bundled next to install.ps1. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    Write-Host "Installing Python with uv..."
    & $uv python install 3.14 --reinstall
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

    # The shared runtime is user-writable. Execute its python.exe only when this
    # setup just created the venv. An existing tree is not run and not upgraded.
    # A standalone interpreter (python.exe at the runtime root, no pyvenv.cfg) is left alone.
    $createdRuntime = $false
    $runtimeVenv = (Test-Path -LiteralPath $RuntimePy) -and (Test-Path -LiteralPath $RuntimeCfg)
    if (-not $runtimeVenv) {
        $occupied = Get-ChildItem -Force $Runtime -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($occupied) {
            throw "The suite runtime is already installed. Canvas will not replace it."
        }
        New-Item -ItemType Directory -Force $Shared | Out-Null
        & $Base -m venv $Runtime
        if ($LASTEXITCODE -ne 0) { throw "Python could not create the suite runtime." }
        $createdRuntime = $true
    }

    if (-not (Test-Path $Py)) {
        & $Base -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "Python could not create the environment." }
    } else {
        & $Base -m venv --upgrade $Venv
        if ($LASTEXITCODE -ne 0) { throw "Python could not refresh the environment." }
    }

    $Constraints = Join-Path $App "constraints.txt"
    if (-not (Test-Path $Constraints)) {
        throw "constraints.txt is missing from the setup. The install cannot continue."
    }
    Invoke-Py @("-m", "pip", "install", "--upgrade", "pip")
    # PyPI is the main index. torch and torchvision come from the CUDA 13.0 index.
    # torchao is installed from PyPI with an exact pin.
    $Uv = Join-Path $App "uv.exe"
    if (-not (Test-Path $Uv)) {
        Invoke-Py @("-m", "pip", "install", "--index-url", "https://pypi.org/simple", "uv==0.13.0")
        $Uv = Join-Path $Venv "Scripts\uv.exe"
    }
    if (-not (Test-Path $Uv)) {
        throw "uv 0.13.0 is required to install PyTorch."
    }
    $Fetch = Join-Path $env:TEMP ("voxprint-wheels-" + [guid]::NewGuid().ToString("n"))
    New-Item -ItemType Directory -Force $Fetch | Out-Null
    if ($createdRuntime) {
        $runtimeReady = Test-SuiteGpuPackages $RuntimePy
    } else {
        $runtimeReady = Test-SuiteGpuFiles $RuntimeSite
    }
    $appReady = Test-SuiteGpuPackages $Py
    if ($runtimeReady -and $appReady) {
        Write-Host "PyTorch is already installed."
    } else {
        Write-Host "Installing PyTorch..."
        Invoke-Py @(
            "-m", "pip", "download", "-c", $Constraints,
            "torch==2.11.0+cu130", "torchvision==0.26.0+cu130",
            "--index-url", $TorchIndex, "--extra-index-url", "https://pypi.org/simple", "-d", $Fetch
        )
        Invoke-Py @(
            "-m", "pip", "download", "-c", $Constraints,
            "onnxruntime-gpu==1.31.0", "torchao==0.18.0",
            "--index-url", "https://pypi.org/simple", "-d", $Fetch
        )
    }
    if (-not $runtimeReady) {
        Write-Host "Repairing the suite runtime..."
        if ($createdRuntime) {
            Install-SuiteGpuPackages $RuntimePy ""
            if (-not (Test-SuiteGpuPackages $RuntimePy)) { throw "CUDA 13.0 PyTorch is still missing." }
        } else {
            Install-SuiteGpuPackages $Base $RuntimeSite
            if (-not (Test-SuiteGpuFiles $RuntimeSite)) { throw "CUDA 13.0 PyTorch is still missing." }
        }
    }
    if (-not $appReady) {
        Install-SuiteGpuPackages $Py ""
        if (-not (Test-SuiteGpuPackages $Py)) { throw "Setup stopped." }
    }
    & $Uv pip install --python $Py --index-url https://pypi.org/simple "torchao==0.18.0"
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    & $Uv pip install --python $Py --index-url https://pypi.org/simple -c $Constraints "${App}[gen]"
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    & $Py -m canvas.infra.layout migrate --install-dir $App
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    & $Py -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "TensorRT is skipped on Python 3.14."
    } elseif (-not (Get-ChildItem -Force $Trt -ErrorAction SilentlyContinue | Select-Object -First 1)) {
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
    Clear-WheelFetch
    Write-Host "Installed."
    if ($TranscriptOn) {
        Stop-Transcript | Out-Null
        $TranscriptOn = $false
    }
} catch {
    Write-Host $_
    Clear-WheelFetch
    if ($TranscriptOn) {
        try { Stop-Transcript | Out-Null } catch { }
    }
    exit 1
}
