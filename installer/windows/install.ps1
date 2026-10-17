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
$PythonDir = Join-Path $App "python"
$PythonBin = Join-Path $PythonDir "bin"
$UvCache = Join-Path $App "uv-cache"
$PipCache = Join-Path $App "pip-cache"
$Work = Join-Path $App "setup-work"
$AdminTemp = Join-Path $App "setup-temp"
$UserRoots = @()
foreach ($envName in @("LOCALAPPDATA", "APPDATA", "TEMP", "TMP")) {
    $raw = [Environment]::GetEnvironmentVariable($envName)
    if (-not [string]::IsNullOrWhiteSpace($raw)) {
        $UserRoots += [System.IO.Path]::GetFullPath($raw)
    }
}
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
    New-Item -ItemType Directory -Force $Work | Out-Null
    $probe = Join-Path $Work ("voxprint-gpu-probe-" + [guid]::NewGuid().ToString("n") + ".py")
    Set-Content -LiteralPath $probe -Value $code -Encoding ascii
    try {
        Assert-AdminPath $Python
        Assert-AdminPath $probe
        & $Python $probe | Out-Null
        return $LASTEXITCODE -eq 0
    } finally {
        Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
    }
}

function Assert-AdminPath([string]$Path) {
    # Elevated setup only runs files under the install folder. Program Files
    # needs admin rights to write. Profile, roaming, and temp folders do not.
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Setup will not run an empty program path."
    }
    $full = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetFullPath($App)
    if (-not $root.EndsWith('\')) { $root = $root + '\' }
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Setup will not run a program outside the install folder."
    }
    foreach ($userRoot in $UserRoots) {
        $prefix = $userRoot
        if (-not $prefix.EndsWith('\')) { $prefix = $prefix + '\' }
        if ($full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Setup will not run a program from a user-writable folder."
        }
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
    Assert-AdminPath $Python
    Assert-AdminPath $Uv
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
    foreach ($dir in @($AdminTemp, $UvCache, $PipCache)) {
        if ($dir -and (Test-Path -LiteralPath $dir)) {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-Py([string[]]$ArgList) {
    Assert-AdminPath $Py
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
    New-Item -ItemType Directory -Force $PythonDir, $PythonBin, $UvCache, $PipCache, $Work, $AdminTemp | Out-Null
    # Keep uv, pip, and child temp files inside the install folder. A user
    # uv.toml or the default profile install dir must not redirect them.
    $env:UV_PYTHON_INSTALL_DIR = $PythonDir
    $env:UV_PYTHON_BIN_DIR = $PythonBin
    $env:UV_CACHE_DIR = $UvCache
    $env:UV_NO_CONFIG = "1"
    $env:UV_MANAGED_PYTHON = "1"
    $env:PYTHONNOUSERSITE = "1"
    $env:PIP_CACHE_DIR = $PipCache
    $env:TEMP = $AdminTemp
    $env:TMP = $AdminTemp
    foreach ($name in @("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE")) {
        if (Test-Path "Env:$name") { Remove-Item "Env:$name" }
    }
    Write-Host "Installing Python with uv..."
    Assert-AdminPath $uv
    & $uv python install 3.14 --reinstall --no-registry --install-dir $PythonDir
    if ($LASTEXITCODE -ne 0) {
        throw "uv could not install Python 3.14. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    $found = (& $uv python find 3.14 --managed-python | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $found) {
        throw "uv installed Python 3.14 but the interpreter could not be found. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    $Base = (($found -split '\r?\n') | Where-Object { $_.Trim() } | Select-Object -Last 1).Trim()
    if (-not (Test-Path $Base)) {
        throw "uv reported Python at '$Base', but that file does not exist. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    Assert-AdminPath $Base

    # The shared runtime is user-writable. Never execute its python.exe.
    # Packages are installed with the uv interpreter via --target.
    # A standalone interpreter (python.exe at the runtime root, no pyvenv.cfg) is left alone.
    $runtimeVenv = (Test-Path -LiteralPath $RuntimePy) -and (Test-Path -LiteralPath $RuntimeCfg)
    if (-not $runtimeVenv) {
        $occupied = Get-ChildItem -Force $Runtime -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($occupied) {
            throw "The suite runtime is already installed. Canvas will not replace it."
        }
        New-Item -ItemType Directory -Force $Shared | Out-Null
        # --without-pip keeps venv from launching the new interpreter. That
        # interpreter lives in the user-writable shared tree.
        & $Base -m venv --without-pip $Runtime
        if ($LASTEXITCODE -ne 0) { throw "Python could not create the suite runtime." }
    }

    Assert-AdminPath $Base
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
    $Fetch = Join-Path $Work ("voxprint-wheels-" + [guid]::NewGuid().ToString("n"))
    New-Item -ItemType Directory -Force $Fetch | Out-Null
    $runtimeReady = Test-SuiteGpuFiles $RuntimeSite
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
        Install-SuiteGpuPackages $Base $RuntimeSite
        if (-not (Test-SuiteGpuFiles $RuntimeSite)) {
            Write-Host "CUDA 13.0 PyTorch is still missing from $RuntimeSite"
            Get-ChildItem -LiteralPath $RuntimeSite -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like "*.dist-info" } |
                ForEach-Object { Write-Host $_.Name }
            throw "CUDA 13.0 PyTorch is still missing."
        }
    }
    if (-not $appReady) {
        Install-SuiteGpuPackages $Py ""
        if (-not (Test-SuiteGpuPackages $Py)) { throw "Setup stopped." }
    }
    Assert-AdminPath $Uv
    Assert-AdminPath $Py
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
