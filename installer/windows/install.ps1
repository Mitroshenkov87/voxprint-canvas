# Install VOXPrint AI Canvas. Program files live beside this script.
# Per-user data stays in %LOCALAPPDATA%\VOXPrint AI Suite (voxprint-common 0.2.5 data home).
# The setup does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
# Scope Machine writes only under the install folder and runs elevated.
# Scope User writes the profile and runs as the original user.
# Scope All is the direct script path: profile writes stay in-process and every path is checked for links.
param(
    [ValidateSet("All", "Machine", "User", "Cleanup")]
    [string]$Scope = "All"
)
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$scriptPath = $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($scriptPath)) { $scriptPath = $PSCommandPath }
$App = Split-Path -Parent $scriptPath
$Venv = Join-Path $App "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$SuiteHome = Join-Path $env:LOCALAPPDATA "VOXPrint AI Suite"
$LegacyHome = Join-Path $env:LOCALAPPDATA "Voxprint"
$Shared = Join-Path $SuiteHome "shared"
$RuntimeRel = "runtime/py314-torch2.11-cu130"
$Runtime = Join-Path $Shared ($RuntimeRel -replace '/', '\')
$RuntimePy = Join-Path $Runtime "Scripts\python.exe"
$RuntimeCfg = Join-Path $Runtime "pyvenv.cfg"
$RuntimeSite = Join-Path $Runtime "Lib\site-packages"
$PythonDir = Join-Path $App "python"
$PythonBin = Join-Path $PythonDir "bin"
$UvCache = Join-Path $App "uv-cache"
$PipCache = Join-Path $App "pip-cache"
$Work = Join-Path $App "setup-work"
$AdminTemp = Join-Path $App "setup-temp"
$Fetch = Join-Path $Work "wheels"
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
$Constraints = Join-Path $App "constraints.txt"
$CudaConstraints = Join-Path $App "constraints-cuda.txt"
$script:TranscriptOn = $false
$script:Uv = Join-Path $App "uv.exe"
$script:Base = $null

function Test-StorePythonAlias([string]$CommandPath) {
    if ([string]::IsNullOrWhiteSpace($CommandPath)) {
        return $false
    }
    $root = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
    $full = [System.IO.Path]::GetFullPath($CommandPath)
    return $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-NoReparse([string]$Path) {
    # Fail closed. A junction or symlink on any existing component would redirect an admin write.
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Setup stopped because this path is empty."
    }
    $cursor = [System.IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        $attrs = $null
        try {
            $attrs = [System.IO.File]::GetAttributes($cursor)
        } catch {
            $attrs = $null
        }
        if ($null -ne $attrs -and (($attrs -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "Setup stopped because this path contains a link: $cursor"
        }
        $parent = [System.IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
            break
        }
        $cursor = $parent
    }
}

function Test-Elevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-MachineProgramFiles {
    # HKLM is not writable by the user who launches an elevated setup. The
    # ProgramFiles environment variable is, so it cannot authorize a path.
    $roots = New-Object System.Collections.Generic.List[string]
    $key = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion"
    foreach ($name in @("ProgramFilesDir", "ProgramFilesDir (x86)")) {
        $item = Get-ItemProperty -LiteralPath $key -Name $name -ErrorAction SilentlyContinue
        if ($null -eq $item) { continue }
        $raw = $item.$name
        if ([string]::IsNullOrWhiteSpace([string]$raw)) { continue }
        $full = [System.IO.Path]::GetFullPath([string]$raw)
        if (-not $roots.Contains($full)) { $roots.Add($full) }
    }
    if ($roots.Count -eq 0) {
        throw "Setup stopped because Program Files could not be read from Windows."
    }
    return $roots
}

function Test-PathUnder([string]$Path, [string]$Root) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $base = [System.IO.Path]::GetFullPath($Root)
    if ($full.Equals($base, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    if (-not $base.EndsWith('\')) { $base = $base + '\' }
    return $full.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-PathInside([string]$Path, [string]$Root) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $base = [System.IO.Path]::GetFullPath($Root)
    if (-not $base.EndsWith('\')) { $base = $base + '\' }
    return $full.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-FinalPathName([string]$Path) {
    # Windows PowerShell 5.1 has no Directory.ResolveLinkTarget. Follow every
    # junction with the Win32 final-path call, then drop the \\?\ prefix.
    if (-not ("VoxprintFinalPath" -as [type])) {
        $definition = @(
            "using System;"
            "using System.Runtime.InteropServices;"
            "using System.Text;"
            "public static class VoxprintFinalPath {"
            "    [DllImport(`"kernel32.dll`", CharSet = CharSet.Unicode, SetLastError = true)]"
            "    public static extern IntPtr CreateFileW("
            "        string lpFileName,"
            "        uint dwDesiredAccess,"
            "        uint dwShareMode,"
            "        IntPtr lpSecurityAttributes,"
            "        uint dwCreationDisposition,"
            "        uint dwFlagsAndAttributes,"
            "        IntPtr hTemplateFile);"
            "    [DllImport(`"kernel32.dll`", CharSet = CharSet.Unicode, SetLastError = true)]"
            "    public static extern uint GetFinalPathNameByHandleW("
            "        IntPtr hFile,"
            "        StringBuilder lpszFilePath,"
            "        uint cchFilePath,"
            "        uint dwFlags);"
            "    [DllImport(`"kernel32.dll`", SetLastError = true)]"
            "    public static extern bool CloseHandle(IntPtr hObject);"
            "}"
        ) -join "`n"
        Add-Type -TypeDefinition $definition
    }
    $handle = [VoxprintFinalPath]::CreateFileW(
        $Path,
        [uint32]128,
        [uint32]7,
        [IntPtr]::Zero,
        [uint32]3,
        [uint32]0x02000000,
        [IntPtr]::Zero
    )
    $invalid = New-Object IntPtr -1
    if ($handle -eq $invalid) {
        throw "Setup stopped because this path could not be opened: $Path"
    }
    try {
        $builder = New-Object System.Text.StringBuilder 32768
        $len = [VoxprintFinalPath]::GetFinalPathNameByHandleW($handle, $builder, [uint32]32768, [uint32]0)
        if ($len -eq 0 -or $len -ge 32768) {
            throw "Setup stopped because this path could not be resolved: $Path"
        }
        $resolved = $builder.ToString()
        if ($len -lt $resolved.Length) {
            $resolved = $resolved.Substring(0, [int]$len)
        }
    } finally {
        [void][VoxprintFinalPath]::CloseHandle($handle)
    }
    if ($resolved.StartsWith("\\?\UNC\", [System.StringComparison]::OrdinalIgnoreCase)) {
        $resolved = "\\" + $resolved.Substring(8)
    } elseif ($resolved.StartsWith("\\?\")) {
        $resolved = $resolved.Substring(4)
    }
    return [System.IO.Path]::GetFullPath($resolved)
}

function Resolve-ContainedPath([string]$Path, [string]$Root) {
    # uv's managed Python alias is a junction inside {app}\python. Follow it
    # only when every link stays inside that folder. A link on the folder
    # itself, or a link that lands outside it, stops the setup.
    if ([string]::IsNullOrWhiteSpace($Path) -or [string]::IsNullOrWhiteSpace($Root)) {
        throw "Setup stopped because this path is empty."
    }
    $full = [System.IO.Path]::GetFullPath($Path)
    $rootFull = [System.IO.Path]::GetFullPath($Root)
    $cursor = $full
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        $attrs = $null
        try {
            $attrs = [System.IO.File]::GetAttributes($cursor)
        } catch {
            $attrs = $null
        }
        if ($null -ne $attrs -and (($attrs -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
            if (-not (Test-PathInside $cursor $rootFull)) {
                throw "Setup stopped because this path contains a link: $cursor"
            }
        }
        $parent = [System.IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
            break
        }
        $cursor = $parent
    }
    $resolved = Get-FinalPathName $full
    if (-not (Test-PathUnder $resolved $rootFull)) {
        throw "Setup stopped because this link points outside the Python folder: $full"
    }
    Assert-NoReparse $resolved
    return $resolved
}

function Assert-AdminPath([string]$Path) {
    # An elevated process may run only a program under the install folder, and
    # that folder must be Program Files. Downloads and other profile folders
    # are writable before UAC, so they cannot authorize an elevated program.
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Setup will not run an empty program path."
    }
    $full = [System.IO.Path]::GetFullPath($Path)
    Assert-NoReparse $full
    if (-not (Test-PathUnder $full $App)) {
        throw "Setup will not run a program outside the install folder."
    }
    foreach ($userRoot in $UserRoots) {
        if (Test-PathUnder $full $userRoot) {
            throw "Setup will not run a program from a user-writable folder."
        }
    }
    if (-not (Test-Elevated)) { return }
    $profile = [Environment]::GetEnvironmentVariable("USERPROFILE")
    if (-not [string]::IsNullOrWhiteSpace($profile) -and (Test-PathUnder $full $profile)) {
        throw "Setup will not run a program from a user-writable folder."
    }
    foreach ($adminRoot in @(Get-MachineProgramFiles)) {
        if (Test-PathUnder $full $adminRoot) { return }
    }
    throw "Setup stopped because an elevated install must live under Program Files."
}

function Reset-InstallerEnvironment([string]$Mode) {
    # Scrub inherited UV_*, PIP_*, PYTHON*, VIRTUAL_ENV, CONDA_*, and proxy overrides.
    $prefixes = @("UV_", "PIP_", "PYTHON", "CONDA_")
    $exact = @(
        "VIRTUAL_ENV",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY",
        "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "GIT_SSL_CAINFO"
    )
    foreach ($name in @([Environment]::GetEnvironmentVariables().Keys)) {
        $drop = $false
        foreach ($prefix in $prefixes) {
            if ($name.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $drop = $true
            }
        }
        foreach ($item in $exact) {
            if ($name.Equals($item, [System.StringComparison]::OrdinalIgnoreCase)) {
                $drop = $true
            }
        }
        if ($drop) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
    }
    $env:UV_PYTHON_INSTALL_DIR = $PythonDir
    $env:UV_PYTHON_BIN_DIR = $PythonBin
    $env:UV_NO_CONFIG = "1"
    $env:UV_MANAGED_PYTHON = "1"
    $env:PYTHONNOUSERSITE = "1"
    Set-Location -LiteralPath $App
    if ($Mode -eq "User") {
        $userCache = Join-Path $SuiteHome "uv-cache"
        $userTemp = Join-Path $SuiteHome "setup-temp"
        Assert-NoReparse $userCache
        Assert-NoReparse $userTemp
        New-Item -ItemType Directory -Force $userCache, $userTemp | Out-Null
        $env:UV_CACHE_DIR = $userCache
        $env:PIP_CACHE_DIR = $userTemp
        $env:TEMP = $userTemp
        $env:TMP = $userTemp
        return
    }
    Assert-NoReparse $App
    Assert-AdminPath $App
    Assert-NoReparse $PythonDir
    Assert-NoReparse $UvCache
    Assert-NoReparse $PipCache
    Assert-NoReparse $Work
    Assert-NoReparse $AdminTemp
    New-Item -ItemType Directory -Force $PythonDir, $PythonBin, $UvCache, $PipCache, $Work, $AdminTemp | Out-Null
    $env:UV_CACHE_DIR = $UvCache
    $env:PIP_CACHE_DIR = $PipCache
    $env:TEMP = $AdminTemp
    $env:TMP = $AdminTemp
}

function Start-SetupLog([string]$Path) {
    $dir = Split-Path -Parent $Path
    Assert-NoReparse $dir
    New-Item -ItemType Directory -Force $dir | Out-Null
    try {
        Start-Transcript -Path $Path -Force | Out-Null
        $script:TranscriptOn = $true
    } catch {
        Write-Host "The install log could not be opened: $Path"
    }
}

function Stop-SetupLog {
    if ($script:TranscriptOn) {
        try { Stop-Transcript | Out-Null } catch { }
        $script:TranscriptOn = $false
    }
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
    Assert-NoReparse $Work
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

function Test-TorchCudaFiles([string]$SitePackages) {
    $versionPy = Join-Path $SitePackages "torch\version.py"
    if (-not (Test-Path -LiteralPath $versionPy)) {
        return $false
    }
    $text = Get-Content -LiteralPath $versionPy -Raw -ErrorAction SilentlyContinue
    if (-not $text -or -not $text.Contains("13.0")) {
        return $false
    }
    return Test-MetadataVersion $SitePackages "torch-" "2.11.0+cu130"
}

function Test-CpuOnnxDist([string]$SitePackages) {
    if (-not (Test-Path -LiteralPath $SitePackages)) {
        return $false
    }
    $cpuName = "onnx" + "runtime" + "-"
    foreach ($dir in @(Get-ChildItem -LiteralPath $SitePackages -Directory -ErrorAction SilentlyContinue)) {
        $name = $dir.Name
        if ($name.StartsWith($cpuName, [System.StringComparison]::OrdinalIgnoreCase) -and $name.EndsWith(".dist-info", [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Install-PinnedPackage([string]$Python, [string]$Target, [string]$IndexUrl, [string]$Strategy, [string]$Package) {
    Assert-AdminPath $Python
    Assert-AdminPath $script:Uv
    if (-not (Test-Path -LiteralPath $Fetch)) {
        Assert-NoReparse $Fetch
        New-Item -ItemType Directory -Force $Fetch | Out-Null
    }
    $into = @("pip", "install", "--no-config", "--python", $Python, "--find-links", $Fetch, "-c", $Constraints, "-c", $CudaConstraints)
    if (-not [string]::IsNullOrWhiteSpace($Target)) {
        Assert-NoReparse $Target
        New-Item -ItemType Directory -Force $Target | Out-Null
        $into += @("--target", $Target)
    }
    if ($Strategy) {
        & $script:Uv @into --index $IndexUrl --index-strategy $Strategy $Package
    } else {
        & $script:Uv @into --index-url $IndexUrl $Package
    }
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
}

function Repair-SharedRuntimePackages {
    # Other suite apps share this tree. Install a pin only when it is missing
    # or the CUDA build is wrong. Do not reinstall a package that already matches.
    $visionOk = Test-MetadataVersion $RuntimeSite "torchvision-" "0.26.0+cu130"
    $gpuName = "onnx" + "runtime" + "_gpu-"
    $gpuOk = Test-MetadataVersion $RuntimeSite $gpuName "1.31.0"
    $cpu = Test-CpuOnnxDist $RuntimeSite
    if (-not (Test-TorchCudaFiles $RuntimeSite)) {
        Write-Host "Repairing torch in the suite runtime..."
        Install-PinnedPackage $script:Base $RuntimeSite $TorchIndex "unsafe-best-match" "torch==2.11.0+cu130"
    }
    if (-not $visionOk) {
        Write-Host "Repairing torchvision in the suite runtime..."
        Install-PinnedPackage $script:Base $RuntimeSite $TorchIndex "unsafe-best-match" "torchvision==0.26.0+cu130"
    }
    if (-not $gpuOk) {
        Write-Host "Repairing ONNX Runtime in the suite runtime..."
        Install-PinnedPackage $script:Base $RuntimeSite "https://pypi.org/simple" "" "onnxruntime-gpu==1.31.0"
    }
    if ($cpu) {
        $cpuPackage = "onnx" + "runtime"
        Assert-AdminPath $script:Uv
        Assert-AdminPath $script:Base
        & $script:Uv pip uninstall --no-config --python $script:Base --target $RuntimeSite $cpuPackage
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    }
}

function Install-SuiteGpuPackages([string]$Python, [string]$Target) {
    Assert-AdminPath $Python
    Assert-AdminPath $script:Uv
    if (-not (Test-Path -LiteralPath $Fetch)) {
        Assert-NoReparse $Fetch
        New-Item -ItemType Directory -Force $Fetch | Out-Null
    }
    $into = @("pip", "install", "--no-config", "--python", $Python, "--find-links", $Fetch, "-c", $Constraints, "-c", $CudaConstraints)
    if (-not [string]::IsNullOrWhiteSpace($Target)) {
        Assert-NoReparse $Target
        New-Item -ItemType Directory -Force $Target | Out-Null
        $into += @("--target", $Target)
    }
    & $script:Uv @into --index $TorchIndex --index-strategy unsafe-best-match "torch==2.11.0+cu130" "torchvision==0.26.0+cu130"
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    & $script:Uv @into --index-url "https://pypi.org/simple" "onnxruntime-gpu==1.31.0"
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    $cpu = "onnx" + "runtime"
    if (-not [string]::IsNullOrWhiteSpace($Target)) {
        if (Test-MetadataVersion $Target ($cpu + "-") "") {
            & $script:Uv pip uninstall --no-config --python $Python --target $Target $cpu
            if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
        }
    } else {
        & $script:Uv pip uninstall --no-config --python $Python $cpu
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    }
}

function Clear-WheelFetch {
    if ($Fetch -and (Test-Path -LiteralPath $Fetch)) {
        try {
            Assert-NoReparse $Fetch
            Remove-Item -LiteralPath $Fetch -Recurse -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host $_
        }
    }
    foreach ($dir in @($AdminTemp, $UvCache, $PipCache)) {
        if ($dir -and (Test-Path -LiteralPath $dir)) {
            try {
                Assert-NoReparse $dir
                Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
            } catch {
                Write-Host $_
            }
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

function Assert-ConstraintFiles {
    if (-not (Test-Path -LiteralPath $Constraints)) {
        throw "constraints.txt is missing from the setup. The install cannot continue."
    }
    if (-not (Test-Path -LiteralPath $CudaConstraints)) {
        throw "constraints-cuda.txt is missing from the setup. The install cannot continue."
    }
}

function Resolve-BundledUv {
    $bundled = Join-Path $App "uv.exe"
    if (Test-Path -LiteralPath $bundled) {
        Assert-AdminPath $bundled
        $script:Uv = $bundled
        return
    }
    $local = Join-Path $Venv "Scripts\uv.exe"
    if (Test-Path -LiteralPath $local) {
        Assert-AdminPath $local
        $script:Uv = $local
        return
    }
    throw "uv 0.13.0 is required to install PyTorch."
}

function Resolve-BasePython {
    $cfg = Join-Path $Venv "pyvenv.cfg"
    if (Test-Path -LiteralPath $cfg) {
        $pyHome = $null
        $executable = $null
        foreach ($line in Get-Content -LiteralPath $cfg) {
            if ($line.StartsWith("home = ")) { $pyHome = $line.Substring(7).Trim() }
            if ($line.StartsWith("executable = ")) { $executable = $line.Substring(13).Trim() }
        }
        if ($executable -and (Test-Path -LiteralPath $executable)) {
            $executable = Resolve-ContainedPath $executable $PythonDir
            Assert-AdminPath $executable
            return $executable
        }
        if ($pyHome) {
            $candidate = Join-Path $pyHome "python.exe"
            if (Test-Path -LiteralPath $candidate) {
                $candidate = Resolve-ContainedPath $candidate $PythonDir
                Assert-AdminPath $candidate
                return $candidate
            }
        }
    }
    $found = Get-ChildItem -LiteralPath $PythonDir -Filter python.exe -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $found) {
        throw "uv installed Python 3.14 but the interpreter could not be found. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
    }
    $resolved = Resolve-ContainedPath $found.FullName $PythonDir
    Assert-AdminPath $resolved
    return $resolved
}

function Move-OccupiedRuntimeAside {
    # A non-venv tree is user data. Rename it and build a new runtime beside the backup.
    Assert-NoReparse $Runtime
    $stamp = Get-Date -Format "yyyyMMddHHmmss"
    $backup = "$Runtime.bak-$stamp"
    $suffix = 0
    while (Test-Path -LiteralPath $backup) {
        $suffix += 1
        $backup = "$Runtime.bak-$stamp-$suffix"
    }
    Write-Host "Moving the existing suite runtime aside: $backup"
    Move-Item -LiteralPath $Runtime -Destination $backup
}

function Ensure-SharedRuntime([string]$BasePython) {
    Assert-NoReparse $Shared
    Assert-NoReparse $Runtime
    $runtimeVenv = (Test-Path -LiteralPath $RuntimePy) -and (Test-Path -LiteralPath $RuntimeCfg)
    if ($runtimeVenv) {
        return
    }
    if (Test-Path -LiteralPath $Runtime) {
        $occupied = Get-ChildItem -Force -LiteralPath $Runtime -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($occupied) {
            Move-OccupiedRuntimeAside
        }
    }
    Assert-NoReparse $Shared
    Assert-NoReparse $Runtime
    New-Item -ItemType Directory -Force $Shared | Out-Null
    Assert-AdminPath $BasePython
    # --without-pip keeps venv from launching the new interpreter. That
    # interpreter lives in the user-writable shared tree.
    & $BasePython -m venv --without-pip $Runtime
    if ($LASTEXITCODE -ne 0) { throw "Python could not create the suite runtime." }
}

function Invoke-Machine {
    Reset-InstallerEnvironment "Machine"
    Start-SetupLog (Join-Path $Work "canvas-install.log")
    try {
        Write-Host "Checking Python..."
        if (-not (Test-Path -LiteralPath $script:Uv)) {
            throw "uv.exe is not bundled next to install.ps1. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
        }
        Assert-ConstraintFiles
        Assert-AdminPath $script:Uv
        Write-Host "Installing Python with uv..."
        & $script:Uv python install 3.14 --reinstall --no-registry --no-config --install-dir $PythonDir
        if ($LASTEXITCODE -ne 0) {
            throw "uv could not install Python 3.14. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
        }
        $found = (& $script:Uv python find 3.14 --managed-python --no-config | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $found) {
            throw "uv installed Python 3.14 but the interpreter could not be found. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
        }
        $script:Base = (($found -split '\r?\n') | Where-Object { $_.Trim() } | Select-Object -Last 1).Trim()
        if (-not (Test-Path -LiteralPath $script:Base)) {
            throw "uv reported Python at '$script:Base', but that file does not exist. Install Python 3.14 from $WindowsPythonUrl and run this setup again."
        }
        $script:Base = Resolve-ContainedPath $script:Base $PythonDir
        Assert-AdminPath $script:Base
        Write-Host "Python is $script:Base"
        if (-not (Test-Path -LiteralPath $Py)) {
            & $script:Base -m venv $Venv
            if ($LASTEXITCODE -ne 0) { throw "Python could not create the environment." }
        } else {
            & $script:Base -m venv --upgrade $Venv
            if ($LASTEXITCODE -ne 0) { throw "Python could not refresh the environment." }
        }
        Invoke-Py @("-m", "pip", "--isolated", "install", "--upgrade", "pip")
        if (-not (Test-Path -LiteralPath $script:Uv)) {
            Invoke-Py @("-m", "pip", "--isolated", "install", "--index-url", "https://pypi.org/simple", "uv==0.13.0")
        }
        Resolve-BundledUv
        Assert-NoReparse $Fetch
        New-Item -ItemType Directory -Force $Fetch | Out-Null
        $appReady = Test-SuiteGpuPackages $Py
        if ($appReady) {
            Write-Host "PyTorch is already installed."
        } else {
            Write-Host "Installing PyTorch..."
            Invoke-Py @(
                "-m", "pip", "--isolated", "download", "-c", $Constraints, "-c", $CudaConstraints,
                "torch==2.11.0+cu130", "torchvision==0.26.0+cu130",
                "--index-url", $TorchIndex, "--extra-index-url", "https://pypi.org/simple", "-d", $Fetch
            )
        }
        Invoke-Py @(
            "-m", "pip", "--isolated", "download", "-c", $Constraints, "-c", $CudaConstraints,
            "onnxruntime-gpu==1.31.0", "torchao==0.18.0",
            "--index-url", "https://pypi.org/simple", "-d", $Fetch
        )
        if (-not $appReady) {
            Install-SuiteGpuPackages $Py ""
            if (-not (Test-SuiteGpuPackages $Py)) { throw "Setup stopped." }
        }
        $CommonUrl = "https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.2.6/voxprint_common-0.2.6-py3-none-any.whl"
        $CommonSha = "aeb2b180025967a02a81e206dfa0dd01c3d77a2aa20447485288496123851505"
        $CommonWheel = Join-Path $Fetch "voxprint_common-0.2.6-py3-none-any.whl"
        Assert-NoReparse $CommonWheel
        Invoke-WebRequest -Uri $CommonUrl -OutFile $CommonWheel
        $got = (Get-FileHash -Algorithm SHA256 -Path $CommonWheel).Hash.ToLowerInvariant()
        if ($got -ne $CommonSha) { throw "voxprint-common 0.2.6 failed the SHA-256 check." }
        Assert-AdminPath $script:Uv
        Assert-AdminPath $Py
        & $script:Uv pip install --no-config --python $Py $CommonWheel
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
        & $script:Uv pip install --no-config --python $Py --index-url https://pypi.org/simple "torchao==0.18.0"
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
        & $script:Uv pip install --no-config --python $Py --index $TorchIndex --index-url https://pypi.org/simple --index-strategy unsafe-best-match -c $Constraints -c $CudaConstraints "${App}[gen]"
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
        & $Py -m canvas.infra.layout migrate --install-dir $App
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
    } finally {
        Stop-SetupLog
    }
}

function Move-LegacySuiteHome {
    # voxprint-common 0.2.5 moved the data home. Move the old folder before
    # anything runs from it: Windows cannot rename a folder that holds a
    # running interpreter or an open log. When both exist, the new one is kept
    # and the old one is left in place, as voxprint-common does.
    if (-not (Test-Path -LiteralPath $LegacyHome -PathType Container)) { return }
    if (Test-Path -LiteralPath $SuiteHome) { return }
    Assert-NoReparse $LegacyHome
    Assert-NoReparse (Split-Path -Parent $SuiteHome)
    Write-Host "Moving the suite data from $LegacyHome to $SuiteHome"
    Move-Item -LiteralPath $LegacyHome -Destination $SuiteHome
}

function Invoke-User {
    if ($Scope -eq "All") {
        Write-Host "Profile folders are written by this process because -Scope All was requested. Each path is checked for links."
    }
    Move-LegacySuiteHome
    Reset-InstallerEnvironment "User"
    $logDir = Join-Path $SuiteHome "logs"
    Start-SetupLog (Join-Path $logDir "canvas-install.log")
    try {
        Assert-ConstraintFiles
        Resolve-BundledUv
        $script:Base = Resolve-BasePython
        Write-Host "Python is $script:Base"
        Ensure-SharedRuntime $script:Base
        $runtimeReady = $false
        if (Test-Path -LiteralPath $RuntimeSite) {
            Assert-NoReparse $RuntimeSite
            $runtimeReady = Test-SuiteGpuFiles $RuntimeSite
        }
        if (-not $runtimeReady) {
            Write-Host "Repairing the suite runtime..."
            Repair-SharedRuntimePackages
            if (-not (Test-SuiteGpuFiles $RuntimeSite)) {
                Write-Host "CUDA 13.0 PyTorch is still missing from $RuntimeSite"
                Get-ChildItem -LiteralPath $RuntimeSite -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like "*.dist-info" } |
                    ForEach-Object { Write-Host $_.Name }
                throw "CUDA 13.0 PyTorch is still missing."
            }
        }
        & $Py -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"
        Assert-NoReparse $Trt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "TensorRT is skipped on Python 3.14."
        } elseif (-not (Get-ChildItem -Force -LiteralPath $Trt -ErrorAction SilentlyContinue | Select-Object -First 1)) {
            New-Item -ItemType Directory -Force $Trt | Out-Null
            Invoke-Py @("-m", "pip", "--isolated", "install", "-c", $Constraints, "--target", $Trt, "--upgrade-strategy", "only-if-needed", "tensorrt-cu13==10.16.1.11")
        }
        $dataRoot = $SuiteHome
        Assert-NoReparse $dataRoot
        $Mdl = Join-Path $App "mdl.json"
        if (-not (Test-Path -LiteralPath $Mdl)) { throw "mdl.json is missing from the setup. The install cannot continue." }
        Write-Host "Installing the default image pack..."
        Invoke-Py @("-m", "voxprint_common.mdl", "download", "--app", "canvas", "--manifest", $Mdl)
        $tierDir = Join-Path $dataRoot "Canvas"
        Assert-NoReparse $tierDir
        New-Item -ItemType Directory -Force $tierDir | Out-Null
        $tierFile = Join-Path $tierDir "tier.txt"
        Assert-NoReparse $tierFile
        & $Py -m canvas.models.tiers | Set-Content -Encoding ascii $tierFile
        if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
        Invoke-Py @("-m", "canvas.app.main", "--register-models")
        Write-Host "Installed."
    } finally {
        Stop-SetupLog
    }
}

function Invoke-Cleanup {
    Reset-InstallerEnvironment "Machine"
    Clear-WheelFetch
}

if ($env:VOXPRINT_INSTALL_LIB -ne "1") {
    Write-Host "Setup scope: $Scope"
    try {
        switch ($Scope) {
            "Machine" { Invoke-Machine }
            "User" { Invoke-User }
            "Cleanup" { Invoke-Cleanup }
            "All" {
                Invoke-Machine
                Invoke-User
                Invoke-Cleanup
            }
            default { throw "Setup stopped because the scope is not Machine, User, Cleanup, or All." }
        }
    } catch {
        Write-Host $_
        if ((Test-Elevated) -or ($Scope -ne "User")) {
            Clear-WheelFetch
        }
        Stop-SetupLog
        exit 1
    }
}
