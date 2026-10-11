# Install Voxprint AI Canvas into a local environment.
# The setup does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $App "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Shared = Join-Path $env:LOCALAPPDATA "Voxprint\shared"
$PyHome = Join-Path $Shared "python"
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
foreach ($spec in @("-3.12", "-3")) {
    & py $spec -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
    if ($LASTEXITCODE -eq 0) {
        $Base = $spec
        break
    }
}
if (-not $Base) {
    $Installed = Join-Path $PyHome "python.exe"
    if (-not (Test-Path $Installed)) {
        Write-Host "Installing Python..."
        New-Item -ItemType Directory -Force $PyHome | Out-Null
        $exe = Join-Path $env:TEMP "voxprint-python-3.12.10-amd64.exe"
        Invoke-WebRequest -Uri $WindowsPythonUrl -OutFile $exe
        $proc = Start-Process -FilePath $exe -ArgumentList @(
            "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_test=0", "Include_pip=1", "TargetDir=$PyHome"
        ) -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "Python could not be installed." }
    }
    $Base = $Installed
}

if (-not (Test-Path $Py)) {
    if ($Base -eq "-3.12" -or $Base -eq "-3") {
        & py $Base -m venv $Venv
    } else {
        & $Base -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Python could not create the environment." }
}

Invoke-Py @("-m", "pip", "install", "--upgrade", "pip")
Write-Host "Installing PyTorch..."
New-Item -ItemType Directory -Force $Wheels | Out-Null
Invoke-Py @("-m", "pip", "download", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128", "-d", $Wheels)
Invoke-Py @("-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "--find-links", $Wheels, "torch", "torchvision")
Invoke-Py @("-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "${App}[gen]")

Write-Host "Installing the default image pack..."
Invoke-Py @("-m", "canvas.app.provision")
$tierDir = Join-Path $env:LOCALAPPDATA "Voxprint\Canvas"
New-Item -ItemType Directory -Force $tierDir | Out-Null
& $Py -m canvas.models.tiers | Set-Content -Encoding ascii (Join-Path $tierDir "tier.txt")
if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
Invoke-Py @("-m", "canvas.app.main", "--register-models")
Write-Host "Installed."
