# Install Voxprint AI Canvas into a local environment.
# The setup does not contain GPU libraries or model files.
# The first picture downloads the pinned models.
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $App "venv"
$Py = Join-Path $Venv "Scripts\python.exe"

function Invoke-Py([string[]]$ArgList) {
    & $Py @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "Setup stopped."
    }
}

if (-not (Test-Path $Py)) {
    $created = $false
    foreach ($spec in @("-3.12", "-3")) {
        & py $spec -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) {
            & py $spec -m venv $Venv
            if ($LASTEXITCODE -ne 0) { throw "Python could not create the environment." }
            $created = $true
            break
        }
    }
    if (-not $created) {
        throw "Python 3.12 or newer is required."
    }
}

Invoke-Py @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Py @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")
Invoke-Py @("-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "${App}[gen]")

$tierDir = Join-Path $env:LOCALAPPDATA "Voxprint\Canvas"
New-Item -ItemType Directory -Force $tierDir | Out-Null
& $Py -m canvas.models.tiers | Set-Content -Encoding ascii (Join-Path $tierDir "tier.txt")
if ($LASTEXITCODE -ne 0) { throw "Setup stopped." }
Invoke-Py @("-m", "canvas.app.main", "--register-models")
Write-Host "Installed. The first picture downloads the models."
