# 16 GB smoke run: install Canvas from this checkout, fetch the models through
# VOXPrint MDL (mdl.json), then generate five 1024x1024 pictures with
# Z-Image-Turbo BF16 on the 16 GB group-offload path. Every output goes through
# the content gate. Writes <out>\result.json.
# Exit 0 pass, 1 a picture failed, 2 setup, 3 no GPU, 4 gate unavailable,
# 5 a manifest entry failed its download or hash check.
# Usage: powershell -ExecutionPolicy Bypass -File tools\smoke_16gb.ps1 [out-folder]
param([string]$Out = "smoke-16gb")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$TorchIndex = "https://download.pytorch.org/whl/cu130"
$CommonUrl = "https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.2.4/voxprint_common-0.2.4-py3-none-any.whl"
$CommonSha = "3051bc64700964705bf5cee5182829d1311560a38a9f7436195f26481e4fa239"
$Py = if ($env:PYTHON) { $env:PYTHON } else { "py" }
$PyArgs = if ($env:PYTHON) { @() } else { @("-3.14") }
if (-not (Get-Command $Py -ErrorAction SilentlyContinue)) {
    Write-Host "Python 3.14 is not installed. Set PYTHON to a Python 3.14 interpreter."
    exit 2
}
function Step([string]$Exe, [string[]]$Arguments, [int]$Code = 2) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { exit $Code }
}
$Venv = Join-Path $Root ".smoke-venv"
$VPy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VPy)) { Step $Py ($PyArgs + @("-m", "venv", $Venv)) }
$Wheels = Join-Path $Venv "wheels"
New-Item -ItemType Directory -Force -Path $Wheels | Out-Null
$Wheel = Join-Path $Wheels "voxprint_common-0.2.4-py3-none-any.whl"
Invoke-WebRequest -Uri $CommonUrl -OutFile $Wheel -UseBasicParsing
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Wheel).Hash.ToLower() -ne $CommonSha) {
    Write-Host "voxprint-common wheel hash mismatch."
    exit 2
}
Step $VPy @("-m", "pip", "install", "-q", "--upgrade", "pip")
Step $VPy @("-m", "pip", "install", "-q", $Wheel)
Step $VPy @("-m", "pip", "install", "-q", "--extra-index-url", $TorchIndex,
    "-c", "constraints.txt", "-c", "constraints-cuda.txt", "-e", ".[gen]")
# A failed download or a hash mismatch stops the run here with exit 5.
Step $VPy @("-m", "voxprint_common.mdl", "download", "--app", "canvas", "--manifest", (Join-Path $Root "mdl.json")) 5
& $VPy -m tools.smoke_16gb --python $VPy --out $Out
if ($null -eq $LASTEXITCODE) { exit 2 }
exit $LASTEXITCODE
