# Canvas smoke run for Proving Ground on Windows (Windows PowerShell 5.1 or PowerShell 7).
# No WSL and no bash. The Setup .exe installs this script as
# C:\Program Files\VOXPrint AI Suite\Canvas\tools\smoke_16gb.ps1; a checkout works too.
# Results go to .\smoke-results under the current folder unless -Out is given.
#
#   -VramCapGB 16  16 GB pass: torch.cuda.set_per_process_memory_fraction plus the
#                  16 GB group-offload path, so an RTX 4090 runs as a 16 GB card.
#   (no cap, or 0) 24 GB pass: uncapped, on the 24 GB tier path.
#
# Steps: find Python (the one the Canvas installer ships, or -Python), fetch and
# verify the voxprint-common 0.3.1 wheel, download the models through VOXPrint MDL
# (mdl.json), generate five neutral 1024x1024 pictures through the content gate,
# then time the gate by stage. Writes <Out>\result-16gb.json or result-24gb.json.
#
# Exit 0 pass, 1 a picture failed, 2 setup (no Python, wheel hash, wrong tier),
# 3 no GPU, 4 gate unavailable, 5 an MDL download or hash check failed.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\smoke_16gb.ps1 -VramCapGB 16
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\smoke_16gb.ps1
param(
    [double]$VramCapGB = 0,
    [string]$Python = "",
    [string]$Out = ""
)
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Root = Split-Path -Parent $PSScriptRoot
$CommonVersion = "0.3.1"
$CommonUrl = "https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.3.1/voxprint_common-0.3.1-py3-none-any.whl"
$CommonSha = "f14036fc70db07e40e5e5376427cea28c3173a94f1d031997a3259ed0afc91fd"
$Installed = Join-Path $env:ProgramFiles "VOXPrint AI Suite\Canvas\venv\Scripts\python.exe"

if ($Python -eq "") { $Python = $Installed }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Host "Python was not found at $Python. Install Canvas from the Setup .exe, or pass -Python <python.exe>."
    exit 2
}
$Python = (Resolve-Path -LiteralPath $Python).Path
if ($VramCapGB -gt 0) { $Pass = "16gb" } else { $Pass = "24gb" }
if ($Out -eq "") { $Out = "smoke-results" }
if (-not [System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path (Get-Location).Path $Out }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path -LiteralPath $Out).Path
Write-Host "Python: $Python"
Write-Host "Pass: $Pass"

# voxprint-common 0.3.1: fetch, verify, and use it when the interpreter has another version.
$Work = Join-Path $Out ".smoke-work"
$Lib = Join-Path $Work "lib"
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$Wheel = Join-Path $Work "voxprint_common-0.3.1-py3-none-any.whl"
try {
    Invoke-WebRequest -Uri $CommonUrl -OutFile $Wheel -UseBasicParsing
} catch {
    Write-Host "The voxprint-common $CommonVersion wheel could not be downloaded: $_"
    exit 2
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Wheel).Hash.ToLower() -ne $CommonSha) {
    Write-Host "The voxprint-common $CommonVersion wheel failed the SHA-256 check."
    exit 2
}
$ErrorActionPreference = "Continue"
$have = & $Python -c "import importlib.metadata as m; print(m.version('voxprint-common'))" 2>$null
$probe = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($probe -ne 0) { $have = "" }
if ("$have".Trim() -ne $CommonVersion) {
    Write-Host "voxprint-common is '$have' in this Python. Using the verified $CommonVersion wheel for this run."
    if (Test-Path -LiteralPath $Lib) { Remove-Item -Recurse -Force -LiteralPath $Lib }
    & $Python -m pip install --disable-pip-version-check -q --no-deps --target $Lib $Wheel
    if ($LASTEXITCODE -ne 0) { Write-Host "The voxprint-common wheel could not be unpacked."; exit 2 }
    # voxprint-common verifies the signed model index with cryptography; pins match constraints.txt.
    & $Python -m pip install --disable-pip-version-check -q --no-deps --only-binary=:all: --target $Lib `
        "cryptography==50.0.2" "cffi==2.1.1" "pycparser==3.11"
    if ($LASTEXITCODE -ne 0) { Write-Host "The cryptography wheels for voxprint-common could not be unpacked."; exit 2 }
    $env:PYTHONPATH = $Lib
}

# Use the mdl.json that belongs to the Canvas this Python runs (the install folder
# above venv\Scripts), else the checkout's.
$AppDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Python))
$Manifest = Join-Path $AppDir "mdl.json"
if (-not (Test-Path -LiteralPath $Manifest)) { $Manifest = Join-Path $Root "mdl.json" }
Push-Location $Out
try {
    $ErrorActionPreference = "Continue"
    & $Python -c "import canvas" 2>$null
    $probe = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($probe -ne 0) { Write-Host "Canvas is not installed in this Python."; exit 2 }
    $Generator = if ($Pass -eq "16gb") { "z-image-turbo-bf16-16gb" } else { "z-image-turbo-bf16" }
    Write-Host "Models: python -m voxprint_common.mdl download --app canvas --manifest $Manifest --model $Generator"
    & $Python -m voxprint_common.mdl download --app canvas --manifest $Manifest --model $Generator
    if ($LASTEXITCODE -ne 0) { Write-Host "MDL download or hash check failed."; exit 5 }

    $toolArgs = @((Join-Path $Root "tools\smoke_16gb.py"), "--python", $Python, "--out", $Out, "--pass", $Pass, "--manifest", $Manifest)
    if ($Pass -eq "16gb") {
        $toolArgs += @("--vram-cap-gb", [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $VramCapGB))
    }
    & $Python @toolArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($null -eq $code) { exit 2 }
Write-Host "Result: $(Join-Path $Out "result-$Pass.json") (exit $code)"
exit $code
