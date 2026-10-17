# Non-interactive Canvas check. Exit 0 pass, 1 check failed, 2 setup, 3 no GPU.
# --smoke is the short run. --full is the longer picture set. --json prints the report.
# --vram-cap-gb 16 runs text-to-image at 2K, an inpaint, and Qwen-Image-Edit.
# Peak VRAM is torch.cuda.max_memory_allocated plus nvidia-smi samples.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed."
    exit 2
}
& python -m tools.vm_test @args
if ($null -eq $LASTEXITCODE) { exit 2 }
exit $LASTEXITCODE
