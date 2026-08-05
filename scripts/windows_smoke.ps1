param(
    [string]$Python = "python",
    [string]$OutputRoot = "runs"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $Python -m tamoe.smoke `
    --config (Join-Path $ProjectRoot "configs\m0_smoke.json") `
    --output-root (Join-Path $ProjectRoot $OutputRoot) `
    --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "M0 smoke test failed with exit code $LASTEXITCODE"
}
