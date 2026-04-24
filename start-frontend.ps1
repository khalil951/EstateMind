$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$nodeDir = Join-Path $root ".tools\node-v22.14.0-win-x64"
$npmCmd = Join-Path $nodeDir "npm.cmd"
$frontendDir = Join-Path $root "frontend_react"

if (-not (Test-Path $npmCmd)) {
    throw "Local Node runtime not found at $nodeDir. Run the initial setup first."
}

if (-not (Test-Path $frontendDir)) {
    throw "Frontend folder not found at $frontendDir"
}

$env:Path = "$nodeDir;$env:Path"
Set-Location $frontendDir
npm run dev
