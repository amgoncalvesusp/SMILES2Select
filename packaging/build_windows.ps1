[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Version = "3.0.3"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
Set-Location $repo

& $Python -m PyInstaller packaging/smiles2select.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

& $Python tools/verify_windows_bundle.py dist/SMILES2Select --manifest dist/SMILES2Select-manifest.json
if ($LASTEXITCODE -ne 0) { throw "Bundle dependency verification failed with exit code $LASTEXITCODE" }

$candidates = @(
    (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source,
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $candidates) {
    throw "ISCC.exe not found. Install Inno Setup 6 before building the installer."
}

$iscc = $candidates[0]
& $iscc "/DAppVersion=$Version" "/DSourceDir=$repo\dist\SMILES2Select" "/DOutputDir=$repo\dist" "packaging\smiles2select.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }

Write-Host "Created dist\SMILES2Select-Setup-$Version.exe"
