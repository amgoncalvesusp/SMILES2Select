param(
    [Parameter(Mandatory = $true)][string]$Installer,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
# /DIR does not isolate Inno Setup's AppId registration. Never test over a
# real user's installation, even when the application files go into TEMP.
$uninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{B1D0B6C7-6D51-4F18-9D15-6D9CEB390300}_is1'
foreach ($hive in @([Microsoft.Win32.RegistryHive]::CurrentUser, [Microsoft.Win32.RegistryHive]::LocalMachine)) {
    foreach ($view in @([Microsoft.Win32.RegistryView]::Registry32, [Microsoft.Win32.RegistryView]::Registry64)) {
        $registry = [Microsoft.Win32.RegistryKey]::OpenBaseKey($hive, $view)
        try {
            $existing = $registry.OpenSubKey($uninstallKey)
            if ($null -ne $existing) {
                $existing.Dispose()
                throw "An existing SMILES2Select installation is registered. Run this smoke test on a clean Windows account or CI runner."
            }
        }
        finally { $registry.Dispose() }
    }
}
$installerPath = (Resolve-Path -LiteralPath $Installer).Path
$projectRoot = Split-Path -Parent $PSScriptRoot
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("s2s install test-" + [guid]::NewGuid().ToString("N"))
$installDir = Join-Path $testRoot "app"
$userData = Join-Path $testRoot "example.s2s.sqlite"
New-Item -ItemType Directory -Path $testRoot | Out-Null
Set-Content -LiteralPath $userData -Value "User-owned session sentinel" -NoNewline
$previousBundle = $env:S2S_FROZEN_BUNDLE
$previousQt = $env:QT_QPA_PLATFORM
try {
    foreach ($attempt in 1..2) {
        $log = Join-Path $testRoot "install-$attempt.log"
        $installArgs = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", "/NOICONS", "/TASKS=", "/DIR=`"$installDir`"", "/LOG=`"$log`"")
        $process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -ne 0) { throw "Installer failed with exit code $($process.ExitCode). See $log" }
        if (-not (Test-Path -LiteralPath (Join-Path $installDir "SMILES2Select.exe"))) {
            throw "Installer did not create the executable. See $log"
        }
    }
    $env:S2S_FROZEN_BUNDLE = $installDir
    $env:QT_QPA_PLATFORM = "offscreen"
    & $Python -m pytest -q (Join-Path $projectRoot "tests/test_frozen_inference.py")
    if ($LASTEXITCODE -ne 0) { throw "Installed model inference failed with exit code $LASTEXITCODE." }
}
finally {
    $env:S2S_FROZEN_BUNDLE = $previousBundle
    $env:QT_QPA_PLATFORM = $previousQt
    $uninstaller = Join-Path $installDir "unins000.exe"
    if (Test-Path -LiteralPath $uninstaller) {
        $uninstallLog = Join-Path $testRoot "uninstall.log"
        $uninstallArgs = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=`"$uninstallLog`"")
        $process = Start-Process -FilePath $uninstaller -ArgumentList $uninstallArgs -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -ne 0) { throw "Uninstaller failed with exit code $($process.ExitCode). See $uninstallLog" }
    }
}
if (Test-Path -LiteralPath (Join-Path $installDir "SMILES2Select.exe")) { throw "Uninstaller left the executable behind." }
if ((Get-Content -LiteralPath $userData -Raw) -ne "User-owned session sentinel") { throw "User data changed during installation/uninstallation." }
Write-Output "Windows install, update, installed inference and uninstall passed. Logs: $testRoot"
