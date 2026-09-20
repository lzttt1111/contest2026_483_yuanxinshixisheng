[CmdletBinding()]
param(
    [string]$Adb = "adb",
    [string]$BoardRoot = "/userdata/medvision",
    [switch]$SkipStart
)

$ErrorActionPreference = "Stop"

if ($BoardRoot -notmatch '^/userdata/[A-Za-z0-9._/-]+$') {
    throw "BoardRoot must be an explicit path below /userdata/."
}

$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
$Checksums = Join-Path $PackageRoot "SHA256SUMS"
if (-not (Test-Path -LiteralPath $Checksums -PathType Leaf)) {
    throw "SHA256SUMS is missing. Run package_release.py first."
}

function Invoke-Adb {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $Adb @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Invoke-Adb -Arguments @("start-server")
$DeviceLines = @(& $Adb devices) | Where-Object { $_ -match "`tdevice$" }
if ($LASTEXITCODE -ne 0) {
    throw "Unable to query adb devices."
}
if ($DeviceLines.Count -ne 1) {
    throw "Expected exactly one online RK3568 adb device; found $($DeviceLines.Count)."
}

Write-Host "Stopping an existing MedVision service, if present..."
Invoke-Adb -Arguments @(
    "shell",
    "if [ -x '$BoardRoot/deploy/stop.sh' ]; then '$BoardRoot/deploy/stop.sh'; fi; mkdir -p '$BoardRoot'"
)

$Files = Get-ChildItem -LiteralPath $PackageRoot -Recurse -File
foreach ($File in $Files) {
    $Relative = [IO.Path]::GetRelativePath($PackageRoot, $File.FullName)
    if ($Relative -eq '..' -or $Relative.StartsWith("..$([IO.Path]::DirectorySeparatorChar)")) {
        throw "Package file escaped the release root: $($File.FullName)"
    }
    $Relative = $Relative.Replace('\', '/')
    $RemotePath = "$BoardRoot/$Relative"
    $Slash = $RemotePath.LastIndexOf('/')
    $RemoteDirectory = $RemotePath.Substring(0, $Slash)
    Invoke-Adb -Arguments @("shell", "mkdir -p '$RemoteDirectory'")
    Invoke-Adb -Arguments @("push", $File.FullName, $RemotePath)
}

Invoke-Adb -Arguments @(
    "shell",
    "cd '$BoardRoot' && chmod 0755 bin/medvision_rk3568 deploy/start.sh deploy/stop.sh && sha256sum -c SHA256SUMS"
)

if (-not $SkipStart) {
    Invoke-Adb -Arguments @("shell", "'$BoardRoot/deploy/start.sh'")
}
Invoke-Adb -Arguments @("forward", "tcp:18080", "tcp:8080")

Write-Host "DEPLOY=PASS"
Write-Host "Board service: http://127.0.0.1:18080"
Write-Host "Demo command:  pwsh -File `"$PSScriptRoot/serve_demo.ps1`""
Write-Host "Demo page:     http://127.0.0.1:8090"
