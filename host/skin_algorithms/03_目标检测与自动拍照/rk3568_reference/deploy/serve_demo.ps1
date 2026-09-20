[CmdletBinding()]
param([ValidateRange(1024, 65535)][int]$Port = 8090)

$ErrorActionPreference = "Stop"
$WebRoot = (Resolve-Path (Join-Path $PSScriptRoot "../web/rk3568_demo")).ProviderPath

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    Write-Host "Serving $WebRoot at http://127.0.0.1:$Port"
    & $Python.Source -m http.server $Port --bind 127.0.0.1 --directory $WebRoot
    exit $LASTEXITCODE
}

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    Write-Host "Serving $WebRoot at http://127.0.0.1:$Port"
    & $PythonLauncher.Source -3 -m http.server $Port --bind 127.0.0.1 --directory $WebRoot
    exit $LASTEXITCODE
}

throw "Python 3 was not found. Serve web/rk3568_demo on 127.0.0.1:$Port with any static HTTP server."
