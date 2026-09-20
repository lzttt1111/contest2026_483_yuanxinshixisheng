param(
    [string]$Distribution = 'Ubuntu-22.04',
    [ValidateSet('cpu', 'cuda')][string]$RuntimeGroup = 'cpu'
)
$ErrorActionPreference = 'Stop'
$linuxPackagePath = (& wsl.exe -d $Distribution -- wslpath -a $PSScriptRoot)
if ($LASTEXITCODE -ne 0) { throw '无法转换WSL路径，请先用 wsl -l -v 检查发行版名称。' }
$linuxPackagePath = $linuxPackagePath.Trim()
Write-Host '服务就绪后在 Windows Edge/Chrome 打开 http://127.0.0.1:8876/'
& wsl.exe -d $Distribution -- env "LAB_RUNTIME_GROUP=$RuntimeGroup" bash "$linuxPackagePath/start_uv.sh"
if ($LASTEXITCODE -ne 0) { throw "启动失败，退出码 $LASTEXITCODE。请查看上方错误和README。" }
