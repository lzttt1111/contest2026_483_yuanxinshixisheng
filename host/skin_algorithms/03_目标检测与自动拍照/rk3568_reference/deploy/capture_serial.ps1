[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^COM\d+$')]
    [string]$Port,
    [ValidateRange(300, 4000000)]
    [int]$BaudRate = 1500000,
    [ValidateRange(0, 86400)]
    [int]$DurationSeconds = 0,
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
if (-not $OutputPath) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputPath = Join-Path (
        Split-Path $PSScriptRoot -Parent
    ) "diagnostics/serial-$Timestamp.log"
}
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
$OutputDirectory = Split-Path $OutputPath -Parent
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null

$Serial = [IO.Ports.SerialPort]::new(
    $Port,
    $BaudRate,
    [IO.Ports.Parity]::None,
    8,
    [IO.Ports.StopBits]::One
)
$Serial.Handshake = [IO.Ports.Handshake]::None
$Serial.DtrEnable = $false
$Serial.RtsEnable = $false
$Serial.ReadTimeout = 250
$Serial.WriteTimeout = 1000
$Serial.Encoding = [Text.Encoding]::UTF8
$Writer = [IO.StreamWriter]::new(
    $OutputPath,
    $true,
    [Text.UTF8Encoding]::new($false)
)
$Writer.AutoFlush = $true
$Started = [Diagnostics.Stopwatch]::StartNew()

try {
    $Serial.Open()
    Write-Host (
        "SERIAL_CAPTURE=START port=$Port baud=$BaudRate data=8 parity=None " +
        "stop=1 flow=None output=$OutputPath"
    )
    if ($DurationSeconds -eq 0) {
        Write-Host "Press Ctrl+C to stop; the finally block closes the port and file."
    }
    while ($DurationSeconds -eq 0 -or $Started.Elapsed.TotalSeconds -lt $DurationSeconds) {
        $Chunk = $Serial.ReadExisting()
        if ($Chunk.Length -gt 0) {
            $Writer.Write($Chunk)
            Write-Host -NoNewline $Chunk
        }
        Start-Sleep -Milliseconds 20
    }
}
finally {
    $Started.Stop()
    if ($Serial.IsOpen) {
        $Chunk = $Serial.ReadExisting()
        if ($Chunk.Length -gt 0) {
            $Writer.Write($Chunk)
        }
        $Serial.Close()
    }
    $Serial.Dispose()
    $Writer.Dispose()
    Write-Host (
        "SERIAL_CAPTURE=STOP elapsed_s=$([math]::Round($Started.Elapsed.TotalSeconds, 3)) " +
        "output=$OutputPath"
    )
}
