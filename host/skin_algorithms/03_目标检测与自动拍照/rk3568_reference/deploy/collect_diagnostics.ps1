[CmdletBinding()]
param(
    [string]$Adb = "adb",
    [ValidateRange(1, 120)][int]$CommandTimeoutSeconds = 8,
    [ValidateRange(1, 120)][int]$HttpTimeoutSeconds = 5,
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path (Split-Path $PSScriptRoot -Parent) "diagnostics/$Timestamp"
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null

function Convert-ToSafeName {
    param([Parameter(Mandatory)][string]$Value)
    return ($Value -replace '[^A-Za-z0-9._-]', '_')
}

function Invoke-AdbCapture {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    $SafeName = Convert-ToSafeName -Value $Name
    $StdoutPath = Join-Path $OutputDirectory "$SafeName.stdout.txt"
    $StderrPath = Join-Path $OutputDirectory "$SafeName.stderr.txt"
    $Started = [Diagnostics.Stopwatch]::StartNew()
    $Info = [Diagnostics.ProcessStartInfo]::new()
    $Info.FileName = $Adb
    $Info.UseShellExecute = $false
    $Info.CreateNoWindow = $true
    $Info.RedirectStandardOutput = $true
    $Info.RedirectStandardError = $true
    foreach ($Argument in $Arguments) {
        $Info.ArgumentList.Add($Argument)
    }

    $ExitCode = $null
    $TimedOut = $false
    $StartError = $null
    $Stdout = ""
    $Stderr = ""
    try {
        $Process = [Diagnostics.Process]::Start($Info)
        if (-not $Process) {
            throw "Unable to start adb."
        }
        $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
        $StderrTask = $Process.StandardError.ReadToEndAsync()
        if (-not $Process.WaitForExit($CommandTimeoutSeconds * 1000)) {
            $TimedOut = $true
            $Process.Kill($true)
            $Process.WaitForExit()
        }
        $Stdout = $StdoutTask.GetAwaiter().GetResult()
        $Stderr = $StderrTask.GetAwaiter().GetResult()
        $ExitCode = $Process.ExitCode
        $Process.Dispose()
    }
    catch {
        $StartError = $_.Exception.Message
        $Stderr = $StartError
    }
    finally {
        $Started.Stop()
        [IO.File]::WriteAllText($StdoutPath, $Stdout, [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText($StderrPath, $Stderr, [Text.UTF8Encoding]::new($false))
    }

    return [pscustomobject]@{
        name = $Name
        arguments = $Arguments
        exit_code = $ExitCode
        timed_out = $TimedOut
        start_error = $StartError
        duration_ms = [math]::Round($Started.Elapsed.TotalMilliseconds, 3)
        stdout_file = [IO.Path]::GetRelativePath($OutputDirectory, $StdoutPath)
        stderr_file = [IO.Path]::GetRelativePath($OutputDirectory, $StderrPath)
    }
}

$Results = [Collections.Generic.List[object]]::new()
$Commands = @(
    @("adb_version", @("version")),
    @("adb_devices", @("devices", "-l")),
    @("adb_state", @("get-state")),
    @("uptime", @("shell", "cat /proc/uptime")),
    @("kernel_cmdline", @("shell", "cat /proc/cmdline")),
    @("kernel_version", @("shell", "cat /proc/version")),
    @("memory", @("shell", "cat /proc/meminfo")),
    @("processes", @("shell", "ps")),
    @("thermal_zones", @("shell", 'for f in /sys/class/thermal/thermal_zone*/temp; do echo $f; cat $f; done')),
    @("cpu_governors", @("shell", 'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo $f; cat $f; done')),
    @("npu_governor", @("shell", "cat /sys/class/devfreq/fde40000.npu/governor")),
    @("npu_load", @("shell", "cat /sys/kernel/debug/rknpu/load")),
    @("npu_driver", @("shell", "cat /sys/kernel/debug/rknpu/driver_version")),
    @("service_process", @("shell", 'p=/userdata/medvision/run/medvision_rk3568.pid; if [ -f $p ]; then pid=$(cat $p); echo PID=$pid; cat /proc/$pid/status; echo FD_COUNT=$(ls /proc/$pid/fd 2>/dev/null | wc -l); fi')),
    @("service_log", @("shell", "tail -n 300 /userdata/medvision/logs/service.log")),
    @("kernel_log", @("shell", "dmesg"))
)

foreach ($Command in $Commands) {
    Write-Host "Collecting $($Command[0])..."
    $Results.Add((Invoke-AdbCapture -Name $Command[0] -Arguments $Command[1]))
}

$HealthPath = Join-Path $OutputDirectory "http_health.txt"
$HealthError = $null
try {
    $Health = Invoke-WebRequest -Uri "http://127.0.0.1:18080/health" -TimeoutSec $HttpTimeoutSeconds
    [IO.File]::WriteAllText(
        $HealthPath,
        $Health.Content,
        [Text.UTF8Encoding]::new($false)
    )
}
catch {
    $HealthError = $_.Exception.Message
    [IO.File]::WriteAllText(
        $HealthPath,
        "ERROR: $HealthError",
        [Text.UTF8Encoding]::new($false)
    )
}

$Successful = @($Results | Where-Object { -not $_.timed_out -and $_.exit_code -eq 0 }).Count
$Summary = [ordered]@{
    schema_version = 1
    collected_at = (Get-Date).ToString("o")
    adb = $Adb
    command_timeout_seconds = $CommandTimeoutSeconds
    output_directory = $OutputDirectory
    successful_commands = $Successful
    total_commands = $Results.Count
    http_health_error = $HealthError
    commands = $Results
}
$SummaryPath = Join-Path $OutputDirectory "diagnostics_summary.json"
$Summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryPath -Encoding utf8NoBOM

$State = if ($Successful -eq $Results.Count -and -not $HealthError) { "PASS" } else { "PARTIAL" }
Write-Host "DIAGNOSTICS=$State directory=$OutputDirectory"
