# RK3568 board hang observation — 2026-08-31

## Scope

This records the first 30-minute HTTP stability attempt for RK3568-DEPLOY-V1.
The original stability runner was manually interrupted after the board stopped
responding and did not persist its final JSON report. The figures below are
transcribed from the live progress output; they are not a completed acceptance
report.

## Observed sequence

- Board service PID at start: `2056`.
- Initial process state: `VmSize=42284 kB`, `VmRSS=39304 kB`,
  `VmData=3316 kB`, one thread, and 19 file descriptors.
- CPU, DMC, and NPU governors were set to `performance`.
- Through minutes 1–9, the HTTP stream reported no request errors and sustained
  approximately `5.22 FPS`.
- At the five-minute snapshot, `VmRSS` remained `39304 kB`, the process
  still had one thread, and the active-connection file-descriptor count was 20.
  No monotonic memory growth was observed before the hang.
- At approximately `601.6 s`: attempted `3149`, completed `3051`,
  errors `98`; the health endpoint timed out.
- At approximately `661.8 s`: attempted `3161`, completed `3051`,
  errors `110`; completed requests had stopped increasing.
- Windows `adb devices` still listed the board as `device`, but
  `adb shell` did not return.

## Interpretation boundary

The observation proves that the V1 stack did not pass the 30-minute stability
gate. It does not yet identify the cause. The simultaneous HTTP and ADB-shell
stall is consistent with a wider board userspace, kernel, USB-gadget, thermal,
power, or NPU-driver problem; none of those causes is established without a
UART/kernel log.

## Required rerun

1. Start `deploy/capture_serial.ps1` on UART2 at 1500000 baud before board
   power-on or reset.
2. Deploy the unchanged, checksum-verified release.
3. Run `deploy/collect_diagnostics.ps1` once as a healthy baseline.
4. Run `tools/run_http_stability.py` for 1800 seconds. The updated runner
   stops after three consecutive errors and always writes a JSON report.
5. If the board stalls, keep serial capture running and collect a second bounded
   diagnostic snapshot without rebooting first.
6. Record thermal readings, kernel/NPU messages, service PID state, and whether
   the serial console itself remains interactive before deciding on a fix.
