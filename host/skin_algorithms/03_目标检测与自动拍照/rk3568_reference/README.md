# RK3568 Offline Deployment V1

This directory is the isolated implementation area for the ATK-DLRK3568 deployment.

## Boundaries

- Source worktree: `codex/rk3568-deploy-v1` at commit `e8798373ad89af17f622f10ebc421a74d7a02b2c`.
- Do not modify training, capture, review, or the existing research web pages.
- Do not overwrite source images, annotations, V1-V7 models, or reports.
- Do not access the network without explicit user approval. Reuse `/home/lzttttt/.cache/medvision-rk3568/`.
- Board runtime: Buildroot 2021.11, ARM64, RKNN Runtime 1.5.2.
- V1 deploys only V5 REAL-ONLY YOLOX-S and the V5 MobileNetV3-Small contact-point model.
- SAM2, Face Mesh, contact-state classification, and `/dev/video0` are outside V1.

## Current gated selection

- Device: `int8_hybrid_all_heads`, minimum matched-box IoU `0.956373`, recall drop `0`.
- Contact point: `int8_hybrid_proposal`, maximum normalized point error `0.014033`, visibility agreement `1.0`.
- Deployment readiness: `INT8_DEFAULT_READY`.
- 100-frame board HTTP run: `5.492 FPS`, RTT P95 `192.205 ms`, 100/100 requests passed.
- The device-only result is approximately `6.6 FPS`, below the planned `8 FPS` gate.
- The 30-minute acceptance remains open: the first run became unresponsive after about 10 minutes. See `BOARD_HANG_OBSERVATION_2026-08-31.md`.

## Layout

- `tools/export_v5_onnx.py`: exports RKNN-friendly opset-12 ONNX models and verifies CPU parity.
- `tools/prepare_calibration.py`: creates deterministic 100-image calibration sets and a 50-image golden set without changing source data.
- `tools/convert_rknn.py`: builds FP16 and INT8 RKNN models with Toolkit2 1.5.2.
- `tools/prepare_hybrid_quant.py`, `tools/make_hybrid_profile.py`, and `tools/build_hybrid_quant.py`: create reproducible mixed-precision candidates.
- `tools/prepare_parity_inputs.py` and `tools/evaluate_rknn_parity.py`: run the fixed 50-image ONNX/RKNN accuracy gates.
- `tools/run_http_stability.py`: runs the board HTTP soak test and always persists a PASS/FAIL report.
- `tools/package_release.py`: selects only passing variants and assembles `release/rk3568-v1/` with checksums and licenses.
- `cpp/`: standalone C++14 board runtime.
- `web/rk3568_demo/`: PC-only camera capture and result overlay.
- `artifacts/`: generated local outputs; ignored by Git.
- `deploy/deploy.ps1`: deploys, verifies, starts the service, and creates the ADB forward.
- `deploy/capture_serial.ps1`: captures UART2 at 1500000 baud, 8N1, no flow control.
- `deploy/collect_diagnostics.ps1`: captures bounded ADB, thermal, NPU, process, service, and kernel evidence.

## Release workflow

```bash
rk3568_deploy/tools/build_cpp.sh
python3 rk3568_deploy/tools/package_release.py
cd release/rk3568-v1
sha256sum -c SHA256SUMS
```

On Windows, with exactly one online ADB board:

```powershell
pwsh -File .\deploy\deploy.ps1 -Adb <SOURCE_OR_OUTPUT_PATH>
pwsh -File .\deploy\serve_demo.ps1
```

Then open `http://127.0.0.1:8090`. The page rejects any health or frame response that does not report `runtime_provider=RKNN_NPU`.

The package generator follows `artifacts/parity/rknn_parity_report.json`; a smaller or faster model is never selected unless its gate has passed. The passing hybrid INT8 pair above is now the default, while FP16 artifacts remain available as a conversion/parity fallback.

## Failure capture and stability rerun

Start debug-UART capture before powering or resetting the board:

```powershell
[IO.Ports.SerialPort]::GetPortNames()
pwsh -ExecutionPolicy Bypass -File .\deploy\capture_serial.ps1 -Port COMx
```

The debug console is UART2 at `1500000` baud, 8 data bits, no parity, 1 stop bit, and no flow control. USB ADB is separate and has no baud-rate setting.

After deployment, capture a baseline and run the soak test:

```powershell
pwsh -ExecutionPolicy Bypass -File .\deploy\collect_diagnostics.ps1 -Adb <SOURCE_OR_OUTPUT_PATH>
python .\tools\run_http_stability.py --image <SOURCE_OR_OUTPUT_PATH>
```

The soak runner stops after three consecutive frame failures by default. Normal completion, consecutive failures, setup failures, final-health failures, and Ctrl+C all produce a JSON report.

## Pinned source assets

- Device checkpoint: `/home/lzttttt/mubiao--test/models/camera_active_v5/device/real_only/camera_active_v5_real_only_yolox_s/best_ckpt.pth`
- Device checkpoint SHA256: `89dd527662a758a62856978e2638dcf4a38a7a3fb6de4e97c058ec652cb0c6b6`
- Contact checkpoint: `/home/lzttttt/mubiao--test/models/camera_active_v5/contact/best.pt`
- Contact checkpoint SHA256: `299f4ea45c2d0aae742fac2c67e71d4b61dbdf0d59a44ca1d96b394075086fa2`
- YOLOX source commit: `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`

Generated files record source and output SHA256 values. The source root remains read-only during deployment.
