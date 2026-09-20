#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MODEL_FILES = {
    "device": {
        "fp16": "device_yolox_s_v5_fp16.rknn",
        "int8": "device_yolox_s_v5_int8.rknn",
        "int8_mmse": "device_yolox_s_v5_int8_mmse.rknn",
        "int8_opt2": "device_yolox_s_v5_int8_opt2.rknn",
        "int8_hybrid_proposal": (
            "device_yolox_s_v5_int8_hybrid_proposal.rknn"
        ),
        "int8_hybrid_regression_head": (
            "device_yolox_s_v5_int8_hybrid_regression_head.rknn"
        ),
        "int8_hybrid_all_heads": (
            "device_yolox_s_v5_int8_hybrid_all_heads.rknn"
        ),
    },
    "contact": {
        "fp16": "contact_mobilenet_v3_small_v5_fp16.rknn",
        "int8": "contact_mobilenet_v3_small_v5_int8.rknn",
        "int8_mmse": "contact_mobilenet_v3_small_v5_int8_mmse.rknn",
        "int8_opt2": "contact_mobilenet_v3_small_v5_int8_opt2.rknn",
        "int8_hybrid_proposal": (
            "contact_mobilenet_v3_small_v5_int8_hybrid_proposal.rknn"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, target: Path, executable: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if executable:
        target.chmod(target.stat().st_mode | 0o111)


def git_value(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def write_config(
    target: Path, device_sha: str, contact_sha: str, quantization: str
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            (
                "DEVICE_MODEL=models/device.rknn",
                "CONTACT_MODEL=models/contact.rknn",
                f"DEVICE_SHA256={device_sha}",
                f"CONTACT_SHA256={contact_sha}",
                f"MODEL_QUANTIZATION={quantization}",
                "BOARD_MODEL=ATK-DLRK3568",
                "SERVICE_HOST=127.0.0.1",
                "SERVICE_PORT=8080",
                "ENABLE_PERFORMANCE_MODE=1",
                "DEVICE_CONFIDENCE=0.25",
                "NMS_IOU=0.65",
                "DEVICE_STALE_MS=250",
                "CONTACT_STALE_MS=150",
                "CONTACT_INTERVAL_VALID_FRAMES=2",
                "",
            )
        ),
        encoding="utf-8",
    )


def write_readme(target: Path, selection: dict[str, object]) -> None:
    target.write_text(
        f"""# MedVision RK3568 V1

This is the offline ARM64 release for ATK-DLRK3568.

Selected accuracy-gated models:

- Device: {selection['device']}
- Contact point: {selection['contact']}
- Readiness: {selection['deployment_readiness']}

## Deploy from Windows

1. Connect exactly one board and ensure it appears in adb devices.
2. Run pwsh -File .\\deploy\\deploy.ps1 -Adb <path-to-adb.exe>.
3. Run pwsh -File .\\deploy\\serve_demo.ps1.
4. Open http://127.0.0.1:8090.

The deployment script copies the package to /userdata/medvision/, verifies
SHA256SUMS on the board, starts the NPU service, and configures
adb forward tcp:18080 tcp:8080.

## Capture evidence before a stability run

The ATK-DLRK3568 UART2 debug console defaults to 1500000 baud, 8N1, with no
flow control. Start serial capture before powering or resetting the board:

    [IO.Ports.SerialPort]::GetPortNames()
    pwsh -ExecutionPolicy Bypass -File .\\deploy\\capture_serial.ps1 -Port COMx

Collect a bounded ADB, thermal, NPU, process, service-log, and kernel-log
snapshot at baseline or immediately after a failure:

    pwsh -ExecutionPolicy Bypass -File .\\deploy\\collect_diagnostics.ps1 -Adb <path-to-adb.exe>

Run the 30-minute HTTP test with a fixed JPEG:

    python .\\tools\\run_http_stability.py --image C:\\path\\fixed.jpg --duration-seconds 1800

The stability runner stops after three consecutive failures by default and
always writes a JSON report, including on Ctrl+C or a final health timeout.

The PC performs camera capture, JPEG encoding, and drawing only. Device
detection and contact-point inference run on the RK3568 NPU.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    deploy_root = Path(__file__).resolve().parents[1]
    repo_root = deploy_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=deploy_root / "artifacts" / "parity" / "rknn_parity_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "release" / "rk3568-v1",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MEDVISION_RK3568_CACHE", str(Path.home() / ".cache" / "medvision-rk3568"))),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deploy_root = Path(__file__).resolve().parents[1]
    repo_root = deploy_root.parent
    output = args.output.resolve()
    expected_parent = (repo_root / "release").resolve()
    if output.parent != expected_parent or output.name != "rk3568-v1":
        raise RuntimeError(
            f"refusing to replace unexpected output directory: {output}"
        )

    parity_report = json.loads(args.parity_report.read_text(encoding="utf-8"))
    if parity_report.get("result") != "PASS":
        raise RuntimeError("parity report did not pass the FP16 baseline gates")
    selection = parity_report["selection"]
    selected_paths = {}
    selected_metadata = {}
    for family in ("device", "contact"):
        variant = selection[family]
        family_results = parity_report[family]
        if variant not in family_results:
            raise RuntimeError(f"selected {family} variant is absent: {variant}")
        if not family_results[variant]["gate"]["pass"]:
            raise RuntimeError(f"selected {family} variant failed its gate: {variant}")
        try:
            filename = MODEL_FILES[family][variant]
        except KeyError as error:
            raise RuntimeError(f"no artifact mapping for {family}:{variant}") from error
        path = deploy_root / "artifacts" / "rknn" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        selected_paths[family] = path
        selected_metadata[family] = {
            "variant": variant,
            "source_path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "parity": family_results[variant],
        }

    stage = (deploy_root / "release-stage" / "rk3568-v1").resolve()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    cache = args.cache_root.resolve()
    runtime_root = cache / "rknpu2" / "runtime" / "RK356X" / "Linux"
    zoo_root = cache / "rknn_model_zoo"
    copies = (
        (deploy_root / "build" / "medvision_rk3568", stage / "bin" / "medvision_rk3568", True),
        (selected_paths["device"], stage / "models" / "device.rknn", False),
        (selected_paths["contact"], stage / "models" / "contact.rknn", False),
        (runtime_root / "librknn_api" / "aarch64" / "librknnrt.so", stage / "lib" / "librknnrt.so", False),
        (zoo_root / "3rdparty" / "librga" / "Linux" / "aarch64" / "librga.so", stage / "lib" / "librga.so", False),
        (deploy_root / "deploy" / "start.sh", stage / "deploy" / "start.sh", True),
        (deploy_root / "deploy" / "stop.sh", stage / "deploy" / "stop.sh", True),
        (deploy_root / "deploy" / "deploy.ps1", stage / "deploy" / "deploy.ps1", False),
        (deploy_root / "deploy" / "serve_demo.ps1", stage / "deploy" / "serve_demo.ps1", False),
        (deploy_root / "deploy" / "collect_diagnostics.ps1", stage / "deploy" / "collect_diagnostics.ps1", False),
        (deploy_root / "deploy" / "capture_serial.ps1", stage / "deploy" / "capture_serial.ps1", False),
        (deploy_root / "tools" / "run_http_stability.py", stage / "tools" / "run_http_stability.py", True),
        (cache / "rknpu2" / "LICENSE", stage / "licenses" / "rknpu2-LICENSE", False),
        (zoo_root / "LICENSE", stage / "licenses" / "rknn-model-zoo-LICENSE", False),
        (zoo_root / "3rdparty" / "jpeg_turbo" / "LICENSE.md", stage / "licenses" / "libjpeg-turbo-LICENSE.md", False),
    )
    for source, target, executable in copies:
        copy_file(source, target, executable)

    web_source = deploy_root / "web" / "rk3568_demo"
    shutil.copytree(web_source, stage / "web" / "rk3568_demo")

    device_sha = selected_metadata["device"]["sha256"]
    contact_sha = selected_metadata["contact"]["sha256"]
    quantization = (
        f"device_{selection['device'].upper()}+contact_{selection['contact'].upper()}"
    )
    write_config(stage / "config" / "service.env", device_sha, contact_sha, quantization)
    write_readme(stage / "README.md", selection)

    status = git_value(repo_root, "status", "--porcelain")
    manifest = {
        "schema_version": 1,
        "service_version": "RK3568-DEPLOY-V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "board_model": "ATK-DLRK3568",
        "deploy_path": "/userdata/medvision/",
        "runtime_provider": "RKNN_NPU",
        "rknn_runtime_version": "1.5.2",
        "toolkit_version": "1.5.2",
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "parity_report_sha256": sha256(args.parity_report),
        "selection": selection,
        "models": selected_metadata,
    }
    manifest_path = stage / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    checksum_rows = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        relative = path.relative_to(stage).as_posix()
        checksum_rows.append(f"{sha256(path)}  {relative}")
    (stage / "SHA256SUMS").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    shutil.move(str(stage), str(output))
    file_count = sum(1 for item in output.rglob("*") if item.is_file())
    print(
        f"PACKAGE_RELEASE=PASS output={output} files={file_count} "
        f"device={selection['device']} contact={selection['contact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
