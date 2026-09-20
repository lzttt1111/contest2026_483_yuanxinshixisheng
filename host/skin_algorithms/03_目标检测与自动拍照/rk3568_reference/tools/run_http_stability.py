#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PROVIDER = "RKNN_NPU"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def metric(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def request_json(
    url: str,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers or {},
        method=method,
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
        status = response.status
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if status < 200 or status >= 300:
        raise RuntimeError(f"HTTP {status}")
    if not payload.get("ok"):
        raise RuntimeError(f"service returned ok=false: {payload}")
    return payload, elapsed_ms


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-root", default="http://127.0.0.1:18080")
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--progress-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=3,
        help="stop and persist a FAIL report after this many failures; 0 disables",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="JPEG input; defaults to the first fixed golden image",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=root / "artifacts" / "calibration" / "calibration_manifest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "artifacts" / "stability" / "http_stability_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0 or args.progress_seconds <= 0:
        raise ValueError("durations must be positive")
    if args.timeout_seconds <= 0 or args.max_consecutive_errors < 0:
        raise ValueError("timeout must be positive and max errors non-negative")
    if args.image:
        image_path = args.image.resolve()
    else:
        calibration = json.loads(
            args.calibration_manifest.read_text(encoding="utf-8")
        )
        image_path = Path(calibration["golden_set"][0]["derived"]).resolve()
    jpeg = image_path.read_bytes()
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise RuntimeError(f"input is not a complete JPEG: {image_path}")

    origin = "http://127.0.0.1:8090"
    common_headers = {
        "Origin": origin,
        "Content-Type": "image/jpeg",
    }
    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()
    deadline = started + args.duration_seconds
    next_progress = started + args.progress_seconds
    frame_id = 0
    errors: list[dict[str, object]] = []
    rtt_ms: list[float] = []
    board_total_ms: list[float] = []
    device_inference_ms: list[float] = []
    contact_inference_ms: list[float] = []
    initial_health: dict[str, Any] | None = None
    final_health: dict[str, Any] | None = None
    final_health_error: dict[str, object] | None = None
    fatal_error: dict[str, object] | None = None
    termination_reason = "completed"
    consecutive_errors = 0
    max_observed_consecutive_errors = 0

    try:
        initial_health, _ = request_json(
            args.api_root + "/health",
            headers={"Origin": origin},
            timeout=args.timeout_seconds,
        )
        if initial_health.get("runtime_provider") != EXPECTED_PROVIDER:
            raise RuntimeError("health did not report RKNN_NPU")
        request_json(
            args.api_root + "/v1/stream/reset",
            method="POST",
            body=b"",
            headers={"Origin": origin},
            timeout=args.timeout_seconds,
        )

        while time.monotonic() < deadline:
            frame_id += 1
            headers = dict(common_headers)
            headers["X-Frame-Id"] = str(frame_id)
            headers["X-Capture-Timestamp-Ms"] = str(int(time.time() * 1000))
            try:
                payload, elapsed_ms = request_json(
                    args.api_root + "/v1/stream/frame",
                    method="POST",
                    body=jpeg,
                    headers=headers,
                    timeout=args.timeout_seconds,
                )
                result = payload["result"]
                if result.get("runtime_provider") != EXPECTED_PROVIDER:
                    raise RuntimeError("frame did not report RKNN_NPU")
                timings = result.get("timings_ms", {})
                rtt_ms.append(elapsed_ms)
                board_total_ms.append(float(timings["total"]))
                device_inference_ms.append(float(timings["device_inference"]))
                contact_value = float(
                    timings.get("contact_inference", 0.0) or 0.0
                )
                if contact_value > 0:
                    contact_inference_ms.append(contact_value)
                consecutive_errors = 0
            except Exception as error:
                consecutive_errors += 1
                max_observed_consecutive_errors = max(
                    max_observed_consecutive_errors, consecutive_errors
                )
                if len(errors) < 100:
                    errors.append(
                        {
                            "phase": "frame",
                            "frame_id": frame_id,
                            "elapsed_seconds": time.monotonic() - started,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                if (
                    args.max_consecutive_errors
                    and consecutive_errors >= args.max_consecutive_errors
                ):
                    termination_reason = "consecutive_errors"
                    break
            now = time.monotonic()
            if now >= next_progress:
                completed = len(rtt_ms)
                elapsed = now - started
                print(
                    f"STABILITY_PROGRESS elapsed_s={elapsed:.1f} "
                    f"attempted={frame_id} completed={completed} "
                    f"errors={frame_id - completed} "
                    f"fps={completed / elapsed:.3f}",
                    flush=True,
                )
                next_progress += args.progress_seconds
    except KeyboardInterrupt as error:
        termination_reason = "keyboard_interrupt"
        fatal_error = {
            "phase": "runner",
            "elapsed_seconds": time.monotonic() - started,
            "type": type(error).__name__,
            "message": "test interrupted by operator",
        }
    except Exception as error:
        termination_reason = "fatal_error"
        fatal_error = {
            "phase": "setup_or_runner",
            "elapsed_seconds": time.monotonic() - started,
            "type": type(error).__name__,
            "message": str(error),
        }

    finished = time.monotonic()
    try:
        final_health, _ = request_json(
            args.api_root + "/health",
            headers={"Origin": origin},
            timeout=args.timeout_seconds,
        )
    except (Exception, KeyboardInterrupt) as error:
        final_health_error = {
            "phase": "final_health",
            "elapsed_seconds": time.monotonic() - started,
            "type": type(error).__name__,
            "message": str(error) or "final health check interrupted",
        }

    elapsed_seconds = finished - started
    completed = len(rtt_ms)
    error_count = frame_id - completed
    passed = (
        termination_reason == "completed"
        and error_count == 0
        and completed > 0
        and initial_health is not None
        and final_health is not None
        and final_health.get("runtime_provider") == EXPECTED_PROVIDER
        and final_health.get("model_quantization")
        == initial_health.get("model_quantization")
    )
    report = {
        "schema_version": 2,
        "result": "PASS" if passed else "FAIL",
        "termination_reason": termination_reason,
        "started_at_utc": started_wall.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_duration_seconds": args.duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": args.timeout_seconds,
        "max_consecutive_errors": args.max_consecutive_errors,
        "max_observed_consecutive_errors": max_observed_consecutive_errors,
        "attempted_frames": frame_id,
        "completed_frames": completed,
        "error_count": error_count,
        "actual_fps": completed / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "image": str(image_path),
        "initial_health": initial_health,
        "final_health": final_health,
        "final_health_error": final_health_error,
        "fatal_error": fatal_error,
        "metrics_ms": {
            "rtt": metric(rtt_ms),
            "board_total": metric(board_total_ms),
            "device_inference": metric(device_inference_ms),
            "contact_inference": metric(contact_inference_ms),
        },
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"STABILITY_TEST={report['result']} elapsed_s={elapsed_seconds:.1f} "
        f"completed={completed} errors={error_count} "
        f"reason={termination_reason} fps={report['actual_fps']:.3f} "
        f"report={args.report.resolve()}",
        flush=True,
    )
    if termination_reason == "keyboard_interrupt":
        return 130
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
