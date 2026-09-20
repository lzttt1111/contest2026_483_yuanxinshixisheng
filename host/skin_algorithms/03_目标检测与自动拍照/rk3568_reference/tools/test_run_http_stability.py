#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_http_stability.py")
SPEC = importlib.util.spec_from_file_location("run_http_stability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STABILITY)


class StabilityReportTest(unittest.TestCase):
    def make_args(
        self,
        root: Path,
        *,
        duration_seconds: float = 0.01,
        max_consecutive_errors: int = 3,
    ) -> argparse.Namespace:
        image = root / "input.jpg"
        image.write_bytes(b"\xff\xd8mock-jpeg\xff\xd9")
        return argparse.Namespace(
            api_root="http://mock",
            duration_seconds=duration_seconds,
            progress_seconds=60.0,
            timeout_seconds=0.1,
            max_consecutive_errors=max_consecutive_errors,
            image=image,
            calibration_manifest=root / "unused.json",
            report=root / "report.json",
        )

    @staticmethod
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "runtime_provider": "RKNN_NPU",
            "model_quantization": "test-quantization",
        }

    @staticmethod
    def frame() -> dict[str, object]:
        return {
            "ok": True,
            "result": {
                "runtime_provider": "RKNN_NPU",
                "timings_ms": {
                    "total": 10.0,
                    "device_inference": 8.0,
                    "contact_inference": 1.0,
                },
            },
        }

    def test_pass_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))

            def request(url: str, **_: object) -> tuple[dict[str, object], float]:
                if url.endswith("/health"):
                    return self.health(), 1.0
                if url.endswith("/v1/stream/reset"):
                    return {"ok": True}, 1.0
                return self.frame(), 11.0

            with mock.patch.object(STABILITY, "parse_args", return_value=args):
                with mock.patch.object(STABILITY, "request_json", side_effect=request):
                    result = STABILITY.main()

            report = json.loads(args.report.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(report["result"], "PASS")
            self.assertEqual(report["termination_reason"], "completed")
            self.assertGreater(report["completed_frames"], 0)

    def test_consecutive_errors_persist_fail_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory), duration_seconds=60.0)
            health_calls = 0

            def request(url: str, **_: object) -> tuple[dict[str, object], float]:
                nonlocal health_calls
                if url.endswith("/health"):
                    health_calls += 1
                    return self.health(), 1.0
                if url.endswith("/v1/stream/reset"):
                    return {"ok": True}, 1.0
                raise TimeoutError("mock board timeout")

            with mock.patch.object(STABILITY, "parse_args", return_value=args):
                with mock.patch.object(STABILITY, "request_json", side_effect=request):
                    result = STABILITY.main()

            report = json.loads(args.report.read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertEqual(report["result"], "FAIL")
            self.assertEqual(report["termination_reason"], "consecutive_errors")
            self.assertEqual(report["attempted_frames"], 3)
            self.assertEqual(report["error_count"], 3)
            self.assertEqual(health_calls, 2)

    def test_keyboard_interrupt_persists_fail_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory), duration_seconds=60.0)
            with mock.patch.object(STABILITY, "parse_args", return_value=args):
                with mock.patch.object(
                    STABILITY, "request_json", side_effect=KeyboardInterrupt
                ):
                    result = STABILITY.main()

            report = json.loads(args.report.read_text(encoding="utf-8"))
            self.assertEqual(result, 130)
            self.assertEqual(report["result"], "FAIL")
            self.assertEqual(report["termination_reason"], "keyboard_interrupt")
            self.assertEqual(report["fatal_error"]["type"], "KeyboardInterrupt")
            self.assertEqual(
                report["final_health_error"]["type"], "KeyboardInterrupt"
            )


if __name__ == "__main__":
    unittest.main()
