from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_utils import json_safe


PROTOCOL_PREFIX = "@@NINE_JSON@@"


def emit(payload: dict[str, Any]) -> None:
    print(
        PROTOCOL_PREFIX
        + json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False),
        flush=True,
    )


def gpu_snapshot(reset_peak: bool = False) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(0)
        return {
            "available": True,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "allocated_mib": round(torch.cuda.memory_allocated(0) / 1024**2, 3),
            "reserved_mib": round(torch.cuda.memory_reserved(0) / 1024**2, 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 3),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


class ServiceRuntime:
    def __init__(self, service: str, service_root: Path) -> None:
        self.service = service
        self.service_root = service_root.resolve()
        os.chdir(self.service_root)
        sys.path.insert(0, str(self.service_root))
        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        self.pipeline = self._make_pipeline()

    def _make_pipeline(self):
        if self.service == "dermavision":
            from src.pipeline import DermaVisionPipeline

            return DermaVisionPipeline()
        if self.service == "acne":
            from src.acne.pipeline import Pipeline

            return Pipeline()
        if self.service == "wrinkle":
            from src.wrinkle.pipeline import Pipeline

            return Pipeline()
        raise ValueError(f"未知服务: {self.service}")

    def warmup(self) -> dict[str, Any]:
        started = time.perf_counter()
        metadata = self.pipeline.warmup()
        return {
            "seconds": round(time.perf_counter() - started, 4),
            "metadata": metadata,
            "gpu": gpu_snapshot(),
        }

    def _configure_output(self, request_root: Path) -> None:
        service_output = request_root / self.service
        service_output.mkdir(parents=True, exist_ok=True)
        if self.service == "dermavision":
            mapping = {
                "output_dir": service_output,
                "preprocessed_dir": service_output / "preprocessed",
                "rbx_dir": service_output / "redness",
                "spots_dir": service_output / "spots",
                "brown_dir": service_output / "brown",
                "texture_dir": service_output / "texture",
                "pores_dir": service_output / "pores",
                "purple_dir": service_output / "purple",
                "surface_gloss_dir": service_output / "surface_gloss",
                "vascular_dir": service_output / "vascular",
                "contour_firmness_dir": service_output / "contour_firmness",
                "wrinkles_dir": service_output / "wrinkles",
            }
            for name, path in mapping.items():
                path.mkdir(parents=True, exist_ok=True)
                setattr(self.pipeline, name, str(path))
            if self.pipeline.preprocessor is not None:
                self.pipeline.preprocessor.output_dir = str(mapping["preprocessed_dir"])
        else:
            self.pipeline.output_root = service_output

    def analyze(
        self,
        image_path: Path,
        output_root: Path,
        algorithms: list[str] | None = None,
    ) -> dict[str, Any]:
        self._configure_output(output_root)
        gpu_before = gpu_snapshot(reset_peak=True)
        started = time.perf_counter()
        if self.service == "dermavision":
            requested = algorithms or ["redness", "spots", "brown", "texture", "pores", "purple"]
        elif self.service == "acne":
            from src.acne.artifact_policy import FORMAL_FAST_TOKEN

            requested = ["acne", FORMAL_FAST_TOKEN]
        else:
            requested = ["wrinkle", "balanced"]
        result = self.pipeline.process_single(str(image_path), algorithms=requested)
        return {
            "service": self.service,
            "status": result.get("status", "failed"),
            "seconds": round(time.perf_counter() - started, 4),
            "algorithms": requested,
            "result": result,
            "gpu_before": gpu_before,
            "gpu_after": gpu_snapshot(),
        }

    def close(self) -> None:
        close = getattr(self.pipeline, "close", None)
        if callable(close):
            close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True, choices=("dermavision", "acne", "wrinkle"))
    parser.add_argument("--service-root", required=True, type=Path)
    args = parser.parse_args()

    runtime: ServiceRuntime | None = None
    try:
        process_started = time.perf_counter()
        runtime = ServiceRuntime(args.service, args.service_root)
        warmup = runtime.warmup()
        if not warmup["gpu"].get("available"):
            raise RuntimeError(f"{args.service} CUDA 不可用: {warmup['gpu']}")
        emit({
            "event": "ready",
            "service": args.service,
            "python_executable": sys.executable,
            "process_start_seconds": round(time.perf_counter() - process_started, 4),
            "warmup": warmup,
        })
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            request_id = str(request.get("id", ""))
            if request.get("command") == "close":
                emit({"event": "closed", "service": args.service, "id": request_id})
                break
            if request.get("command") != "analyze":
                emit({"event": "error", "service": args.service, "id": request_id, "message": "unknown command"})
                continue
            try:
                response = runtime.analyze(
                    Path(request["image_path"]).resolve(),
                    Path(request["output_root"]).resolve(),
                    algorithms=request.get("algorithms"),
                )
                emit({"event": "result", "id": request_id, **response})
            except Exception as exc:
                emit({
                    "event": "result",
                    "id": request_id,
                    "service": args.service,
                    "status": "failed",
                    "seconds": 0.0,
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "gpu_after": gpu_snapshot(),
                })
    except Exception as exc:
        emit({
            "event": "fatal",
            "service": args.service,
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "gpu": gpu_snapshot(),
        })
        return 1
    finally:
        if runtime is not None:
            runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
