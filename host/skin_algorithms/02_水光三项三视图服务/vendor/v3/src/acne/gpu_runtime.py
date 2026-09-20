"""Strict process-local CUDA runtime shared by YOLO and Acne-LDS."""

from __future__ import annotations

import os
import threading

import torch


_LOCK = threading.Lock()
_INITIALIZED = False


def require_cuda(device: str = "cuda:0") -> torch.device:
    global _INITIALIZED
    requested = str(device).strip().lower()
    if requested in {"0", "cuda", "cuda:0"}:
        requested = "cuda:0"
    if not requested.startswith("cuda"):
        raise RuntimeError(
            f"Acne production inference requires cuda:0; received {device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for acne inference; CPU fallback is disabled"
        )
    result = torch.device(requested)
    index = result.index if result.index is not None else 0
    with _LOCK:
        if not _INITIALIZED:
            torch.cuda.set_device(index)
            fraction = float(os.getenv("ACNE_GPU_MEMORY_FRACTION", "0.20"))
            torch.cuda.set_per_process_memory_fraction(fraction, index)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            _INITIALIZED = True
    return result


def cuda_metadata(device: str = "cuda:0") -> dict[str, object]:
    resolved = require_cuda(device)
    index = resolved.index if resolved.index is not None else 0
    return {
        "device": str(resolved),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(index),
        "compute_capability": list(torch.cuda.get_device_capability(index)),
    }
