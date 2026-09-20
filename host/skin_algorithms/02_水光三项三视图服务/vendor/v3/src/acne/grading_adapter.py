from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import torch
from PIL import Image

from src.acne.gpu_runtime import cuda_metadata, require_cuda


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "acne_lds_vendor"
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def skipped_grading(reason: str = "input_not_eligible_for_grading") -> dict:
    return {
        "status": "skipped",
        "reason": reason,
        "predicted_count": None,
        "severity_level": None,
        "severity_probabilities": [],
        "model": "acne-lds",
        "device": "cuda:0",
    }


def _load_model(checkpoint: Path, model_type: str, device: str):
    key = (str(checkpoint.resolve()), model_type, device)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        if str(VENDOR_ROOT) not in sys.path:
            sys.path.insert(0, str(VENDOR_ROOT))
        from predict_on_img import ModelInit

        model = ModelInit(
            model_type=model_type,
            path_checkpoint=str(checkpoint),
            device=device,
        )
        _MODEL_CACHE[key] = model
        return model


def warmup_acne_lds(
    checkpoint_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, object]:
    """在 Worker 子进程冷启动阶段加载并缓存 Acne-LDS。"""
    checkpoint = Path(checkpoint_path)
    resolved = str(require_cuda(device))
    if not checkpoint.exists():
        raise FileNotFoundError(f"grading checkpoint missing: {checkpoint}")
    _load_model(checkpoint, "model_ld_smoothing", resolved)
    return {
        "device": resolved,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
    }


def run_acne_lds_grading(
    image_path: str | Path,
    output_json: str | Path,
    checkpoint_path: str | Path,
    python_path: str | Path = PROJECT_ROOT / ".venv" / "bin" / "python",
    device: str = "cuda:0",
    timeout_seconds: int = 300,
) -> dict:
    """Run Acne-LDS in-process on the same CUDA runtime as YOLO.

    ``python_path`` and ``timeout_seconds`` remain in the signature for Worker
    configuration compatibility; production no longer starts a CPU subprocess.
    """
    del python_path, timeout_seconds
    started = time.perf_counter()
    image_path = Path(image_path)
    output_json = Path(output_json)
    checkpoint = Path(checkpoint_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved = str(require_cuda(device))
        if not checkpoint.exists():
            raise FileNotFoundError(f"grading checkpoint missing: {checkpoint}")
        model = _load_model(checkpoint, "model_ld_smoothing", resolved)
        image = Image.open(image_path).convert("RGB")
        cls, cou, cou2cls = model.predict_on_img(image)
        probabilities = (0.5 * (cls + cou2cls)).detach().cpu()[0]
        probabilities = probabilities / probabilities.sum()
        payload = {
            "status": "ok",
            "predicted_count": float(torch.argmax(cou.detach().cpu()[0]).item() + 1),
            "severity_level": int(torch.argmax(probabilities).item() + 1),
            "severity_probabilities": [float(value) for value in probabilities.tolist()],
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": "acne-lds",
            "model_type": "model_ld_smoothing",
            "device": resolved,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "cuda": cuda_metadata(resolved),
        }
    except Exception as exc:
        payload = {
            "status": "unavailable",
            "reason": type(exc).__name__,
            "message": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": "acne-lds",
            "device": str(device),
        }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
