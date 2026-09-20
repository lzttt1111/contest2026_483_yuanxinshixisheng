from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import threading

from src.acne.gpu_runtime import cuda_metadata, require_cuda


_YOLO_MODELS: dict[str, Any] = {}
_YOLO_MODEL_LOCK = threading.Lock()


class DetectorUnavailableError(RuntimeError):
    """检测器后端或权重无法加载时抛出。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_detector_runtime(
    model_path: str | Path,
    device: str = "cpu",
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise DetectorUnavailableError(f"detector checkpoint missing: {path}")

    try:
        import torch
        from ultralytics import YOLO  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DetectorUnavailableError(f"detector dependencies unavailable: {exc}") from exc

    if expected_sha256:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise DetectorUnavailableError(
                f"detector checkpoint sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )

    try:
        resolved = require_cuda(device)
    except RuntimeError as exc:
        raise DetectorUnavailableError(str(exc)) from exc

    return {
        "model_path": str(path),
        "device": str(resolved),
        "expected_sha256": expected_sha256,
        "cuda": cuda_metadata(str(resolved)),
    }


def _load_yolo_once(model_path: Path):
    from ultralytics import YOLO

    key = str(model_path.resolve())
    with _YOLO_MODEL_LOCK:
        model = _YOLO_MODELS.get(key)
        if model is None:
            model = YOLO(str(model_path))
            _YOLO_MODELS[key] = model
        return model


class AcneDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
        max_det: int = 300,
    ) -> None:
        self.model_path = Path(model_path)
        runtime_info = validate_detector_runtime(self.model_path, device=device)
        self.model = _load_yolo_once(self.model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = str(runtime_info["device"])
        self.max_det = max_det
        self.runtime_info = runtime_info

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        start = time.perf_counter()
        height, width = image_bgr.shape[:2]
        results = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].detach().cpu().tolist()
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    x1 = max(0.0, min(float(width), x1))
                    x2 = max(0.0, min(float(width), x2))
                    y1 = max(0.0, min(float(height), y1))
                    y2 = max(0.0, min(float(height), y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    class_id = int(box.cls[0].detach().cpu()) if box.cls is not None else 0
                    if class_id != 0:
                        continue
                    confidence = float(box.conf[0].detach().cpu()) if box.conf is not None else 0.0
                    detections.append(
                        {
                            "bbox_xyxy": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                            "confidence": confidence,
                            "class_id": 0,
                            "label": "acne_candidate",
                            "label_zh": "疑似痤疮病灶",
                            "source": "yolo",
                        }
                    )
        return {
            "status": "ok",
            "model_path": str(self.model_path),
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "detections": detections,
            "count": len(detections),
            "empty_detections_are_valid": True,
        }
