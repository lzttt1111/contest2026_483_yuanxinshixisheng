from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from .preprocessing import contact_tensor, device_tensor, expand_box, normalize_orientation
from .version import ARCHITECTURE_GENERATION, MODEL_LINEAGE, RELEASE_STATUS, SOFTWARE_VERSION
VISIBILITY_CLASSES = ("visible", "occluded_inferable", "not_annotatable")


def _preload_cuda_libraries() -> None:
    """Let ONNX Runtime discover CUDA/cuDNN wheels installed in this environment."""
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload) and "CUDAExecutionProvider" in ort.get_available_providers():
        preload(cuda=True, cudnn=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_build_commit(project_root: Path) -> str:
    configured = os.environ.get("TARGET_DETECTION_BUILD_COMMIT")
    if configured:
        return configured
    release_manifest = project_root / "release_manifest.json"
    if release_manifest.is_file():
        try:
            value = json.loads(release_manifest.read_text(encoding="utf-8")).get("build_commit")
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "uncommitted"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


def _box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    left_top = np.maximum(one[:2], many[:, :2])
    right_bottom = np.minimum(one[2:], many[:, 2:])
    extent = np.maximum(0.0, right_bottom - left_top)
    intersection = extent[:, 0] * extent[:, 1]
    one_area = max(0.0, float(one[2] - one[0])) * max(0.0, float(one[3] - one[1]))
    many_area = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(0.0, many[:, 3] - many[:, 1])
    return intersection / np.maximum(one_area + many_area - intersection, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[_box_iou(boxes[current], boxes[remaining]) <= threshold]
    return kept


def _decode_device(
    output: np.ndarray,
    ratio: float,
    image_size: tuple[int, int],
    confidence_threshold: float,
    nms_iou: float,
) -> tuple[list[float] | None, float]:
    rows = np.asarray(output).reshape(-1, np.asarray(output).shape[-1])
    if rows.shape[1] < 6:
        raise RuntimeError(f"unexpected detector output shape: {np.asarray(output).shape}")
    scores = rows[:, 4] * np.max(rows[:, 5:], axis=1)
    selected = scores >= confidence_threshold
    if not np.any(selected):
        return None, 0.0
    rows = rows[selected]
    scores = scores[selected]
    xywh = rows[:, :4]
    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
    kept = _nms(boxes, scores, nms_iou)
    if not kept:
        return None, 0.0
    index = kept[0]
    width, height = image_size
    box = boxes[index] / ratio
    box[0::2] = np.clip(box[0::2], 0, width)
    box[1::2] = np.clip(box[1::2], 0, height)
    return [round(float(value), 3) for value in box], round(float(scores[index]), 6)


def _decode_contact(
    heatmap_logits: np.ndarray,
    visibility_logits: np.ndarray,
    roi: tuple[int, int, int, int],
    radius: int = 3,
) -> tuple[list[float], str, float, float]:
    heatmap = np.asarray(heatmap_logits, dtype=np.float64).squeeze()
    if heatmap.ndim != 2:
        raise RuntimeError(f"unexpected contact heatmap shape: {np.asarray(heatmap_logits).shape}")
    height, width = heatmap.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    x1 = max(0, peak_x - radius)
    x2 = min(width, peak_x + radius + 1)
    y1 = max(0, peak_y - radius)
    y2 = min(height, peak_y + radius + 1)
    weights = _softmax(heatmap[y1:y2, x1:x2].reshape(-1)).reshape(y2 - y1, x2 - x1)
    grid_y, grid_x = np.mgrid[y1:y2, x1:x2]
    heatmap_x = float(np.sum(weights * grid_x))
    heatmap_y = float(np.sum(weights * grid_y))
    global_probabilities = _softmax(heatmap.reshape(-1))
    entropy = -float(np.sum(global_probabilities * np.log(np.maximum(global_probabilities, 1e-12))))
    confidence = max(0.0, min(1.0, 1.0 - entropy / math.log(max(global_probabilities.size, 2))))
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    point = [
        round(roi_x1 + heatmap_x / max(width - 1, 1) * (roi_x2 - roi_x1), 3),
        round(roi_y1 + heatmap_y / max(height - 1, 1) * (roi_y2 - roi_y1), 3),
    ]
    probabilities = _softmax(np.asarray(visibility_logits).reshape(-1))
    visibility_index = int(np.argmax(probabilities))
    return point, VISIBILITY_CLASSES[visibility_index], round(float(probabilities[visibility_index]), 6), round(confidence, 6)


class V5Runtime:
    def __init__(self, project_root: Path | None = None) -> None:
        inferred_root = Path(__file__).resolve().parents[2]
        configured_root = os.environ.get("TARGET_DETECTION_ROOT")
        self.project_root = Path(project_root or configured_root or inferred_root).resolve()
        default_model_root = self.project_root / "models"
        self.model_root = Path(os.environ.get("TARGET_DETECTION_MODEL_ROOT", default_model_root)).resolve()
        self.device_path = self.model_root / "device" / "model.onnx"
        self.contact_path = self.model_root / "contact" / "model.onnx"
        self.device_manifest = self._read_manifest("device")
        self.contact_manifest = self._read_manifest("contact")
        self._validate_model(self.device_path, self.device_manifest["sha256"])
        self._validate_model(self.contact_path, self.contact_manifest["sha256"])
        _preload_cuda_libraries()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.device_session = ort.InferenceSession(str(self.device_path), providers=providers)
        self.contact_session = ort.InferenceSession(str(self.contact_path), providers=providers)
        self.device_input = self.device_session.get_inputs()[0].name
        self.contact_input = self.contact_session.get_inputs()[0].name
        self.build_commit = _resolve_build_commit(self.project_root)

    def _read_manifest(self, role: str) -> dict[str, Any]:
        path = self.model_root / role / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _validate_model(path: Path, expected: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"model SHA256 mismatch for {path}: expected {expected}, got {actual}")

    @property
    def runtime_provider(self) -> str:
        device = self.device_session.get_providers()[0]
        contact = self.contact_session.get_providers()[0]
        return device if device == contact else f"device={device};contact={contact}"

    def metadata(self) -> dict[str, Any]:
        return {
            "software_version": SOFTWARE_VERSION,
            "architecture_generation": ARCHITECTURE_GENERATION,
            "model_lineage": MODEL_LINEAGE,
            "release_status": RELEASE_STATUS,
            "runtime_provider": self.runtime_provider,
            "build_commit": self.build_commit,
            "device_model_sha256": self.device_manifest["sha256"],
            "contact_model_sha256": self.contact_manifest["sha256"],
        }

    def infer_bytes(self, payload: bytes, frame_id: int = 0, timestamp_ms: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(payload)) as source:
            image = normalize_orientation(source)
        device_started = time.perf_counter()
        tensor, ratio = device_tensor(image)
        outputs = self.device_session.run(None, {self.device_input: tensor})
        device_bbox, device_score = _decode_device(outputs[0], ratio, image.size, 0.25, 0.65)
        device_ms = (time.perf_counter() - device_started) * 1000.0
        contact_point: list[float] | None = None
        visibility = "not_annotatable"
        visibility_confidence = 0.0
        heatmap_confidence = 0.0
        contact_ms = 0.0
        if device_bbox is not None:
            contact_started = time.perf_counter()
            roi = expand_box(tuple(device_bbox), image.size)
            contact_outputs = self.contact_session.run(None, {self.contact_input: contact_tensor(image, roi)})
            contact_point, visibility, visibility_confidence, heatmap_confidence = _decode_contact(contact_outputs[0], contact_outputs[1], roi)
            if visibility == "not_annotatable":
                contact_point = None
            contact_ms = (time.perf_counter() - contact_started) * 1000.0
        result: dict[str, Any] = {
            **self.metadata(),
            "frame_id": int(frame_id),
            "timestamp_ms": timestamp_ms,
            "image_width": image.width,
            "image_height": image.height,
            "device_bbox": device_bbox,
            "device_score": device_score,
            "contact_point": contact_point,
            "contact_visibility": visibility,
            "contact_confidence": visibility_confidence,
            "contact_heatmap_confidence": heatmap_confidence,
            "contact_region": "unknown",
            "coordinate_space": "exif_normalized_unmirrored_original",
            "timings_ms": {
                "device": round(device_ms, 3),
                "contact": round(contact_ms, 3),
                "total": round((time.perf_counter() - started) * 1000.0, 3),
            },
        }
        return result
