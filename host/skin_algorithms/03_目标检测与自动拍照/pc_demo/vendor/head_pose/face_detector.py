"""Lightweight YuNet face detection and stable pose-model cropping."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np

DEFAULT_FACE_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "miniprogram_demo"
    / "models"
    / "face_detection_yunet.onnx"
)


@dataclass(frozen=True)
class FaceDetection:
    x: float
    y: float
    width: float
    height: float
    score: float
    landmarks: tuple[tuple[float, float], ...]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def to_dict(self) -> dict:
        return {
            "x": float(round(self.x, 2)),
            "y": float(round(self.y, 2)),
            "width": float(round(self.width, 2)),
            "height": float(round(self.height, 2)),
            "score": float(round(self.score, 4)),
            "landmarks": [
                {"x": float(round(x, 2)), "y": float(round(y, 2))}
                for x, y in self.landmarks
            ],
        }


class YuNetFaceDetector:
    """Detect the primary face with a fixed 640x640 YuNet ONNX model."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_FACE_MODEL_PATH,
        input_size: int = 640,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        crop_scale: float = 1.35,
        upward_shift: float = 0.04,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        self.crop_scale = float(crop_scale)
        self.upward_shift = float(upward_shift)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"face detector model not found: {self.model_path}")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0 < self.score_threshold < 1:
            raise ValueError("score_threshold must be between 0 and 1")
        if self.crop_scale < 1:
            raise ValueError("crop_scale must be at least 1")

        # The buffer overload avoids OpenCV path decoding failures on Chinese paths.
        model_buffer = np.fromfile(self.model_path, dtype=np.uint8)
        self._detector = cv2.FaceDetectorYN.create(
            "onnx",
            model_buffer,
            np.empty(0, dtype=np.uint8),
            (self.input_size, self.input_size),
            self.score_threshold,
            float(nms_threshold),
            int(top_k),
        )
        self._lock = threading.Lock()

    def _letterbox(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        height, width = image_bgr.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _map_point(
        x: float,
        y: float,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> tuple[float, float]:
        return float((x - pad_x) / scale), float((y - pad_y) / scale)

    def detect_all(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy array")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
            raise ValueError("image_bgr must have shape [height, width, 3]")

        frame_height, frame_width = image_bgr.shape[:2]
        model_input, scale, pad_x, pad_y = self._letterbox(image_bgr)
        with self._lock:
            _retval, faces = self._detector.detect(model_input)
        if faces is None:
            return []

        detections = []
        for row in faces:
            x, y = self._map_point(row[0], row[1], scale, pad_x, pad_y)
            width = float(row[2]) / scale
            height = float(row[3]) / scale
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(frame_width), x + width)
            y2 = min(float(frame_height), y + height)
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = tuple(
                self._map_point(row[index], row[index + 1], scale, pad_x, pad_y)
                for index in range(4, 14, 2)
            )
            detections.append(
                FaceDetection(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    score=float(row[14]),
                    landmarks=landmarks,
                )
            )
        return detections

    @staticmethod
    def select_primary(
        detections: Sequence[FaceDetection], frame_width: int, frame_height: int
    ) -> FaceDetection | None:
        if not detections:
            return None
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        half_diagonal = max(1.0, math.hypot(frame_width, frame_height) / 2)

        def priority(face: FaceDetection) -> float:
            center_x, center_y = face.center
            distance = math.hypot(
                center_x - frame_center_x, center_y - frame_center_y
            ) / half_diagonal
            center_weight = 1.25 - 0.25 * min(distance, 1.0)
            return face.area * face.score * center_weight

        return max(detections, key=priority)

    def detect_primary(self, image_bgr: np.ndarray) -> FaceDetection | None:
        height, width = image_bgr.shape[:2]
        return self.select_primary(self.detect_all(image_bgr), width, height)

    def crop_box(
        self, face: FaceDetection, frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int]:
        side = min(
            max(face.width, face.height) * self.crop_scale,
            float(frame_width),
            float(frame_height),
        )
        center_x, center_y = face.center
        center_y -= side * self.upward_shift
        x1 = int(round(center_x - side / 2))
        y1 = int(round(center_y - side / 2))
        side_int = max(1, int(round(side)))
        x1 = min(max(x1, 0), max(0, frame_width - side_int))
        y1 = min(max(y1, 0), max(0, frame_height - side_int))
        x2 = min(frame_width, x1 + side_int)
        y2 = min(frame_height, y1 + side_int)
        return x1, y1, x2, y2

    def detect_and_crop(
        self, image_bgr: np.ndarray
    ) -> tuple[FaceDetection, tuple[int, int, int, int], np.ndarray] | None:
        face = self.detect_primary(image_bgr)
        if face is None:
            return None
        height, width = image_bgr.shape[:2]
        crop_box = self.crop_box(face, width, height)
        x1, y1, x2, y2 = crop_box
        return face, crop_box, image_bgr[y1:y2, x1:x2]
