"""Image preprocessing shared by the Python tools.

FSA-Net was exported from an OpenCV based demo, so channel order intentionally
remains BGR. The model expects NCHW float32 values normalized to roughly
[-1, 1] with ``(pixel - 127.5) / 128``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

MODEL_HEIGHT = 64
MODEL_WIDTH = 64


@dataclass(frozen=True)
class NormalizedCrop:
    """A crop rectangle expressed as fractions of image width and height."""

    x: float
    y: float
    width: float
    height: float

    def validate(self) -> "NormalizedCrop":
        values = (self.x, self.y, self.width, self.height)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("crop values must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("crop width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("crop x and y must be non-negative")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("crop rectangle must stay inside the image")
        return self


def read_image_bgr(path: Union[str, Path]) -> np.ndarray:
    """Read an image through imdecode so Unicode Windows paths work reliably."""

    image_path = Path(path)
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode image: {image_path}")
    return image


def center_square_box(
    image_width: int,
    image_height: int,
    vertical_center: float = 0.46,
) -> Tuple[int, int, int, int]:
    """Return a top-biased square suitable for already centered portrait faces."""

    side = min(image_width, image_height)
    x1 = max(0, (image_width - side) // 2)
    center_y = int(round(image_height * vertical_center))
    y1 = max(0, min(image_height - side, center_y - side // 2))
    return x1, y1, x1 + side, y1 + side


def _crop_pixels(image_bgr: np.ndarray, crop: NormalizedCrop) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    crop.validate()
    x1 = int(round(crop.x * width))
    y1 = int(round(crop.y * height))
    x2 = int(round((crop.x + crop.width) * width))
    y2 = int(round((crop.y + crop.height) * height))
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return image_bgr[y1:y2, x1:x2]


def preprocess_bgr(
    image_bgr: np.ndarray,
    crop: Optional[NormalizedCrop] = None,
    center_square: bool = False,
) -> np.ndarray:
    """Convert a BGR image to the model's ``[1, 3, 64, 64]`` input tensor."""

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape [height, width, 3]")
    if image_bgr.size == 0:
        raise ValueError("image_bgr must not be empty")
    if crop is not None and center_square:
        raise ValueError("crop and center_square are mutually exclusive")

    working = image_bgr
    if crop is not None:
        working = _crop_pixels(working, crop)
    elif center_square:
        x1, y1, x2, y2 = center_square_box(
            image_width=working.shape[1], image_height=working.shape[0]
        )
        working = working[y1:y2, x1:x2]

    resized = cv2.resize(
        working,
        (MODEL_WIDTH, MODEL_HEIGHT),
        # Matches the upstream OpenCV demo and the mini-program bilinear path.
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = (resized.astype(np.float32) - 127.5) / 128.0
    chw = normalized.transpose(2, 0, 1)
    return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)
