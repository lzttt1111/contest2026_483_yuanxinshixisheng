from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def normalize_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation once and define the unmirrored coordinate space."""
    return ImageOps.exif_transpose(image).convert("RGB")


def device_tensor(image: Image.Image, size: int = 640) -> tuple[np.ndarray, float]:
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = max(1, int(width * ratio))
    resized_height = max(1, int(height * ratio))
    resized = cv2.resize(bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return tensor[None, ...], ratio


def expand_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    ratio: float = 0.20,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    image_width, image_height = image_size
    return (
        max(0, int(math.floor(x1 - width * ratio))),
        max(0, int(math.floor(y1 - height * ratio))),
        min(image_width, int(math.ceil(x2 + width * ratio))),
        min(image_height, int(math.ceil(y2 + height * ratio))),
    )


def contact_tensor(image: Image.Image, roi: tuple[int, int, int, int]) -> np.ndarray:
    crop = np.asarray(image.crop(roi).resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    normalized = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...], dtype=np.float32)
