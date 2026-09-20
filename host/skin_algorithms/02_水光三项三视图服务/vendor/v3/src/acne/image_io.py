from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageReadError(ValueError):
    """Raised when an image cannot be decoded."""


class ImageWriteError(ValueError):
    """Raised when an image cannot be encoded or written."""


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image through numpy buffers so UTF-8 paths work on Linux/WSL."""
    image_path = Path(path)
    if not image_path.exists():
        raise ImageReadError(f"image does not exist: {image_path}")
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise ImageReadError(f"image file is empty: {image_path}")
    # IMREAD_COLOR applies JPEG EXIF orientation while still normalizing every
    # supported source into the BGR three-channel contract used downstream.
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(f"failed to decode image: {image_path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageReadError(f"unsupported image shape {image.shape}: {image_path}")
    return image


def write_image(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> None:
    """Write an image through numpy buffers so UTF-8 paths work on Linux/WSL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if not suffix:
        raise ImageWriteError(f"output path has no suffix: {out_path}")
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise ImageWriteError(f"failed to encode image as {suffix}: {out_path}")
    encoded.tofile(str(out_path))


def collect_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in VALID_IMAGE_EXTS else []
    if not root.exists():
        return []
    return sorted(
        item
        for item in root.iterdir()
        if item.is_file()
        and item.suffix.lower() in VALID_IMAGE_EXTS
        and ":Zone.Identifier" not in item.name
    )
