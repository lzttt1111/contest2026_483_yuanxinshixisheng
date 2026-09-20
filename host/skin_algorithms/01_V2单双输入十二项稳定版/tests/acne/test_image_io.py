from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.acne.image_io import read_image_bgr


def test_read_image_bgr_applies_exif_orientation(tmp_path: Path) -> None:
    source = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (20, 30), (120, 80, 40))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)

    decoded = read_image_bgr(source)

    assert decoded.shape == (20, 30, 3)
