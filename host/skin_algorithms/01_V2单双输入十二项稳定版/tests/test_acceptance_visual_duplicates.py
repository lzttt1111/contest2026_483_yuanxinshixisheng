from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from scripts.acceptance.deep_compare_inventory import duplicate_file_groups


def test_duplicate_gate_detects_pixel_identical_images_with_different_metadata(
    tmp_path: Path,
) -> None:
    # Given
    wrinkle = tmp_path / "十二项检测" / "08_皱纹"
    wrinkle.mkdir(parents=True)
    image = Image.new("RGB", (16, 16), color=(20, 40, 60))
    first_metadata = PngInfo()
    first_metadata.add_text("variant", "first")
    second_metadata = PngInfo()
    second_metadata.add_text("variant", "second")
    image.save(wrinkle / "01_main.png", pnginfo=first_metadata)
    image.save(wrinkle / "02_region.png", pnginfo=second_metadata)

    # When
    groups = duplicate_file_groups(tmp_path, {".png"}, "十二项检测")

    # Then
    assert groups == [
        {
            "match_kind": "pixel_exact",
            "paths": [
                "十二项检测/08_皱纹/01_main.png",
                "十二项检测/08_皱纹/02_region.png",
            ],
            "phash_distance": 0,
        }
    ]


def test_duplicate_gate_allows_documented_near_base_overlay_pair(
    tmp_path: Path,
) -> None:
    # Given
    uv = tmp_path / "十二项检测" / "06_UV色斑"
    uv.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(
        uv / "00_紫外线色斑底图.png"
    )
    Image.new("RGB", (16, 16), color=(21, 41, 61)).save(
        uv / "01_UV色斑检测结果图.png"
    )

    # When / Then
    assert duplicate_file_groups(tmp_path, {".png"}, "十二项检测") == []
