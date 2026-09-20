"""Tracked-file, tree, and image evidence."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from scripts.acceptance.acceptance_types import JsonDict, JsonValue
from scripts.acceptance.deep_compare_pairing import pair_file_trees


IMAGE_SUFFIXES: Final = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PERCEPTUAL_DUPLICATE_DISTANCE: Final = 2
NEAR_DUPLICATE_ROLE_ALLOWLIST: Final = {
    "01_红区",
    "03_棕区",
    "06_UV色斑",
    "07_卟啉",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_tree(root: Path) -> list[JsonValue]:
    if not root.is_dir():
        return []
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def duplicate_file_groups(
    root: Path,
    suffixes: set[str],
    required_part: str | None = None,
) -> list[JsonValue]:
    images: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if not (
            path.is_file()
            and path.suffix.lower() in suffixes
            and (required_part is None or required_part in path.parts)
        ):
            continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            pixel_digest = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
            ).hexdigest()
            phash = _phash(rgb)
        images.append((path.relative_to(root).as_posix(), pixel_digest, phash))

    exact: dict[str, list[tuple[str, str]]] = {}
    for relative, pixel_digest, phash in images:
        exact.setdefault(pixel_digest, []).append((relative, phash))
    findings: list[JsonValue] = [
        {
            "match_kind": "pixel_exact",
            "paths": [relative for relative, _ in group],
            "phash_distance": 0,
        }
        for _, group in sorted(exact.items())
        if len(group) > 1
    ]
    exact_paths = {
        relative
        for group in exact.values()
        if len(group) > 1
        for relative, _ in group
    }
    for index, (left_path, _, left_hash) in enumerate(images):
        for right_path, _, right_hash in images[index + 1:]:
            if left_path in exact_paths and right_path in exact_paths:
                continue
            distance = (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()
            if distance > PERCEPTUAL_DUPLICATE_DISTANCE:
                continue
            shared_parts = set(Path(left_path).parts) & set(Path(right_path).parts)
            if shared_parts & NEAR_DUPLICATE_ROLE_ALLOWLIST:
                continue
            findings.append(
                {
                    "match_kind": "perceptual_near",
                    "paths": [left_path, right_path],
                    "phash_distance": distance,
                }
            )
    return findings


def tracked_files(root: Path) -> JsonDict:
    if not (root / ".git").exists():
        return {"status": "unavailable", "files": []}
    result = subprocess.run(
        ("git", "-C", str(root), "ls-files", "-z"),
        check=True,
        capture_output=True,
        text=False,
    )
    paths = sorted(
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    )
    files: list[JsonValue] = []
    for relative in paths:
        path = root / relative
        files.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {"status": "available", "files": files}


def compare_tracked_files(baseline_root: Path, merged_root: Path) -> JsonDict:
    baseline = tracked_files(baseline_root)
    merged = tracked_files(merged_root)
    baseline_items = {
        str(item["path"]): item
        for item in baseline["files"]
        if isinstance(item, dict)
    }
    merged_items = {
        str(item["path"]): item
        for item in merged["files"]
        if isinstance(item, dict)
    }
    all_paths = sorted(set(baseline_items) | set(merged_items))
    rows: list[JsonValue] = []
    for path in all_paths:
        baseline_item = baseline_items.get(path)
        merged_item = merged_items.get(path)
        rows.append(
            {
                "path": path,
                "status": (
                    "added"
                    if baseline_item is None
                    else "removed"
                    if merged_item is None
                    else "same"
                    if baseline_item.get("sha256") == merged_item.get("sha256")
                    else "changed"
                ),
                "baseline_sha256": baseline_item.get("sha256") if baseline_item else None,
                "merged_sha256": merged_item.get("sha256") if merged_item else None,
            }
        )
    return {"baseline": baseline, "merged": merged, "comparison": rows}


def _phash(image: Image.Image) -> str:
    grayscale = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    indices = np.arange(32, dtype=np.float64)
    transform = np.cos((2 * indices[:, None] + 1) * indices[None, :] * np.pi / 64)
    coefficients = transform.T @ grayscale @ transform
    low_frequency = coefficients[:8, :8]
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = (low_frequency >= median).flatten()
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def compare_images(baseline_path: Path, merged_path: Path) -> JsonDict:
    with Image.open(baseline_path) as baseline_image, Image.open(merged_path) as merged_image:
        baseline_rgb = np.asarray(baseline_image.convert("RGB"), dtype=np.int16)
        merged_rgb = np.asarray(merged_image.convert("RGB"), dtype=np.int16)
        same_dimensions = baseline_rgb.shape == merged_rgb.shape
        if same_dimensions:
            absolute = np.abs(baseline_rgb - merged_rgb)
            changed_pixels = int(np.count_nonzero(np.any(absolute != 0, axis=2)))
            mean_absolute = float(absolute.mean())
            maximum_absolute = int(absolute.max())
        else:
            changed_pixels = None
            mean_absolute = None
            maximum_absolute = None
        return {
            "baseline_dimensions": list(baseline_image.size),
            "merged_dimensions": list(merged_image.size),
            "baseline_sha256": sha256_file(baseline_path),
            "merged_sha256": sha256_file(merged_path),
            "baseline_phash": _phash(baseline_image),
            "merged_phash": _phash(merged_image),
            "pixel_diff": {
                "same_dimensions": same_dimensions,
                "changed_pixels": changed_pixels,
                "mean_absolute_channel_delta": mean_absolute,
                "max_absolute_channel_delta": maximum_absolute,
            },
        }


def compare_image_trees(baseline_root: Path, merged_root: Path) -> JsonDict:
    pairing = pair_file_trees(baseline_root, merged_root, IMAGE_SUFFIXES)
    comparisons: list[JsonValue] = [
        {
            "baseline_path": pair.baseline.relative_to(baseline_root).as_posix(),
            "merged_path": pair.merged.relative_to(merged_root).as_posix(),
            "match_kind": pair.match_kind,
            **compare_images(pair.baseline, pair.merged),
        }
        for pair in pairing.pairs
    ]
    return {
        "comparisons": comparisons,
        "unmatched_baseline": list(pairing.unmatched_baseline),
        "unmatched_merged": list(pairing.unmatched_merged),
        "ambiguous": list(pairing.ambiguous),
    }
