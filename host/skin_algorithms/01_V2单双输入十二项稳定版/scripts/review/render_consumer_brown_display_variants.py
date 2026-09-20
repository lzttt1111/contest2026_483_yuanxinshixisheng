#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

# ─── How to run ───
# PYTHONPATH=. ../../.venv/bin/python -B \
#   scripts/review/render_consumer_brown_display_variants.py <output-dir>
# ──────────────────

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.capture_profile import CaptureProfile
from src.consumer_pigment.brown_tone import (
    BrownToneInput,
    ConsumerBrownTonePreset,
    render_brown_tone,
)
from src.consumer_pigment.comparison_export import (
    VariantImage,
    marker_result_document,
    render_contact_sheet,
    sha256_array,
    sha256_file,
    write_image,
)
from src.consumer_pigment.markers import consolidate_existing_marker_mask
from src.consumer_pigment.specular import (
    build_source_and_render_specular_evidence,
)
from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.vendor_skin_overlay import render_vendor_brown_result
from src.engines.visia_regions import build_visia_regions
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.profile_strategy import strategy_for_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT.parents[1] / "data/test_images"
SAMPLES = (
    "22.png",
    "1000005_1.jpg",
    "visia示例图2.png",
    "1000043_1.jpg",
    "1000059_1.jpg",
    "1000065_1.jpg",
    "visia示例图1.png",
    "侧脸1.png",
    "侧脸2.png",
    "测试.png",
)


@dataclass(frozen=True, slots=True)
class ReviewInputError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ReviewInputError(f"cannot decode image: {path}")
    return image


def display_foreground(base: np.ndarray) -> np.ndarray:
    seed = (np.min(base, axis=2) < 248).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    output = np.zeros(seed.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(output, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return output


def configuration_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "src/consumer_pigment/brown_tone.py",
        root / "src/consumer_pigment/markers.py",
        root / "src/consumer_pigment/specular.py",
        root / "src/engines/brown_engine.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def render_review(output_dir: Path) -> None:
    if output_dir.exists():
        raise ReviewInputError(f"refuse existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    git_sha = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    config_sha = configuration_sha256()
    preprocessor = ImagePreprocessor(capture_profile=CaptureProfile.CONSUMER)
    analyzer = BrownAreaAnalyzer()
    strategy = strategy_for_profile(CaptureProfile.CONSUMER)
    documents = []
    try:
        for index, filename in enumerate(SAMPLES, 1):
            source = INPUT_ROOT / filename
            print(f"START {index:02d}/10 {filename}", flush=True)
            image = read_image(source)
            preprocess = preprocessor.preprocess_image(image)
            result = analyzer.detect_brown(preprocess)
            regions = build_visia_regions(
                preprocess.analysis_image,
                preprocess.skin_mask,
                preprocess.landmarks,
                preprocess.quality_flags,
                include_chin=True,
                mode="full",
                feature_margin_px=30,
            )
            with tempfile.TemporaryDirectory(prefix="brown-display-review-") as scratch_raw:
                scratch = Path(scratch_raw)
                base_path = scratch / "base.jpg"
                overlay_path = scratch / "overlay.jpg"
                write_image(base_path, result.brown_rbx_image)
                write_image(overlay_path, result.brown_overlay)
                bundle = strategy.build(preprocess, image, scratch / "vendor", None)
                strategy.finalize_brown(
                    bundle,
                    str(base_path),
                    str(overlay_path),
                    result.instance_mask,
                    render_vendor_brown_result,
                )
                formal_base = read_image(base_path)
                formal_overlay = read_image(overlay_path)
            specular = build_source_and_render_specular_evidence(
                preprocess.analysis_image,
                formal_base,
                regions.analysis_mask,
            )
            exclusion = cv2.bitwise_or(
                specular.mask,
                regions.feature_exclusion_mask,
            )
            marker_result = consolidate_existing_marker_mask(
                result.instance_mask,
                result.brown_score_map,
                regions.analysis_mask,
                exclusion,
                minimum_peak_score=0.78,
                minimum_distance_px=12,
            )
            foreground = display_foreground(formal_base)
            protected_highlight = np.maximum(
                specular.alpha,
                (specular.mask > 0).astype(np.float32),
            )
            render_input = BrownToneInput(
                formal_base,
                foreground,
                protected_highlight,
            )
            variants = [VariantImage("CURRENT", formal_overlay)]
            sample_dir = output_dir / "full" / source.stem
            write_image(sample_dir / "00_current.jpg", formal_overlay)
            output_hashes = {"current": sha256_array(formal_overlay)}
            for variant_index, preset in enumerate(ConsumerBrownTonePreset, 1):
                tone = render_brown_tone(render_input, preset)
                overlay = render_vendor_brown_result(
                    preprocess,
                    tone,
                    marker_result.marker_mask,
                )
                write_image(
                    sample_dir / f"0{variant_index}_{preset.value}.jpg",
                    overlay,
                )
                variants.append(VariantImage(preset.value.upper(), overlay))
                output_hashes[preset.value] = sha256_array(overlay)
            write_image(sample_dir / "04_marker_mask.png", marker_result.marker_mask)
            write_image(sample_dir / "05_highlight_mask.png", specular.mask)
            contact_path = output_dir / "contacts" / f"{index:02d}_{source.stem}.jpg"
            write_image(
                contact_path,
                render_contact_sheet(
                    tuple(variants),
                    f"SAMPLE-{index:02d} CONSUMER BROWN DISPLAY",
                ),
            )
            documents.append(
                {
                    "sample": source.stem,
                    "input_sha256": sha256_file(source),
                    "quality_status": preprocess.quality_status,
                    "quality_flags": list(preprocess.quality_flags),
                    "original_count": result.brown_spot_count,
                    "optimized_markers": marker_result_document(marker_result),
                    "highlight_pixels": int(np.count_nonzero(specular.mask)),
                    "marker_mask_sha256": sha256_array(marker_result.marker_mask),
                    "outputs": output_hashes,
                    "contact_sha256": sha256_file(contact_path),
                }
            )
            print(
                f"DONE {index:02d}/10 {source.stem} markers={len(marker_result.findings)}",
                flush=True,
            )
    finally:
        analyzer.close()
        preprocessor.close()
    manifest = {
        "schema": "consumer_brown_display_review_v1",
        "git_sha": git_sha,
        "config_sha256": config_sha,
        "sample_count": len(documents),
        "samples": documents,
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise ReviewInputError("expected one output directory")
    render_review(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
