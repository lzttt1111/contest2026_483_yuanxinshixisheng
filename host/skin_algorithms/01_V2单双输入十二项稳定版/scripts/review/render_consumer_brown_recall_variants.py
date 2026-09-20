"""Render current plus four consumer-only Brown recall thresholds."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.capture_profile import CaptureProfile
from src.consumer_pigment.comparison_export import (
    VariantImage,
    render_contact_sheet,
    sha256_file,
    write_image,
)
from src.consumer_pigment.formal_markers import (
    ConsumerBrownRecallPreset,
    policy_for_consumer_brown_recall,
)
from src.consumer_pigment.brown_detection_policy import (
    policy_for_consumer_brown_detection_recall,
)
from src.engines.brown_engine import BrownAreaAnalyzer
from src.preprocess.image_preprocessor import ImagePreprocessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    preprocessor = ImagePreprocessor(capture_profile=CaptureProfile.CONSUMER)
    analyzer = BrownAreaAnalyzer(capture_profile=CaptureProfile.CONSUMER)
    manifest: dict[str, object] = {
        "schema": "consumer_brown_recall_variants_v1",
        "git_sha": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip(),
        "invariants": {
            "capture_profile": "consumer",
            "minimum_peak_score": 0.50,
            "current_minimum_distance_px": 12,
            "recall_minimum_distance_px": 15,
            "institution_unchanged": True,
        },
        "samples": [],
    }
    try:
        for filename in args.sample:
            source = args.input_dir / filename
            image = cv2.imdecode(
                np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError(f"cannot decode {source}")
            preprocess = preprocessor.preprocess_image(image)
            sample_root = args.output_dir / source.stem
            sample_root.mkdir()
            write_image(sample_root / "00_input.jpg", image)
            variants: list[VariantImage] = []
            rows: list[dict[str, object]] = []
            for preset in ConsumerBrownRecallPreset:
                analyzer.consumer_brown_recall_preset = preset
                result = analyzer.detect_brown(preprocess)
                target = sample_root / f"{preset.value}.jpg"
                write_image(target, result.brown_overlay)
                policy = policy_for_consumer_brown_recall(preset)
                detection = policy_for_consumer_brown_detection_recall(preset)
                rows.append({
                    "preset": preset.value,
                    "minimum_local_prominence_z": policy.minimum_local_prominence_z,
                    "minimum_distance_px": policy.minimum_distance_px,
                    "minimum_marker_edge_gap_px": policy.minimum_marker_edge_gap_px,
                    "large_component_prominence_relief": (
                        policy.large_component_prominence_relief
                    ),
                    "scale_z_threshold": detection.scale_z_threshold,
                    "minimum_scale_votes": detection.minimum_scale_votes,
                    "candidate_z_threshold": detection.candidate_z_threshold,
                    "minimum_raw_response": detection.minimum_raw_response,
                    "count": result.brown_spot_count,
                    "area_ratio": result.brown_spot_area_ratio,
                    "image": target.name,
                    "sha256": sha256_file(target),
                })
                variants.append(VariantImage(
                    f"{preset.value} | n={result.brown_spot_count}",
                    result.brown_overlay,
                ))
            write_image(
                sample_root / "comparison.jpg",
                render_contact_sheet(tuple(variants), source.stem),
            )
            (sample_root / "counts.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            samples = manifest["samples"]
            assert isinstance(samples, list)
            samples.append({
                "sample": source.stem,
                "input_sha256": sha256_file(source),
                "variants": rows,
            })
    finally:
        analyzer.close()
        preprocessor.close()
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
