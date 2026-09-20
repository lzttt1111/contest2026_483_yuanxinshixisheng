# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Render small consumer pigment variants from the existing formal engines."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.capture_profile import CaptureProfile
from src.consumer_pigment.brown_render import (
    BrownRenderInput,
    ConsumerBrownPalette,
    render_consumer_brown_base,
)
from src.consumer_pigment.comparison_export import (
    VariantImage,
    marker_result_document,
    render_contact_sheet,
    sha256_array,
    sha256_file,
    write_html_index,
    write_image,
)
from src.consumer_pigment.markers import (
    CompactMarkerStyle,
    consolidate_existing_marker_mask,
    render_compact_marker_mask,
)
from src.consumer_pigment.purple_render import (
    ConsumerPurplePalette,
    PurpleRenderInput,
    render_consumer_purple_palette,
)
from src.consumer_pigment.specular import build_specular_evidence
from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.purple_analysis_engine import PurpleAnalysisEngine
from src.engines.visia_regions import build_visia_regions, draw_region_boundaries
from src.preprocess.image_preprocessor import ImagePreprocessor


@dataclass(frozen=True, slots=True)
class CliArgs:
    input_dir: Path
    output_dir: Path
    samples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComparisonInputError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Render consumer pigment variants from formal mother images"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True)
    namespace = parser.parse_args()
    return CliArgs(namespace.input_dir, namespace.output_dir, tuple(namespace.sample))


def _with_boundaries(image: np.ndarray, regions) -> np.ndarray:
    return draw_region_boundaries(
        image,
        regions.display_regions,
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )

def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise ComparisonInputError(
            f"refuse existing output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    git_sha = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    source_root = Path(__file__).resolve().parents[2]
    config_paths = tuple(
        source_root / "src" / "consumer_pigment" / name
        for name in ("brown_render.py", "purple_render.py", "markers.py", "specular.py")
    )
    config_sha = "-".join(sha256_file(path)[:12] for path in config_paths)
    styles = {
        "brown": CompactMarkerStyle((190, 220, 55), (110, 150, 25), 0.90),
        "uv": CompactMarkerStyle((36, 190, 238), (20, 125, 175), 0.90),
        "porphyrin": CompactMarkerStyle((45, 126, 245), (25, 75, 170), 0.90),
    }
    preprocessor = ImagePreprocessor(capture_profile=CaptureProfile.CONSUMER)
    brown_analyzer = BrownAreaAnalyzer()
    purple_analyzer = PurpleAnalysisEngine()
    sample_names: list[str] = []
    root_manifest: dict[str, object] = {
        "schema": "consumer_pigment_mother_image_review_v2",
        "git_sha": git_sha,
        "config_sha": config_sha,
        "samples": [],
    }
    try:
        for sample_index, filename in enumerate(args.samples, 1):
            source = args.input_dir / filename
            image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ComparisonInputError(f"cannot decode input: {source}")
            preprocess = preprocessor.preprocess_image(image)
            regions = build_visia_regions(
                preprocess.analysis_image,
                preprocess.skin_mask,
                preprocess.landmarks,
                preprocess.quality_flags,
                include_chin=True,
                mode="full",
                feature_margin_px=30,
            )
            brown = brown_analyzer.detect_brown(preprocess)
            purple = purple_analyzer.analyze(preprocess)
            specular = build_specular_evidence(
                preprocess.analysis_image,
                regions.analysis_mask,
            )
            shared_exclusion = cv2.bitwise_or(
                regions.feature_exclusion_mask,
                specular.mask,
            )
            brown_markers = consolidate_existing_marker_mask(
                brown.instance_mask,
                brown.brown_score_map,
                regions.analysis_mask,
                shared_exclusion,
                minimum_peak_score=0.78,
                minimum_distance_px=12,
            )
            uv_markers = consolidate_existing_marker_mask(
                purple.uv_spots_mask,
                purple.uv_spots_score_map,
                purple.analysis_mask,
                cv2.bitwise_or(purple.feature_exclusion_mask, specular.mask),
                minimum_peak_score=0.90,
                minimum_distance_px=10,
            )
            porphyrin_markers = consolidate_existing_marker_mask(
                purple.porphyrin_mask,
                purple.porphyrin_score_map,
                purple.analysis_mask,
                cv2.bitwise_or(purple.feature_exclusion_mask, specular.mask),
                minimum_peak_score=0.86,
                minimum_distance_px=10,
            )
            sample_name = source.stem
            sample_names.append(sample_name)
            sample_dir = args.output_dir / sample_name
            sample_dir.mkdir()
            write_image(sample_dir / "input.jpg", image)
            write_image(sample_dir / "brown_mother.jpg", brown.brown_rbx_image)
            write_image(sample_dir / "uv_mother.png", purple.uv_like_base)
            write_image(sample_dir / "porphyrin_mother.png", purple.porphyrin_base)
            brown_bases = {
                palette: render_consumer_brown_base(
                    BrownRenderInput(
                        preprocess.analysis_image,
                        brown.brown_rbx_image,
                        regions.analysis_mask,
                    ),
                    palette,
                )
                for palette in ConsumerBrownPalette
            }
            purple_bases = {
                palette: render_consumer_purple_palette(
                    PurpleRenderInput(purple.uv_like_base, purple.porphyrin_base),
                    palette,
                )
                for palette in ConsumerPurplePalette
            }
            contacts: dict[str, list[VariantImage]] = {
                "brown": [],
                "uv": [],
                "porphyrin": [],
            }
            for palette, base in brown_bases.items():
                overlay = render_compact_marker_mask(
                    base,
                    brown_markers.marker_mask,
                    styles["brown"],
                )
                overlay = _with_boundaries(overlay, regions)
                write_image(sample_dir / "brown" / f"{palette.value}.jpg", overlay)
                contacts["brown"].append(VariantImage(palette.value.upper(), overlay))
            for palette, bases in purple_bases.items():
                for kind, base, marker in (
                    ("uv", bases.uv_base, uv_markers),
                    ("porphyrin", bases.porphyrin_base, porphyrin_markers),
                ):
                    overlay = render_compact_marker_mask(
                        base,
                        marker.marker_mask,
                        styles[kind],
                    )
                    overlay = _with_boundaries(overlay, regions)
                    write_image(sample_dir / kind / f"{palette.value}.jpg", overlay)
                    contacts[kind].append(VariantImage(palette.value.upper(), overlay))
            display_label = f"SAMPLE-{sample_index:02d}"
            for kind, variants in contacts.items():
                write_image(
                    sample_dir / f"{kind}_comparison.jpg",
                    render_contact_sheet(tuple(variants), f"{display_label} {kind}"),
                )
            manifest = {
                "sample": sample_name,
                "input_sha256": sha256_file(source),
                "quality_status": preprocess.quality_status,
                "quality_flags": preprocess.quality_flags,
                "mother_sha256": {
                    "brown": sha256_array(brown.brown_rbx_image),
                    "uv": sha256_array(purple.uv_like_base),
                    "porphyrin": sha256_array(purple.porphyrin_base),
                },
                "original_counts": {
                    "brown": brown.brown_spot_count,
                    "uv": purple.uv_spots_count,
                    "porphyrin": purple.porphyrin_count,
                },
                "optimized_markers": {
                    "brown": marker_result_document(brown_markers),
                    "uv": marker_result_document(uv_markers),
                    "porphyrin": marker_result_document(porphyrin_markers),
                },
            }
            (sample_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            samples = root_manifest["samples"]
            assert isinstance(samples, list)
            samples.append({"sample": sample_name, "input_sha256": manifest["input_sha256"]})
    finally:
        purple_analyzer.close()
        brown_analyzer.close()
        preprocessor.close()
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html_index(args.output_dir, tuple(sample_names))
    (args.output_dir / "README.md").write_text(
        "# Consumer pigment mother-image review\n\n"
        "All variants reuse the existing formal Brown/UV/Porphyrin mother images.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
