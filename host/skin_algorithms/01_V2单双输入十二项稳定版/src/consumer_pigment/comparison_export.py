"""Image and static-report export helpers for pigmentation preset review."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.consumer_pigment.markers import ExistingMarkerResult


@dataclass(frozen=True, slots=True)
class VariantImage:
    label: str
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class ComparisonExportError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".jpg"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 96] if extension in {".jpg", ".jpeg"} else []
    ok, payload = cv2.imencode(extension, np.asarray(image), parameters)
    if not ok:
        raise ComparisonExportError(f"failed to encode comparison image: {path.name}")
    payload.tofile(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def marker_result_document(result: ExistingMarkerResult) -> dict[str, object]:
    return {
        "displayed_count": len(result.findings),
        "rejected_low_score": result.rejected_low_score,
        "rejected_exclusion": result.rejected_exclusion,
        "rejected_nms": result.rejected_nms,
        "findings": [
            {
                "centroid": list(item.centroid),
                "source_area_px": item.source_area_px,
                "marker_radius_px": item.marker_radius_px,
                "peak_score": item.peak_score,
            }
            for item in result.findings
        ],
    }


def render_contact_sheet(
    variants: tuple[VariantImage, ...],
    title: str,
) -> np.ndarray:
    if not variants:
        raise ComparisonExportError("comparison contact sheet requires variants")
    height = max(item.image.shape[0] for item in variants)
    width = max(item.image.shape[1] for item in variants)
    cards: list[np.ndarray] = []
    for item in variants:
        resized = cv2.resize(item.image, (width, height), interpolation=cv2.INTER_AREA)
        header = np.full((72, width, 3), 245, dtype=np.uint8)
        cv2.putText(
            header,
            item.label,
            (24, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (28, 28, 28),
            2,
            cv2.LINE_AA,
        )
        cards.append(np.vstack((header, resized)))
    sheet = np.hstack(cards)
    title_bar = np.full((64, sheet.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        title_bar,
        title,
        (24, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (238, 238, 238),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((title_bar, sheet))


def write_html_index(root: Path, sample_names: tuple[str, ...]) -> None:
    cards = "\n".join(
        f"""<article><h2>{name}</h2>
        <img src=\"{name}/brown_detection.jpg\" alt=\"{name} brown detection\">
        <img src=\"{name}/brown_palettes.jpg\" alt=\"{name} brown palettes\">
        <img src=\"{name}/uv_detection.jpg\" alt=\"{name} UV detection\">
        <img src=\"{name}/porphyrin_detection.jpg\" alt=\"{name} porphyrin detection\">
        <img src=\"{name}/uv_palettes.jpg\" alt=\"{name} UV palettes\">
        <img src=\"{name}/porphyrin_palettes.jpg\" alt=\"{name} porphyrin palettes\">
        <a href=\"{name}/manifest.json\">manifest.json</a></article>"""
        for name in sample_names
    )
    document = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>Consumer pigment preset review</title><style>
    :root{{color-scheme:dark}}body{{margin:0;background:#111;color:#eee;font:15px system-ui;padding:24px}}
    main{{max-width:1600px;margin:auto}}article{{background:#1b1b1b;padding:18px;margin:18px 0;border-radius:14px}}
    img{{display:block;width:100%;height:auto;margin:12px 0;border-radius:8px;background:#000}}a{{color:#8fd7ff}}
    </style><main><h1>Consumer 棕区 / 紫区 Preset 对比</h1>{cards}</main></html>"""
    (root / "index.html").write_text(document, encoding="utf-8")


__all__ = [
    "ComparisonExportError",
    "VariantImage",
    "render_contact_sheet",
    "marker_result_document",
    "sha256_array",
    "sha256_file",
    "write_html_index",
    "write_image",
]
