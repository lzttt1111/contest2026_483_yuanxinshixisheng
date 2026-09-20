from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, TypeAlias

from typing_extensions import assert_never

from src.capture_profile import CaptureProfile
from .final_output_artifacts import export_item_documents
from .production_proxy_scoring import (
    score_production_proxy,
)
from .public_sanitize import sanitize_public_document
from .v2_labels import LABELS_PATH, enrich_v2_modules, load_v2_labels
from .v2_proxy_projection import build_v2_proxy_modules
from .zero_target_metrics import normalize_zero_target_detector_results


ITEM_DEFINITIONS = {
    "redness": ("01_红区", "红区"),
    "spots": ("02_可见斑点", "可见斑点"),
    "brown": ("03_棕区", "棕区"),
    "texture": ("04_纹理", "纹理"),
    "pores": ("05_毛孔", "毛孔"),
    "uv_spots": ("06_UV色斑", "UV色斑"),
    "porphyrin": ("07_卟啉", "卟啉"),
    "wrinkle": ("08_皱纹", "皱纹"),
    "acne": ("09_痤疮", "痤疮"),
    "surface_gloss": ("10_油光", "油光"),
    "vascular": ("11_血管样结构", "血管样结构"),
    "contour_firmness": ("12_轮廓紧致度", "轮廓紧致度"),
}
EXTRA_IMAGES = {
    "redness": (("06_VISIA红色区实例图.jpg", "02_VISIA红色区实例图.jpg", False),),
    "brown": (("02_VISIA棕色斑实例图.jpg", "02_VISIA棕色斑实例图.jpg", False),),
    "uv_spots": (("01_紫外线色斑底图.png", "00_紫外线色斑底图.png", False),),
    "porphyrin": (("03_紫质荧光底图.png", "00_紫质荧光底图.png", False),),
    "wrinkle": (
        ("08_稳定性线性皱纹全脸分区结果图.jpg", "02_稳定性线性皱纹全脸分区结果图.jpg", True),
        ("09_结构性沟纹全脸分区结果图.jpg", "03_结构性沟纹全脸分区结果图.jpg", True),
    ),
}
REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "calibration"
    / "metric_registry_v2_proxy_20260728.json"
)
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class DuplicateFormalImageError(ValueError):
    def __init__(self, label: str) -> None:
        self.label = label
        super().__init__(f"{label}正式结果图内容重复")


def _load_json(path: str | Path) -> JsonObject:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_complete_document(
    *,
    detector_results: dict[str, dict],
    capture_profile: CaptureProfile,
    input_signature: Mapping[str, JsonValue],
    provenance: Mapping[str, JsonValue],
    quality_control: Mapping[str, JsonValue],
) -> JsonObject:
    """Build the durable detector truth used for later population scoring."""
    if capture_profile is CaptureProfile.CONSUMER:
        detector_results = normalize_zero_target_detector_results(detector_results)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    projected_modules, scoring_features = build_v2_proxy_modules(
        detector_results,
        registry,
    )
    medical_modules = enrich_v2_modules(projected_modules, load_v2_labels())
    provenance_document = sanitize_public_document(provenance)
    provenance_document["capture_profile"] = capture_profile.value
    match capture_profile:
        case CaptureProfile.INSTITUTION:
            provenance_document["route"] = "institution_vendor_rgb"
        case CaptureProfile.CONSUMER:
            provenance_document["route"] = "consumer_rgb"
        case unreachable:
            assert_never(unreachable)
    scoring_results = score_production_proxy(medical_modules)
    provenance_document["v2_registry_sha256"] = _sha256(REGISTRY_PATH)
    provenance_document["v2_label_registry_sha256"] = _sha256(LABELS_PATH)
    return {
        "schema_version": "single_rgb_twelve_complete_v2_20260728",
        "medical_requirement_version": "face_detection_v2_20260728",
        "input_signature": sanitize_public_document(input_signature),
        "provenance": provenance_document,
        "detector_results": detector_results,
        "medical_v2_modules": medical_modules,
        "scoring_features": scoring_features,
        "scoring_results": {
            **scoring_results,
            "complete_proxy_inputs": all(
                module["completeness"]["complete"]
                for module in medical_modules.values()
            ),
        },
        "quality_control": sanitize_public_document(quality_control),
        "compatibility": {"legacy_nine_files_emitted": False},
    }


def write_final_output(
    target: Path,
    *,
    items: Mapping[str, Mapping[str, JsonValue]],
    input_signature: Mapping[str, JsonValue],
    provenance: Mapping[str, JsonValue],
    quality_control: Mapping[str, JsonValue],
    timing: Mapping[str, JsonValue],
) -> Path:
    capture_profile = CaptureProfile(str(
        provenance.get(
            "capture_profile",
            CaptureProfile.INSTITUTION.value,
        )
    ))
    target.mkdir(parents=True, exist_ok=True)
    image_root = target / "十二项检测"
    image_root.mkdir()
    index_items: dict[str, dict] = {}
    detector_results: dict[str, dict] = {}
    for key, item in items.items():
        folder, label = ITEM_DEFINITIONS[key]
        destination = image_root / folder
        destination.mkdir()
        source_image = Path(str(item["主结果图"]))
        configured_images = EXTRA_IMAGES.get(key, ())
        required_sources = [source_image]
        for source_name, _, required in configured_images:
            if required and not (source_image.parent / source_name).is_file():
                raise FileNotFoundError(f"缺少皱纹正式结果图: {source_name}")
            if required:
                required_sources.append(source_image.parent / source_name)
        if (
            len(required_sources) > 1
            and len({_sha256(path) for path in required_sources})
            != len(required_sources)
        ):
            raise DuplicateFormalImageError(label)
        image_path = destination / f"01_{label}检测结果图{source_image.suffix.lower()}"
        shutil.copy2(source_image, image_path)
        image_paths = [image_path]
        for source_name, target_name, _ in configured_images:
            source_extra = source_image.parent / source_name
            if source_extra.is_file():
                target_extra = destination / target_name
                shutil.copy2(source_extra, target_extra)
                image_paths.append(target_extra)
        documents = export_item_documents(key, item, destination)
        metrics = documents.public_metrics
        complete_metrics = sanitize_public_document(_load_json(
            str(item.get("完整量化JSON") or item["量化JSON"])
        ))
        relative_image = image_path.relative_to(target).as_posix()
        index_items[key] = {
            "项目": label,
            "状态": "success",
            "主结果图": relative_image,
            "附加结果图": [
                path.relative_to(target).as_posix()
                for path in image_paths[1:]
            ],
            "images": [
                {
                    "path": path.relative_to(target).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in image_paths
            ],
            "量化CSV": documents.compact_csv.relative_to(target).as_posix(),
            "医学V2CSV": documents.medical_csv.relative_to(target).as_posix(),
            "量化JSON": documents.public_json.relative_to(target).as_posix(),
        }
        detector_results[key] = {
            "status": "success",
            "images": [
                path.relative_to(target).as_posix()
                for path in image_paths
            ],
            "public_metrics": metrics,
            "metrics": complete_metrics,
        }
    index = {
        "schema_version": "single_rgb_twelve_index_v3",
        "status": "success",
        "capture_profile": capture_profile.value,
        "items": index_items,
    }
    complete = build_complete_document(
        detector_results=detector_results,
        capture_profile=capture_profile,
        input_signature=input_signature,
        provenance=provenance,
        quality_control=quality_control,
    )
    (target / "十二项检测结果索引.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "十二项完整量化指标.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (target / "运行回执.json").write_text(
        json.dumps({
            "capture_profile": capture_profile.value,
            "timing": sanitize_public_document(timing),
            "input_signature": sanitize_public_document(input_signature),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = ["build_complete_document", "write_final_output"]
