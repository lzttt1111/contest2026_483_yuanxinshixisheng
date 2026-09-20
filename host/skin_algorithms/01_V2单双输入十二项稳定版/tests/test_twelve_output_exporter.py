from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection_runtime.contracts import (
    LEGACY_NINE_IDS,
    TWELVE_DETECTION_ITEMS,
    RuntimeRoute,
)
from src.detection_runtime.exporter import (
    TwelveOutputExporter,
    provider_receipt_for_export,
    split_institution_provider_receipts,
)
from src.detection_runtime.final_delivery_layout import (
    build_institution_run_receipt,
)


def _file(path: Path, value: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _clinic_receipt(consumer: str, geometry: str) -> dict[str, object]:
    return {
        "actual_vendor_function_executed": True,
        "vendor_function": "core.skin_generate.Enhence_images",
        "module_sha256": "a" * 64,
        "source_role": "CP_M",
        "generated_sha256": {
            "RED_M.jpg": "b" * 64,
            "BROWN_M.jpg": "c" * 64,
            "REDNE_M.jpg": "d" * 64,
        },
        "consumer": consumer,
        "pixel_source_role": "CP_M",
        "geometry_mask_source_role": geometry,
        "fallback_allowed": False,
        "object_aliasing_allowed": False,
    }


def test_consumer_provider_receipt_snapshot_is_returned_byte_unchanged() -> None:
    receipt = {
        "actual_vendor_function_executed": True,
        "source_role": "RGB_M",
        "generated_sha256": {"RED_M.jpg": "a" * 64},
    }
    before = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"

    exported = provider_receipt_for_export(
        {"红棕底图生成回执": receipt},
        RuntimeRoute.CONSUMER_RGB,
    )

    assert exported is receipt
    assert json.dumps(exported, ensure_ascii=False, indent=2) + "\n" == before
    assert "vascular_auxiliary_red" not in exported


def test_institution_provider_receipts_round_trip_to_stable_final_keys() -> None:
    red = _clinic_receipt("redness_brown", "RGB_M")
    vascular = _clinic_receipt("vascular_auxiliary_red", "CP_M")
    container = provider_receipt_for_export(
        {
            "红棕底图生成回执": red,
            "血管RED辅助生成回执": vascular,
        },
        RuntimeRoute.CLINIC_FOUR_LIGHT,
    )
    persisted = json.loads(json.dumps(container, ensure_ascii=False))

    round_trip_red, round_trip_vascular = split_institution_provider_receipts(
        persisted
    )
    final = build_institution_run_receipt(persisted, {"seconds": 1.0})

    assert round_trip_red == red
    assert round_trip_vascular == vascular
    assert final["red_brown_provider"] == red
    assert final["vascular_red_provider"] == vascular
    assert set(final) == {
        "capture_profile",
        "route",
        "timing",
        "red_brown_provider",
        "vascular_red_provider",
    }


@pytest.mark.parametrize(
    ("target", "key", "value"),
    (
        ("red", "fallback_allowed", True),
        ("red", "object_aliasing_allowed", True),
        ("red", "pixel_source_role", "RGB_M"),
        ("red", "geometry_mask_source_role", "CP_M"),
        ("vascular", "consumer", "redness_brown"),
        ("vascular", "geometry_mask_source_role", "RGB_M"),
    ),
)
def test_institution_provider_receipt_rejects_role_or_safety_drift(
    target: str,
    key: str,
    value: object,
) -> None:
    red = _clinic_receipt("redness_brown", "RGB_M")
    vascular = _clinic_receipt("vascular_auxiliary_red", "CP_M")
    (red if target == "red" else vascular)[key] = value

    with pytest.raises(RuntimeError, match="invalid"):
        provider_receipt_for_export(
            {
                "红棕底图生成回执": red,
                "血管RED辅助生成回执": vascular,
            },
            RuntimeRoute.CLINIC_FOUR_LIGHT,
        )


def test_institution_provider_receipt_rejects_missing_or_aliased_view() -> None:
    red = _clinic_receipt("redness_brown", "RGB_M")
    with pytest.raises(RuntimeError):
        provider_receipt_for_export(
            {"红棕底图生成回执": red},
            RuntimeRoute.CLINIC_FOUR_LIGHT,
        )
    with pytest.raises(RuntimeError, match="distinct"):
        provider_receipt_for_export(
            {
                "红棕底图生成回执": red,
                "血管RED辅助生成回执": red,
            },
            RuntimeRoute.CLINIC_FOUR_LIGHT,
        )


def test_exporter_creates_one_twelve_tree_and_legacy_references(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image = _file(source / "input.jpg", b"image")
    items = {}
    metrics = {}
    for index, definition in enumerate(TWELVE_DETECTION_ITEMS, start=1):
        root = source / definition.item_id
        overlay = _file(root / "overlay.jpg", f"image-{index}".encode())
        metric_json = root / "metrics.json"
        raw_metrics: dict[str, object] = {"value": index}
        if definition.item_id == "wrinkle":
            raw_metrics = {
                "weights": "/home/private/wrinkle.pt",
                "region_analysis_status": "ok",
                "region_metrics": [{
                    "region_name": "额头纹",
                    "area_px": 100,
                    "segment_count": 2,
                    "wrinkle_pixels": 10,
                    "mean_segment_length": 5,
                    "max_segment_length": 6,
                    "relative_score": 20,
                }],
            }
        elif definition.item_id == "acne":
            raw_metrics = {
                "output_dir": "/tmp/private",
                "detection": {"status": "ok", "input_mode": "full_face", "detection_scope": "global", "detections": [], "region_counts": {}},
                "preprocess": {"skin_pixels": 1000, "face_count": 1},
                "grading": {"status": "ok", "severity_level": 1},
            }
        elif definition.item_id in {"uv_spots", "porphyrin"}:
            raw_metrics = {
                "project": definition.display_name,
                "metrics_version": f"{definition.item_id}-Real365-v1",
                "scoring_status": "uncalibrated",
                "imaging_and_units": {"imaging_type": "真实365nm正面图像", "area_unit": "pixel"},
                "overall_metrics": {"core_metrics": {"count_and_density": {"feature_count": index}}},
                "region_metrics": [{"region_name": "forehead", "feature_count": index + 100}],
                "left_right_comparison": {"preserved": index},
                "quality_control": {"preserved": index},
                "medical_limitations": ["旧的真实365说明"],
            }
        metric_json.write_text(json.dumps(raw_metrics), encoding="utf-8")
        metric_csv = _file(root / "metrics.csv", b"name,value\nvalue,1\n")
        v2_csv = _file(root / "v2.csv", b"name,value\nvalue,1\n")
        items[definition.item_id] = {
            "项目": definition.display_name,
            "状态": "success",
            "主结果图": str(overlay),
            "量化JSON": str(metric_json),
            "量化CSV": str(metric_csv),
            "医学V2CSV": str(v2_csv),
        }
        if definition.item_id in {"redness", "brown"}:
            items[definition.item_id]["附加结果图"] = [
                str(_file(root / "vendor_base.png", f"base-{index}".encode())),
            ]
        elif definition.item_id == "wrinkle":
            items[definition.item_id]["附加结果图"] = [
                str(_file(root / "08_稳定性线性皱纹全脸分区结果图.jpg", b"stable")),
                str(_file(root / "09_结构性沟纹全脸分区结果图.jpg", b"groove")),
            ]
        elif definition.item_id in {"uv_spots", "porphyrin"}:
            items[definition.item_id]["附加结果图"] = [
                str(_file(root / "00_真实365_source_base.jpg", f"base-{index}".encode())),
            ]
        metrics[definition.item_id] = {
            **items[definition.item_id],
            "核心总体指标": {"value": index},
            "评分输入": {},
        }
    (source / "twelve_metrics.json").write_text(
        json.dumps({"十二项": metrics}), encoding="utf-8"
    )
    _file(source / "timing.json", b"{}")
    _file(source / "timing.csv", b"stage,seconds\nall,1\n")
    manifest = {
        "路线": "consumer_rgb",
        "输入图片": str(image),
        "状态": "success",
        "成功项目数": 12,
        "十二项结果": items,
        "红棕底图生成回执": {
            "actual_vendor_function_executed": True,
            "vendor_function": "core.skin_generate.Enhence_images",
            "input_mode": "front_only_rgb_m_repeated_for_vendor_signature",
            "source_role": "RGB_M",
            "module_sha256": "a" * 64,
            "generated_sha256": {
                "RED_M.jpg": "b" * 64,
                "BROWN_M.jpg": "c" * 64,
                "REDNE_M.jpg": "d" * 64,
            },
            "reference_image_used": False,
        },
    }
    destination = TwelveOutputExporter(tmp_path / "out").export(
        source, manifest, "sample"
    )
    twelve = json.loads((destination / "十二项检测结果索引.json").read_text(encoding="utf-8"))
    legacy = json.loads((destination / "九项检测结果索引.json").read_text(encoding="utf-8"))
    assert tuple(twelve["十二项结果"]) == tuple(
        item.item_id for item in TWELVE_DETECTION_ITEMS
    )
    assert tuple(legacy["九项结果"]) == LEGACY_NINE_IDS
    assert twelve["路线"] == "consumer_rgb"
    assert tuple(twelve["输入通道"]) == ("RGB_M",)
    assert {
        row["主结果图"] for row in legacy["九项结果"].values()
    } == {
        twelve["十二项结果"][key]["主结果图"] for key in LEGACY_NINE_IDS
    }
    assert len(list((destination / "十二项检测").glob("*"))) == 12
    receipt = json.loads(
        (destination / "红棕底图生成回执.json").read_text(encoding="utf-8")
    )
    assert receipt["actual_vendor_function_executed"] is True
    assert receipt["reference_image_used"] is False
    assert len(twelve["十二项结果"]["redness"]["附加结果图"]) == 1
    assert len(twelve["十二项结果"]["brown"]["附加结果图"]) == 1
    assert [
        Path(value).name
        for value in twelve["十二项结果"]["wrinkle"]["附加结果图"]
    ] == [
        "08_稳定性线性皱纹全脸分区结果图.jpg",
        "09_结构性沟纹全脸分区结果图.jpg",
    ]
    assert all(
        (destination / relative).is_file()
        for item_id in ("redness", "brown")
        for relative in twelve["十二项结果"][item_id]["附加结果图"]
    )
    for item_id in ("uv_spots", "porphyrin"):
        exported = twelve["十二项结果"][item_id]
        public_json = json.loads(
            (destination / exported["量化JSON"]).read_text(encoding="utf-8")
        )
        source_json = json.loads(Path(items[item_id]["量化JSON"]).read_text(encoding="utf-8"))
        assert set(public_json) == {
            "project",
            "metrics_version",
            "scoring_status",
            "imaging_and_units",
            "overall_metrics",
            "region_metrics",
            "left_right_comparison",
            "quality_control",
            "medical_limitations",
        }
        assert "input_roles" not in public_json
        assert public_json["imaging_and_units"]["imaging_type"] == "单RGB正面图像（工程代理）"
        assert public_json["imaging_and_units"]["area_unit"] == "pixel"
        for field in ("overall_metrics", "region_metrics", "left_right_comparison", "quality_control"):
            assert public_json[field] == source_json[field]
        assert public_json["overall_metrics"]["core_metrics"]["count_and_density"]["feature_count"] > 0
        assert public_json["region_metrics"][0]["feature_count"] > 100
        assert (destination / exported["量化CSV"]).read_bytes() == b"name,value\nvalue,1\n"
        assert (destination / exported["医学V2CSV"]).read_bytes() == b"name,value\nvalue,1\n"
        assert "真实365" not in exported["附加结果图"][0]
        assert "RGB代理" in exported["附加结果图"][0]
    assert not (destination / "七项检测").exists()
    assert not (destination / "皱纹").exists()
    assert not (destination / "痤疮").exists()
    public_text = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in destination.rglob("*")
        if path.suffix.lower() in {".json", ".csv"}
    )
    public_paths = "\n".join(
        path.relative_to(destination).as_posix() for path in destination.rglob("*")
    )
    assert "Real365" not in public_text
    assert "真实365" not in public_text
    assert "Real365" not in public_paths
    assert "真实365" not in public_paths
    assert "/home/" not in public_text
    assert "/tmp/" not in public_text


def test_clinic_versioned_uv_and_porphyrin_exports_remain_byte_stable(
    tmp_path: Path,
) -> None:
    # Given: Clinic's already-versioned real-365 UV and porphyrin artifacts.
    exporter = TwelveOutputExporter(tmp_path / "out")
    temporary = tmp_path / "temporary"
    expected_fields = {
        "项目",
        "状态",
        "主结果图",
        "量化JSON",
        "量化CSV",
        "医学V2CSV",
        "附加结果图",
    }
    fixtures = (
        ("uv_spots", "00_真实365_UV底图"),
        ("porphyrin", "00_真实365荧光底图"),
    )

    for item_id, expected_base_stem in fixtures:
        definition = next(
            item for item in TWELVE_DETECTION_ITEMS if item.item_id == item_id
        )
        source = tmp_path / "clinic" / item_id
        main_bytes = f"{item_id}-main".encode()
        base_bytes = f"{item_id}-base".encode()
        json_bytes = json.dumps(
            {
                "metrics_version": f"{item_id}-real365-v1",
                "overall_metrics": {},
                "imaging_and_units": {"imaging_type": "真实365nm正面图像"},
            },
            ensure_ascii=False,
        ).encode()
        csv_bytes = f"{item_id},real365\n".encode()
        v2_bytes = f"{item_id},real365-v2\n".encode()
        item = {
            "状态": "success",
            "主结果图": str(_file(source / "main.jpg", main_bytes)),
            "量化JSON": str(_file(source / "metrics.json", json_bytes)),
            "量化CSV": str(_file(source / "metrics.csv", csv_bytes)),
            "医学V2CSV": str(_file(source / "v2.csv", v2_bytes)),
            "附加结果图": [str(_file(source / "base.jpg", base_bytes))],
        }

        # When: the item is exported on the explicit Clinic route.
        exported = exporter._export_item(
            definition,
            item,
            temporary,
            route=RuntimeRoute.CLINIC_FOUR_LIGHT,
        )

        # Then: fields, names, and all four result/base image bytes remain stable.
        assert set(exported) == expected_fields
        assert (temporary / exported["主结果图"]).read_bytes() == main_bytes
        assert (temporary / exported["附加结果图"][0]).read_bytes() == base_bytes
        assert Path(exported["主结果图"]).stem == f"01_{definition.display_name}检测结果图"
        assert Path(exported["附加结果图"][0]).stem == expected_base_stem
        assert Path(exported["量化JSON"]).name == f"{definition.display_name}量化指标.json"
        assert Path(exported["量化CSV"]).name == f"{definition.display_name}量化指标.csv"
        assert Path(exported["医学V2CSV"]).name == f"{definition.display_name}医学量化指标_V2.csv"
        assert (temporary / exported["量化JSON"]).read_bytes() == json_bytes
        assert (temporary / exported["量化CSV"]).read_bytes() == csv_bytes
        assert (temporary / exported["医学V2CSV"]).read_bytes() == v2_bytes
