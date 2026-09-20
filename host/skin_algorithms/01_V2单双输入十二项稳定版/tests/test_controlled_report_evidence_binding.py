from __future__ import annotations

from pathlib import Path
from datetime import date
import importlib
import inspect
import json

from docx import Document
from PIL import Image

from src.aisia_medical_report.single_rgb_v012_delivery import _formal_images
from src.aisia_medical_report.report import (
    FORMAL_RESULT_IMAGE_WIDTH,
    _add_result_image,
    _user_result_image_limit,
)
from src.aisia_medical_report.twelve_delivery import bind_twelve_report_images
from src.aisia_medical_report.controlled_evidence import (
    ControlledEvidencePaths,
    bind_controlled_evidence,
)


def _module(module_id: str, groups: list[str] | None = None):
    return {
        "模块编号": module_id,
        "结果图": [],
        "医生结果分组": [
            {"title": title, "images": []}
            for title in (groups or [])
        ],
    }


def test_controlled_report_keeps_three_approved_derived_image_roles(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    names = (
        "01", "02_oil", "02_porphyrin", "03_visible", "03_uv", "03_brown",
        "03_brown_base", "03_brown_marker", "03_red_brown_mixed",
        "03_priority_pigment", "04", "04_base", "04_marker", "05", "06",
        "07", "08", "09", "10", "10_surface_irregularity", "11",
    )
    images = {}
    for name in names:
        path = tmp_path / f"{name}.jpg"
        path.write_bytes(name.encode())
        images[name] = path
    payload = {
        "检测模块": [
            _module("01"),
            _module("02", ["表面油光表现", "毛囊荧光表现"]),
            _module("03", [
                "可见色斑",
                "UV色斑与UV下更明显区域",
                "Brown综合色素",
                "红褐混合印记",
                "重点色斑区域",
            ]),
            _module("04", ["泛红强度"]),
            _module("05", ["血管样结构观察结果"]),
            _module("06"),
            _module("07", ["双眼下细纹表现"]),
            _module("08"),
            _module("09"),
            _module("10", ["二维纹理分布", "表面不规则分布"]),
            _module("11", ["面部轮廓几何测量"]),
        ]
    }

    bind_twelve_report_images(payload, images, source)

    modules = {row["模块编号"]: row for row in payload["检测模块"]}
    pigment_groups = modules["03"]["医生结果分组"]
    assert pigment_groups[0]["images"][0]["path"] == str(images["03_visible"])
    assert pigment_groups[1]["images"][0]["path"] == str(images["03_uv"])
    assert pigment_groups[2]["images"][0]["path"] == str(images["03_brown_marker"])
    assert pigment_groups[3]["images"][0]["path"] == str(images["03_red_brown_mixed"])
    assert pigment_groups[4]["images"][0]["path"] == str(images["03_priority_pigment"])
    assert modules["05"]["医生结果分组"][0]["images"][0]["path"] == str(images["05"])
    assert modules["07"]["医生结果分组"][0]["images"][0]["path"] == str(images["07"])
    texture_groups = modules["10"]["医生结果分组"]
    assert texture_groups[0]["images"][0]["path"] == str(images["10"])
    assert texture_groups[1]["images"][0]["path"] == str(images["10_surface_irregularity"])
    assert modules["03"]["结果图"] == [
        str(images["03_brown_marker"]),
        str(images["03_visible"]),
        str(images["03_uv"]),
        str(images["03_red_brown_mixed"]),
        str(images["03_priority_pigment"]),
    ]
    assert modules["10"]["结果图"] == [
        str(images["10"]),
        str(images["10_surface_irregularity"]),
    ]
    assert modules["04"]["结果图"] == [str(images["04_marker"])]
    assert modules["04"]["医生结果分组"][0]["images"][0]["path"] == str(images["04_marker"])


def test_single_rgb_report_uses_marker_media_for_red_and_brown(
    tmp_path: Path,
) -> None:
    item_ids = (
        "pores", "surface_gloss", "porphyrin", "spots", "uv_spots",
        "brown", "redness", "vascular", "acne", "wrinkle", "texture",
        "contour_firmness",
    )
    items = {}
    for item_id in item_ids:
        main = tmp_path / f"{item_id}-base.jpg"
        main.write_bytes(item_id.encode())
        items[item_id] = {"主结果图": main.name, "附加结果图": []}
    for item_id in ("redness", "brown"):
        marker = tmp_path / f"{item_id}-marker.jpg"
        marker.write_bytes(f"{item_id}-marker".encode())
        items[item_id]["附加结果图"] = [marker.name]
    for index in range(2):
        extra = tmp_path / f"wrinkle-{index}.jpg"
        extra.write_bytes(str(index).encode())
        items["wrinkle"]["附加结果图"].append(extra.name)

    images = _formal_images(tmp_path, {"items": items})

    assert images["04_base"].name == "redness-base.jpg"
    assert images["04_marker"].name == "redness-marker.jpg"
    assert images["03_brown_base"].name == "brown-base.jpg"
    assert images["03_brown_marker"].name == "brown-marker.jpg"


def test_red_brown_report_uses_one_marker_image(tmp_path: Path) -> None:
    marker = tmp_path / "marker.jpg"
    Image.new("RGB", (64, 64), "red").save(marker)
    document = Document()

    _add_result_image(
        document,
        {"模块编号": "03", "结果图": [str(marker)]},
        maximum=1,
    )

    assert len(document.inline_shapes) == 1
    assert document.inline_shapes[0].width == FORMAL_RESULT_IMAGE_WIDTH


def test_user_report_keeps_same_core_image_roles_as_doctor_report() -> None:
    assert _user_result_image_limit({"模块编号": "03"}) == 5
    assert _user_result_image_limit({"模块编号": "04"}) == 1
    assert _user_result_image_limit({"模块编号": "10"}) == 2


def test_current_formal_view_refreshes_report_and_collection_dates() -> None:
    module = importlib.import_module("src.aisia_medical_report.formal_view")
    build = getattr(module, "build_formal_report_view", None)
    assert callable(build)
    payload = {
        "报告信息": {"报告日期": "2026-08-18", "报告编号": "fixture"},
        "受检者信息": {"采集日期": "2026-08-18"},
        "检测模块": [],
    }

    current = build(
        payload,
        generated_on=date(2026, 8, 25),
        collected_on=date(2026, 8, 25),
    )

    assert current["报告信息"]["报告日期"] == "2026-08-25"
    assert current["受检者信息"]["采集日期"] == "2026-08-25"


def test_formal_report_requires_runtime_evidence_for_all_doctor_groups() -> None:
    module = importlib.import_module(
        "src.aisia_medical_report.single_rgb_v012_delivery"
    )
    parameter = inspect.signature(
        module.generate_single_rgb_v012_reports
    ).parameters["runtime_items"]

    assert parameter.default is inspect.Parameter.empty


def test_controlled_evidence_binds_current_derived_metrics(
    tmp_path: Path,
) -> None:
    red_brown = tmp_path / "red-brown.jpg"
    priority = tmp_path / "priority.jpg"
    irregular = tmp_path / "irregular.jpg"
    for path in (red_brown, priority, irregular):
        path.write_bytes(path.stem.encode())
    metrics = tmp_path / "pigmentation.json"
    metrics.write_text(
        json.dumps(
            {
                "metrics": {
                    "visible_pigment": {"area_px": 200},
                    "red_brown_mixed": {
                        "component_count": 7,
                        "area_px": 40,
                        "area_ratio": 0.004,
                    },
                    "pure_red_suspected": {"area_ratio": 0.002},
                },
                "region_distribution": [
                    {
                        "available": True,
                        "component_count": 3,
                        "feature_area_px": 50,
                        "valid_area_px": 1000,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "检测模块": [
            _module(
                "03",
                ["可见色斑", "UV色斑", "Brown综合色素", "红褐混合印记", "重点色斑区域"],
            ),
            _module("10", ["表面纹理表现", "表面不规则表现"]),
        ]
    }

    bind_controlled_evidence(
        payload,
        ControlledEvidencePaths(
            red_brown_mixed=red_brown,
            priority_pigment=priority,
            surface_irregularity=irregular,
            pigmentation_metrics=metrics,
        ),
    )

    pigment = payload["检测模块"][0]["医生结果分组"]
    assert pigment[3]["metrics"][0]["value"] == 7
    assert pigment[3]["metrics"][1]["value"] == 0.004
    assert pigment[4]["metrics"][0]["value"] == 3
    assert pigment[4]["metrics"][1]["value"] == 0.05
    assert all(
        row.get("explanation")
        for group in pigment[3:]
        for row in group["metrics"]
    )
    assert payload["检测模块"][0]["结果图"] == [
        str(red_brown),
        str(priority),
    ]
    assert payload["检测模块"][1]["结果图"] == [str(irregular)]
