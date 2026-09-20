from __future__ import annotations

import csv
import hashlib
import importlib
import json
from datetime import date
from pathlib import Path

import pytest

from src.aisia_medical_report.controlled_report_explanations import (
    attach_missing_metric_explanations,
)
from src.aisia_medical_report.controlled_report_metrics import _apply_contour
from src.aisia_medical_report.controlled_report_truth import apply_complete_scoring
from src.aisia_medical_report.single_rgb_v012_delivery import (
    _formal_images,
    _normalized_gloss_metrics,
    _normalized_vascular_metrics,
    _profile_paths,
    _write_combined_purple_csv,
)
from src.aisia_medical_report.twelve_delivery_contract import validate_index_media
from src.detection_runtime.contracts import TWELVE_DETECTION_ITEMS
from src.detection_runtime.final_delivery_layout import (
    build_institution_complete_document,
)


def test_clinic_delivery_imports_without_deleted_payload_builder() -> None:
    module = importlib.import_module(
        "src.aisia_medical_report.clinic_twelve_delivery"
    )

    assert callable(module.generate_clinic_dual_reports_from_twelve_result)


def test_bundled_v011_scoring_assets_are_sha_locked(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AISIA_SCORING_V011_ROOT", raising=False)

    official, shadow = _profile_paths()

    assert official.name == "全量历史ECDF评分配置.json"
    assert shadow.name == "综合色素V0.1.2_shadow评分配置.json"


def test_institution_purple_csv_is_normalized_for_shared_v011_scoring(
    tmp_path: Path,
) -> None:
    fields = [
        "检测范围",
        "评估状态",
        "有效皮肤面积（像素）",
        "核心-特征数量（个）",
        "核心-P50强度（0～1）",
        "核心-P90强度（0～1）",
    ]
    items = {}
    for item_id, count, area in (
        ("uv_spots", 203, 423607),
        ("porphyrin", 1282, 467976),
    ):
        source = tmp_path / f"{item_id}.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "检测范围": "全面部",
                "评估状态": "可评估",
                "有效皮肤面积（像素）": area,
                "核心-特征数量（个）": count,
                "核心-P50强度（0～1）": 0.58,
                "核心-P90强度（0～1）": 0.86,
            })
        items[item_id] = {"医学V2CSV": source.name}
    target = tmp_path / "combined.csv"

    _write_combined_purple_csv(tmp_path, items, target)

    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["检测项目"] == "标准化UV紫外线色斑工程代理"
    assert float(rows[0]["核心-单位面积密度（个/10万有效皮肤像素）"]) == pytest.approx(47.9218)
    assert rows[1]["检测项目"] == "标准化荧光UV紫质工程代理"
    assert float(rows[1]["核心-实例P50强度（0～1）"]) == 0.58
    assert float(rows[1]["核心-实例P90强度（0～1）"]) == 0.86


def test_institution_gloss_is_normalized_for_shared_report_staging() -> None:
    normalized = _normalized_gloss_metrics({
        "overall_metrics": {
            "core_metrics": {
                "scope_and_morphology": {
                    "gloss_area_ratio": 0.0424,
                    "high_gloss_area_ratio": 0.0021,
                },
                "signal_intensity": {
                    "p50_gloss_intensity": 0.424,
                    "p90_gloss_intensity": 0.621,
                },
                "count_and_density": {"gloss_patch_count": 32},
            }
        },
        "region_metrics": [{
            "analysis_region": "鼻部",
            "evaluation_status": "ASSESSABLE",
            "core_metrics": {
                "scope_and_morphology": {"gloss_area_ratio": 0.1259},
                "signal_intensity": {"p90_gloss_intensity": 0.676},
                "count_and_density": {"gloss_patch_count": 7},
            },
        }],
    })

    assert normalized["full_face"]["gloss_area_ratio"] == 0.0424
    assert normalized["full_face"]["patch_count"] == 32
    assert normalized["full_face"]["largest_gloss_component_area_ratio"] == 0.0
    assert normalized["region_metrics"]["institution_0"]["label"] == "鼻部"


def test_institution_vascular_is_normalized_for_shared_report_staging() -> None:
    normalized = _normalized_vascular_metrics({
        "overall_metrics": {
            "core_metrics": {
                "count_and_density": {
                    "vascular_count": 68,
                    "vascular_line_density_per_10k_face_px": 27.96,
                    "branch_point_count": 12,
                },
                "scope_and_morphology": {
                    "vascular_total_length_px": 932.0,
                    "vascular_area_ratio": 0.00583,
                    "p50_vascular_width_px": 1.91,
                    "p90_vascular_width_px": 2.74,
                },
                "signal_intensity": {
                    "p50_redness_response": 2.72,
                    "p90_redness_response": 4.36,
                },
            },
            "auxiliary_metrics": {
                "analysis_scope": {"valid_skin_area_px": 333275}
            },
        },
        "region_metrics": [],
    })

    assert normalized["vascular_count"] == 68
    assert normalized["vascular_total_length_px"] == 932.0
    assert normalized["branch_point_density"] == pytest.approx(0.360063)


def test_vascular_report_summary_preserves_values_and_handles_unavailable() -> None:
    import src.aisia_medical_report.v012 as v012

    assert v012._vascular_result_summary({
        "vascular_count": 68,
        "vascular_total_length_px": 932.0,
    }) == "检测到68个红色线状血管样候选结构，总可见长度约932像素。"
    assert v012._vascular_result_summary({
        "vascular_count": None,
        "vascular_total_length_px": None,
    }) == "红色线状血管样候选结构数量不可评估，总可见长度不可评估。"


def test_institution_report_uses_official_v011_without_consumer_shadow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import src.aisia_medical_report.v012 as v012

    official = {"features": "official"}
    monkeypatch.setattr(v012, "_scoring_features", lambda root: official)
    monkeypatch.setattr(
        v012,
        "_enriched_features",
        lambda root: pytest.fail("institution must not require consumer shadow fields"),
    )

    features, use_shadow = v012._report_features(
        tmp_path,
        {"provenance": {"capture_profile": "institution"}},
    )

    assert features is official
    assert use_shadow is False


def test_v011_scoring_override_fails_closed_on_tamper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.aisia_medical_report.single_rgb_v012_delivery as delivery

    official = tmp_path / "official.json"
    shadow = tmp_path / "shadow.json"
    official.write_bytes(b"official")
    shadow.write_bytes(b"shadow")
    assets = {
        official.name: (official.stat().st_size, hashlib.sha256(official.read_bytes()).hexdigest()),
        shadow.name: (shadow.stat().st_size, hashlib.sha256(shadow.read_bytes()).hexdigest()),
    }
    monkeypatch.setattr(delivery, "PROFILE_ASSETS", assets)
    monkeypatch.setenv("AISIA_SCORING_V011_ROOT", str(tmp_path))
    assert delivery._profile_paths() == (official, shadow)

    shadow.write_bytes(b"tampered")

    with pytest.raises(delivery.WordDeliveryContractError, match="大小异常|SHA异常"):
        delivery._profile_paths()


def test_formal_images_accepts_institution_twelve_item_index(
    tmp_path: Path,
) -> None:
    items = {}
    for item_id in (
        "pores", "surface_gloss", "porphyrin", "spots", "uv_spots",
        "brown", "redness", "vascular", "acne", "wrinkle", "texture",
        "contour_firmness",
    ):
        main = tmp_path / f"{item_id}-main.jpg"
        main.write_bytes(item_id.encode())
        items[item_id] = {"主结果图": main.name, "附加结果图": []}
    for item_id in ("redness", "brown", "porphyrin"):
        extra = tmp_path / f"{item_id}-extra.jpg"
        extra.write_bytes(f"{item_id}-extra".encode())
        items[item_id]["附加结果图"] = [extra.name]
    for index in range(2):
        extra = tmp_path / f"wrinkle-{index}.jpg"
        extra.write_bytes(str(index).encode())
        items["wrinkle"]["附加结果图"].append(extra.name)

    images = _formal_images(
        tmp_path,
        {"路线": "clinic_four_light", "十二项结果": items},
    )

    assert images["01"].name == "pores-main.jpg"
    assert images["03_brown_base"].name == "brown-extra.jpg"
    assert images["03_brown_marker"].name == "brown-main.jpg"
    assert images["04_base"].name == "redness-extra.jpg"
    assert images["04_marker"].name == "redness-main.jpg"
    assert images["08"].name == "wrinkle-0.jpg"
    assert images["09"].name == "wrinkle-1.jpg"


def test_word_media_contract_accepts_hashed_institution_index(
    tmp_path: Path,
) -> None:
    from src.detection_runtime.final_delivery_layout import (
        attach_institution_media_hashes,
    )

    items = {}
    for definition in TWELVE_DETECTION_ITEMS:
        main = tmp_path / f"{definition.item_id}-main.jpg"
        main.write_bytes(definition.item_id.encode())
        extras = []
        for index in range({"redness": 1, "brown": 1, "uv_spots": 1, "porphyrin": 1, "wrinkle": 2}.get(definition.item_id, 0)):
            extra = tmp_path / f"{definition.item_id}-extra-{index}.jpg"
            extra.write_bytes(
                b"shared-365"
                if definition.item_id in {"uv_spots", "porphyrin"}
                else f"{definition.item_id}-{index}".encode()
            )
            extras.append(extra.name)
        items[definition.item_id] = {
            "状态": "success",
            "主结果图": main.name,
            "附加结果图": extras,
        }
    institution = {
        "路线": "clinic_four_light",
        "状态": "success",
        "成功项目数": 12,
        "项目总数": 12,
        "十二项结果": items,
    }

    attach_institution_media_hashes(tmp_path, institution)
    validate_index_media(institution, tmp_path)

    (tmp_path / items["spots"]["主结果图"]).write_bytes(
        (tmp_path / items["pores"]["主结果图"]).read_bytes()
    )
    attach_institution_media_hashes(tmp_path, institution)
    consumer = {
        "schema_version": "single_rgb_twelve_index_v3",
        "status": "success",
        "items": items,
    }
    with pytest.raises(RuntimeError, match="非授权重复"):
        validate_index_media(consumer, tmp_path)


def test_unavailable_contour_never_retains_frozen_numeric_summary() -> None:
    payload = {
        "检测模块": [{
            "模块编号": "11",
            "结果摘要": "冻结旧值0.832/0.022",
            "核心指标": [{"name": "冻结指标", "value": 0.832}],
            "医生结果分组": [{"title": "冻结表", "metrics": []}],
            "分区指标": [{"分区名称": "冻结区", "指标": {"旧值": 0.022}}],
        }]
    }
    complete = {
        "detector_results": {
            "contour_firmness": {
                "metrics": {
                    "overall_metrics": {
                        "core_metrics": {},
                        "auxiliary_metrics": {
                            "analysis_scope": {"valid_skin_area_px": 0}
                        },
                    }
                }
            }
        }
    }

    _apply_contour(payload, complete)

    contour = payload["检测模块"][0]
    assert contour["核心指标"] == []
    assert contour["医生结果分组"] == []
    assert contour["分区指标"] == []
    assert "不可评估" in contour["结果摘要"]
    assert "0.832" not in json.dumps(contour, ensure_ascii=False)


def test_controlled_explanations_distinguish_length_density_and_analysis_area() -> None:
    payload = {
        "检测模块": [{
            "模块编号": "05",
            "核心指标": [
                {"name": "单位面积长度密度", "value": 12.3, "unit": "像素/万有效像素"},
                {"name": "最大连续网络长度", "value": 44, "unit": "像素"},
                {"name": "眼下有效皮肤面积", "value": 3000, "unit": "像素"},
                {"name": "已批准解释", "value": 1, "explanation": "保持原字节"},
            ],
            "医生结果分组": [],
        }]
    }

    attach_missing_metric_explanations(payload)

    metrics = {row["name"]: row["explanation"] for row in payload["检测模块"][0]["核心指标"]}
    assert "总长度" in metrics["单位面积长度密度"]
    assert "骨架长度" in metrics["最大连续网络长度"]
    assert "参与" in metrics["眼下有效皮肤面积"]
    assert metrics["已批准解释"] == "保持原字节"


def test_institution_complete_document_retains_current_runtime_summary(
    tmp_path: Path,
) -> None:
    exported = {}
    runtime_items = {}
    compact = {}
    for definition in TWELVE_DETECTION_ITEMS:
        item_id = definition.item_id
        public = tmp_path / f"{item_id}-public.json"
        raw = tmp_path / f"{item_id}-raw.json"
        public.write_text("{}\n", encoding="utf-8")
        raw.write_text("{}\n", encoding="utf-8")
        exported[item_id] = {
            "主结果图": f"{item_id}.jpg",
            "附加结果图": [],
            "量化JSON": public.name,
        }
        runtime_items[item_id] = {"量化JSON": str(raw)}
        compact[item_id] = {"核心总体指标": {}}
    compact["contour_firmness"]["核心总体指标"] = {
        "下颌缘连续性": 0.832419,
        "左右轮廓差异": 0.02216,
        "中面部曲面连续性": 0.260078,
    }
    (tmp_path / "十二项检测结果索引.json").write_text(
        json.dumps({"十二项结果": exported, "输入通道": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "十二项核心量化指标.json").write_text(
        json.dumps({"十二项": compact}, ensure_ascii=False),
        encoding="utf-8",
    )
    generated = {
        "RED_M.jpg": "a" * 64,
        "BROWN_M.jpg": "b" * 64,
        "REDNE_M.jpg": "c" * 64,
    }
    (tmp_path / "红棕底图生成回执.json").write_text(
        json.dumps({
            "consumer": "redness_brown",
            "pixel_source_role": "CP_M",
            "geometry_mask_source_role": "RGB_M",
            "fallback_allowed": False,
            "object_aliasing_allowed": False,
            "generated_sha256": generated,
            "vascular_auxiliary_red": {
                "consumer": "vascular_auxiliary_red",
                "pixel_source_role": "CP_M",
                "geometry_mask_source_role": "CP_M",
                "fallback_allowed": False,
                "object_aliasing_allowed": False,
                "generated_sha256": generated,
            },
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    complete = build_institution_complete_document(tmp_path, runtime_items)

    summary = complete["detector_results"]["contour_firmness"]["metrics"]["runtime_summary"]
    assert summary["下颌缘连续性"] == 0.832419
    assert summary["左右轮廓差异"] == 0.02216


def test_controlled_redness_uses_current_detector_truth() -> None:
    module = importlib.import_module(
        "src.aisia_medical_report.controlled_report_metrics"
    )
    apply_redness = getattr(module, "_apply_redness", None)
    assert callable(apply_redness)
    payload = {
        "检测模块": [{
            "模块编号": "04",
            "结果摘要": "冻结弥漫红区覆盖40.03%",
            "核心指标": [{"name": "冻结红度", "value": 0.4003}],
            "医生结果分组": [],
            "分区指标": [],
        }]
    }
    complete = {
        "detector_results": {
            "redness": {
                "metrics": {
                    "mean_redness": 0.373383,
                    "p90_redness": 0.800557,
                    "high_red_area_ratio": 0.142273,
                    "red_feature_count": 65,
                    "medical_metrics_v2": {
                        "overall_metrics": {
                            "core_metrics": {
                                "scope_and_burden": {
                                    "diffuse_red_area_ratio": 0.358432,
                                    "max_continuous_region_area_px": 98514,
                                    "redness_continuity": 0.762457,
                                },
                                "auxiliary_statistics": {
                                    "diffuse_redness_uniformity": 0.519758,
                                    "redness_boundary_gradient": 0.231634,
                                },
                            },
                            "auxiliary_metrics": {
                                "analysis_scope": {"valid_skin_area_px": 360476}
                            },
                        },
                        "region_metrics": [],
                    },
                }
            }
        }
    }

    apply_redness(payload, complete)

    redness = payload["检测模块"][0]
    values = {row["name"]: row["value"] for row in redness["核心指标"]}
    assert values["弥漫红区面积占比"] == 0.358432
    assert values["高红度响应区域面积占比"] == 0.142273
    assert values["局灶红色实例数量"] == 65
    assert "35.84%" in redness["结果摘要"]
    assert "40.03" not in redness["结果摘要"]


def test_word_keeps_current_v011_scores_and_hides_proxy_zeroes() -> None:
    payload = {
        "报告信息": {},
        "检测模块": [
            {
                "模块编号": f"{index:02d}",
                "综合得分": 40.478152 if index == 1 else None,
                "程度等级": "中度" if index == 1 else None,
                "结果摘要": "当前结果",
                "正式评分说明": {
                    "summary_text": "本次评分由当前检测结果重算。",
                    "drivers": [],
                } if index == 1 else None,
            }
            for index in range(1, 12)
        ],
    }
    complete = {
        "scoring_results": {
            "scoring_profile_version": "production_proxy_v1",
            "calibration_boundary": {"statement": "三锚点工程代理分"},
            "module_scores": {
                "pores": {
                    "score": 0.0,
                    "grade": "未见明显",
                    "score_valid": True,
                    "groups": {},
                }
            },
        }
    }

    apply_complete_scoring(payload, complete)

    pores = payload["检测模块"][0]
    assert pores["综合得分"] == 40.478152
    assert pores["程度等级"] == "中度"
    assert pores["正式评分说明"]["summary_text"] == "本次评分由当前检测结果重算。"
    assert payload["检测模块"][1]["综合得分"] is None
    assert "production_proxy_v1" not in json.dumps(payload, ensure_ascii=False)
    assert "工程代理分" not in json.dumps(payload, ensure_ascii=False)
