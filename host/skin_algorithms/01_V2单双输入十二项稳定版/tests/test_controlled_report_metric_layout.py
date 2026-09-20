from __future__ import annotations

from src.aisia_medical_report.controlled_report_metrics import (
    apply_brown_metric_layout,
    apply_controlled_metric_layout,
    apply_vascular_metric_layout,
)


def _module(module_id: str):
    return {
        "模块编号": module_id,
        "模块名称": module_id,
        "核心指标": [],
        "分区指标": [],
        "医生结果分组": [],
    }


def test_current_detector_truth_populates_approved_controlled_tables() -> None:
    payload = {"检测模块": [_module(f"{index:02d}") for index in range(1, 12)]}
    modules = {row["模块编号"]: row for row in payload["检测模块"]}
    modules["03"]["核心指标"] = [
        {"name": "可见斑点数量", "value": 67},
        {"name": "棕色实例数量", "value": 480},
        {"name": "UV色斑数量", "value": 153},
        {"name": "可见色斑主要集中区域", "value": "鼻部"},
    ]
    modules["03"]["医生结果分组"] = [
        {"title": "可见色斑", "metrics": []},
        {"title": "UV色斑与UV下更明显区域", "metrics": []},
        {"title": "Brown综合色素", "metrics": []},
        {"title": "红褐混合印记", "metrics": []},
        {"title": "重点色斑区域", "metrics": []},
    ]
    modules["10"]["核心指标"] = [
        {"name": "纹理特征总数", "value": 710},
        {"name": "纹理面积占比", "value": 0.069257},
        {"name": "P90纹理响应强度", "value": 0.565257},
    ]
    modules["10"]["医生结果分组"] = [
        {"title": "表面纹理表现", "images": [{"path": "texture.jpg"}]},
        {"title": "表面不规则表现", "images": [{"path": "irregular.jpg"}]},
    ]
    complete = {
        "detector_results": {
            "uv_spots": {"metrics": {"uv_spots": {"总体指标": {
                "核心指标": {
                    "范围与负担": {"特征面积占比": 0.021591},
                    "信号强度": {"P90强度（0～1）": 0.865209},
                },
                "辅助指标": {"数量与密度": {"特征数量（个）": 153}},
            }}}},
            "vascular": {"metrics": {
                "vascular_count": 41,
                "vascular_total_length_px": 490.0,
                "vascular_line_density_per_10k_face_px": 14.6455,
                "vascular_area_ratio": 0.003231,
                "p90_width_px": 2.7386,
                "p90_redness": 5.6,
                "branch_point_count": 7,
                "max_continuous_network_length_px": 29,
                "cp_rgb_visible_support_ratio": 0.967347,
                "region_distribution": {
                    "forehead": {
                        "count": 11,
                        "total_length_px": 112.0,
                        "area_ratio": 0.0043,
                        "line_density_per_10k_px": 22.5,
                    },
                    "left_periocular": {
                        "count": 4,
                        "total_length_px": 47.0,
                        "area_ratio": 0.0036,
                        "line_density_per_10k_px": 15.8,
                    },
                },
            }},
            "wrinkle": {"metrics": {
                "region_metrics": [],
                "report_group_overlays": {"07": {"missing_region_keys": [
                    "left_under_eye", "right_under_eye",
                ]}},
            }},
            "texture": {"metrics": {"medical_metrics_v2": {"overall_metrics": {
                "core_metrics": {
                    "scope_and_burden": {
                        "feature_count": 710,
                        "feature_area_ratio": 0.069257,
                    },
                    "signal_intensity": {"p90_intensity": 0.565257},
                },
            }}}},
            "contour_firmness": {"metrics": {
                "jaw_continuity_ratio": 0.832419,
                "jaw_arc_asymmetry_ratio": 0.02216,
                "jaw_width_to_face_height_ratio": 0.805471,
                "jaw_curve_p90": 0.263236,
                "midface_surface_continuity_ratio": 0.260078,
            }},
        },
    }

    apply_controlled_metric_layout(payload, complete)

    assert len(modules["03"]["核心指标"]) == 6
    assert len(modules["05"]["医生结果分组"]) == 1
    assert len(modules["05"]["医生结果分组"][0]["metrics"]) == 9
    assert modules["05"]["医生结果分组"][0]["metrics"][0]["value"] == 41
    assert [row["分区名称"] for row in modules["05"]["分区指标"]] == [
        "额部",
        "画面左眼周",
    ]
    assert modules["05"]["分区指标"][0]["状态"] == "不纳入评估"
    assert modules["05"]["分区指标"][0]["指标"]["目标数量"] == "不可评估"
    assert len(modules["07"]["医生结果分组"][0]["metrics"]) == 7
    assert modules["07"]["医生结果分组"][0]["metrics"][0]["value"] == 0
    assert [row["title"] for row in modules["10"]["医生结果分组"]] == [
        "表面纹理表现",
        "表面不规则表现",
    ]
    assert modules["10"]["医生结果分组"][1]["images"] == [{"path": "irregular.jpg"}]
    assert len(modules["10"]["医生结果分组"][1]["metrics"]) == 3
    assert len(modules["11"]["医生结果分组"][0]["metrics"]) == 5
    controlled_metrics = [
        metric
        for module in modules.values()
        for collection in [
            module.get("核心指标", []),
            *[group.get("metrics", []) for group in module.get("医生结果分组", [])],
        ]
        for metric in collection
    ]
    assert controlled_metrics
    assert all(str(metric.get("explanation", "")).strip() for metric in controlled_metrics)


def test_clinic_brown_table_uses_current_marker_truth() -> None:
    payload = {"检测模块": [_module(f"{index:02d}") for index in range(1, 12)]}
    module = payload["检测模块"][2]
    module["核心指标"] = [{"name": "棕色实例数量", "value": 478}]
    module["医生结果分组"] = [{
        "title": "Brown综合色素",
        "metrics": [{"name": "Brown综合色素目标数量", "value": 478}],
    }]
    raw = {
        "总计": 338,
        "medical_metrics_v2": {"overall_metrics": {
            "core_metrics": {
                "signal_intensity": {"p90_intensity": 0.970537},
                "scope_and_burden": {"brown_spot_area_ratio": 0.048761},
            },
            "auxiliary_metrics": {
                "scope_and_burden": {"brown_spot_area_ratio": 0.048761},
                "signal_intensity": {"mean_intensity": 0.280257},
            },
        }},
    }

    apply_brown_metric_layout(payload, raw)

    assert module["核心指标"][0]["value"] == 338
    brown = module["医生结果分组"][0]["metrics"]
    assert [metric["value"] for metric in brown] == [
        338, 0.048761, 0.280257, 0.970537,
    ]


def test_clinic_vascular_nested_medical_metrics_populate_regions() -> None:
    payload = {"检测模块": [_module(f"{index:02d}") for index in range(1, 12)]}
    vascular_module = payload["检测模块"][4]
    vascular_module["结果摘要"] = "旧值66/912"
    vascular_module["左右比较摘要"] = "旧值"
    vascular_module["医生结果分组"] = [{"title": "血管样结构观察结果", "summary": "旧值66/912", "metrics": []}]
    raw = {
        "overall_metrics": {
            "core_metrics": {
                "count_and_density": {
                    "vascular_count": 68,
                    "vascular_line_density_per_10k_face_px": 27.9649,
                    "branch_point_count": 12,
                },
                "scope_and_morphology": {
                    "vascular_total_length_px": 932.0,
                    "vascular_area_ratio": 0.00583,
                    "p90_vascular_width_px": 2.7386,
                    "max_continuous_network_length_px": 42,
                },
                "signal_intensity": {"p90_redness_response": 4.3562},
                "auxiliary_statistics": {"rgb_visible_support_ratio": 0.31545},
            },
        },
        "region_metrics": [{
            "analysis_region": "额头",
            "evaluation_status": "ASSESSABLE",
            "valid_skin_area_px": 50000,
            "core_metrics": {
                "count_and_density": {"vascular_count": 4},
                "scope_and_morphology": {
                    "vascular_total_length_px": 50.0,
                    "vascular_area_px": 100,
                },
            },
        }],
        "left_right_comparison": {
            "left_total_length_px": 397.0,
            "right_total_length_px": 405.0,
        },
    }

    apply_vascular_metric_layout(payload, raw)

    module = payload["检测模块"][4]
    assert module["核心指标"][0]["value"] == 68
    assert "68" in module["结果摘要"]
    assert "932" in module["结果摘要"]
    assert module["医生结果分组"][0]["summary"] == module["结果摘要"]
    assert "397" in module["左右比较摘要"]
    assert "405" in module["左右比较摘要"]
    assert module["分区指标"] == [{
        "分区名称": "额部",
        "状态": "不纳入评估",
        "指标": {
            "目标数量": "不可评估",
            "总长度（像素）": "不可评估",
            "长度密度（像素/万有效像素）": "不可评估",
            "面积占比": "不可评估",
        },
    }]
