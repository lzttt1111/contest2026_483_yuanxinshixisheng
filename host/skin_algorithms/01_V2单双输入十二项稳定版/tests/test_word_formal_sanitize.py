from __future__ import annotations

from src.aisia_medical_report.formal_view import (
    _formal_metric,
    build_formal_report_view,
)


def test_formal_view_removes_internal_proxy_unit_from_visible_word_metric() -> None:
    metric = _formal_metric({
        "name": "分区P90视觉对比度最大值",
        "medical_name": "分区P90视觉对比度最大值",
        "value": 0.7,
        "unit": "0～1工程代理",
        "display_type": "INTENSITY_ENGINEERING",
    }, module_id="10")

    assert metric["unit"] == "无量纲（0～1）"
    assert "代理" not in str(metric)


def _acne_payload(count: int) -> dict:
    return {
        "报告信息": {"报告编号": "ACNE-CONDITIONAL-TABLE"},
        "受检者信息": {"姓名或编号": "clinic-test"},
        "检测模块": [{
            "模块编号": "06",
            "模块名称": "毛囊炎症/痤疮样活动",
            "综合得分": None,
            "程度等级": None,
            "结果摘要": f"检测到{count}个痤疮样特征。",
            "核心指标": [
                {"name": "有效皮肤面积", "value": 1000, "unit": "像素"},
                {"name": "痤疮样特征数量", "value": count, "unit": "个"},
                {"name": "单位面积密度", "value": float(count), "unit": "个/10万有效皮肤像素"},
                {"name": "主要集中区域", "value": "额头", "unit": ""},
            ],
            "分区指标": [{
                "分区名称": "额头",
                "状态": "可评估",
                "指标": {"痤疮样特征数量（个）": count},
            }],
            "结果图": [],
            "医生结果分组": [],
        }],
    }


def test_zero_acne_omits_region_table_payload() -> None:
    module = build_formal_report_view(_acne_payload(0))["检测模块"][0]

    assert [row["name"] for row in module["核心指标"]] == ["痤疮样特征数量"]
    assert module["分区指标"] == []


def test_positive_acne_keeps_full_core_and_region_table_payload() -> None:
    module = build_formal_report_view(_acne_payload(1))["检测模块"][0]

    assert {row["name"] for row in module["核心指标"]} == {
        "有效皮肤面积", "痤疮样特征数量", "单位面积密度", "主要集中区域",
    }
    assert len(module["分区指标"]) == 1
    assert module["分区指标"][0]["分区名称"] == "额头"
