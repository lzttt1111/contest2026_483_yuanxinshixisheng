from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from cloud_contracts import (
    public_metric_documentation_rows,
    validate_worker_envelope,
)
from src.added_algorithms.public_metrics import (
    write_contour_metrics,
    write_gloss_metrics,
    write_vascular_metrics,
)


REGIONS = ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")


def _envelope(algorithm: str, metrics: dict) -> dict:
    return {
        "record_id": f"contract-{algorithm}",
        "status": "success",
        "schema_version": "1",
        "meta_data": {"name": algorithm, "version": "1"},
        "raw_result": {
            "overlay": f"{algorithm}/result.jpg",
            "metrics": metrics,
            "medical_report_csv_v2": f"{algorithm}/medical.csv",
            "quality_score": 90.0,
            "quality_status": "PASS",
            "quality_flags": [],
        },
        "debug_info": {
            "report_csv": f"{algorithm}/compact.csv",
            "timing_seconds": {"pipeline": 1.0},
            "execution_mode": "single_algorithm",
        },
    }


def test_added_contracts_use_typed_chinese_public_metrics() -> None:
    regional_ratio = {name: 0.1 for name in REGIONS}
    regional_count = {name: 2 for name in REGIONS}
    payloads = {
        "surface_gloss": {
            "油光面积占比": regional_ratio,
            "油光区域数量": regional_count,
        },
        "vascular": {
            "血管样结构数量": regional_count,
            "血管样结构总长度": {
                name: 12.5 for name in REGIONS
            },
        },
        "contour_firmness": {
            "中面部曲面连续性": 0.82,
            "下颌缘连续性": 0.76,
            "左右轮廓差异": 0.08,
        },
    }
    expected_units = {
        "surface_gloss": {"比例（0～1）", "个"},
        "vascular": {"个", "标准化图像像素"},
        "contour_firmness": {"相对代理值（0～1）"},
    }
    for algorithm, metrics in payloads.items():
        envelope = _envelope(algorithm, metrics)
        assert validate_worker_envelope(algorithm, envelope) is envelope
        rows = public_metric_documentation_rows(algorithm, envelope["raw_result"])
        assert rows
        assert {row["unit"] for row in rows} == expected_units[algorithm]
        assert all(any("\u4e00" <= char <= "\u9fff" for char in row["title"]) for row in rows)
        assert all("见字段合同文档" not in row["unit"] for row in rows)
        assert all("新增单RGB检测的公开精简指标" not in row["description"] for row in rows)


def test_added_public_writers_label_every_compact_row(tmp_path: Path) -> None:
    for name in ("gloss", "vascular", "contour"):
        (tmp_path / name).mkdir()
    gloss = SimpleNamespace(
        valid_skin_area_px=1000,
        gloss_area_ratio=0.1,
        patch_count=3,
        region_metrics={
            "forehead": {"valid_area_px": 100, "gloss_area_px": 10, "patch_count": 1},
            "nose": {"valid_area_px": 100, "gloss_area_px": 20, "patch_count": 1},
            "chin": {"valid_area_px": 100, "gloss_area_px": 5, "patch_count": 1},
            "left_nasal_side": {"valid_area_px": 50, "gloss_area_px": 5, "patch_count": 1},
            "left_inner_cheek": {"valid_area_px": 50, "gloss_area_px": 5, "patch_count": 1},
            "left_outer_cheek": {"valid_area_px": 50, "gloss_area_px": 5, "patch_count": 1},
            "right_nasal_side": {"valid_area_px": 50, "gloss_area_px": 5, "patch_count": 1},
            "right_inner_cheek": {"valid_area_px": 50, "gloss_area_px": 5, "patch_count": 1},
            "right_outer_cheek": {"valid_area_px": 50, "gloss_area_px": 5, "patch_count": 1},
        },
    )
    vascular = {
        "vascular_count": 2,
        "vascular_total_length_px": 20.0,
        "valid_face_pixels": 1000,
        "qc_passed": True,
        "region_distribution": {},
    }
    outputs = (
        write_gloss_metrics(gloss, tmp_path / "gloss"),
        write_vascular_metrics(vascular, tmp_path / "vascular"),
        write_contour_metrics(
            {
                "midface_surface_continuity_ratio": 0.8,
                "jaw_continuity_ratio": 0.7,
                "jaw_arc_asymmetry_ratio": 0.1,
            },
            1000,
            tmp_path / "contour",
        ),
    )
    for files in outputs:
        metrics = json.loads(files.metrics_json.read_text(encoding="utf-8"))
        assert all(any("\u4e00" <= char <= "\u9fff" for char in name) for name in metrics)
        rows = list(csv.reader(files.compact_csv.open(encoding="utf-8-sig")))
        assert rows[0][0] == "指标"
        assert rows[0][1] == "单位"
        assert all(row[0].strip() and row[1].strip() for row in rows[1:])
        assert files.full_metrics_json.is_file()
        full_metrics = json.loads(files.full_metrics_json.read_text(encoding="utf-8"))
        assert full_metrics
