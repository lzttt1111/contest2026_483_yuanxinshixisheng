from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.nine_analysis.final_output import write_final_output
from src.nine_analysis.final_output_artifacts import export_item_documents


def test_csv_artifacts_preserve_mature_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "main.jpg"
    compact = source / "compact.csv"
    medical = source / "medical.csv"
    metrics = source / "metrics.json"
    image.write_bytes(b"image")
    compact.write_bytes("\ufeff指标,单位,总计\r\n数量,个,7\r\n".encode())
    medical.write_bytes("\ufeff检测范围,评估状态\r\n全面部,可评估\r\n".encode())
    metrics.write_text('{"总计": 7}', encoding="utf-8")
    destination = tmp_path / "output"

    exported = export_item_documents(
        "redness",
        {
            "主结果图": str(image),
            "量化CSV": str(compact),
            "医学V2CSV": str(medical),
            "量化JSON": str(metrics),
        },
        destination,
    )

    assert exported.compact_csv.read_bytes() == compact.read_bytes()
    assert exported.medical_csv.read_bytes() == medical.read_bytes()
    assert not list(destination.glob("*.xlsx"))


def test_export_rejects_missing_formal_csv(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"总计": 1}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="缺少正式结果来源"):
        export_item_documents(
            "redness",
            {"主结果图": str(tmp_path / "main.jpg"), "量化JSON": str(metrics)},
            tmp_path / "output",
        )


def test_final_output_rejects_stale_detection_tree(tmp_path: Path) -> None:
    target = tmp_path / "sample"
    (target / "十二项检测").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        write_final_output(
            target,
            items={},
            input_signature={},
            provenance={},
            quality_control={},
            timing={},
        )


def test_purple_csv_artifacts_are_isolated_by_detection_item(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "main.png"
    compact = source / "purple.csv"
    medical = source / "purple_medical.csv"
    metrics = source / "purple.json"
    image.write_bytes(b"image")
    compact.write_text("检测项目,总计\n紫外线色斑,2\n紫质,3\n", encoding="utf-8-sig")
    medical.write_text(
        "检测项目,评估状态\n标准化UV紫外线色斑工程代理,可评估\n"
        "标准化荧光UV紫质工程代理,可评估\n",
        encoding="utf-8-sig",
    )
    metrics.write_text(
        '{"uv_spots_total": 2, "porphyrin_total": 3}', encoding="utf-8"
    )
    item = {
        "主结果图": str(image),
        "量化CSV": str(compact),
        "医学V2CSV": str(medical),
        "量化JSON": str(metrics),
    }

    uv = export_item_documents("uv_spots", item, tmp_path / "uv")
    porphyrin = export_item_documents("porphyrin", item, tmp_path / "porphyrin")

    assert uv.compact_csv.read_bytes() != porphyrin.compact_csv.read_bytes()
    assert "紫质,3" not in uv.compact_csv.read_text(encoding="utf-8-sig")
    assert "紫外线色斑,2" not in porphyrin.compact_csv.read_text(encoding="utf-8-sig")
    assert uv.medical_csv.read_bytes() != porphyrin.medical_csv.read_bytes()
    assert "荧光UV紫质" not in uv.medical_csv.read_text(encoding="utf-8-sig")
    assert "UV紫外线色斑" not in porphyrin.medical_csv.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("header_only_field", ["量化CSV", "医学V2CSV"])
def test_export_rejects_header_only_csv(
    tmp_path: Path, header_only_field: str
) -> None:
    image = tmp_path / "main.jpg"
    compact = tmp_path / "compact.csv"
    medical = tmp_path / "medical.csv"
    metrics = tmp_path / "metrics.json"
    image.write_bytes(b"image")
    compact.write_text("指标,单位,总计\n数量,个,1\n", encoding="utf-8-sig")
    medical.write_text("检测范围,评估状态\n全面部,可评估\n", encoding="utf-8-sig")
    metrics.write_text('{"总计": 1}', encoding="utf-8")
    sources = {"量化CSV": compact, "医学V2CSV": medical}
    sources[header_only_field].write_text("列A,列B\n", encoding="utf-8-sig")

    with pytest.raises(FileNotFoundError, match="不存在或为空"):
        export_item_documents(
            "redness",
            {
                "主结果图": str(image),
                "量化CSV": str(compact),
                "医学V2CSV": str(medical),
                "量化JSON": str(metrics),
            },
            tmp_path / "output",
        )


def test_final_output_rejects_wrinkle_without_all_formal_images(
    tmp_path: Path,
) -> None:
    source = tmp_path / "wrinkle"
    source.mkdir()
    image = source / "main.jpg"
    metrics = source / "metrics.json"
    image.write_bytes(b"stage2")
    metrics.write_text(
        json.dumps({
            "region_analysis_status": "ok",
            "region_metrics": [{
                "region_name": "额头纹", "area_px": 1000, "segment_count": 2,
                "wrinkle_pixels": 50, "mean_segment_length": 25,
                "max_segment_length": 30, "relative_score": 80,
            }],
            "stage2_recommended_pixels": 100,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="皱纹.*正式结果图"):
        write_final_output(
            tmp_path / "output",
            items={"wrinkle": {"主结果图": str(image), "量化JSON": str(metrics)}},
            input_signature={},
            provenance={},
            quality_control={},
            timing={},
        )


@pytest.mark.parametrize("invalid_field", ["量化CSV", "医学V2CSV"])
def test_export_rejects_nonempty_unknown_csv_schema(
    tmp_path: Path, invalid_field: str
) -> None:
    image = tmp_path / "main.jpg"
    compact = tmp_path / "compact.csv"
    medical = tmp_path / "medical.csv"
    metrics = tmp_path / "metrics.json"
    image.write_bytes(b"image")
    compact.write_text("指标,单位,总计\n数量,个,1\n", encoding="utf-8-sig")
    medical.write_text("检测范围,评估状态\n全面部,可评估\n", encoding="utf-8-sig")
    metrics.write_text('{"总计": 1}', encoding="utf-8")
    sources = {"量化CSV": compact, "医学V2CSV": medical}
    sources[invalid_field].write_text("foo,bar\n1,2\n", encoding="utf-8-sig")

    with pytest.raises(FileNotFoundError, match="CSV.*表头"):
        export_item_documents(
            "redness",
            {
                "主结果图": str(image),
                "量化CSV": str(compact),
                "医学V2CSV": str(medical),
                "量化JSON": str(metrics),
            },
            tmp_path / "output",
        )


def test_purple_export_rejects_empty_metrics_after_prefix_filter(
    tmp_path: Path,
) -> None:
    image = tmp_path / "main.png"
    compact = tmp_path / "purple.csv"
    medical = tmp_path / "purple_medical.csv"
    metrics = tmp_path / "purple.json"
    image.write_bytes(b"image")
    compact.write_text("检测项目,总计\n紫外线色斑,2\n紫质,3\n", encoding="utf-8-sig")
    medical.write_text(
        "检测项目,评估状态\n标准化UV紫外线色斑工程代理,可评估\n"
        "标准化荧光UV紫质工程代理,可评估\n",
        encoding="utf-8-sig",
    )
    metrics.write_text('{"porphyrin_total": 3}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="UV色斑.*量化 JSON"):
        export_item_documents(
            "uv_spots",
            {
                "主结果图": str(image),
                "量化CSV": str(compact),
                "医学V2CSV": str(medical),
                "量化JSON": str(metrics),
            },
            tmp_path / "output",
        )


def test_final_output_rejects_duplicate_wrinkle_media(tmp_path: Path) -> None:
    source = tmp_path / "wrinkle"
    source.mkdir()
    image = source / "main.jpg"
    metrics = source / "metrics.json"
    for name in (
        "07_干燥性细纹全脸分区结果图.jpg",
        "08_稳定性线性皱纹全脸分区结果图.jpg",
        "09_结构性沟纹全脸分区结果图.jpg",
    ):
        (source / name).write_bytes(b"duplicate")
    metrics.write_text(
        json.dumps({
            "region_analysis_status": "ok",
            "region_metrics": [{
                "region_name": "额头纹", "area_px": 1000, "segment_count": 2,
                "wrinkle_pixels": 50, "mean_segment_length": 25,
                "max_segment_length": 30, "relative_score": 80,
            }],
            "stage2_recommended_pixels": 100,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="皱纹.*内容重复"):
        write_final_output(
            tmp_path / "output",
            items={
                "wrinkle": {
                    "主结果图": str(source / "07_干燥性细纹全脸分区结果图.jpg"),
                    "量化JSON": str(metrics),
                },
            },
            input_signature={},
            provenance={},
            quality_control={},
            timing={},
        )
