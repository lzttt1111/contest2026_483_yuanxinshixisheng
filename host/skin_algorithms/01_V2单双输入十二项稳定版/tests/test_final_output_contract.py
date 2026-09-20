from __future__ import annotations

import json
from pathlib import Path

from src.nine_analysis.final_output import write_final_output


ITEMS = (
    "redness", "spots", "brown", "texture", "pores", "uv_spots",
    "porphyrin", "wrinkle", "acne", "surface_gloss", "vascular",
    "contour_firmness",
)


def _item(root: Path, key: str) -> dict[str, str]:
    item_root = root / key
    item_root.mkdir(parents=True)
    image = item_root / (
        "07_干燥性细纹全脸分区结果图.jpg"
        if key == "wrinkle"
        else "main.jpg"
    )
    metrics = item_root / "metrics.json"
    compact = item_root / "compact.csv"
    medical = item_root / "medical.csv"
    image.write_bytes(b"image")
    extra_names = {
        "redness": "06_VISIA红色区实例图.jpg",
        "brown": "02_VISIA棕色斑实例图.jpg",
        "uv_spots": "01_紫外线色斑底图.png",
        "porphyrin": "03_紫质荧光底图.png",
    }
    if key in extra_names:
        (item_root / extra_names[key]).write_bytes(b"extra-image")
    if key == "wrinkle":
        (item_root / "08_稳定性线性皱纹全脸分区结果图.jpg").write_bytes(b"wrinkle-stable")
        (item_root / "09_结构性沟纹全脸分区结果图.jpg").write_bytes(b"wrinkle-groove")
    payload: dict = {"总计": 1}
    if key in {"uv_spots", "porphyrin"}:
        payload = {
            **{f"uv_spots_{region}": 2 for region in ("total", "forehead", "left_cheek", "right_cheek", "nose", "chin")},
            **{f"porphyrin_{region}": 3 for region in ("total", "forehead", "left_cheek", "right_cheek", "nose", "chin")},
        }
    elif key == "surface_gloss":
        payload = {
            "油光面积占比": {name: 0.1 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")},
            "油光区域数量": {name: 2 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")},
        }
    elif key == "vascular":
        payload = {
            "血管样结构数量": {name: 2 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")},
            "血管样结构总长度": {name: 12.0 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")},
        }
    elif key == "contour_firmness":
        payload = {"中面部曲面连续性": 0.8, "下颌缘连续性": 0.7, "左右轮廓差异": 0.1}
    elif key == "acne":
        payload = {
            "detection": {
                "status": "ok",
                "input_mode": "full_face",
                "detection_scope": "global",
                "detections": [{"box_area": 25, "confidence": 0.9, "region": "forehead"}],
                "region_counts": {"forehead": 1},
            },
            "preprocess": {"skin_pixels": 1000, "face_count": 1},
            "grading": {"status": "ok", "severity_level": 1},
        }
    elif key == "wrinkle":
        payload = {
            "region_analysis_status": "ok",
            "region_metrics": [{
                "region_name": "额头纹", "area_px": 1000, "segment_count": 2,
                "wrinkle_pixels": 50, "mean_segment_length": 25,
                "max_segment_length": 30, "relative_score": 80,
            }],
            "stage2_recommended_pixels": 100,
            "weights": "/home/example/models/wrinkle.pt",
            "output_dir": "/tmp/runtime/wrinkle",
        }
    metrics.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if key in {"wrinkle", "acne"}:
        compact_text = "raw_column\nraw_value\n"
    elif key in {"uv_spots", "porphyrin"}:
        compact_text = "检测项目,总计\n紫外线色斑,2\n紫质,3\n"
    else:
        compact_text = "指标,单位,总计\n数量,个,1\n"
    compact.write_text(compact_text, encoding="utf-8-sig")
    medical_text = "检测范围,评估状态,有效皮肤面积（像素）\n全面部,可评估,100\n"
    if key in {"uv_spots", "porphyrin"}:
        medical_text = (
            "检测项目,评估状态\n标准化UV紫外线色斑工程代理,可评估\n"
            "标准化荧光UV紫质工程代理,可评估\n"
        )
    medical.write_text(medical_text, encoding="utf-8-sig")
    result = {
        "项目": key,
        "状态": "success",
        "主结果图": str(image),
        "量化JSON": str(metrics),
        "量化CSV": str(compact),
    }
    if key not in {"wrinkle", "acne"}:
        result["医学V2CSV"] = str(medical)
    return result


def test_final_output_keeps_mature_artifacts_with_each_item(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    items = {key: _item(runtime, key) for key in ITEMS}
    target = tmp_path / "sample"

    write_final_output(
        target,
        items=items,
        input_signature={
            "path": "/home/example/clinic28-25/RGB_M.jpg",
            "relative_path": "clinic28-25_RGB_M.jpg",
            "sha256": "a" * 64,
        },
        provenance={"route": "consumer_rgb"},
        quality_control={"status": "PASS"},
        timing={"wall_seconds": 25.0},
    )

    assert [path.name for path in target.iterdir() if path.is_dir()] == ["十二项检测"]
    folders = sorted(path for path in (target / "十二项检测").iterdir())
    assert len(folders) == 12
    for folder in folders:
        files = [path for path in folder.iterdir() if path.is_file()]
        assert sum(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in files) >= 1
        assert sum(path.suffix.lower() == ".json" for path in files) == 1
        assert sum(path.suffix.lower() == ".csv" for path in files) == 2
        assert sum(path.suffix.lower() == ".xlsx" for path in files) == 0
        csv_texts = [
            path.read_text(encoding="utf-8-sig")
            for path in files
            if path.suffix == ".csv"
        ]
        assert all(text.strip() for text in csv_texts)
        assert all("raw_column" not in text for text in csv_texts)
        public_json = next(path for path in files if path.suffix == ".json")
        assert json.loads(public_json.read_text(encoding="utf-8"))
    assert len(list((target / "十二项检测" / "01_红区").glob("*.jpg"))) == 2
    assert len(list((target / "十二项检测" / "03_棕区").glob("*.jpg"))) == 2
    assert len(list((target / "十二项检测" / "06_UV色斑").glob("*.*"))) == 5
    assert len(list((target / "十二项检测" / "07_卟啉").glob("*.*"))) == 5
    assert len(list((target / "十二项检测" / "08_皱纹").glob("*.jpg"))) == 3
    assert {path.name for path in target.iterdir() if path.is_file()} == {
        "十二项检测结果索引.json",
        "十二项完整量化指标.json",
        "运行回执.json",
    }
    complete = json.loads((target / "十二项完整量化指标.json").read_text(encoding="utf-8"))
    index = json.loads((target / "十二项检测结果索引.json").read_text(encoding="utf-8"))
    receipt = json.loads((target / "运行回执.json").read_text(encoding="utf-8"))
    assert index["capture_profile"] == "institution"
    assert complete["provenance"]["capture_profile"] == "institution"
    assert complete["provenance"]["route"] == "institution_vendor_rgb"
    assert receipt["capture_profile"] == "institution"
    assert all(
        len(image["sha256"]) == 64
        for item in index["items"].values()
        for image in item["images"]
    )
    wrinkle_item = index["items"]["wrinkle"]
    assert [Path(image["path"]).name for image in wrinkle_item["images"]] == [
        "01_皱纹检测结果图.jpg",
        "02_稳定性线性皱纹全脸分区结果图.jpg",
        "03_结构性沟纹全脸分区结果图.jpg",
    ]
    assert len({image["sha256"] for image in wrinkle_item["images"]}) == 3
    assert all(
        (target / item[field]).is_file()
        for item in index["items"].values()
        for field in ("量化CSV", "医学V2CSV", "量化JSON")
    )
    assert not list(target.rglob("*.xlsx"))
    for key in ("wrinkle", "acne"):
        public_json = json.loads(
            (target / index["items"][key]["量化JSON"]).read_text(encoding="utf-8")
        )
        assert list(public_json) == [
            "project", "metrics_version", "scoring_status", "imaging_and_units",
            "overall_metrics", "region_metrics", "left_right_comparison",
            "quality_control", "medical_limitations",
        ]
    assert list(complete) == [
        "schema_version", "medical_requirement_version", "input_signature",
        "provenance", "detector_results", "medical_v2_modules",
        "scoring_features", "scoring_results", "quality_control",
        "compatibility",
    ]
    assert list(complete["detector_results"]) == list(ITEMS)
    assert len(complete["medical_v2_modules"]) == 11
    assert any(
        not module["completeness"]["complete"]
        for module in complete["medical_v2_modules"].values()
    )
    assert len(complete["scoring_features"]) == 11
    assert complete["scoring_results"]["status"] == "uncalibrated"
    assert complete["scoring_results"]["scoring_profile_version"] == "production_proxy_v1"
    assert complete["scoring_results"]["score_status"] == "production_proxy"
    assert len(complete["scoring_results"]["module_scores"]) == 11
    assert complete["scoring_results"]["complete_proxy_inputs"] is False
    assert complete["scoring_results"]["calibration_boundary"]["population_calibrated"] is False
    assert all(
        module["score"] is None
        and module["grade"] == "不可评估"
        and module["score_valid"] is False
        for module in complete["scoring_results"]["module_scores"].values()
    )
    consumer_target = tmp_path / "consumer-sample"
    write_final_output(
        consumer_target,
        items=items,
        input_signature={"relative_path": "phone.jpg"},
        provenance={"route": "consumer_rgb", "capture_profile": "consumer"},
        quality_control={"status": "PASS"},
        timing={"wall_seconds": 10.0},
    )
    consumer = json.loads(
        (consumer_target / "十二项完整量化指标.json").read_text(encoding="utf-8")
    )
    assert consumer["scoring_results"]["scoring_profile_version"] == (
        "production_proxy_v1"
    )
    assert consumer["provenance"]["route"] == "consumer_rgb"
    assert consumer["scoring_results"]["complete_proxy_inputs"] is False
    uv_json = json.loads((target / index["items"]["uv_spots"]["量化JSON"]).read_text(encoding="utf-8"))
    porphyrin_json = json.loads((target / index["items"]["porphyrin"]["量化JSON"]).read_text(encoding="utf-8"))
    assert uv_json and all(key.startswith("uv_spots_") for key in uv_json)
    assert porphyrin_json and all(key.startswith("porphyrin_") for key in porphyrin_json)
    assert uv_json != porphyrin_json
    for path in target.rglob("*"):
        if path.suffix.lower() in {".json", ".csv"}:
            text = path.read_text(encoding="utf-8-sig")
            assert "/tmp/" not in text
            assert "/home/" not in text
