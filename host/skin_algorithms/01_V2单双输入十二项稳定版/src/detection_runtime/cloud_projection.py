from __future__ import annotations

"""Project local clinic artifacts to the existing DermaVision pipeline contract."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from src.added_algorithms.public_metrics import write_gloss_metrics, write_vascular_metrics
from src.nine_analysis.metrics import extract_item
from src.nine_analysis.public_sanitize import sanitize_public_document


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _scoring_nodes(items):
    nodes = {}
    for item_id, item in items.items():
        raw = load_json(item.get("完整量化JSON") or item["量化JSON"])
        metrics = sanitize_public_document(raw)
        if item_id == "redness":
            # Exactly the same compact summary appended by the local finalizer.
            metrics["runtime_summary"] = sanitize_public_document(extract_item(item_id, raw)[0])
        nodes[item_id] = {"metrics": metrics}
    return nodes


def project_clinic_result(algorithm, items, prepared, root, receipt):
    root.mkdir(parents=True, exist_ok=True)
    output, medical = {}, {}
    if algorithm == "purple":
        metrics, rows = {}, []
        for item_id, label in (("uv_spots", "紫外线色斑"), ("porphyrin", "紫质")):
            item = items[item_id]
            with Path(item["量化CSV"]).open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.reader(handle))
            labels = ("total", "forehead", "left_cheek", "right_cheek", "nose", "chin")
            if len(csv_rows) != 2 or len(csv_rows[1]) != 6:
                raise ValueError("invalid clinic purple count CSV")
            values = [int(v) for v in csv_rows[1]]
            metrics.update({f"{item_id}_{key}": value for key, value in zip(labels, values)})
            rows.append([label, *values])
        path = root / "紫区量化指标.json"
        path.write_text(json.dumps(metrics), encoding="utf-8")
        csv_path = root / "紫区量化指标.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["检测项目", "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"])
            writer.writerows(rows)
        output = dict(purple_metrics=str(path), purple_report=str(csv_path),
                      purple_uv_base=items["uv_spots"]["附加结果图"][0],
                      purple_uv_spots_overlay=items["uv_spots"]["主结果图"],
                      purple_fluorescence_base=items["porphyrin"]["附加结果图"][0],
                      purple_porphyrin_overlay=items["porphyrin"]["主结果图"])
        quality = prepared.preprocessed["365_M"]
    else:
        item = items[algorithm]
        output = {algorithm: item["主结果图"], f"{algorithm}_metrics": item["量化JSON"],
                  f"{algorithm}_report": item["量化CSV"]}
        medical[algorithm] = dict(json=item["量化JSON"], csv=item["医学V2CSV"])
        quality = prepared.preprocessed["PP_M" if algorithm == "surface_gloss" else "CP_M"]
        if algorithm in {"redness", "brown"}:
            red = algorithm == "redness"
            bases = item.get("附加结果图", [])
            if len(bases) != 1 or not Path(bases[0]).is_file():
                raise ValueError("missing clinic red/brown base image")
            # Public contract: base first, boundary + markers second. Reuse both
            # approved local artifacts; do not render a different third image.
            output[algorithm] = str(bases[0])
            output["red_areas_overlay" if red else "brown_spots_overlay"] = item["主结果图"]
        elif algorithm == "surface_gloss":
            raw = load_json(Path(item["主结果图"]).parent / "表面油光量化指标.json")
            full = raw["full_face"]
            result = SimpleNamespace(
                valid_skin_area_px=full["valid_skin_area_px"],
                gloss_area_ratio=full["gloss_area_ratio"], patch_count=full["patch_count"],
                region_metrics=raw["region_metrics"], metrics=lambda: raw,
            )
            paths = write_gloss_metrics(result, root)
            output.update(surface_gloss_metrics=str(paths.metrics_json),
                          surface_gloss_report=str(paths.compact_csv),
                          surface_gloss_medical_report_csv_v2=item["医学V2CSV"])
        elif algorithm == "vascular":
            raw = load_json(Path(item["主结果图"]).parent / "血管样结构量化指标.json")
            if not raw.get("qc_passed") or raw.get("vascular_count") is None:
                raise ValueError("clinic vascular measurement is unassessable")
            paths = write_vascular_metrics(raw, root)
            output.update(vascular_metrics=str(paths.metrics_json),
                          vascular_report=str(paths.compact_csv),
                          vascular_medical_report_csv_v2=item["医学V2CSV"])
    return {
        "status": "success", "results": output, "medical_v2_results": medical,
        # Same per-detector truth consumed by the local institution finalizer.
        # This private pipeline field is pruned by the pinned scoring field list;
        # it is not copied wholesale to the public Worker envelope.
        "scoring_detector_results": _scoring_nodes(items),
        "metadata": {
            "quality_score": quality.quality_score, "quality_status": quality.quality_status,
            "quality_flags": list(quality.quality_flags),
            "timing_seconds": receipt["timing_seconds"],
            "capture_profile": "institution",
        },
        # The surrounding capture task owns and removes the temporary tree.
        "cleanup_paths": [],
    }
