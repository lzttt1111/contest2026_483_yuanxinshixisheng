"""One result-only measurement pipeline shared by review and scoring output modes."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .measurements import read_evidence
from .stage1_schema import SCHEMA_VERSION, VERSIONS, required_items, key_of, coverage, version_identity, input_identity
from .stage1_storage import publish_result
from .stage1_scoring import score_measurements


def collect_evidence(source_root, runtime_items=None):
    source_root = Path(source_root)
    candidates = {}
    for path in source_root.rglob("doctor_v3_*.npz"):
        candidates.setdefault(path.stem.removeprefix("doctor_v3_"), set()).add(path.resolve())
    for project, item in (runtime_items or {}).items():
        parent = Path(item["主结果图"]).parent
        for folder in (parent, parent.parent):
            path = folder / ("doctor_v3_" + project + ".npz")
            if path.is_file():
                candidates.setdefault(project, set()).add(path.resolve())
    data, metadata, assets = {}, {}, {}
    for project, paths in candidates.items():
        if len(paths) != 1:
            raise ValueError("ambiguous current evidence: " + project)
        path = next(iter(paths))
        data[project], metadata[project] = read_evidence(path)
        assets["evidence/" + path.name] = path
        assets["evidence/" + path.with_suffix(".json").name] = path.with_suffix(".json")
    return data, metadata, assets


def analyze(data, metadata, *, subject_id, profile, input_sha256, registration=None,
            thresholds=None, reference=None, allow_test_reference=False):
    from .stage1_two_d import analyze as two_d
    from .stage1_lines import analyze as lines
    from .stage1_geometry import analyze as geometry
    from .stage1_config import resolve_config
    thresholds = resolve_config(thresholds)
    versions = version_identity(thresholds, reference)
    indexed, basis, arrays, visual = {}, {}, {}, {}
    for name, function in (("two_d", two_d), ("lines", lines), ("geometry", geometry)):
        result = function(data, metadata, profile, registration, thresholds)
        for row in result["measurements"]:
            key = key_of(row)
            if key in indexed:
                if row["module"] == "08" and row["metric_id"] == "08.depth":
                    if row.get("value") is not None or indexed[key].get("value") is None:
                        indexed[key] = row
                    continue
                raise ValueError("two producers emitted " + key)
            indexed[key] = row
        basis.update(result.get("basis", {}))
        for array_key, array in result.get("arrays", {}).items():
            if array_key in arrays:raise ValueError("duplicate derived evidence: "+array_key)
            arrays[array_key]=array
        if name=="lines":
            data={**data,"_stage1_line_arrays":result.get("arrays",{})}
        visual.update(result.get("visual_data", {}))
    for item in required_items(profile, thresholds):
        key = key_of(item)
        if key not in indexed:
            indexed[key] = {**item, "value": None, "unit": "标准化指数", "direction": "higher_burden",
                            "source_kind": "unavailable", "status": "unavailable",
                            "reason": "required_row_not_emitted"}
    rows = []
    for key, row in sorted(indexed.items()):
        row = dict(row)
        producer_version = row.get("definition_version", "unspecified")
        row["definition_version"] = VERSIONS["formula"] + "/" + row.get("source_kind", "measured") + "/" + producer_version
        row.update(roi_version=versions["roi"], formula_identity=versions["implementation"],
                   threshold_identity=versions["thresholds"])
        row["measurement_status"] = row.get("measurement_status", basis.get(key, {}).get("measurement_status", row["status"]))
        row["measurement_identity"] = key + "|" + row["definition_version"] + "|" + versions["implementation"] + "|" + versions["thresholds"]
        row.setdefault("direction", "higher_burden")
        rows.append(row)
    scored = score_measurements(rows, reference, allow_test_reference, basis=basis)
    from .stage1_zero_state import summaries
    payload = {"schema_version": SCHEMA_VERSION, "subject_id": subject_id, "capture_profile": profile,
               "input_sha256": input_identity(input_sha256,profile), "versions": versions,
               "measurement_config": thresholds,
               "measurements": rows, "basis": basis, "scores": scored,
               "zero_states": summaries(rows, basis),
               "coverage": coverage(rows, scored, profile, thresholds), "registration": registration or {},
               "reference_purpose": reference.get("purpose") if reference else None}
    return payload, arrays, visual


def configured_reference():
    import os
    source=os.environ.get("DERMAVISION_V3_SCORE_REFERENCE")
    if not source:return None
    from .stage1_runtime import require_output
    path=require_output(source)
    reference=json.loads(path.read_text(encoding="utf8"))
    if reference.get("purpose")!="formal":raise ValueError("test reference is prohibited for production reports")
    from .stage1_reference import validate_reference_qualification
    validate_reference_qualification(reference)
    return reference


def from_result(source_root, *, subject_id, profile, input_sha256, runtime_items=None,
                registration=None, thresholds=None, reference=None):
    data, metadata, assets = collect_evidence(source_root, runtime_items)
    if reference is None:reference=configured_reference()
    payload, derived, visual = analyze(data, metadata, subject_id=subject_id, profile=profile,
                                      input_sha256=input_sha256, registration=registration,
                                      thresholds=thresholds, reference=reference)
    if derived:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **derived)
        assets["evidence/stage1_derived.npz"] = buffer.getvalue()
    return payload, assets, data, visual


def save(destination, payload, assets=None, detailed=False):
    return publish_result(destination, payload, assets if detailed else None)
