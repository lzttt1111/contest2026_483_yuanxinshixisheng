"""Merge fresh RGB/wrinkle ROI measurements with explicitly retained old heads."""
from copy import deepcopy
import hashlib
import numpy as np
from .stage1_config import digest, resolve_config
from .stage1_schema import SCHEMA_VERSION, VERSIONS, key_of, required_items, version_identity, coverage
from .stage1_scoring import score_measurements
from .stage1_zero_state import summaries, verify_saved_states
from .stage1_lines import analyze as analyze_lines
from .stage1_geometry import analyze as analyze_geometry

VERSION = "v301-roi-repair-merge-1"
FRESH_MODULES = frozenset(("07", "08", "09", "11"))
CACHED_MODULES = ("01", "02", "03", "04", "05", "06", "10")
IDENTITY_FIELDS = ("roi_version", "formula_identity", "threshold_identity", "measurement_identity")


class ROIRepairError(ValueError):
    """The retained or fresh evidence cannot support an isolated ROI repair."""


def _require(condition, reason):
    if not condition:
        raise ROIRepairError(reason)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _input(value):
    result = value.get("RGB_M") if isinstance(value, dict) and set(value) == {"RGB_M"} else value
    _require(_sha(result), "invalid_single_rgb_identity")
    return result


def _array_content(value):
    if isinstance(value, np.ndarray):
        _require(not value.dtype.hasobject, "object_array_not_supported")
        array = np.ascontiguousarray(value)
        return {"shape": list(array.shape), "dtype": array.dtype.str,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {k: _array_content(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_array_content(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _references(value, field=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _references(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item, field)
    elif isinstance(value, str) and (value.startswith("evidence:") or field.endswith(("_ref", "_refs"))):
        yield value


def _copy_basis(key, source, output, active=None):
    active = set() if active is None else active
    _require(key not in active, "cyclic_basis_reference:" + key)
    _require(key in source and isinstance(source[key], dict), "missing_basis_reference:" + key)
    value = source[key]
    if key in output:
        _require(digest(output[key]) == digest(value), "conflicting_basis_reference:" + key)
    else:
        output[key] = deepcopy(value)
    for reference in _references(value):
        _copy_basis(reference, source, output, active | {key})


def _unsealed(row):
    return {k: v for k, v in row.items() if k not in IDENTITY_FIELDS}


def repair_payload(previous, data, metadata, *, input_sha256, evidence_provenance):
    """Return a new unscored ROI4 payload, without mutating inputs or writing files.

    provenance requires capture_profile, input_sha256 and evidence_sha256 with
    real saved RGB/wrinkle file hashes. File hashes are caller attestations;
    array-content and metadata hashes are independently computed here.
    """
    _require(previous.get("schema_version") == SCHEMA_VERSION, "unsupported_payload_schema")
    _require(previous.get("capture_profile") == "consumer", "consumer_profile_required")
    identity = _input(input_sha256)
    _require(_input(previous.get("input_sha256")) == identity, "input_identity_mismatch")
    provenance = deepcopy(evidence_provenance)
    _require(isinstance(provenance, dict) and provenance.get("capture_profile") == "consumer", "fresh_profile_mismatch")
    _require(_input(provenance.get("input_sha256")) == identity, "fresh_input_mismatch")
    hashes = provenance.get("evidence_sha256", {})
    _require(isinstance(hashes, dict) and set(hashes) == {"rgb", "wrinkle"}
             and all(_sha(v) for v in hashes.values()), "fresh_evidence_hashes_required")
    _require(set(data) == {"rgb", "wrinkle"}, "only_fresh_rgb_and_wrinkle_allowed")
    _require(set(metadata) <= {"rgb", "wrinkle"}, "unexpected_fresh_metadata")
    for name in data:
        _require(isinstance(data[name], dict), "invalid_fresh_arrays:" + name)
        for field in ("valid", "landmarks", "image") + (("relative_z",) if name == "rgb" else ("skeleton",)):
            _require(isinstance(data[name].get(field), np.ndarray), "missing_fresh_array:" + name + ":" + field)
        meta = metadata.get(name, {})
        _require(meta.get("capture_profile", "consumer") == "consumer", "metadata_profile_mismatch")
        if "input_sha256" in meta:
            _require(_input(meta["input_sha256"]) == identity, "metadata_input_mismatch")
    versions_old = previous.get("versions", {})
    _require(versions_old.get("roi") == "doctor-v3-roi-3"
             and versions_old.get("formula") == "v3-stage1-formula-2"
             and versions_old.get("data") == VERSIONS["data"], "unexpected_previous_versions")
    _require(versions_old.get("reference") == "none" and previous.get("reference_purpose") is None,
             "previous_reference_must_be_disabled")
    _require(isinstance(previous.get("scores"), dict) and all(
        isinstance(s, dict) and s.get("score") is None and s.get("score_raw") is None
        for s in previous["scores"].values()), "previous_scores_must_be_disabled")
    config = deepcopy(previous.get("measurement_config"))
    _require(isinstance(config, dict) and resolve_config(config) == config, "previous_config_not_resolved")
    _require(versions_old.get("thresholds") == digest(config) and _sha(versions_old.get("implementation")),
             "invalid_previous_configuration_identity")
    old_rows, old_basis = previous.get("measurements"), previous.get("basis")
    _require(isinstance(old_rows, list) and isinstance(old_basis, dict), "missing_previous_measurements")
    indexed_old = {}
    for row in old_rows:
        key = key_of(row)
        _require(key not in indexed_old, "duplicate_previous_measurement:" + key)
        _require(row.get("capture_profile") == "consumer" and row.get("module") == row["metric_id"].split(".")[0], "previous_row_profile_or_module_mismatch")
        _require(row.get("roi_version") == versions_old["roi"] and row.get("formula_identity") == versions_old["implementation"]
                 and row.get("threshold_identity") == versions_old["thresholds"]
                 and row.get("definition_version", "").startswith(versions_old["formula"] + "/")
                 and row.get("measurement_identity") == key + "|" + row["definition_version"] + "|" + versions_old["implementation"] + "|" + versions_old["thresholds"],
                 "previous_row_identity_mismatch:" + key)
        indexed_old[key] = row
    _require({key_of(r) for r in required_items("consumer", config)} <= set(indexed_old), "incomplete_previous_required_rows")
    verify_saved_states(previous)
    fresh_data, fresh_meta = deepcopy(data), deepcopy(metadata)
    line = analyze_lines(fresh_data, fresh_meta, "consumer", previous.get("registration"), config)
    geometry = analyze_geometry({**fresh_data, "_stage1_line_arrays": line.get("arrays", {})}, fresh_meta,
                                "consumer", previous.get("registration"), config)
    selected = {}
    for produced in (line, geometry):
        for row in produced["measurements"]:
            if row["module"] not in FRESH_MODULES:
                continue
            key = key_of(row)
            if key in selected:
                _require(row["metric_id"] == "08.depth", "duplicate_fresh_measurement:" + key)
                if row.get("value") is None and selected[key][0].get("value") is not None:
                    continue
            selected[key] = (deepcopy(row), produced["basis"])
    expected = {key_of(r) for r in required_items("consumer", config) if r["module"] in FRESH_MODULES}
    _require(expected <= set(selected), "fresh_required_rows_missing")
    basis, cached_basis, rows = {}, {}, []
    for key, row in indexed_old.items():
        if row["module"] in CACHED_MODULES:
            _copy_basis(key, old_basis, cached_basis)
            rows.append(deepcopy(row))
    basis.update(deepcopy(cached_basis))
    for key, (row, producer_basis) in sorted(selected.items()):
        _copy_basis(key, producer_basis, basis)
        producer_version = row.get("definition_version", "unspecified")
        row["definition_version"] = VERSIONS["formula"] + "/" + row.get("source_kind", "measured") + "/" + producer_version
        row["measurement_status"] = row.get("measurement_status", producer_basis.get(key, {}).get("measurement_status", row["status"]))
        rows.append(row)
    versions = version_identity(config, None)
    _require(versions["roi"] == "doctor-v3-roi-4", "current_roi4_required")
    for row in rows:
        key = key_of(row)
        row.update(roi_version=versions["roi"], formula_identity=versions["implementation"], threshold_identity=versions["thresholds"])
        row["measurement_identity"] = key + "|" + row["definition_version"] + "|" + versions["implementation"] + "|" + versions["thresholds"]
    rows.sort(key=key_of)
    old_cached = sorted((_unsealed(r) for r in old_rows if r["module"] in CACHED_MODULES), key=key_of)
    new_cached = [_unsealed(r) for r in rows if r["module"] in CACHED_MODULES]
    _require(digest(old_cached) == digest(new_cached), "cached_measurements_changed")
    _require(all(basis[k] == v for k, v in cached_basis.items()), "cached_basis_changed")
    scores = score_measurements(rows, None, basis=basis)
    result = deepcopy(previous)
    result.update(measurements=rows, basis=basis, versions=versions, measurement_config=config,
                  scores=scores, zero_states=summaries(rows, basis), coverage=coverage(rows, scores, "consumer", config),
                  reference_purpose=None, roi_repair={"version": VERSION, "source_payload_sha256": digest(previous),
                  "previous_versions": deepcopy(versions_old), "cached_modules": list(CACHED_MODULES),
                  "recomputed_modules": sorted(FRESH_MODULES), "cached_measurements_sha256": digest(old_cached),
                  "cached_basis_sha256": digest(cached_basis), "cached_values_and_basis_unchanged": True,
                  "cached_heads_reinferred": False, "cached_raw_condition": "same_original_input_and_retained_original_head_evidence",
                  "fresh_vs_original_raw_pixel_equality_verified": False,
                  "module10_condition": "retains_original_raw_texture_and_original_line_exclusion; not recomputed with fresh lines",
                  "input_sha256": identity, "fresh_evidence_provenance": provenance,
                  "fresh_array_content_sha256": digest(_array_content(data)), "fresh_metadata_sha256": digest(metadata),
                  "stale_scores_discarded": True, "reference_enabled": False})
    return result
