"""Doctor-required inputs, separate from optional display and compatibility fields."""
from .registry import MODULES, METRIC_LABELS, REGION_LABELS, stage1_weights

SCHEMA_VERSION = "doctor_v3_scoring_sample_v1"
VERSIONS = {"data": "v3-stage1-2", "roi": "doctor-v3-roi-4", "formula": "v3-stage1-formula-2"}


def version_identity(thresholds=None,reference=None):
    from .stage1_config import digest, resolve_config, formula_identity, scoring_identity
    config = resolve_config(thresholds)
    return {**VERSIONS, "thresholds": digest(config), "implementation": formula_identity(config), "scoring": scoring_identity(),
            "reference": digest(reference) if reference else "none"}


def input_identity(hashes,profile):
    if profile=="consumer" and isinstance(hashes,dict) and len(hashes)==1:
        return {"RGB_M":next(iter(hashes.values()))}
    return hashes


def required_items(profile, configuration=None):
    if profile not in ("consumer", "institution"):
        raise ValueError("explicit capture profile required")
    result = []
    for module in MODULES:
        for region in ("full_face", *module.regions):
            keys = list(stage1_weights(module.id))
            if module.id == "07" and configuration is not None and configuration.get("imaging", {}).get("high_precision_texture") is not True:
                keys.remove("contrast_p50")
            if module.id == "03":
                keys = [layer + "." + key for layer in ("spots", "brown", "uv") for key in ("area", "p90")]
            elif module.id == "08":
                if region == "full_face":
                    continue  # Its four-type aggregation is not a new measurement.
                keys = (["main_count", "length_burden", "depth"] if region in ("forehead", "glabella")
                        else ["area", "density", "high_area", "contrast_p50"])
            elif module.id == "11" and region != "full_face":
                keys = (["jaw_continuity"] if region.endswith("_jaw") else
                        ["smoothness", "turning", "jowl"] if region.endswith("_lower_face")
                        else ["smoothness", "turning"])
            for key in keys:
                result.append({"metric_id": module.id + "." + key, "module": module.id,
                               "region": region, "capture_profile": profile,
                               "title": module.title, "region_name": REGION_LABELS.get(region, region),
                               "name": METRIC_LABELS.get(key, key)})
    return result


def key_of(row):
    return row["metric_id"] + ":" + row["region"]


def coverage(rows, scores, profile, configuration=None):
    indexed = {key_of(row): row for row in rows}
    result = []
    for required in required_items(profile, configuration):
        key = key_of(required)
        row = indexed.get(key)
        scoring = scores.get(key, {})
        if row is None:
            state, reason = "implementation_missing", "required_row_not_emitted"
        elif row.get("value") is not None:
            state = "measured_scored" if scoring.get("score") is not None else "measured_needs_reference"
            reason = scoring.get("reference_status", "reference_missing")
        else:
            reason = row.get("reason") or "missing_reason"
            state = row.get("measurement_status", row.get("status", "unavailable"))
            if reason in ("missing_v3_measurement_evidence", "required_row_not_emitted", "missing_reason"):
                state = "implementation_missing"
        result.append({**required, "key": key, "state": state, "reason": reason})
    return result
