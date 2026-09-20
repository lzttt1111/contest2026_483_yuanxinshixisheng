"""Explicit prior-result comparisons. No guessed identity or clinical change threshold."""
from datetime import datetime

def compare(current, previous=None):
    if previous is None:
        return {"status": "no_history", "reason": "首次检测或暂无可比历史记录", "items": []}
    if not current.get("subject_id") or current["subject_id"] != previous.get("subject_id"):
        raise ValueError("previous result belongs to a different or unidentified subject")
    reasons = []
    for key in ("spec_version", "capture_profile", "scale_version", "roi_version", "calibration_version"):
        if not current.get(key) or current[key] != previous.get(key):
            reasons.append(key + "_mismatch")
    try:
        now = datetime.fromisoformat(current["captured_at"])
        before = datetime.fromisoformat(previous["captured_at"])
        if now.tzinfo is None or before.tzinfo is None or before >= now:
            reasons.append("capture_time_not_comparable")
    except (KeyError, ValueError, TypeError):
        reasons.append("missing_capture_time")
    old = {m["metric_id"] + ":" + m["region"]: m for m in previous["measurements"]}
    rows = []
    for item in current["measurements"]:
        prior = old.get(item["metric_id"] + ":" + item["region"])
        local = list(reasons)
        if prior is None:
            local.append("missing_previous_metric")
        else:
            for key in ("unit", "definition_version", "source_kind"):
                if prior.get(key) != item.get(key):
                    local.append(key + "_mismatch")
            if prior.get("status") != "measured" or item.get("status") != "measured":
                local.append("measurement_unavailable")
        row = {"metric_id": item["metric_id"], "region": item["region"], "current": item.get("value"),
               "unit":item["unit"],
               "previous": prior.get("value") if prior else None, "delta": None,
               "status": "not_comparable", "reasons": local}
        if not local:
            row.update(delta=item["value"] - prior["value"], status="comparable",
                       interpretation="仅展示数值变化，变化意义待确认")
        rows.append(row)
    return {"status": "comparable" if rows and all(r["status"] == "comparable" for r in rows) else "partial_or_not_comparable",
            "reason": "按同一指标口径逐项核对", "items": rows}
