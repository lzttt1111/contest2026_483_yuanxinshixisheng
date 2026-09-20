"""Direction-safe scoring; absent calibration never becomes a fabricated score."""
from bisect import bisect_left, bisect_right
from decimal import Decimal, ROUND_HALF_UP
import math

def display_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("score must be finite and within [0,100]")
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def grade(value):
    if value is None:
        return "不可评估"
    n = display_score(value)
    return next(label for limit, label in ((20, "显著"), (40, "较明显"), (60, "中度"), (80, "轻度"), (100, "未见明显")) if n <= limit)

def score_metric(m, reference=None):
    trace = {"metric_id": m["metric_id"], "value": m.get("value"), "unit": m["unit"],
             "definition_version": m["definition_version"], "status": "uncalibrated",
             "score_raw": None, "score": None, "grade": "不可评估", "reference_version": None}
    if m.get("status") != "measured":
        trace.update(status="unavailable", reason=m.get("reason", "missing_measurement"))
        return trace
    if reference is None:
        trace["reason"] = "missing_compatible_reference"
        return trace
    for key in ("metric_id", "unit", "definition_version", "capture_profile", "region", "source_kind", "direction"):
        if reference.get(key) != m.get(key):
            raise ValueError("reference mismatch: " + key)
    values = reference.get("sorted_values", [])
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("invalid reference population")
    if values != sorted(values) or not reference.get("version") or not reference.get("population_sha256"):
        raise ValueError("reference must be sorted, versioned and attributable")
    value = m["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("non-finite measurement")
    rank = (bisect_left(values, value) + bisect_right(values, value)) / (2 * len(values))
    raw = 100 * (1 - rank)
    if value == 0 and values[0] >= 0:
        raw = 100.0
    trace.update(status="candidate", score_raw=raw, score=display_score(raw), grade=grade(raw),
                 reference_version=reference["version"], population_sha256=reference["population_sha256"],
                 method="inverse_midrank_ecdf", population_count=len(values))
    return trace

def aggregate(scores, weights):
    if set(scores) != set(weights) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError("incomplete score/weight mapping")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("invalid weight")
    if any(scores[k].get("score_raw") is None for k in weights):
        return {"score_raw": None, "score": None, "grade": "不可评估", "status": "incomplete",
                "missing": [k for k in weights if scores[k].get("score_raw") is None], "weights": weights}
    raw = sum(scores[k]["score_raw"] * w for k, w in weights.items())
    return {"score_raw": raw, "score": display_score(raw), "grade": grade(raw), "status": "candidate", "weights": weights}
