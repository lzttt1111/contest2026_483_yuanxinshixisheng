"""Current phenotype state comes from current primary measurements, not legacy extras."""
import math


def phenotype_state(module, rows, assessment, zero_states):
    if module + ":full_face" in zero_states or assessment.get("zero_state"):
        return False
    values = {(r["metric_id"].split(".", 1)[1], r["region"]): r.get("value")
              for r in rows if r["module"] == module and r.get("status") == "measured"}
    keys = {
        "01": [("density", "full_face")],
        "02": [("gloss_area", "full_face"), ("porphyrin_p90", "full_face")],
        "03": [(layer + ".area", "full_face") for layer in ("spots", "brown", "uv")],
        "04": [("area", "full_face")], "05": [("clusters", "full_face")],
        "06": [(name + "_count", "full_face") for name in ("erythema", "papule", "pustule")],
        "07": [("area", "full_face")],
        "08": [("main_count", "forehead"), ("main_count", "glabella"),
               ("area", "crow_feet"), ("area", "perioral")],
        "09": [("volume", "full_face")],
        "10": [("raised_area", "full_face"), ("depressed_area", "full_face")],
    }
    if module == "11":
        return assessment.get("grade") != "未见明显" if assessment.get("score") is not None else None
    measured = [values.get(key) for key in keys[module]]
    finite = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if any(finite(v) and v > 0 for v in measured):
        return True
    if module not in ("09", "10") and all(finite(v) and v == 0 for v in measured):
        return False
    return None
