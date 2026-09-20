"""Build one report truth shared by the two PDF editions."""
from datetime import datetime, timezone
from . import BUNDLE_VERSION, SPEC_VERSION
from .registry import MODULES
from .regions import ROI_VERSION
from .scoring import aggregate, score_metric
from .history import compare

def build_report(subject_id, profile, measurements, media, *, captured_at=None, references=None):
    references = references or {}
    scores = {}
    for m in measurements:
        key = m["metric_id"] + ":" + m["region"]
        scores[key] = score_metric(m, references.get(key))
    modules = []
    missing = {"score_raw": None}
    for definition in MODULES:
        regions = {}
        sub_scores = {}
        for region in ("full_face", *definition.regions):
            if definition.id == "08":
                if region == "full_face":
                    continue
                weights = {"main_count":.30,"length_burden":.35,"depth":.35} if region in ("forehead","glabella") else {"area":.30,"density":.25,"high_area":.25,"contrast_p50":.20}
                regions[region] = aggregate({n:scores.get("08."+n+":"+region,missing) for n in weights}, weights)
                continue
            if definition.id == "03":
                components = {
                    name: aggregate({n:scores.get("03."+name+"."+n+":"+region,missing) for n in ("area","p90")}, {"area":.55,"p90":.45})
                    for name in ("spots","brown","uv")}
                sub_scores[region] = dict(components)
            else:
                components = {n:scores.get(definition.id+"."+n+":"+region,missing) for n in definition.weights}
            if definition.id == "02":
                sub_scores[region] = {
                    "surface_gloss":aggregate({n:components[n] for n in ("gloss_area","gloss_high_area","gloss_mean")},
                                             {"gloss_area":.25/.60,"gloss_high_area":.20/.60,"gloss_mean":.15/.60}),
                    "porphyrin":aggregate({n:components[n] for n in ("porphyrin_high_density","porphyrin_p90")},
                                         {"porphyrin_high_density":.70,"porphyrin_p90":.30})}
            weights = definition.weights
            if definition.id == "07":
                matching = [m for m in measurements if m["metric_id"]=="07.contrast_p50" and m["region"]==region]
                if matching and matching[0].get("reason")=="high_precision_not_confirmed":
                    weights = {k:v/.85 for k,v in definition.weights.items() if k!="contrast_p50"}
                    components = {k:components[k] for k in weights}
            regions[region] = aggregate(components, weights)
        if definition.id == "08":
            regions["full_face"] = aggregate({name:regions[name] for name in definition.weights}, definition.weights)
        modules.append({"id": definition.id, "title": definition.title, "note": definition.note,
                        "score": regions["full_face"], "regions": regions,
                        "sub_scores":sub_scores,
                        "metric_keys": [m["metric_id"]+":"+m["region"] for m in measurements if m["module"] == definition.id]})
    payload = {"schema_version": BUNDLE_VERSION, "spec_version": SPEC_VERSION, "subject_id": subject_id,
               "capture_profile": profile, "captured_at": captured_at,
               "generated_at": datetime.now(timezone.utc).isoformat(), "roi_version": ROI_VERSION,
               "scale_version": "standardized_pixels_v1", "calibration_version": "uncalibrated" if not references else "explicit_references",
               "modules": modules, "measurements": measurements, "score_trace": scores, "media": media}
    payload["history"] = compare(payload)
    return payload
