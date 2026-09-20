"""Keep actual V3 measurements distinct from intermediate historical views."""
from copy import deepcopy


def restore(model, measurements):
    index = {(x["metric_id"], x["region"]): x for x in measurements}
    for mid, module in model["modules"].items():
        for region, metrics in module["metrics"].items():
            for key, item in metrics.items():
                if "native_v3_measurement" not in item:
                    continue
                actual = index.get((mid + "." + key, region))
                if actual is None:
                    actual = {"value": None, "unit": None, "status": "unavailable",
                              "reason": "no_native_v3_measurement"}
                native = {k: deepcopy(actual.get(k)) for k in
                          ("value", "unit", "status", "reason", "source_kind")}
                native.update(score=None, grade=None)
                trace = model.get("score_trace", {}).get(mid + "." + key + ":" + region, {})
                for node in (item, trace):
                    old = node.get("native_v3_measurement")
                    if old is not None and (old.get("value"), old.get("unit")) != (native["value"], native["unit"]):
                        node.setdefault("previous_historical_measurement", deepcopy(old))
                    node["native_v3_measurement"] = deepcopy(native)
    return model
