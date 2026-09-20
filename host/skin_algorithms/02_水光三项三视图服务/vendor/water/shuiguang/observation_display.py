"""Presentation accounting, preserving every single-view observation without cross-view addition."""
from .config import MODULES,VIEWS,REGIONS
# VISIA-based detectors do not independently measure an eye-region endpoint.
# Keep any peripheral observations in the unlisted-position account, not a new eye claim.
NO_EYE_REGIONS={"pores","texture","spots","brown","redness"}

def build_display_account(data,statistics):
    output={}
    for module in MODULES:
        names=[r for r in REGIONS if not (module in NO_EYE_REGIONS and r.endswith("_eye"))
               and any(statistics["views"].get(v,{}).get("modules",{}).get(module,{}).get("regions",{}).get(r,{}).get("status")=="available" for v in VIEWS)]
        views={}
        for v in VIEWS:
            original=data["views"][v].get("modules",{}).get(module,{})
            item=statistics["views"].get(v,{}).get("modules",{}).get(module,{})
            if original.get("status")!="success":
                views[v]={"status":"unavailable","total_count":None,"unlisted_count":None,"regional_counts":{}}
                continue
            total=len(original.get("instances",[]))
            counts={r:item.get("regions",{}).get(r,{}).get("count") for r in names}
            listed=sum(value for value in counts.values() if value is not None)
            if total!=original["metrics"]["count"] or listed>total:
                raise ValueError("图像目标数量与分区统计无法对账")
            represented={oid for r in names for oid in item.get("regions",{}).get(r,{}).get("observation_ids",[])}
            remaining=[o["observation_id"] for o in item.get("observations",[]) if o["observation_id"] not in represented]
            assert len(remaining)==total-listed
            views[v]={"status":"available","total_count":total,"regional_counts":counts,
                      "listed_count":listed,"unlisted_count":total-listed,
                      "unlisted_observation_ids":remaining,"reconciliation_ok":True}
        output[module]={"visible_regions":names,"views":views,"whole_face_unique_count":None,
                        "policy":"each_column_matches_its_own_image; total=listed+unlisted; never_sum_views"}
    return {"version":"same_view_display_account_v1","modules":output}

