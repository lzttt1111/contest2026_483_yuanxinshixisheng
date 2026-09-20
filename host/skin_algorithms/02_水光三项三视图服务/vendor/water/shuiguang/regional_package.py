"""Build regional statistics and concern-location data solely from a saved result package."""
from pathlib import Path
from types import SimpleNamespace
import json
import cv2
import numpy as np
from .anatomy import partition, rings, anchor, GROUPS, LEAVES, COLORS, VERSION
from .regional_statistics import module_observations, select_primary
from .measurement_partition import measurement_partition, VERSION as PARTITION_VERSION
from .config import VIEWS, MODULES, REGIONS
from .io import read_image, write_image, write_json, contained, sha256

def bbox(mask):
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return [x, y, x+w, y+h]

def original_box(box, inverse, shape):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    corners = cv2.transform(np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]), inverse)[0]
    lo = corners.min(axis=0); hi = corners.max(axis=0)
    return [max(0, int(lo[0])), max(0, int(lo[1])),
            min(shape[1], int(np.ceil(hi[0]))), min(shape[0], int(np.ceil(hi[1])))]

def cluster_rings(mask):
    # This is a display-only cluster envelope. Metrics use the unmodified source mask.
    clustered = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return rings(clustered)

def restore_effective_domain(module, item, saved, image, skin, points, flags):
    """Restore the exact geometric analysis restriction omitted by the old adapter.

    No response/target image is used to infer where the detector worked. The migrated
    engine's saved-geometry rule is used, and may only reduce the cached upper bound.
    """
    if module not in {"pores","texture","spots","brown"} or item.get("effective_domain_origin"):
        return saved.copy(), "saved_detector_domain"
    from sg_core.engines.visia_regions import build_visia_regions
    margin=2 if module=="spots" else 30
    work=image.copy()
    if module=="spots":work[skin==0]=0
    r=build_visia_regions(work,skin,points,flags,include_chin=True,mode="full",feature_margin_px=margin)
    valid=cv2.bitwise_and(saved,r.analysis_mask)
    return valid,"restored_original_visia_analysis_rule_margin_"+str(margin)

def enrich_result(root):
    root = Path(root).resolve()
    data = json.loads((root/"水光检测完整结果.json").read_text(encoding="utf-8"))
    presentation_path = root/"V2结果图索引.json"
    presentation = json.loads(presentation_path.read_text(encoding="utf-8")) if presentation_path.exists() else {}
    observed_views, map_views = {}, {}
    source_differences = []
    for view in VIEWS:
        v = data["views"][view]
        if not v.get("geometry"):
            observed_views[view] = {"modules": {}}
            map_views[view] = {"state": "uncovered", "reason": "本视角未取得几何证据", "regions": {}, "issues": {}}
            continue
        with np.load(contained(root, v["geometry"])) as geometry:
            skin = geometry["skin_mask"].copy()
            points = geometry["landmarks"].copy()[:468]
            observed_skin = geometry["observed_skin_mask"].copy() if "observed_skin_mask" in geometry else skin
            hair = geometry["hair_mask"].copy() if "hair_mask" in geometry else None
        image = read_image(contained(root, v["input"]))
        matrix = np.float32(v["transform"]); inverse = np.float32(v["inverse_transform"])
        aligned = cv2.warpAffine(image, matrix, (skin.shape[1],skin.shape[0]), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        anatomy = partition(points, observed_skin, view, hair,for_statistics=True)
        folder = root/"06_面部问题地图"/view
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/"anatomy.npz", labels=anatomy.labels,
                            geometric_labels=anatomy.geometric_labels, face=anatomy.face,
                            excluded=anatomy.excluded, hidden=anatomy.hidden)
        write_image(folder/"labels.png", anatomy.labels)
        candidates = presentation.get("views",{}).get(view,{}).get("modules",{})
        base = next((candidates[n]["base"] for n in ("pores","texture","spots","acne") if candidates.get(n,{}).get("base")), None)
        if base is None:
            write_image(folder/"rgb_base.png", aligned)
            base = (folder/"rgb_base.png").relative_to(root).as_posix()
        regions = {}
        masks = anatomy.masks()
        for name in LEAVES:
            current = masks[name]
            regions[name] = {**anatomy.coverage[name], "label": REGIONS[name], "color": COLORS[name],
                             "rings": rings(current), "geometry_rings": rings(
                                 (anatomy.geometric_labels == LEAVES.index(name)+1).astype(np.uint8)),
                             "bbox_analysis": bbox(current),
                             "bbox_original": original_box(bbox(current),inverse,image.shape)}
        original_points = cv2.transform(points[None],inverse)[0]
        map_views[view] = {"base": base, "state": "observed", "regions": regions,
                          "labels": (folder/"labels.png").relative_to(root).as_posix(),
                          "landmarks": points.round(3).tolist(),
                          "landmarks_original": original_points.round(3).tolist(),
                          "issues": {}, "module_regions": {}}
        near_side = max(("left","right"), key=lambda side: int(np.count_nonzero(masks[side+"_cheek"])))
        observed_views[view] = {"modules": {}, "coverage": anatomy.coverage,
                                "near_anatomical_side": near_side}
        for module, item in v.get("modules",{}).items():
            if item.get("status") != "success" or not item.get("evidence"):
                observed_views[view]["modules"][module] = {"status":item.get("status"),"regions":{}}
                continue
            with np.load(contained(root,item["evidence"])) as e:
                evidence={key:e[key].copy() for key in ("mask","response","valid_mask")}
            effective,domain_origin=restore_effective_domain(module,item,evidence["valid_mask"],
                aligned,skin,points,v.get("quality",{}).get("flags",[]))
            evidence["valid_mask"]=effective
            effective_path=folder/(module+"_effective_domain.png")
            write_image(effective_path,effective)
            measured,partition_policy,boundary_radius=measurement_partition(module,aligned,skin,points,
                v.get('quality',{}).get('flags',[]),effective,observed_skin,view,hair)
            module_masks=measured.masks()
            write_image(folder/(module+'_region_labels.png'),measured.labels)
            map_views[view]['module_regions'][module]={name:{**measured.coverage[name],
                'label':REGIONS[name],'color':COLORS[name],'rings':rings(mask),
                'bbox_analysis':bbox(mask),'bbox_original':original_box(bbox(mask),inverse,image.shape)}
                for name,mask in module_masks.items()}
            values = module_observations(data["capture_id"],view,module,item,evidence,measured,
                                         points,aligned,v.get("quality",{}).get("flags",[]),boundary_radius=boundary_radius)
            values.update(partition_policy=partition_policy,partition_version=PARTITION_VERSION,
                region_labels=(folder/(module+'_region_labels.png')).relative_to(root).as_posix())
            values.update(effective_domain=effective_path.relative_to(root).as_posix(),
                          effective_domain_sha256=sha256(effective_path),effective_domain_origin=domain_origin)
            observed_views[view]["modules"][module] = values
            region_issues = {}
            for region in LEAVES:
                support = (evidence["mask"]>0) & (evidence["valid_mask"]>0) & (module_masks[region]>0)
                # Cluster envelopes are only a navigation aid, not area measurements.
                region_issues[region] = {"rings": cluster_rings(support),
                    "bbox_original": original_box(bbox(support),inverse,image.shape),
                    "polygon_semantics": ("candidate_box_navigation_not_lesion_area" if module=="acne"
                        else "display_cluster_envelope_not_measurement_area")}
                before=item.get('legacy_regions_before_v22',item.get("regions",{})).get(region,{})
                after=values["regions"][region]
                for field in ("count","coverage","analysis_pixels","density_per_100k_pixels"):
                    if before.get(field)!=after.get(field):
                        source_differences.append({"view":view,"module":module,"region":region,
                            "field":field,"before":before.get(field),"after":after.get(field),
                            "reason":"统一解剖分区/边界待定归属/本项有效域与局部质量门；未重新推理"})
            map_views[view]["issues"][module] = region_issues
            # Keep the full-result JSON aligned with the authoritative per-view statistics.
            item.setdefault('legacy_regions_before_v22',item.get('regions',{}))
            item['regions']=values['regions']
            item['region_statistics_version']='regional_observation_statistics_v2.2'
            assigned={o['local_id']:o['region_id'] for o in values['observations']}
            for instance in item.get('instances',[]):
                instance.setdefault('legacy_region_before_v22',instance.get('region'))
                instance['region']=assigned.get(instance['id'])
            for observation in values['observations']:
                observation['source_instance']['region']=observation['region_id']
            # Per-item metrics.json is a result alias, never a second calculation.
            item_json=root/v['geometry'].split('/')[0]/module/'metrics.json'
            if item_json.exists():write_json(item_json,item)
    statistics = select_primary(observed_views)
    statistics.update(anatomy_version='anatomy_visible_v2',partition_version=PARTITION_VERSION, views=observed_views,
                      measurement_origin="saved_detector_evidence",
                      missing_extent_policy="never_expand_saved_detector_valid_mask",
                      location_guidance_only=True)
    from .observation_display import build_display_account
    statistics["display_reconciliation"]=build_display_account(data,statistics)
    groups = {}
    for key,(label,leaves) in GROUPS.items():
        concerns = []
        for leaf in leaves:
            for module, record in statistics["regions"][leaf]["modules"].items():
                if leaf not in statistics["display_reconciliation"]["modules"][module]["visible_regions"]:
                    continue
                m=record["metrics"]
                supplemental=any(a["has_findings"] for a in record["alternatives"])
                if (m and m.get("has_evidence")) or supplemental:
                    concerns.append({"leaf_region":leaf,"module":module,"label":MODULES[module],
                                     "primary_view":record["primary_view"],
                                     "statement":record["statement"],"metrics":m,
                                     "supplementary_views":[a["view"] for a in record["alternatives"] if a["has_findings"]]})
        states=[map_views[v].get("regions",{}).get(n,{}).get("state","uncovered") for v in VIEWS for n in leaves]
        state="observed" if "observed" in states else "occluded" if "occluded" in states else "uncovered"
        groups[key]={"label":label,"leaf_regions":list(leaves),"state":state,"concerns":concerns}
    location = {"version":"concern_region_map_v2","anatomy_version":"anatomy_visible_v2","partition_version":PARTITION_VERSION,
                "purpose":"image_concern_localization_not_treatment_instructions",
                "coordinate_space":"aligned_1024_with_per_view_original_transform",
                "operation_points":None,"dose":None,"depth":None,
                "groups":groups,"views":map_views,
                "unassigned_observations":[{"view":v,"module":m,**o} for v,view in observed_views.items()
                    for m,item in view["modules"].items() for o in item.get("uncertain_observations",[])]}
    write_json(root/"区域统计V2.json",statistics)
    data['regional_statistics_version']=statistics['version']
    if 'region_summary' in data:
        data.setdefault('legacy_region_summary_before_v22',data.pop('region_summary'))
    data['regional_statistics_file']='区域统计V2.json'
    write_json(root/"水光检测完整结果.json",data)
    write_json(root/"面部问题地图.json",location)
    write_json(root/"区域统计变化说明.json",{"changes":source_differences,
        "reason":"旧单视角观测保持；新分区和主视角口径使用独立版本，不回填旧值"})
    return statistics,location
