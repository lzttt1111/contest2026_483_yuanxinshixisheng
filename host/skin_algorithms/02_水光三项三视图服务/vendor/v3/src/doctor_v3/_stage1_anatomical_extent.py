"""Anatomical extent evidence and explicit, basis-only range-state mapping."""
import numpy as np

VERSION = "v3-anatomical-extent-1"
PARAMETERS = {"sampling_step_px": 1.0, "interval_merge_gap_px": 1.5,
              "minimum_valid_path_fraction": .50,
              "minimum_valid_stage_fraction": .80}


def merge_intervals(intervals, gap=0.0):
    result = []
    for a, b in sorted((max(0., float(a)), min(1., float(b))) for a, b in intervals if b > a):
        if result and a <= result[-1][1] + gap:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def intersect_intervals(a, b):
    return merge_intervals([[max(x0,y0),min(x1,y1)] for x0,x1 in a for y0,y1 in b
                            if min(x1,y1)>max(x0,y0)])


def interval_length(intervals):
    return float(sum(b-a for a,b in intervals))


def extent_evidence(path, target, domain, groove_type, target_ids=None):
    path = np.asarray(path, dtype=float)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():
        return {"definition_version":VERSION,"groove_type":groove_type,"measurement_status":"missing_evidence","reason":"degenerate_anatomical_path","path_xy":path.tolist()}
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    total = float(lengths.sum())
    context = {"definition_version": VERSION, "groove_type": groove_type,
               "path_xy": path.tolist(), "path_length_px": total,
               "target_ids": list(target_ids or []), "parameters": dict(PARAMETERS)}
    if len(path)<2 or not np.isfinite(path).all() or np.any(lengths<1) or total<10:
        return {**context, "measurement_status": "missing_evidence", "reason": "degenerate_anatomical_path"}
    cumulative = np.r_[0., np.cumsum(lengths)]
    sample_s = np.linspace(0,total,int(np.ceil(total/PARAMETERS["sampling_step_px"]))+1)
    index = np.minimum(np.searchsorted(cumulative[1:],sample_s,side="right"),len(lengths)-1)
    sampled_xy = path[index]+segments[index]*((sample_s-cumulative[index])/lengths[index])[:,None]
    xy = np.rint(sampled_xy).astype(int)
    inside = (xy[:,0]>=0)&(xy[:,0]<domain.shape[1])&(xy[:,1]>=0)&(xy[:,1]<domain.shape[0])
    valid = np.zeros(len(xy),bool)
    valid[inside] = domain[xy[inside,1],xy[inside,0]]
    half = .5/max(len(xy)-1,1)
    intervals = merge_intervals([[s/total-half,s/total+half] for s in sample_s[valid]])
    y,x = np.nonzero(np.asarray(target,dtype=bool)&domain)
    points = np.column_stack((x,y))
    projection = []
    distances = []
    for point in points:
        t = np.clip(((point-path[:-1])*segments).sum(axis=1)/(lengths**2),0,1)
        distance = np.linalg.norm(path[:-1]+t[:,None]*segments-point,axis=1)
        i = int(np.argmin(distance))
        projection.append(float((cumulative[i]+t[i]*lengths[i])/total))
        distances.append(float(distance[i]))
    hit = merge_intervals([[t-.5/total,t+.5/total] for t in projection],
                          PARAMETERS["interval_merge_gap_px"]/total)
    hit = intersect_intervals(hit,intervals)
    denominator = sum(b*b-a*a for a,b in intervals)
    numerator = sum(b*b-a*a for a,b in hit)
    valid_fraction = interval_length(intervals)
    return {**context, "measurement_status": "measured" if valid_fraction>=PARAMETERS["minimum_valid_path_fraction"] else "missing_evidence",
            "reason": None if valid_fraction>=PARAMETERS["minimum_valid_path_fraction"] else "insufficient_valid_anatomical_path",
            "valid_intervals": intervals, "target_intervals": hit,
            "projected_target_t": sorted(projection), "projected_target_distances_px": distances,
            "valid_path_fraction": valid_fraction, "covered_length_ratio": interval_length(hit)/max(valid_fraction,1e-12),
            "longest_contiguous_ratio": max([b-a for a,b in hit],default=0.),
            "distal_progress": max([b for a,b in hit],default=0.),
            "anatomical_extent_burden": numerator/denominator if denominator else None,
            "burden_numerator": numerator*total, "burden_denominator": denominator*total}


def classify_extent_basis(evidence, config=None):
    """Return explicit range-state only; never calibrates or calls a percentile CDF."""
    if evidence.get("measurement_status") != "measured":
        return {"status":"missing_evidence","extent_score":None,"extent_grade":None,
                "reason":evidence.get("reason","missing_anatomical_evidence"),"config_version":None}
    if config is None:
        return {"status":"pending_parameter","extent_score":None,"extent_grade":None,
                "reason":"anatomical_extent_threshold_pending","config_version":None}
    if config.get("definition_version") != VERSION or config.get("groove_type") != evidence.get("groove_type"):
        raise ValueError("anatomical extent definition/type mismatch")
    if not isinstance(config.get("version"),str) or not config["version"].strip():
        raise ValueError("anatomical configuration version must be a nonempty string")
    if any(k in config and not isinstance(config[k],bool) for k in ("test_only","approved")):
        raise ValueError("configuration approval markers must be booleans")
    if bool(config.get("test_only")) == bool(config.get("approved")):
        raise ValueError("exactly one of test_only or approved must be true")
    if config.get("approved") and (not isinstance(config.get("approval_id"),str) or not config["approval_id"].strip() or config.get("purpose") != "formal"):
        raise ValueError("approved anatomical range configuration requires formal approval identity")
    if config.get("test_only") and config.get("purpose") != "test":
        raise ValueError("test configuration cannot claim formal approval")
    boundaries = np.asarray(config["segment_boundaries"],float)
    if boundaries.shape != (4,) or not np.isfinite(boundaries).all() or boundaries[0]!=0 or boundaries[-1]!=1 or np.any(np.diff(boundaries)<=0):
        raise ValueError("anatomical segment boundaries must strictly partition [0,1] into three stages")
    minimum = float(config["min_total_coverage"])
    segment_minimum = float(config["min_segment_coverage"])
    minimum_valid_stage = float(config.get("min_valid_segment_coverage",PARAMETERS["minimum_valid_stage_fraction"]))
    if not 0<minimum<=1 or not 0<segment_minimum<=1 or not 0<minimum_valid_stage<=1:
        raise ValueError("invalid anatomical support thresholds")
    scores = config["grade_scores"]
    if set(scores)!={"none","localized","moderate","extensive"} or any(isinstance(v,bool) or not np.isfinite(v) or not 0<=v<=100 for v in scores.values()):
        raise ValueError("explicit four grade scores required")
    grade_values = [scores[k] for k in ("none","localized","moderate","extensive")]
    if any(a<=b for a,b in zip(grade_values,grade_values[1:])):
        raise ValueError("anatomical grade scores must decrease with extension")
    if evidence.get("groove_type")=="full_face":
        numerator=np.zeros(3)
        denominator=np.zeros(3)
        nominal=np.zeros(3)
        covered,total=0.,0.
        for child in evidence["structure_evidence"].values():
            if child.get("measurement_status")!="measured":
                continue
            limits=np.asarray(config["type_segment_boundaries"][child["groove_type"]],float)
            if limits.shape!=(4,) or not np.isfinite(limits).all() or limits[0]!=0 or limits[-1]!=1 or np.any(np.diff(limits)<=0):
                raise ValueError("invalid per-type anatomical stages")
            length=child["path_length_px"]
            covered+=interval_length(child["target_intervals"])*length
            total+=interval_length(child["valid_intervals"])*length
            for i,(a,b) in enumerate(zip(limits,limits[1:])):
                numerator[i]+=interval_length(intersect_intervals(child["target_intervals"],[[a,b]]))*length
                denominator[i]+=interval_length(intersect_intervals(child["valid_intervals"],[[a,b]]))*length
                nominal[i]+=(b-a)*length
        ratios=np.divide(numerator,denominator,out=np.zeros(3),where=denominator>0).tolist()
        valid_ratios=np.divide(denominator,nominal,out=np.zeros(3),where=nominal>0).tolist()
        stage_lengths={"effective_length":denominator.tolist(),"affected_length":numerator.tolist(),
                       "nominal_length":nominal.tolist(),"unit":"standardized_px"}
        total_coverage=covered/total if total>0 else 0.
    else:
        target = evidence["target_intervals"]
        effective=[interval_length(intersect_intervals(evidence["valid_intervals"],[[a,b]])) for a,b in zip(boundaries,boundaries[1:])]
        affected=[interval_length(intersect_intervals(target,[[a,b]])) for a,b in zip(boundaries,boundaries[1:])]
        nominal=np.diff(boundaries).tolist()
        valid_ratios=[v/n for v,n in zip(effective,nominal)]
        ratios=[a/v if v>0 else 0. for a,v in zip(affected,effective)]
        stage_lengths={"effective_length":effective,"affected_length":affected,
                       "nominal_length":nominal,"unit":"normalized_path_fraction"}
        total_coverage=evidence["covered_length_ratio"]
    stage_details={"stage_valid_fraction":valid_ratios,"stage_lengths":stage_lengths,
                   "effective_min_valid_segment_coverage":minimum_valid_stage}
    if any(v<minimum_valid_stage for v in valid_ratios):
        return {"status":"missing_evidence","extent_score":None,"extent_grade":None,
                "reason":"insufficient_valid_anatomical_stages","config_version":config["version"],
                "configuration":dict(config),**stage_details}
    supported = [i for i,ratio in enumerate(ratios) if ratio>=segment_minimum]
    grade = "none" if total_coverage<minimum or not supported else ("localized","moderate","extensive")[max(supported)]
    return {"status":"test_only" if config.get("test_only") else "approved_state",
            "extent_score":float(scores[grade]),"extent_grade":grade,"supported_segments":supported,
            "segment_coverage":ratios,"reason":None,"config_version":config["version"],
            "configuration":dict(config),"definition_version":VERSION,**stage_details}
