"""V3 image-phenotype postprocessing. Parameters are engineering candidates, not diagnoses."""
import cv2
import numpy as np
from .regions import build
from .registry import measurement

PARAMETERS = {"version":"v3-phenotype-candidate-1","grid_px":16,"fine_max_length_px":70,
              "follicular_red_delta_a":4.0,"white_center_delta_l":12.0,
              "max_alignment_rms_fraction":.005}

def warp_mask(source, target, mask):
    a,b = np.asarray(source["landmarks"])[:,:2], np.asarray(target["landmarks"])[:,:2]
    if a.shape != b.shape or len(a)<400:
        return None
    design = np.column_stack((a,np.ones(len(a))))
    matrix = np.linalg.lstsq(design,b,rcond=None)[0].T
    error = np.sqrt(np.mean(np.sum((design@matrix.T-b)**2,axis=1)))
    # This is a secondary coordinate-space sanity check after acquisition
    # authorization, not a replacement for capture registration.
    limit=PARAMETERS["max_alignment_rms_fraction"]*max(target["valid"].shape)
    if not np.isfinite(error) or error>limit or np.linalg.det(matrix[:,:2])<=0:
        return None
    return cv2.warpAffine(mask.astype(np.uint8),matrix,target["valid"].shape[::-1],flags=cv2.INTER_NEAREST)>0

def diffuse(red, vascular=None, acne=None, *, cross_channel_authorized=False):
    if vascular is None or acne is None or cross_channel_authorized is not True:
        return None
    vessels = warp_mask(vascular,red,vascular["instances"]>0)
    lesions = warp_mask(acne,red,acne["instances"]>0)
    if vessels is None or lesions is None:
        return None
    excluded = (red["instances"]>0) | vessels | lesions
    excluded = cv2.dilate(excluded.astype(np.uint8),np.ones((5,5),np.uint8))>0
    valid = (red["valid"]>0)&~excluded
    out = dict(red)
    out["valid"] = valid.astype(np.uint8)*255
    out["continuous"] = ((red.get("continuous",red["instances"])>0)&valid).astype(np.uint8)*255
    out["high"] = ((red.get("high",np.zeros_like(valid))>0)&valid).astype(np.uint8)*255
    return out

def measure_diffuse(arrays, profile):
    rows = []
    for region,domain in build(arrays["valid"],arrays["landmarks"],"04").items():
        area = domain.sum()
        affected = (arrays["continuous"]>0)&domain
        vals = arrays["score"][affected]
        values = {"area":float(affected.sum()/area) if area>=100 else None,
                  "high_area":float(((arrays["high"]>0)&domain).sum()/area) if area>=100 else None,
                  "mean":float(vals.mean()) if len(vals) and area>=100 else None,
                  "p90":float(np.percentile(vals,90)) if len(vals) and area>=100 else None}
        for name,value in values.items():
            rows.append(measurement("04",name,region,profile,value,"比例" if "area" in name else "0-1",
                                    "no_valid_signal","rgb_measured" if profile=="consumer" else "cp_measured"))
    return rows

def classify_acne(arrays,meta,pores=None):
    """Only supported follicular targets are classified; unknowns are retained separately."""
    image = arrays["image"]
    valid = arrays["valid"]>0
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(float)
    follicles = None if pores is None else warp_mask(pores,arrays,pores["instances"]>0)
    labels = []
    mask = np.zeros(valid.shape,np.uint8)
    candidates = list(meta.get("instances",[]))
    if follicles is not None:
        n,_,stats,centers = cv2.connectedComponentsWithStats(follicles.astype(np.uint8),8)
        candidates += [{"bbox_xyxy":[x-5,y-5,x+5,y+5],"follicle_candidate":True} for x,y in centers[1:]]
    occupied = np.zeros(valid.shape,np.uint8)
    for candidate in candidates:
        box = candidate.get("bbox_standardized_xyxy",candidate.get("bbox_xyxy"))
        if not box or len(box)!=4:
            continue
        x0,y0,x1,y1 = [int(round(v)) for v in box]
        x0,x1 = max(0,x0),min(valid.shape[1],x1)
        y0,y1 = max(0,y0),min(valid.shape[0],y1)
        if x1-x0<3 or y1-y0<3:
            continue
        cx,cy = (x0+x1)//2,(y0+y1)//2
        if not valid[cy,cx] or occupied[cy,cx]:
            continue
        radius = max(3,min(x1-x0,y1-y0)//3)
        yy,xx = np.ogrid[:valid.shape[0],:valid.shape[1]]
        distance = (xx-cx)**2+(yy-cy)**2
        center = (distance<=radius**2)&valid
        ring = (distance>(radius*1.3)**2)&(distance<=(radius*2.2)**2)&valid
        if ring.sum()<12 or center.sum()<5:
            continue
        da = float(np.median(lab[:,:,1][center])-np.median(lab[:,:,1][ring]))
        dl = float(np.percentile(lab[:,:,0][center],90)-np.median(lab[:,:,0][ring]))
        anchored = follicles is not None and bool(np.any(follicles[center]))
        if da < PARAMETERS["follicular_red_delta_a"]:
            continue
        kind = "unknown"
        if anchored:
            if not candidate.get("follicle_candidate") and dl>=PARAMETERS["white_center_delta_l"]:
                kind="pustule"
            elif not candidate.get("follicle_candidate") and float(np.std(lab[:,:,0][center]))>=5:
                kind="papule"
            elif candidate.get("follicle_candidate"):
                kind="erythema"
        labels.append({"centroid":[cx,cy],"class":kind,"red_delta_a":da,"center_delta_l":dl,
                       "follicle_supported":anchored,"classification_version":PARAMETERS["version"]})
        occupied[distance<=(radius*2)**2]=1
        mask[center]=255
    return {**arrays,"instances":mask,"classification_available":follicles is not None},labels

def measure_acne(arrays,labels,profile):
    rows=[]
    for region,domain in build(arrays["valid"],arrays["landmarks"],"06").items():
        area=int(domain.sum())
        local=[r for r in labels if domain[r["centroid"][1],r["centroid"][0]]]
        uncertain=sum(r["class"]=="unknown" for r in local)
        for kind in ("erythema","papule","pustule"):
            count=sum(r["class"]==kind for r in local)
            reason="insufficient_classification_evidence" if uncertain else "insufficient_valid_region_or_rejected_quality"
            value=count*100000/area if area>=100 and not uncertain and arrays.get("classification_available",False) else None
            rows.append(measurement("06",kind+"_density",region,profile,value,"个/10万标准化有效像素",reason))
            rows.append(measurement("06",kind+"_count",region,profile,count if area>=100 else None,"个",reason))
    return rows

def supported_spots(spots, brown=None, uv=None):
    supports=[]
    for other in (brown,uv):
        if other is not None:
            aligned=warp_mask(other,spots,other["instances"]>0)
            if aligned is not None:
                supports.append(aligned)
    if not supports:
        return None
    support=np.logical_or.reduce(supports)
    n,labels,stats,_=cv2.connectedComponentsWithStats((spots["instances"]>0).astype(np.uint8),8)
    accepted=np.zeros_like(support)
    for i in range(1,n):
        member=labels==i
        if np.count_nonzero(member&support)/max(int(stats[i,cv2.CC_STAT_AREA]),1)>=.25:
            accepted|=member
    return {**spots,"instances":accepted.astype(np.uint8)*255,"continuous":accepted.astype(np.uint8)*255}

def measure_lines(arrays,profile,high_precision=False):
    rows=[]
    line=arrays["skeleton"]>0
    n,labels,stats,_=cv2.connectedComponentsWithStats(line.astype(np.uint8),8)
    lengths=stats[:,cv2.CC_STAT_AREA]
    fine_ids=np.flatnonzero((lengths>=3)&(lengths<=PARAMETERS["fine_max_length_px"]))
    fine_ids=fine_ids[fine_ids!=0]
    fine=np.isin(labels,fine_ids)
    gray=cv2.cvtColor(arrays["image"],cv2.COLOR_BGR2GRAY).astype(np.float32)
    contrast=np.abs(cv2.GaussianBlur(gray,(0,0),3)-gray)/255
    for region,domain in build(arrays["valid"],arrays["landmarks"],"07").items():
        area=int(domain.sum()); local=fine&domain
        spread=cv2.dilate(local.astype(np.uint8),np.ones((9,9),np.uint8))>0
        affected=spread&domain
        density=cv2.boxFilter(local.astype(np.float32),-1,(16,16))
        # Geometric network support, not a clinical severity cutoff.
        high=(density>=.10)&affected
        values={"area":float(affected.sum()/area) if area>=100 else None,
                "high_area":float(high.sum()/area) if area>=100 else None,
                "density":float(local.sum()*10000/affected.sum()) if affected.any() else None,
                "contrast_p50":float(np.percentile(contrast[local],50)) if local.any() and high_precision else None}
        for name,value in values.items():
            rows.append(measurement("07",name,region,profile,value,"比例" if "area" in name else "标准化指数",
                                    "high_precision_not_confirmed" if name=="contrast_p50" else "no_valid_signal"))
    return rows
