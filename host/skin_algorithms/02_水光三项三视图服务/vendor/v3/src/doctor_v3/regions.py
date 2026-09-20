"""V3-only masks: no changes to any shared detector preprocessing mask."""
import numpy as np
from .registry import MODULES

ROI_VERSION = "doctor-v3-roi-2"

def build(valid, landmarks, module):
    valid = np.asarray(valid) > 0
    empty = lambda: np.zeros(valid.shape, dtype=bool)
    if np.count_nonzero(valid) < 100 or np.asarray(landmarks).shape[0] < 400:
        return {"full_face": empty(), **{name: empty() for name in MODULES[int(module)-1].regions}}
    if module in ("09", "11"):
        from .geometry_regions import build_geometry_regions
        return build_geometry_regions(valid, landmarks, module)
    from src.utils.detailed_metrics import build_medical_report_regions
    old = build_medical_report_regions(valid.astype(np.uint8)*255, landmarks).regions
    old = {k: v > 0 for k,v in old.items()}
    base = {}
    rename = {"image_left_nasal":"left_nasal", "image_right_nasal":"right_nasal",
              "image_left_zygomatic":"left_zygoma", "image_right_zygomatic":"right_zygoma",
              "image_left_cheek":"left_cheek", "image_right_cheek":"right_cheek",
              "image_left_jaw":"left_jaw", "image_right_jaw":"right_jaw"}
    for k,v in old.items():
        base[rename.get(k,k)] = v
    yy, xx = np.indices(valid.shape)
    cx = float(landmarks[1,0])
    if module == "01":
        base["forehead"] |= base["glabella"]
        for side in ("left","right"):
            cheek = base[side+"_zygoma"] | base[side+"_cheek"] | base[side+"_jaw"]
            # Split each row's cheek span radially; not a relabeling of zygoma.
            inner = empty()
            for y in np.flatnonzero(np.any(cheek, axis=1)):
                xs = np.flatnonzero(cheek[y])
                middle = (float(xs.min()) + float(xs.max())) / 2
                inner[y, xs] = xs >= middle if side == "left" else xs < middle
            base[side+"_inner_cheek"] = inner
            base[side+"_outer_cheek"] = cheek & ~inner
        base["chin"] |= base["perioral"]
    elif module in ("02", "05"):
        base["forehead"] |= base["glabella"]
        base["chin"] |= base["perioral"] | base["left_jaw"] | base["right_jaw"]
        base["jaw"] = base["chin"]
    elif module in ("03","04","06","10"):
        if module=="03":
            base["nose"] |= base["left_nasal"] | base["right_nasal"]
        base["jaw"] = base["left_jaw"] | base["right_jaw"] | base["chin"]
        base["left_jaw"] |= base["chin"] & (xx < cx)
        base["right_jaw"] |= base["chin"] & (xx >= cx)
        if module != "04":
            base["forehead"] |= base["glabella"]
    if module in ("03","07"):
        for side, indices in (("left", (33,133,145)), ("right",(362,263,374))):
            pts = np.asarray(landmarks)[list(indices),:2]
            x0,x1 = np.min(pts[:,0]),np.max(pts[:,0])
            bottom = np.max(pts[:,1])
            eye = valid & (xx >= x0) & (xx <= x1) & (yy >= bottom) & (yy < bottom+max(12,(x1-x0)*.45))
            base[side+"_under_eye"] = eye
            base[side+"_eye"] = eye
            base[side+"_zygoma"] &= ~eye
            base[side+"_cheek"] &= ~eye
    result = {}
    assigned = empty()
    for name in MODULES[int(module)-1].regions:
        mask = base.get(name, empty()) & valid & ~assigned
        result[name] = mask
        assigned |= mask
    return {"full_face": assigned, **result}
