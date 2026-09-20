"""V3 anatomical observation domains, independent of any detector's public mask."""
import cv2
import numpy as np

VERSION = "v3-geometry-roi-anatomical-3"
PARAMETERS = {"midcheek_half_width_eye_fraction": .10,
              "mouth_corner_half_width_eye_fraction": .085,
              "tear_trough_path_downshift_eye_fraction": .07}


def anatomical_groove_paths(landmarks):
    """Oriented from anatomical origin towards the distal extension site."""
    p = np.asarray(landmarks, dtype=float)[:, :2]
    eye_width = float(np.linalg.norm(p[33] - p[263]))
    ids = {
        "nasolabial": ([49, 203, 92, 57], [279, 423, 322, 287]),
        "marionette": ([57, 106, 194, 208], [287, 335, 418, 428]),
        "midcheek": ([50, 205, 216, 212], [280, 425, 436, 432]),
        "mouth_corner": ([57, 202, 211, 32], [287, 422, 431, 262]),
        "tear_trough": ([133, 145, 33], [362, 374, 263]),
    }
    result = {}
    for kind, pairs in ids.items():
        for side, indices in zip(("left", "right"), pairs):
            path = p[indices].copy()
            if kind == "tear_trough":
                path[:, 1] += eye_width * PARAMETERS["tear_trough_path_downshift_eye_fraction"]
            result[side+"_"+kind] = path
    return result


def build_geometry_regions(valid, landmarks, module):
    p = np.asarray(landmarks, dtype=float)[:, :2]
    valid = np.asarray(valid) > 0
    yy, xx = np.indices(valid.shape)
    center = float(p[1, 0])
    width = max(abs(float(p[263, 0] - p[33, 0])), 1.0)

    def polygon(indices):
        mask = np.zeros(valid.shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(p[indices]).astype(np.int32)], 1)
        return (mask > 0) & valid

    result = {}
    if module == "09":
        paths = anatomical_groove_paths(p)
        indices = {
            "nasolabial": ([49, 98, 61, 91, 205], [279, 327, 291, 321, 425]),
            "marionette": ([61, 91, 176, 148, 172], [291, 321, 400, 377, 397]),
            "midcheek": ([116, 123, 205, 61, 50], [345, 352, 425, 291, 280]),
            "tear_trough": ([133, 145, 33, 116, 50], [362, 374, 263, 345, 280]),
            "mouth_corner": ([61, 40, 91, 181], [291, 270, 321, 405]),
        }
        # Structure-specific domains may overlap anatomically; instances are
        # resolved by the geometry classifier, not silently renamed/averaged.
        for name, pairs in indices.items():
            for side, ids in zip(("left", "right"), pairs):
                region = side + "_" + name
                if name in ("midcheek", "mouth_corner"):
                    # A skin observation band around the anatomical path,
                    # not a polygon whose vertices all belong to the lip.
                    band = np.zeros(valid.shape, np.uint8)
                    half_width = width * PARAMETERS[name+"_half_width_eye_fraction"]
                    cv2.polylines(band, [np.rint(paths[region]).astype(np.int32)], False,
                                  1, max(3, int(round(2*half_width))))
                    result[region] = valid & (band > 0)
                else:
                    result[region] = polygon(ids)
    elif module == "11":
        from src.utils.detailed_metrics import build_medical_report_regions
        medical = build_medical_report_regions(valid.astype(np.uint8) * 255, landmarks).regions
        for name in ("nose", "image_left_nasal", "image_right_nasal"):
            if name in medical:
                valid &= ~(medical[name] > 0)
        eye_y = max(float(p[145, 1]), float(p[374, 1]))
        mouth_y = (float(p[61, 1]) + float(p[291, 1])) / 2
        for side, side_mask, arc in (
            ("left", xx < center, [132, 58, 172, 136, 150, 149, 176, 148, 152]),
            ("right", xx >= center, [361, 288, 397, 365, 379, 378, 400, 377, 152]),
        ):
            jaw = np.zeros(valid.shape, np.uint8)
            cv2.polylines(jaw, [np.rint(p[arc]).astype(np.int32)], False, 1, max(3, int(width * .075)))
            result[side + "_midface"] = valid & side_mask & (yy >= eye_y) & (yy < mouth_y)
            result[side + "_lower_face"] = valid & side_mask & (yy >= mouth_y) & ~(jaw > 0)
            result[side + "_jaw"] = valid & side_mask & (jaw > 0)
    union = np.zeros_like(valid)
    for mask in result.values():
        union |= mask
    return {"full_face": union, **result}
