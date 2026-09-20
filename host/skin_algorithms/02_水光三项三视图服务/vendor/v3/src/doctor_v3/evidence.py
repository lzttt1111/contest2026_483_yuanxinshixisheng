"""Opt-in internal evidence; never adds fields to a Worker public envelope."""
import os
import shutil
from pathlib import Path
import numpy as np
from .bundle import write_json

def enabled():
    value = os.environ.get("DERMAVISION_CLINICAL_SPEC", "v2")
    if value not in ("v2", "v3"):
        raise ValueError("invalid clinical spec")
    return value == "v3"

class EvidencePayload(dict):
    """Arrays are attributes, not JSON keys: old JSON contract is unchanged."""
    pass

def attach(payload, *, project, analysis_mask, landmarks, instances, score_map,
           instance_mask, continuous_mask=None, high_mask=None, quality_control=None):
    result = EvidencePayload(payload)
    result.v3_arrays = {"valid": np.asarray(analysis_mask), "landmarks": np.asarray(landmarks),
                        "score": np.asarray(score_map), "instances": np.asarray(instance_mask)}
    if continuous_mask is not None:
        result.v3_arrays["continuous"] = np.asarray(continuous_mask)
    if high_mask is not None:
        result.v3_arrays["high"] = np.asarray(high_mask)
    result.v3_meta = {"project": project, "instances": instances, "quality": quality_control or {},
                      "coordinate_system": "aligned_1024_image_left_right", "measurement_version": "v3-evidence-1"}
    return result

def save(directory, project, arrays, metadata=None):
    if not enabled():
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clean_arrays = {}
    for key, value in arrays.items():
        if value is not None:
            array = np.asarray(value)
            if array.dtype.hasobject or not np.all(np.isfinite(array)):
                raise ValueError("invalid evidence array: " + key)
            clean_arrays[key] = array
    from src.utils.detailed_metrics import json_safe
    np.savez_compressed(directory / ("doctor_v3_" + project + ".npz"), **clean_arrays)
    write_json(directory / ("doctor_v3_" + project + ".json"),
               json_safe({"project": project, **(metadata or {})}))
    sink_value = os.environ.get("DERMAVISION_V3_EVIDENCE_SINK")
    if sink_value:
        sink = Path(sink_value).resolve()
        if not sink.is_relative_to(Path(__file__).resolve().parents[3]):
            raise ValueError("V3 evidence sink must remain in the V3 workspace")
        sink.mkdir(parents=True, exist_ok=True)
        for suffix in (".npz",".json"):
            source = directory / ("doctor_v3_"+project+suffix)
            if source.resolve() != (sink/source.name).resolve():
                shutil.copyfile(source,sink/source.name)

def save_payload(directory, payload, project):
    arrays = getattr(payload, "v3_arrays", None)
    if arrays is not None:
        save(directory, project, arrays, payload.v3_meta)

def save_preprocess(directory, preprocess, role="rgb"):
    if enabled():
        save(directory, role, {"image": preprocess.analysis_image, "valid": preprocess.skin_mask,
             "landmarks": preprocess.landmarks, "transform": preprocess.face_transform_matrix,
             "relative_z": getattr(preprocess, "landmarks_relative_z", None)},
             {"quality_status": preprocess.quality_status, "quality_flags": list(preprocess.quality_flags)})
