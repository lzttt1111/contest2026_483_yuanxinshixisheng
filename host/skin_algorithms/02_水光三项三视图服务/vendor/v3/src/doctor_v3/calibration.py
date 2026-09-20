"""Explicit V3 reference configuration; no automatic conversion of V2 final scores."""
import json
import hashlib
import math
from pathlib import Path
from . import SPEC_VERSION

def load_reference(path, profile):
    if path is None:
        return {"references":{},"thresholds":{},"version":"uncalibrated"}
    raw=Path(path).read_bytes()
    doc=json.loads(raw)
    if doc.get('schema_version')=='doctor_v3_scoring_reference_v1':
        from .stage2_reference_loader import validate_scoring_reference_config
        # Do not add file metadata to signed reference content.
        return validate_scoring_reference_config(doc,profile)['reference']
    if doc.get("schema_version")!="doctor_v3_reference_v1" or doc.get("definition_version")!=SPEC_VERSION:
        raise ValueError("incompatible V3 reference definition")
    if doc.get("capture_profile")!=profile or not doc.get("version"):
        raise ValueError("reference profile/version mismatch")
    if doc.get("status") not in ("candidate","calibrated"):
        raise ValueError("reference status must be explicit")
    refs=doc.get("references",{})
    if not isinstance(refs,dict):
        raise ValueError("references must be a keyed object")
    for key,ref in refs.items():
        if ref.get("capture_profile")!=profile or ref.get("definition_version")!=SPEC_VERSION:
            raise ValueError("incompatible metric reference: "+key)
        if key!=ref.get("metric_id","")+":"+ref.get("region",""):
            raise ValueError("incorrect reference key")
        if not ref.get("population_sha256") or not ref.get("version"):
            raise ValueError("reference lineage required")
    thresholds=doc.get("thresholds",{})
    for value in thresholds.get("large_pore_area",{}).values():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError("invalid pore area threshold")
    high=thresholds.get("porphyrin_high_intensity")
    if high is not None and (isinstance(high,bool) or not isinstance(high,(int,float)) or not 0<=high<=1):
        raise ValueError("invalid porphyrin threshold")
    return {**doc,"sha256":hashlib.sha256(raw).hexdigest()}


def resolve_reference_inputs(reference_path,score_reference_path,profile):
    """A unified new reference supplies both measurement parameters and scores.

    Legacy parameter and score files retain their independent behavior.
    Two explicitly supplied files must agree on the effective new configuration.
    """
    from .stage1_config import measurement_config
    from .stage2_reference_loader import SCHEMA,validate_scoring_reference_config
    score_doc=json.loads(Path(score_reference_path).read_bytes()) if score_reference_path is not None else None
    new_score=isinstance(score_doc,dict) and score_doc.get('schema_version')==SCHEMA
    if reference_path is None and new_score:
        reference_path=score_reference_path
    calibration=load_reference(reference_path,profile)
    if calibration.get('schema_version')==SCHEMA and score_reference_path is None:
        score_reference_path=reference_path
        score_doc=calibration
        new_score=True
    if new_score:
        context=validate_scoring_reference_config(score_doc,profile)
        if measurement_config(calibration)!=context['measurement_config']:
            raise ValueError('measurement parameters differ from scoring reference parameters')
    return {'reference_path':reference_path,'score_reference_path':score_reference_path,
            'calibration':calibration,'reference_file_sha256':hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()
            if reference_path is not None else 'uncalibrated'}
