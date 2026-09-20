"""Structured three-item scoring: independent spots never enters mixed V3 scoring."""
import hashlib,json
from pathlib import Path
from .settings import VENDOR

def apply_legacy_fallback(scores):
    """Compatibility shim: sample scores must never fill another input's gaps."""
    return []

def score_saved(source,sample):
    from src.doctor_v3.stage1_pipeline import collect_evidence,analyze
    from src.doctor_v3.stage1_config import measurement_config
    from src.doctor_v3.calibration import load_reference
    from src.doctor_v3.stage1_word import attach_stage1
    from src.doctor_v3.severity_guard import POLICY
    from .v3_saved import extract
    from .independent_spots import score as score_spots
    source=Path(source);data,metadata,_=collect_evidence(source)
    if set(data)-{'rgb','pores','spots','surface_gloss'}:
        raise ValueError('Unexpected detector evidence in independent three-item task')
    selected={k:data[k] for k in ('rgb','pores','surface_gloss')}
    meta={k:metadata[k] for k in selected}
    identity=hashlib.sha256(selected['rgb']['image'].tobytes()).hexdigest()
    reference_path=VENDOR/'v3_reference.json';reference=json.loads(reference_path.read_text())
    payload,_,_=analyze(selected,meta,subject_id=sample,profile='consumer',input_sha256={'aligned_rgb':identity},thresholds=measurement_config(load_reference(reference_path,'consumer')),reference=reference)
    model={'subject_id':sample,'capture_profile':'consumer','modules':{f'{i:02d}':{'metrics':{},'regions':{}} for i in range(1,12)},'scoring_system_version':'V3.0.1','severity_policy':POLICY,'legacy_input_proof':{'input_sha256':[identity],'proof_kind':'current_input_and_quality'}}
    scores=extract(attach_stage1(model,payload))
    files=list((source/'dermavision/spots').rglob('02_Spots量化指标.json'))
    if len(files)!=1:raise ValueError('Independent spot evidence is missing or ambiguous')
    reference_path=VENDOR/'independent_spots_reference.json'
    scores['spots'],trace=score_spots(json.loads(files[0].read_text()),json.loads(reference_path.read_text()))
    from .observed_zero import apply_observed_zero
    zero_trace=apply_observed_zero(scores,payload)
    from .independent_gloss import score_gloss
    gloss_trace=score_gloss(scores,payload)
    return scores,{'doctor_pipeline_projects':list(selected),'independent_spots_measurements':trace,'aligned_rgb_sha256':identity,'independent_spots_reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),'legacy_fallback_used':[], 'scoring_policy':'current_image_evidence_only_v3','observed_zero':zero_trace,'independent_gloss':gloss_trace}
