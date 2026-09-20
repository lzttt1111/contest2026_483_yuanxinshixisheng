"""Model-free replay of three explicitly retained parameter-dependent metrics."""
from copy import deepcopy
import math
from .stage1_config import digest, resolve_config
from .stage1_schema import SCHEMA_VERSION, version_identity, key_of, coverage
from .stage1_scoring import score_measurements
from .stage1_zero_state import summaries, verify_saved_states
from .stage1_two_d import _cutoff, quality_reason
from .registry import GROOVE_REGIONS
from ._stage1_anatomical_extent import classify_extent_basis
from .stage2_parameter_fit import _percentile

VERSION = 'v301-parameter-replay-1'
ALLOWED_PARAMETERS = frozenset(('large_pore_area','porphyrin_high_intensity','anatomical_extent'))
TARGETS = {'01.large_density':('large_pore_area','instance_areas_px2'),
           '02.porphyrin_high_density':('porphyrin_high_intensity','instance_intensities')}


class ParameterReplayError(ValueError):
    """Evidence is not sufficient for a same-definition parameter replay."""


def _require(condition, reason):
    if not condition:
        raise ParameterReplayError(reason)


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _context(key, basis):
    local = basis.get(key)
    _require(isinstance(local,dict),'missing_basis:'+key)
    shared_key = local.get('evidence_ref')
    shared = basis.get(shared_key,{}) if shared_key is not None else {}
    _require(isinstance(shared,dict) and (shared_key is None or shared_key in basis),'missing_shared_basis:'+key)
    return {**shared,**local},local,shared_key


def _set(row, local, value, reason, status):
    row.update(value=value,status='measured' if value is not None else 'unavailable',
               measurement_status=status,reason=None if value is not None else reason)
    local.update(measurement_status=status,reason=None if value is not None else reason)


def _density(row, basis, config, indexed):
    key = key_of(row)
    parameter,sample_key = TARGETS[row['metric_id']]
    threshold = _cutoff(config,parameter,row['region'])
    missing_states = ('missing_evidence','insufficient_quality','unavailable')
    # Missing observations may legitimately have count=None or a partial list.
    # They are not claims that the retained population can be recalculated.
    if row.get('measurement_status',row.get('status')) in missing_states:
        _require(row.get('value') is None,'inconsistent_unavailable_measurement:'+key)
        local=basis.setdefault(key,{})
        _require(isinstance(local,dict),'invalid_basis:'+key)
        local['threshold']=threshold
        _set(row,local,None,row.get('reason') or local.get('reason') or 'missing_evidence',
             row.get('measurement_status','missing_evidence'))
        return
    context,local,_ = _context(key,basis)
    local['threshold'] = threshold
    rejected=context.get('quality_reason') or quality_reason(context)
    if rejected:
        _set(row,local,None,rejected,'insufficient_quality')
        return
    def observation(address):
        proof,_,_ = _context(address,basis)
        sibling=indexed.get(address)
        _require(isinstance(sibling,dict),'missing_observation_measurement:'+address)
        state=sibling.get('measurement_status',sibling.get('status'))
        _require(state==proof.get('measurement_status') and sibling.get('reason')==proof.get('reason'),
                 'inconsistent_observation_state:'+address)
        if state in missing_states:
            _require(sibling.get('value') is None,'inconsistent_unavailable_observation:'+address)
            _set(row,local,None,sibling.get('reason') or 'missing_evidence',
                 'insufficient_quality' if state=='insufficient_quality' else 'missing_evidence')
            return None
        _require(proof.get('quality_status')==context.get('quality_status')
                 and proof.get('quality_reason')==context.get('quality_reason'),
                 'sibling_quality_mismatch:'+address)
        for field in ('capture_profile','region','source_kind','definition_version','roi_version','formula_identity','threshold_identity'):
            _require(sibling.get(field)==row.get(field),'sibling_identity_mismatch:'+field+':'+address)
        return proof
    if parameter=='large_pore_area':
        density=observation('01.density:'+row['region'])
        if density is None:return
        observed=observation('01.area_p50:'+row['region'])
        if observed is None:return
        _require(density.get('measurement_status')=='measured' and density.get('reason') is None,
                 'unproven_pore_observation:'+key)
        _require(density.get('count')==context.get('count') and density.get('instance_areas_px2')==context.get(sample_key)
                 and density.get('effective_area_px')==context.get('effective_area_px'),'inconsistent_pore_observation:'+key)
    else:
        # Pending threshold can coexist with a missing score array. Its empty
        # placeholder list is not evidence of an observed zero population.
        observed=observation('02.porphyrin_p90:'+row['region'])
        if observed is None:return
    samples,count,area = context.get(sample_key),context.get('count'),context.get('effective_area_px')
    _require(isinstance(samples,list),'missing_instance_samples:'+key+':'+sample_key)
    _require(type(count) is int and count>=0 and len(samples)==count,'incomplete_instance_samples:'+key)
    _require(all(_finite(v) and (v>0 if parameter=='large_pore_area' else True) for v in samples),
             'invalid_instance_samples:'+key)
    _require(_finite(area) and area>=0,'missing_effective_area:'+key)
    state,reason = observed.get('measurement_status'),observed.get('reason')
    valid = (state=='measured' and reason is None and count>0) or (
        state in ('no_target','no_targets') and reason=='no_valid_targets' and count==0)
    _require(valid,'unproven_instance_observation:'+key)
    _require(observed.get(sample_key)==samples and observed.get('count')==count
             and observed.get('effective_area_px')==area,'inconsistent_instance_observation:'+key)
    percentile_key=('01.area_p50:' if parameter=='large_pore_area' else '02.porphyrin_p90:')+row['region']
    percentile_row=indexed[percentile_key]
    if count:
        expected=_percentile(samples,.5 if parameter=='large_pore_area' else .9)
        _require(percentile_row.get('status')=='measured' and _finite(percentile_row.get('value'))
                 and math.isclose(percentile_row['value'],expected,rel_tol=1e-9,abs_tol=1e-9),
                 'inconsistent_observed_percentile:'+percentile_key)
    else:
        _require(percentile_row.get('value') is None,'unproven_empty_observation:'+percentile_key)
    if parameter=='large_pore_area':
        _require(context.get('unlocated_instance_count')==0 and density.get('unlocated_instance_count')==0
                 and observed.get('unlocated_instance_count')==0,'missing_instance_geometry:'+key)
        density_row=indexed['01.density:'+row['region']]
        _require(density_row.get('status')=='measured' and area>0 and _finite(density_row.get('value'))
                 and math.isclose(density_row['value'],count*100000/area,rel_tol=1e-9,abs_tol=1e-9),
                 'inconsistent_observed_density:'+key)
    rejected = context.get('quality_reason') or quality_reason(context)
    if area<100 or rejected:
        _set(row,local,None,rejected or 'insufficient_valid_region','insufficient_quality')
    elif threshold is None:
        reason = 'large_pore_threshold_pending' if parameter=='large_pore_area' else 'porphyrin_threshold_pending'
        _set(row,local,None,reason,'pending_parameter')
    else:
        value = sum(v>=threshold for v in samples)*100000/area
        _set(row,local,value,None,'measured')


def _extent(row, basis, config):
    key = key_of(row)
    context,local,shared_key = _context(key,basis)
    _require(_finite(context.get('valid_area_px')),'missing_valid_anatomical_area:'+key)
    anatomy = context.get('anatomical_extent')
    _require(isinstance(anatomy,dict),'missing_anatomical_extent_basis:'+key)
    _require(row.get('source_kind')=='anatomical_extent_state' and row.get('direction')=='higher_health',
             'incompatible_anatomical_measurement_identity:'+key)
    kind = 'full_face' if row['region']=='full_face' else row['region'].split('_',1)[1]
    rules = config.get('anatomical_extent') or {}
    _require(isinstance(rules,dict),'anatomical_extent_must_be_type_mapping')
    state = classify_extent_basis(anatomy,rules.get(kind))
    if state['status']=='test_only':
        state = {**state,'extent_score':None,'extent_grade':None,'status':'pending_parameter',
                 'reason':'test_anatomical_configuration_not_publishable'}
    if context.get('quality_status') in ('REJECT','REJECTED','FAILED','FAIL') or context.get('quality_reason'):
        state = {**state,'extent_score':None,'extent_grade':None,'status':'missing_evidence',
                 'reason':context.get('quality_reason') or 'rejected_quality'}
    if context['valid_area_px']<100:
        state = {**state,'extent_score':None,'extent_grade':None,'status':'missing_evidence',
                 'reason':'insufficient_valid_region'}
    value,reason = state.get('extent_score'),state.get('reason')
    status = 'measured' if value is not None else 'pending_parameter' if state['status']=='pending_parameter' else 'missing_evidence'
    local['extent_classification'] = deepcopy(state)
    if shared_key is not None:
        basis[shared_key]['extent_classification'] = deepcopy(state)
    _set(row,local,value,reason,status)


def replay_parameters(payload, parameters, *, reference=None, allow_test_reference=False):
    """Return a new complete payload; never write files or reuse stale scores.

    parameters is a patch limited to ALLOWED_PARAMETERS. A None value removes
    that parameter and restores its pending state; an omitted key is retained.
    Retained source hashes must match the currently frozen stage-one formulas.
    A new reference must be supplied explicitly to recalculate any score.
    """
    _require(isinstance(parameters,dict),'parameters_must_be_object')
    _require(set(parameters)<=ALLOWED_PARAMETERS,'unsupported_parameter_change:'+','.join(sorted(set(parameters)-ALLOWED_PARAMETERS)))
    _require(payload.get('schema_version')==SCHEMA_VERSION,'unsupported_payload_schema')
    old_config = resolve_config(payload.get('measurement_config'))
    old_versions = payload.get('versions',{})
    expected_old = version_identity(old_config,None)
    for field in ('data','roi','formula','implementation','thresholds'):
        _require(old_versions.get(field)==expected_old[field],'stale_measurement_implementation_or_configuration:'+field)
    # A newer scorer may recognize additional evidence-proven negative states.
    # Existing states must still match byte-for-byte; only additive states are allowed.
    verify_saved_states(payload, allow_additive=True)
    _require(isinstance(payload.get('basis'),dict) and isinstance(payload.get('measurements'),list),'missing_measurement_payload')
    seen = set()
    for row in payload['measurements']:
        key = key_of(row)
        _require(key not in seen,'duplicate_measurement:'+key); seen.add(key)
        expected_identity = key+'|'+row['definition_version']+'|'+old_versions['implementation']+'|'+old_versions['thresholds']
        _require(row.get('measurement_identity')==expected_identity and row.get('roi_version')==old_versions['roi']
                 and row.get('formula_identity')==old_versions['implementation']
                 and row.get('threshold_identity')==old_versions['thresholds'],'inconsistent_measurement_identity:'+key)
    result = deepcopy(payload)
    config = deepcopy(old_config)
    for key,value in parameters.items():
        if value is None: config.pop(key,None)
        else: config[key]=deepcopy(value)
    config = resolve_config(config)
    _require(config.get('anatomical_extent') is None or isinstance(config['anatomical_extent'],dict),
             'anatomical_extent_must_be_type_mapping')
    anatomy = config.get('anatomical_extent') or {}
    allowed_types = {'full_face',*(region.split('_',1)[1] for region in GROOVE_REGIONS)}
    _require(set(anatomy)<=allowed_types and all(isinstance(v,dict) for v in anatomy.values()),
             'unsupported_anatomical_configuration_type')
    for name in ('large_pore_area','porphyrin_high_intensity'):
        value = config.get(name)
        for region in value if isinstance(value,dict) else ('full_face',):
            _cutoff(config,name,region)
    changed = {name for name in ALLOWED_PARAMETERS if old_config.get(name)!=config.get(name)}
    indexed = {key_of(row):row for row in result['measurements']}
    for row in result['measurements']:
        target = TARGETS.get(row['metric_id'])
        if target and target[0] in changed:
            _density(row,result['basis'],config,indexed)
        elif row['metric_id']=='09.extent' and 'anatomical_extent' in changed:
            _extent(row,result['basis'],config)
    versions = version_identity(config,reference)
    for row in result['measurements']:
        key = key_of(row)
        row.update(roi_version=versions['roi'],formula_identity=versions['implementation'],threshold_identity=versions['thresholds'])
        row['measurement_identity']=key+'|'+row['definition_version']+'|'+versions['implementation']+'|'+versions['thresholds']
    scored = score_measurements(result['measurements'],reference,allow_test_reference,basis=result['basis'])
    result.update(measurement_config=config,versions=versions,scores=scored,
                  zero_states=summaries(result['measurements'],result['basis']),
                  coverage=coverage(result['measurements'],scored,result['capture_profile'],config),
                  reference_purpose=reference.get('purpose') if reference else None,
                  parameter_replay={'version':VERSION,'source_payload_sha256':digest(payload),
                                    'previous_versions':deepcopy(old_versions),'changed_parameters':sorted(changed),
                                    'model_inference_executed':False,'stale_scores_discarded':True})
    return result
