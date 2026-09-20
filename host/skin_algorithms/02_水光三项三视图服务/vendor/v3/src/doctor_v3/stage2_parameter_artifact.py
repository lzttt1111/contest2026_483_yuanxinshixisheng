"""Applicability gate for provenance-bearing parameter candidates; never creates approval."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .stage1_config import digest, resolve_config
from .stage1_schema import version_identity
from .stage1_scoring import CONTRACT_FIELDS
from .stage1_reference import validate_reference_qualification
from . import stage2_parameter_fit as fit

VERSION='v301-parameter-application-1'
PARAMETER_METRICS={'large_pore_area':'01.large_density','porphyrin_high_intensity':'02.porphyrin_high_density'}
WEIGHTING='one_total_weight_per_nonempty_subject; each_instance_weight=1/n'
METHOD='exact_rational_empirical_inverse_cdf_no_interpolation'


def _require(condition,reason):
    if not condition:raise ValueError('parameter_artifact:'+reason)


def _sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _source(row):
    _require(isinstance(row,dict) and isinstance(row.get('subject_id'),str) and bool(row['subject_id'].strip()),'invalid_source_subject')
    _require(all(_sha(row.get(k)) for k in ('input_sha256','result_sha256','manifest_sha256')),'invalid_source_hash')
    return {k:row[k] for k in ('subject_id','input_sha256','result_sha256','manifest_sha256')}


def prepare_parameter_application(payload,artifact,*,research=False,qualified_reference=None,expected_artifact_sha256=None):
    """Return {parameters, provenance}, to retain together around parameter replay.

    Default requires an existing stage1-qualified reference whose independent
    metric validation results bind this exact parameter artifact/protocol/source
    identity and cutoff. research=True is explicitly non-formal and pending.
    No scores, approval record, or validation artifact is generated here.
    """
    _require(type(research) is bool,'research_must_be_boolean')
    _require(isinstance(artifact,dict) and artifact.get('schema_version')==fit.VERSION,'unsupported_artifact_version')
    artifact_sha=digest(artifact)
    if expected_artifact_sha256 is not None:_require(expected_artifact_sha256==artifact_sha,'artifact_sha256_mismatch')
    _require(artifact.get('status')=='candidate_not_validated','artifact_has_no_applicable_candidate')
    _require(payload.get('capture_profile')=='consumer','consumer_parameter_cannot_transfer_to_institution')
    parameters,protocol=artifact.get('candidate_parameters'),artifact.get('protocol')
    _require(isinstance(parameters,dict) and set(parameters)<=set(PARAMETER_METRICS),'unsupported_parameter_family')
    _require(digest(parameters)==artifact.get('candidate_parameters_sha256'),'parameter_sha256_mismatch')
    _require(isinstance(protocol,dict) and digest(protocol)==artifact.get('protocol_sha256'),'protocol_sha256_mismatch')
    _require(protocol.get('schema_version')==fit.VERSION and protocol.get('version')==artifact.get('version')
             and isinstance(artifact.get('version'),str) and bool(artifact['version'].strip()),'version_mismatch')
    _require(protocol.get('implementation_sha256')==hashlib.sha256(Path(fit.__file__).read_bytes()).hexdigest(),'stale_parameter_fitter')
    _require(protocol.get('selection_split')=='train' and protocol.get('confirmation_used_for_parameter_selection') is False
             and artifact.get('confirmation_measurements_used') is False and protocol.get('roi_pooling')=='none'
             and protocol.get('weighting')==WEIGHTING and protocol.get('quantile_method')==METHOD,'unapproved_selection_protocol')
    q,n=protocol.get('quantile'),protocol.get('min_subjects')
    _require(type(q) in (int,float) and math.isfinite(q) and 0<q<=1 and type(n) is int and n>0,'invalid_protocol_inputs')
    for field in ('data','roi','formula','implementation','thresholds'):
        _require(protocol.get('input_measurement_versions',{}).get(field)==payload.get('versions',{}).get(field)
                 and isinstance(payload.get('versions',{}).get(field),str) and bool(payload['versions'][field]),'source_version_mismatch:'+field)
    development=artifact.get('development_evidence')
    _require(isinstance(development,dict) and digest(development)==artifact.get('development_evidence_sha256'),'development_evidence_hash_mismatch')
    declared={}
    for source in development.get('development_sources',[]):
        base=_source(source)
        _require(base['subject_id'] not in declared,'duplicate_development_subject')
        declared[base['subject_id']]=base
    indexed={}
    for row in payload.get('measurements',[]):
        key=row['metric_id']+':'+row['region']
        _require(key not in indexed,'duplicate_target_measurement')
        indexed[key]=row
    checked={};patch={}
    for parameter,regions in parameters.items():
        _require(isinstance(regions,dict),'parameter_must_be_roi_mapping')
        if not regions:continue
        metric=PARAMETER_METRICS[parameter]
        previous=payload.get('measurement_config',{}).get(parameter)
        merged=deepcopy(previous) if isinstance(previous,dict) else {
            r['region']:previous for r in indexed.values() if r['metric_id']==metric} if previous is not None else {}
        for region,cutoff in regions.items():
            _require(type(cutoff) in (int,float) and math.isfinite(cutoff) and cutoff>0
                     and (parameter!='porphyrin_high_intensity' or cutoff<=1),'invalid_cutoff')
            key=metric+':'+region;row=indexed.get(key);report=artifact.get('roi_reports',{}).get(key)
            _require(isinstance(row,dict) and isinstance(report,dict),'missing_target_roi')
            identity=report.get('identity')
            _require(isinstance(identity,dict) and all(isinstance(identity.get(k),str) and bool(identity[k].strip())
                     and row.get(k)==identity[k] for k in CONTRACT_FIELDS),'roi_measurement_identity_mismatch:'+key)
            _require(identity['capture_profile']=='consumer' and identity['metric_id']==metric and identity['region']==region,'roi_profile_or_metric_mismatch')
            for field,version in (('roi_version','roi'),('formula_identity','implementation'),('threshold_identity','thresholds')):
                _require(row[field]==payload['versions'][version],'row_payload_identity_mismatch:'+field)
            _require(report.get('parameter')==parameter and report.get('roi')==region and report.get('cutoff')==cutoff
                     and report.get('status')=='candidate' and report.get('method')==METHOD
                     and report.get('protocol_sha256')==artifact['protocol_sha256'],'roi_candidate_binding_mismatch')
            sources,empty=report.get('sources'),report.get('empty_sources')
            _require(isinstance(sources,list) and isinstance(empty,list),'missing_roi_sources')
            _require(report.get('source_evidence_sha256')==digest({'sources':sources,'empty_sources':empty}),'roi_source_hash_mismatch')
            seen=set();instances=0
            for source in sources+empty:
                base=_source(source);subject=base['subject_id']
                _require(subject not in seen and declared.get(subject)==base,'roi_source_not_in_development')
                seen.add(subject)
                _require(_sha(source.get('instance_values_sha256')) and type(source.get('instance_count')) is int,'invalid_instance_provenance')
                if source in sources:
                    _require(source['instance_count']>0,'nonempty_source_required');instances+=source['instance_count']
                else:_require(source['instance_count']==0,'empty_source_not_empty')
            _require(len(sources)>=n and report.get('contributing_subjects')==len(sources)
                     and report.get('instance_count')==instances and report.get('observed_empty_subjects')==len(empty),'roi_source_count_mismatch')
            checked[key]={'parameter_name':parameter,'region':region,'cutoff':cutoff,
                          'source_identity_sha256':digest(identity),'parameter_artifact_sha256':artifact_sha,
                          'protocol_sha256':artifact['protocol_sha256'],'candidate_parameters_sha256':artifact['candidate_parameters_sha256']}
            merged[region]=cutoff
        patch[parameter]=merged
    _require(bool(checked) and artifact.get('fitted_roi_count')==len(checked),'fitted_roi_count_mismatch')
    provenance={'schema_version':VERSION,'artifact_version':artifact['version'],'artifact_sha256':artifact_sha,
                'candidate_parameters_sha256':artifact['candidate_parameters_sha256'],'protocol_sha256':artifact['protocol_sha256'],
                'development_evidence_sha256':artifact['development_evidence_sha256'],'roi_checks':checked,
                'replay_parameters_sha256':digest(patch),'identity_basis':'legacy_manifest_subject_and_sha',
                'patient_identity_verified':False,'clinical_validation':'not_performed'}
    if research:
        _require(qualified_reference is None,'research_and_formal_reference_are_mutually_exclusive')
        provenance.update(status='research_candidate',formal_report_eligible=False,independent_validation_status='independent_validation_pending')
    else:
        _require(qualified_reference is not None,'independent_parameter_validation_required')
        proof=validate_reference_qualification(qualified_reference)
        _require(proof['status']=='qualified' and proof['formal_report_eligible'],'formal_reference_qualification_required')
        config=resolve_config({**deepcopy(payload.get('measurement_config',{})),**patch})
        projected=version_identity(config,qualified_reference)
        validation=qualified_reference['qualification']['validation_artifact']['metric_results']
        train_inputs={r['subject_id']:r['input_sha256'] for r in qualified_reference['qualification']['input_manifest'] if r['split']=='train'}
        for key,binding in checked.items():
            row=indexed[key];metric=qualified_reference.get('metrics',{}).get(key,{})
            applied={**{k:row[k] for k in CONTRACT_FIELDS},'roi_version':projected['roi'],
                     'formula_identity':projected['implementation'],'threshold_identity':projected['thresholds']}
            _require(all(metric.get(k)==v for k,v in applied.items()),'qualified_reference_applied_identity_mismatch:'+key)
            result=validation[key]['result']
            _require(result.get('parameter_application')==binding,'independent_validation_does_not_bind_parameter_application:'+key)
            for source in artifact['roi_reports'][key]['sources']:
                _require(train_inputs.get(source['subject_id'])==source['input_sha256'],'parameter_source_not_in_qualified_training')
        provenance.update(status='qualified_engineering_candidate',formal_report_eligible=True,
                          independent_validation_status='qualified',reference_qualification=proof,
                          qualified_reference_sha256=proof['proof']['reference_sha256'])
    return {'parameters':deepcopy(patch),'provenance':provenance}
