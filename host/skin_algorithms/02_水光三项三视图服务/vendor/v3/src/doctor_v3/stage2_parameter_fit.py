"""Development-only, subject-equal instance quantiles; no medical cutoff is implied."""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import math
from pathlib import Path

from .stage1_config import digest
from .stage1_schema import required_items
from .stage1_scoring import CONTRACT_FIELDS
from .stage2_reference import _check_dataset, _read_sample

VERSION='v301-subject-equal-parameter-fit-1'
TARGETS={'01.large_density':('large_pore_area','instance_areas_px2',('01.density','01.area_p50')),
         '02.porphyrin_high_density':('porphyrin_high_intensity','instance_intensities',('02.porphyrin_p90',))}
MISSING=('missing_evidence','insufficient_quality','unavailable')


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


def _context(key,basis):
    local=basis.get(key)
    if not isinstance(local,dict):raise ValueError('missing_basis')
    ref=local.get('evidence_ref')
    shared=basis.get(ref,{}) if ref is not None else {}
    if not isinstance(shared,dict) or (ref is not None and ref not in basis):raise ValueError('missing_shared_basis')
    return {**shared,**local}


def _state(row,context):
    status=row.get('measurement_status',row.get('status'))
    if status!=context.get('measurement_status') or row.get('reason')!=context.get('reason'):
        raise ValueError('inconsistent_row_basis_observation')
    if status in MISSING:raise ValueError(row.get('reason') or status)
    if context.get('quality_status')!='PASS' or context.get('quality_reason'):
        raise ValueError('quality_not_pass')
    area=context.get('effective_area_px')
    if not _finite(area) or area<100:raise ValueError('insufficient_effective_area')
    return status


def _percentile(values,p):
    values=sorted(values);x=(len(values)-1)*p;i=int(x);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(x-i)


def _observed_samples(row,indexed,basis,versions):
    metric,region=row['metric_id'],row['region'];key=metric+':'+region
    parameter,sample_field,siblings=TARGETS[metric]
    if any(not isinstance(row.get(k),str) or not row[k].strip() for k in CONTRACT_FIELDS):
        raise ValueError('incomplete_measurement_identity')
    for field,version in (('roi_version','roi'),('formula_identity','implementation'),('threshold_identity','thresholds')):
        if row[field]!=versions.get(version):raise ValueError('measurement_identity_disagrees_with_payload')
    context=_context(key,basis);status=_state(row,context)
    if status not in ('pending_parameter','measured','zero_target'):raise ValueError('unproven_parameter_measurement')
    proofs=[]
    for sibling in siblings:
        address=sibling+':'+region;observed=indexed.get(address)
        if observed is None:raise ValueError('missing_observation_measurement:'+sibling)
        proof=_context(address,basis);state=_state(observed,proof)
        for field in ('capture_profile','region','source_kind','definition_version','roi_version','formula_identity','threshold_identity'):
            if observed.get(field)!=row[field]:raise ValueError('sibling_identity_mismatch:'+field)
        proofs.append((sibling,observed,proof,state))
    samples,count,area=context.get(sample_field),context.get('count'),context.get('effective_area_px')
    if not isinstance(samples,list) or type(count) is not int or count<0 or len(samples)!=count:
        raise ValueError('incomplete_instance_population')
    if any(not _finite(v) or (v<=0 if parameter=='large_pore_area' else not 0<=v<=1) for v in samples):
        raise ValueError('invalid_instance_values')
    for sibling,observed,proof,state in proofs:
        if proof.get(sample_field)!=samples or proof.get('count')!=count or proof.get('effective_area_px')!=area:
            raise ValueError('inconsistent_instance_observation')
        if parameter=='large_pore_area' and proof.get('unlocated_instance_count')!=0:
            raise ValueError('missing_instance_geometry')
        if sibling.endswith('.density'):
            if state!='measured' or not _finite(observed.get('value')) or not math.isclose(observed['value'],count*100000/area,rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('inconsistent_observed_density')
        elif count:
            p=.5 if parameter=='large_pore_area' else .9
            if state!='measured' or not _finite(observed.get('value')) or not math.isclose(observed['value'],_percentile(samples,p),rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('inconsistent_observed_percentile')
        elif state not in ('no_target','no_targets') or observed.get('value') is not None or observed.get('reason')!='no_valid_targets':
            raise ValueError('unproven_empty_observation')
    if parameter=='large_pore_area' and context.get('unlocated_instance_count')!=0:
        raise ValueError('missing_instance_geometry')
    return samples,{k:row[k] for k in CONTRACT_FIELDS}


def _weighted_quantile(histogram,subject_count,quantile):
    """Exact empirical inverse CDF; each contributing subject has total mass one."""
    target=Fraction(str(quantile))*subject_count;mass=Fraction(0)
    for value,counts_by_subject_size in sorted(histogram.items()):
        mass+=sum((Fraction(count,size) for size,count in counts_by_subject_size.items()),Fraction(0))
        if mass>=target:return value
    raise ValueError('weighted population has insufficient total mass')


def fit_parameters(dataset,*,quantile,min_subjects,version):
    """Fit two per-ROI candidate parameters from successful development subjects only.

    Observation-empty subjects remain in the audit but have no artificial zero
    instance and contribute no distribution. No confirmation measurement is read.
    """
    if not _finite(quantile) or not 0<quantile<=1:raise ValueError('explicit quantile must be within (0,1]')
    if type(min_subjects) is not int or min_subjects<1:raise ValueError('explicit positive min_subjects required')
    if not isinstance(version,str) or not version.strip():raise ValueError('explicit candidate version required')
    _check_dataset(dataset)
    protocol={'schema_version':VERSION,'version':version,'quantile':quantile,'min_subjects':min_subjects,
              'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_measurement_versions':dict(dataset['versions']),
              'selection_split':'train','confirmation_used_for_parameter_selection':False,
              'weighting':'one_total_weight_per_nonempty_subject; each_instance_weight=1/n',
              'quantile_method':'exact_rational_empirical_inverse_cdf_no_interpolation',
              'quality_rule':'PASS_and_existing_stage1_effective_area_gate',
              'minimum_effective_area_px':100,'effective_area_rule_source':'existing_stage1_measurement_gate_not_fitted',
              'empty_observation_policy':'record_separately_no_zero_imputation',
              'parameters':['large_pore_area','porphyrin_high_intensity'],'roi_pooling':'none',
              'clinical_validation':'not_performed','anatomical_extent_fitted':False}
    keys={r['metric_id']+':'+r['region'] for r in required_items('consumer',dataset['measurement_config']) if r['metric_id'] in TARGETS}
    accum={key:{'hist':defaultdict(Counter),'sources':[],'empty':[],'rejections':Counter(),'identities':{}} for key in keys}
    development_sources=[]
    for sample in dataset['samples']:
        if sample['split']!='train':continue
        payload=_read_sample(sample)
        development_sources.append({k:sample[k] for k in ('subject_id','input_sha256','result_sha256','manifest_sha256')})
        indexed={r['metric_id']+':'+r['region']:r for r in payload['measurements']}
        for key,bucket in accum.items():
            row=indexed.get(key)
            if row is None:bucket['rejections']['missing_required_measurement']+=1;continue
            try:samples,identity=_observed_samples(row,indexed,payload['basis'],payload['versions'])
            except ValueError as exc:bucket['rejections'][str(exc)]+=1;continue
            source={k:sample[k] for k in ('subject_id','input_sha256','result_sha256','manifest_sha256')}
            source.update(instance_count=len(samples),instance_values_sha256=digest(samples))
            if not samples:bucket['empty'].append(source);continue
            bucket['sources'].append(source);bucket['identities'][digest(identity)]=identity
            for value,count in Counter(samples).items():bucket['hist'][value][len(samples)]+=count
    parameters={'large_pore_area':{},'porphyrin_high_intensity':{}};reports={}
    for key,bucket in sorted(accum.items()):
        metric,region=key.split(':',1);parameter=TARGETS[metric][0]
        subjects=len(bucket['sources']);instances=sum(r['instance_count'] for r in bucket['sources'])
        reasons=dict(bucket['rejections']);cutoff=None
        if len(bucket['identities'])>1:reasons['mixed_measurement_identity']=len(bucket['identities'])
        if subjects<min_subjects:reasons['insufficient_contributing_subjects']=subjects
        if not any(value>0 for value in bucket['hist']):reasons['no_positive_instance_values']=instances
        if subjects>=min_subjects and len(bucket['identities'])==1 and any(v>0 for v in bucket['hist']):
            cutoff=_weighted_quantile(bucket['hist'],subjects,quantile)
            if cutoff<=0:reasons['selected_quantile_not_positive']=1;cutoff=None
        if cutoff is not None:parameters[parameter][region]=cutoff
        sources=sorted(bucket['sources'],key=lambda r:r['subject_id'])
        empty=sorted(bucket['empty'],key=lambda r:r['subject_id'])
        reports[key]={'status':'candidate' if cutoff is not None else 'insufficient_evidence','cutoff':cutoff,
                      'parameter':parameter,'roi':region,'contributing_subjects':subjects,'instance_count':instances,
                      'observed_empty_subjects':len(empty),'sources':sources,'empty_sources':empty,
                      'source_evidence_sha256':digest({'sources':sources,'empty_sources':empty}),
                      'identity':next(iter(bucket['identities'].values())) if len(bucket['identities'])==1 else None,
                      'reasons':reasons,'method':protocol['quantile_method'],'protocol_sha256':digest(protocol)}
    failed=deepcopy([{k:v for k,v in sample.items() if k!='terminal_record_file'} for sample in dataset['excluded_samples'] if sample['split']=='train'])
    evidence={'development_sources':sorted(development_sources,key=lambda r:r['subject_id']),'failed_development':failed}
    fitted_count=sum(len(rows) for rows in parameters.values())
    return {'schema_version':VERSION,'version':version,'status':'candidate_not_validated' if fitted_count else 'insufficient_evidence',
            'fitted_roi_count':fitted_count,
            'candidate_parameters':parameters,'candidate_parameters_sha256':digest(parameters),
            'protocol':protocol,'protocol_sha256':digest(protocol),'roi_reports':reports,
            'development_evidence_sha256':digest(evidence),'development_evidence':evidence,
            'dataset_binding_sha256':dataset['binding_sha256'],'frozen_development_count':dataset['cohort_counts']['train'],
            'identity_basis':'legacy_manifest_subject_and_sha','patient_identity_verified':False,
            'anatomical_extent':{'status':'not_fitted','reason':'numeric_anatomical_segments_require_separate_confirmation_and_validation'},
            'confirmation_measurements_used':False,'clinical_validation':'not_performed'}
