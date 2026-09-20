"""Opt-in direct state helpers; no ECDF construction or production integration."""
from copy import deepcopy
import math
from .stage1_config import digest
from .stage1_scoring import CONTRACT_FIELDS
from .stage1_reference import validate_reference_qualification
from ._stage1_anatomical_extent import classify_extent_basis, VERSION as EXTENT_VERSION
from .scoring import display_score, grade

VERSION = 'v301-direct-state-1'
METHODS = ('observed_zero_count','anatomical_range_state_identity','observed_zero_high_density')
COUNT_IDS = {'06.'+kind+'_count' for kind in ('erythema','papule','pustule')}


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _key(row):
    return row['metric_id']+':'+row['region']


def _context(row,basis):
    if not isinstance(basis,dict):return None
    local=basis.get(_key(row))
    if not isinstance(local,dict):return None
    shared={}
    if 'evidence_ref' in local:
        ref=local['evidence_ref']
        if not isinstance(ref,str) or not ref or ref==_key(row):return None
        shared=basis.get(ref)
        if not isinstance(shared,dict) or 'evidence_ref' in shared:return None
    return {**shared,**local}


def _unavailable(method,reason,pending=False):
    return {'method':method,'status':'pending_parameter' if pending else 'unavailable',
            'score_raw':None,'score':None,'grade':'不可评估','reason':reason,
            'formal_report_eligible':False,'clinical_validation':'not_performed'}


def _quality(context):
    area=context.get('effective_area_px') if 'effective_area_px' in context else context.get('valid_area_px')
    return context.get('quality_status')=='PASS' and not context.get('quality_reason') and _finite(area) and area>=100


def _measured(row):
    return row.get('status')=='measured' and row.get('measurement_status',row['status']) in ('measured','zero_target') and row.get('reason') is None


def _candidate(row,context,method,value):
    proof={'version':VERSION,'method':method,'measurement_sha256':digest(row),'basis_sha256':digest(context),
           'metric_id':row['metric_id'],'region':row['region'],'identity':{k:row.get(k) for k in CONTRACT_FIELDS},
           'numeric_statistic_imputed':False,'ecdf_applied':False}
    if method=='anatomical_range_state_identity':
        config=context['extent_classification']['configuration']
        proof['configuration_evidence']={'approval_id':config['approval_id'],'version':config['version'],
            'definition_version':config['definition_version'],'groove_type':config['groove_type'],
            'configuration_sha256':digest(config),'source_sha256':config.get('source_sha256'),
            'approval_scope':'existing_explicit_engineering_configuration_not_clinical_calibration'}
    return {'method':method,'status':'candidate','score_raw':value,'score':display_score(value),'grade':grade(value),
            'reason':None,'formal_report_eligible':False,'clinical_validation':'not_performed',
            'proof':proof,'proof_sha256':digest(proof)}


def observed_zero_count(measurement,basis):
    """Only a genuinely observed negative of the same M06 class yields 100."""
    row=measurement; context=_context(row,basis); method='observed_zero_count'
    if context is None:return _unavailable(method,'missing_or_invalid_observation_basis')
    if row.get('metric_id') not in COUNT_IDS:
        return _unavailable(method,'unsupported_zero_count_metric')
    if not _measured(row) or not _finite(row.get('value')) or int(row['value'])!=row['value'] or row['value']!=0:
        return _unavailable(method,'not_an_observed_zero_class_count')
    if not _quality(context) or context.get('reason') is not None or context.get('measurement_status') not in ('measured','zero_target'):
        return _unavailable(method,'insufficient_zero_count_observation')
    for key in ('unknown_count','observed_classified_count'):
        if not _finite(context.get(key)) or context[key]!=0:
            return _unavailable(method,'unproven_same_class_negative:'+key)
    samples=context.get('classification_samples')
    if not isinstance(samples,list):return _unavailable(method,'missing_classification_observation')
    kind=row['metric_id'].split('.',1)[1].removesuffix('_count')
    if any(not isinstance(s,dict) or not isinstance(s.get('class'),str)
            or s['class'] not in {'erythema','papule','pustule'}
            or s['class']==kind for s in samples):
        return _unavailable(method,'classification_samples_contradict_zero_count')
    return _candidate(row,context,method,100.0)


def anatomical_range_state(measurement,basis):
    """Re-evaluate the existing approved anatomical rule without a second ECDF."""
    row=measurement; context=_context(row,basis); method='anatomical_range_state_identity'
    if context is None:return _unavailable(method,'missing_or_invalid_observation_basis')
    if row.get('metric_id')!='09.extent' or row.get('direction')!='higher_health' or row.get('source_kind')!='anatomical_extent_state':
        return _unavailable(method,'unsupported_anatomical_state_metric')
    anatomy=context.get('anatomical_extent'); saved=context.get('extent_classification',{})
    config=saved.get('configuration')
    if not isinstance(config,dict):return _unavailable(method,'anatomical_extent_threshold_pending',True)
    if config.get('test_only') is not False or config.get('approved') is not True or config.get('purpose')!='formal':
        return _unavailable(method,'test_or_unapproved_anatomical_configuration',True)
    kind='full_face' if row['region']=='full_face' else row['region'].split('_',1)[-1]
    if (not isinstance(anatomy,dict) or anatomy.get('definition_version')!=EXTENT_VERSION
            or config.get('definition_version')!=EXTENT_VERSION or anatomy.get('groove_type')!=kind
            or config.get('groove_type')!=kind):
        return _unavailable(method,'anatomical_definition_or_type_mismatch')
    if (not _quality(context) or not _measured(row) or not _finite(row.get('value'))
            or context.get('measurement_status')!=row.get('measurement_status',row.get('status'))
            or context.get('reason') is not None):
        return _unavailable(method,'insufficient_anatomical_measurement')
    try: actual=classify_extent_basis(anatomy,config)
    except (ValueError,TypeError,KeyError):return _unavailable(method,'invalid_anatomical_configuration_or_evidence')
    if actual.get('status')!='approved_state' or actual!=saved or actual.get('extent_score')!=row['value']:
        return _unavailable(method,'anatomical_classification_does_not_match_saved_measurement')
    return _candidate(row,context,method,float(actual['extent_score']))


def observed_zero_high_density(measurement,basis,*,measurement_config=None):
    """Prove zero over the saved network cells, not a reconstruction of the image.

    Saved phenotype labels remain source observations. A high accepted cell
    anywhere in the saved network blocks every ROI because ROI geometry is absent.
    """
    row=measurement; method=METHODS[2]; context=_context(row,basis)
    if (context is None or row.get('metric_id')!='07.high_area'
            or row.get('direction')!='higher_burden' or row.get('source_kind')!='rgb_line_appearance'
            or not _measured(row) or not _finite(row.get('value')) or row['value']!=0
            or not _quality(context) or context.get('reason') is not None
            or context.get('measurement_status')!='measured'
            or not _finite(context.get('high_area_px')) or context['high_area_px']!=0):
        return _unavailable(method,'unproven_zero_high_density_measurement')
    network_ref=context.get('network_evidence_ref')
    network=basis.get(network_ref) if isinstance(network_ref,str) else None
    if (not isinstance(network,dict) or set(network)!={'cells','fragments','parameters'}
            or not isinstance(network.get('cells'),list) or not network['cells']
            or not isinstance(network.get('fragments'),list)):
        return _unavailable(method,'missing_or_invalid_saved_fine_network')
    from .stage1_lines import PARAMETERS
    params=network['parameters']
    try:
        if (not isinstance(measurement_config,dict) or digest(measurement_config)!=row.get('threshold_identity')
                or not isinstance(params,dict) or set(params)!=set(PARAMETERS)
                or params!=measurement_config.get('lines')
                or any(not _finite(v) or v<=0 for v in params.values())
                or any(type(params[k]) is not int for k in ('grid_px','fine_min_fragments'))
                or not 0<params['fine_min_density']<=params['fine_high_density']<=1
                or params['merge_cosine']>1):
            return _unavailable(method,'fine_network_configuration_mismatch')
        grid=params['grid_px']; seen=set(); total_area=0
        for cell in network['cells']:
            if not isinstance(cell,dict) or set(cell)!={'xyxy','valid_area_px','fragment_count','orientation_bins','density','phenotype_accepted'}:
                return _unavailable(method,'invalid_saved_fine_cell')
            box=cell['xyxy']; area=cell['valid_area_px']; count=cell['fragment_count']
            density=cell['density']; accepted=cell['phenotype_accepted']; bins=cell['orientation_bins']
            if (not isinstance(box,list) or len(box)!=4 or any(type(v) is not int or v<0 for v in box)
                    or box[0]%grid or box[1]%grid or not 0<box[2]-box[0]<=grid or not 0<box[3]-box[1]<=grid
                    or (box[0],box[1]) in seen or type(area) is not int
                    or not max(16,grid*grid//8)<=area<=(box[2]-box[0])*(box[3]-box[1])
                    or type(count) is not int or not 0<=count<=len(network['fragments'])
                    or not _finite(density) or not 0<=density<=1 or type(accepted) is not bool
                    or not isinstance(bins,list) or any(type(v) is not int or v not in range(4) for v in bins)
                    or bins!=sorted(set(bins)) or len(bins)>count
                    or not math.isclose(density*area,round(density*area),rel_tol=0,abs_tol=1e-7)
                    or (count==0 and density!=0)
                    or (accepted and (count<params['fine_min_fragments'] or density<params['fine_min_density']))
                    or (not accepted and count>=params['fine_min_fragments'] and len(bins)>=2 and density>=params['fine_min_density'])):
                return _unavailable(method,'inconsistent_saved_fine_cell')
            if accepted and density>=params['fine_high_density']:
                return _unavailable(method,'positive_high_density_cell_in_saved_network')
            seen.add((box[0],box[1])); total_area+=area
        # Fragment summaries cannot reconstruct original pixels or branch counts;
        # still reject malformed evidence instead of treating it as an empty network.
        ids=set()
        for fragment in network['fragments']:
            if (not isinstance(fragment,dict) or set(fragment)!={'fragment_id','endpoints_xy','length_px','linearity','orientation_radians'}
                    or type(fragment['fragment_id']) is not int or fragment['fragment_id']<0 or fragment['fragment_id'] in ids
                    or type(fragment['length_px']) is not int or fragment['length_px']<=0
                    or not _finite(fragment['linearity']) or fragment['linearity']<0
                    or not _finite(fragment['orientation_radians'])
                    or not isinstance(fragment['endpoints_xy'],list) or len(fragment['endpoints_xy'])!=2
                    or any(not isinstance(p,list) or len(p)!=2 or any(not _finite(v) or v<0 for v in p) for p in fragment['endpoints_xy'])):
                return _unavailable(method,'invalid_saved_fine_fragment')
            ids.add(fragment['fragment_id'])
        candidate=_candidate(row,context,method,100.)
        candidate['proof']['network_evidence']={'network_sha256':digest(network),'configuration_sha256':digest(measurement_config),
            'parameters_sha256':digest(params),'cell_count':len(seen),'saved_cell_valid_area_px':total_area,
            'fine_high_density':params['fine_high_density'],'validation_scope':'saved_grid_consistency_only',
            'original_mask_reconstructed':False,'phenotype_acceptance_independently_validated':False,
            'whole_fine_line_module_scored':False}
        candidate['proof_sha256']=digest(candidate['proof'])
        return candidate
    except (ValueError,TypeError,KeyError,OverflowError):
        return _unavailable(method,'invalid_saved_fine_network_numbers')


def _direct_candidate(measurement,basis,method,measurement_config=None):
    if method==METHODS[0]:return observed_zero_count(measurement,basis)
    if method==METHODS[1]:return anatomical_range_state(measurement,basis)
    if method==METHODS[2]:return observed_zero_high_density(measurement,basis,measurement_config=measurement_config)
    raise ValueError('unsupported direct method')


def verify_direct_evidence(candidate,measurement,basis,*,measurement_config=None):
    method=candidate.get('method')
    if method not in METHODS:raise ValueError('unsupported direct method')
    actual=_direct_candidate(measurement,basis,method,measurement_config)
    if actual['status']!='candidate' or actual!=candidate:raise ValueError('direct state evidence mismatch')
    return deepcopy(actual)


def build_direct_metric_reference(measurement,method,*,version,source_data_sha256):
    """Build only a declaration. The caller must supply independently validated qualification."""
    if method not in METHODS or not isinstance(version,str) or not version.strip():raise ValueError('invalid direct declaration')
    if not isinstance(source_data_sha256,str) or len(source_data_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_data_sha256):
        raise ValueError('source data SHA256 required')
    if any(not isinstance(measurement.get(k),str) or not measurement[k].strip() for k in CONTRACT_FIELDS):
        raise ValueError('complete ten-field measurement identity required')
    if ((method==METHODS[0] and measurement['metric_id'] not in COUNT_IDS)
            or (method==METHODS[1] and (measurement['metric_id']!='09.extent' or measurement['direction']!='higher_health'))
            or (method==METHODS[2] and (measurement['metric_id']!='07.high_area' or measurement['direction']!='higher_burden'
                                      or measurement['source_kind']!='rgb_line_appearance'))):
        raise ValueError('direct declaration metric mismatch')
    return {**{k:measurement[k] for k in CONTRACT_FIELDS},'version':version,'scoring_method':method,
            'source_data_sha256':source_data_sha256,'population_sha256':source_data_sha256,
            'direct_method_version':VERSION,'clinical_validation':'not_performed'}


def validate_direct_reference(reference,measurement,method,*,allow_test_reference=False):
    qualification=validate_reference_qualification(reference)
    testing=qualification['status']=='test_only' and allow_test_reference is True
    if not testing and (qualification['status']!='qualified' or not qualification['formal_report_eligible']):
        raise ValueError('direct formal score requires existing formal qualification')
    metric=reference.get('metrics',{}).get(_key(measurement),{})
    expected=build_direct_metric_reference(measurement,method,version=reference['version'],source_data_sha256=metric.get('source_data_sha256'))
    if any(metric.get(key)!=value for key,value in expected.items()):raise ValueError('direct reference identity or method mismatch')
    if not testing:
        validation=reference['qualification']['validation_artifact']['metric_results'][_key(measurement)]['result']
        if validation.get('direct_score_method')!=method or validation.get('source_data_sha256')!=metric['source_data_sha256']:
            raise ValueError('independent validation does not bind this direct method and data source')
    return {'qualification':qualification,'metric_reference_sha256':digest(metric),'method':method,
            'source_data_sha256':metric['source_data_sha256'],
            'independent_validation_status':'independent_validation_pending' if testing else 'validated',
            'formal_report_eligible':not testing}


def score_direct_state(measurement,basis,reference,method,*,allow_test_reference=False):
    """Qualified helper only. No frozen scorer, guard, Word, or batch is changed."""
    proof=validate_direct_reference(reference,measurement,method,allow_test_reference=allow_test_reference)
    candidate=_direct_candidate(measurement,basis,method,reference.get('thresholds'))
    if candidate['status']!='candidate':return candidate
    metric=reference['metrics'][_key(measurement)]
    trace={**{k:measurement[k] for k in CONTRACT_FIELDS},'value':measurement['value'],
           'population_sha256':metric['population_sha256'],'reference_version':reference['version'],
           'reference_qualification':proof['qualification'],'direct_reference_validation':proof,
           'method':method,'formal_report_eligible':proof['formal_report_eligible'],
           'independent_validation_status':proof['independent_validation_status'],
           'clinical_validation':'not_performed','state_proof_sha256':candidate['proof_sha256']}
    return {**candidate,'status':'candidate' if proof['formal_report_eligible'] else 'test_only',
            'formal_report_eligible':proof['formal_report_eligible'],'reference_proof':proof,'trace':trace,
            'reference_version':reference['version'],'reference_sha256':proof['qualification']['proof']['reference_sha256']}
