"""Scoring with explicit reference lineage and independent availability states."""
from bisect import bisect_left, bisect_right
import math
from .stage1_config import digest
from .scoring import score_metric, display_score, grade
from .severity_guard import apply_score, GROUPS
from .stage1_reference import validate_reference_qualification, ReferenceQualificationError

IDENTITY_FIELDS = ('roi_version', 'formula_identity', 'threshold_identity')
CONTRACT_FIELDS = ('metric_id', 'unit', 'definition_version', 'capture_profile',
                   'region', 'source_kind', 'direction', *IDENTITY_FIELDS)


def _basis_context(measurement, basis):
    key = measurement['metric_id'] + ':' + measurement['region']
    local = basis.get(key) if isinstance(basis, dict) else None
    if not isinstance(local, dict):
        return None
    shared = {}
    if 'evidence_ref' in local:
        ref = local['evidence_ref']
        shared = basis.get(ref) if isinstance(ref, str) else None
        if not isinstance(shared, dict) or 'evidence_ref' in shared:
            return None
    return {**shared, **local}


def _observed_negative(measurement, basis):
    """A valid no-target observation is a health state, not missing data."""
    state = measurement.get('measurement_status', measurement.get('status'))
    if (measurement.get('direction') != 'higher_burden' or measurement.get('value') is not None
            or state not in ('no_target', 'no_targets')
            or measurement.get('reason') not in ('no_target', 'no_valid_targets')):
        return None
    context = _basis_context(measurement, basis)
    if not isinstance(context, dict) or context.get('quality_status') != 'PASS':
        return None
    if context.get('quality_reason') or context.get('reason') not in (None, 'no_target', 'no_valid_targets'):
        return None
    area = context.get('effective_area_px', context.get('valid_area_px'))
    if isinstance(area, bool) or not isinstance(area, (int, float)) or not math.isfinite(area) or area < 100:
        return None
    metric = measurement['metric_id']
    def zero_number(key):
        value = context.get(key)
        return type(value) in (int, float) and value == 0
    if metric.startswith('01.'):
        zero = (zero_number('count') and context.get('instance_areas_px2') == []
                and context.get('instance_ids') == [] and zero_number('unlocated_instance_count'))
    elif metric.startswith('02.gloss_'):
        pixels = context.get('pixel_score_summary')
        zero = isinstance(pixels, dict) and type(pixels.get('count')) is int and pixels['count'] == 0
    elif metric.startswith('02.porphyrin_'):
        zero = zero_number('count') and context.get('instance_intensities') == []
    elif metric.startswith('03.'):
        pixels = context.get('pixel_score_summary')
        zero = (zero_number('affected_area_px') and isinstance(pixels, dict)
                and type(pixels.get('count')) is int and pixels['count'] == 0
                and ('.spots.' not in metric or context.get('support_authorized') is True))
    elif metric.startswith(('07.', '08.')):
        zero = zero_number('line_pixels')
    elif metric.startswith('09.'):
        zero = zero_number('affected_area_px') and zero_number('image_line_pixels')
    elif metric.startswith('10.'):
        zero = (zero_number('class_area_px') and zero_number('unclassified_texture_area_px')
                and context.get('missing_exclusions') == [])
    else:
        zero = False
    if not zero:
        return None
    return {'valid_area_px': area, 'basis_sha256': digest(context), 'metric_id': metric,
            'region': measurement['region']}


def _raw_measurement_score(measurement, reference, measured):
    """An anatomical state has already been mapped by the explicit range rule."""
    if measurement.get('source_kind') == 'anatomical_extent_state':
        if (measurement['metric_id'] != '09.extent' or measurement['direction'] != 'higher_health'
                or reference.get('scoring_method') != 'anatomical_range_state_identity'):
            raise ValueError('anatomical extent needs its explicit state mapping, not ECDF')
        value = measurement.get('value') if measured else 0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('invalid anatomical range state score')
        return {'metric_id': measurement['metric_id'], 'definition_version': measurement['definition_version'],
                'value': measurement.get('value'), 'unit': measurement['unit'], 'status': 'candidate',
                'score_raw': value, 'score': display_score(value), 'grade': grade(value),
                'reference_version': reference['version'], 'population_sha256': reference['population_sha256'],
                'method': 'anatomical_range_state_identity', 'no_second_ecdf': True}
    if reference.get('scoring_method', 'ecdf') != 'ecdf':
        raise ValueError('non-anatomical metric cannot use anatomical state mapping')
    trace = score_metric(dict(measurement, status='measured', value=measurement.get('value') if measured else 0), reference)
    if measurement['direction'] == 'higher_health':
        values = reference['sorted_values']
        value = measurement.get('value') if measured else 0
        raw = 100 * (bisect_left(values, value) + bisect_right(values, value)) / (2 * len(values))
        trace.update(score_raw=raw, score=display_score(raw), grade=grade(raw), method='midrank_ecdf')
    return trace


def _identity_validation(measurement, reference):
    checks = {}
    for field in CONTRACT_FIELDS:
        actual, expected = measurement.get(field), reference.get(field)
        matched = (isinstance(actual, str) and bool(actual.strip())
                   and isinstance(expected, str) and bool(expected.strip()) and actual == expected)
        checks[field] = {'measurement': actual, 'reference': expected, 'matched': matched}
    mismatches = [field for field, check in checks.items() if not check['matched']]
    return {'status': 'mismatched' if mismatches else 'matched',
            'checks': checks, 'mismatched_fields': mismatches}



def _measured_zero(measurement, basis):
    if (measurement.get('direction') != 'higher_burden' or measurement.get('value') != 0
            or measurement.get('status') != 'measured'
            or measurement.get('measurement_status', measurement.get('status')) not in ('measured','zero_target')):
        return None
    context = _basis_context(measurement, basis)
    if not isinstance(context, dict) or context.get('quality_status') != 'PASS' or context.get('quality_reason'):
        return None
    area = context.get('effective_area_px', context.get('valid_area_px'))
    if isinstance(area, bool) or not isinstance(area, (int,float)) or not math.isfinite(area) or area < 100:
        return None
    return {'valid_area_px': area, 'basis_sha256': digest(context)}


def score_measurements(measurements, reference=None, allow_test_reference=False, *, basis=None):
    """Reference: {version, source, purpose: formal|test, metrics: {id:region: row}}.

    Both measurement and reference rows require identical nonempty string
    roi_version, formula_identity and threshold_identity; legacy rows cannot
    implicitly opt into a new formal reference. Identity checks remain in trace.
    """
    qualification = {'status': 'not_checked', 'formal_report_eligible': False, 'proof': {}}
    if reference is not None:
        try:
            qualification = validate_reference_qualification(reference)
        except ReferenceQualificationError as exc:
            qualification = {'status': 'invalid', 'formal_report_eligible': False,
                             'reason': str(exc), 'proof': {}}
    output = {}
    for m in measurements:
        key = m['metric_id'] + ':' + m['region']
        if key in output:
            raise ValueError('duplicate measurement: ' + key)
        status = m.get('measurement_status', m.get('status', 'unavailable'))
        item = {'measurement_status': status, 'reference_status': 'missing',
                'score_status': 'unavailable', 'value': m.get('value'),
                'score_raw': None, 'score': None, 'grade': '不可评估',
                'trace': {'reason': 'missing_compatible_reference', 'formal_report_eligible': False,
                          'reference_qualification': qualification,
                          **{field: m.get(field) for field in IDENTITY_FIELDS},
                          'reference_identity_validation': {'status': 'not_checked', 'checks': {}, 'mismatched_fields': []}}}
        output[key] = item
        measured = m.get('status') == 'measured' and status in ('measured', 'zero_target')
        negative = _observed_negative(m, basis or {})
        scorable = measured or negative is not None
        if not scorable:
            item['trace']['reason'] = m.get('reason') or status
        if reference is None:
            continue
        if not isinstance(reference, dict) or not all(isinstance(reference.get(k), str) and reference[k].strip() for k in ('version', 'source', 'purpose')):
            item['reference_status'] = 'invalid'
            item['trace']['reason'] = 'missing_reference_lineage'
            continue
        purpose = reference['purpose']
        if purpose not in ('formal', 'test') or (purpose == 'test' and not allow_test_reference):
            item['reference_status'] = 'rejected'
            item['trace']['reason'] = 'reference_purpose_not_allowed'
            continue
        populations = reference.get('metrics')
        ref = populations.get(key) if isinstance(populations, dict) else None
        if ref is None:
            zero = _measured_zero(m, basis or {})
            if zero is not None and qualification['status'] in ('qualified','test_only'):
                raw = 100.0
                trace = {'metric_id':m['metric_id'],'value':m['value'],'unit':m['unit'],
                         'definition_version':m['definition_version'],'score_raw':raw,'score':100,
                         'grade':'未见明显','method':'observed_zero_measurement_anchor',
                         'observed_zero_evidence':zero,'numeric_statistic_imputed':False,
                         'reference_version':reference['version'],'reference_source':reference['source'],
                         'reference_purpose':purpose,'formal_report_eligible':purpose=='formal' and qualification['formal_report_eligible'],
                         'reference_qualification':qualification,**{field:m[field] for field in IDENTITY_FIELDS}}
                item.update(reference_status='compatible',score_status='test_only' if purpose=='test' else 'candidate',
                            score_raw=raw,score=100,grade='未见明显',trace=trace)
            continue
        effective_ref = dict(ref) if isinstance(ref,dict) else ref
        validation = _identity_validation(m, effective_ref if isinstance(effective_ref, dict) else {})
        item['trace']['reference_identity_validation'] = validation
        trace = None
        try:
            if not isinstance(ref, dict) or ref.get('version') != reference['version']:
                raise ValueError('reference version mismatch')
            if validation['mismatched_fields']:
                raise ValueError('reference mismatch: ' + ', '.join(validation['mismatched_fields']))
            if qualification['status'] not in ('qualified', 'test_only'):
                raise ValueError('reference_qualification_failed: ' + qualification.get('reason', 'unqualified'))
            if m['direction'] not in ('higher_burden', 'higher_health'):
                raise ValueError('unsupported reference direction')
            population_digest = ref.get('population_sha256')
            if not isinstance(population_digest, str) or len(population_digest) != 64 or any(c not in '0123456789abcdef' for c in population_digest):
                raise ValueError('invalid reference population SHA256')
            method=ref.get('scoring_method','ecdf')
            if method in ('observed_zero_count','anatomical_range_state_identity','observed_zero_high_density'):
                from .stage2_direct_scores import score_direct_state
                state=score_direct_state(m,basis or {},reference,method,
                                         allow_test_reference=allow_test_reference)
                if state.get('score') is None:
                    if method == 'observed_zero_count' and measured and isinstance(m.get('value'), (int,float)) and m['value'] > 0:
                        from .severity_guard import absolute_count_score
                        context = _basis_context(m, basis or {}) or {}
                        valid = (context.get('quality_status') == 'PASS' and not context.get('quality_reason')
                                 and context.get('unknown_count') == 0
                                 and context.get('observed_classified_count') == m['value'])
                        absolute = absolute_count_score(m['value'], context.get('effective_area_px', context.get('valid_area_px')), valid)
                        if absolute.get('score') is not None:
                            trace = {**absolute, 'metric_id':m['metric_id'],'value':m['value'],'unit':m['unit'],
                                     'definition_version':m['definition_version'],'reference_version':reference['version'],
                                     'population_sha256':ref['population_sha256'],
                                     'method':'absolute_observed_class_count_after_validated_zero_population',
                                     'reference_qualification':qualification,'formal_report_eligible':purpose=='formal',
                                     'zero_population_reference_method':'observed_zero_count'}
                            state = {'score':absolute['score']}
                    if state.get('score') is not None:
                        pass
                    else:
                        item['reference_status']='compatible_test' if purpose=='test' else 'compatible'
                        item['trace'].update(reason=state.get('reason','missing_direct_state_evidence'),
                                             formal_report_eligible=False)
                        continue
                if trace is None:
                    trace={**state['trace'],**{name:state[name] for name in ('score_raw','score','grade')},
                           'direct_state_proof':state['proof'],'direct_state_proof_sha256':state['proof_sha256']}
            else:
                trace = _raw_measurement_score(m, effective_ref, measured)
                if negative is not None:
                    trace.update(method='observed_zero_burden_reference_anchor',
                                 observed_negative_evidence=negative,
                                 numeric_statistic_imputed=False,
                                 measurement_value=None, score_input=0)
            trace.update({field: m[field] for field in IDENTITY_FIELDS},
                         reference_identity_validation=validation,
                         reference_qualification=qualification)
        except (ValueError, TypeError, KeyError) as exc:
            item['reference_status'] = 'incompatible'
            item['trace']['reason'] = str(exc)
            continue
        item['reference_status'] = 'compatible_test' if purpose == 'test' else 'compatible'
        if not scorable:
            continue
        module = m.get('module') or m['metric_id'].split('.')[0]
        if module not in GROUPS:
            item['trace']['reason'] = 'unsupported_severity_guard_module'
            continue
        trace.update(reference_source=reference['source'], reference_purpose=purpose,
                       formal_report_eligible=purpose == 'formal' and qualification['formal_report_eligible'])
        item.update(score_status='test_only' if purpose == 'test' else 'candidate',
                    score_raw=trace['score_raw'], score=trace['score'],
                    grade=trace['grade'], trace=trace)
    from .stage1_guard import guard_measurement_scores
    return guard_measurement_scores(measurements, output, basis or {})
