"""Versioned report assessment scope, independent of detector formulas."""
from copy import deepcopy
from .stage1_config import digest

POLICY = {
    'version': 'v3-assessment-scope-20260914-1',
    'scope_basis': 'each_detector_existing_scope',
    'profiles': {
        profile: {'03': {'excluded_regions': ['left_eye', 'right_eye'],
                        'reason': 'outside_existing_pigment_display_scope'}}
        for profile in ('consumer', 'institution')
    },
    'other_regions': 'retain_current_detector_intersection',
    'quality_failure_is_out_of_scope': False,
    'whole_face_measurement': 'retain_original_detector_valid_scope',
}


def excluded_regions(profile, module):
    if profile not in POLICY['profiles']:
        raise ValueError('explicit assessment profile required')
    return set(POLICY['profiles'][profile].get(module, {}).get('excluded_regions', []))


def in_scope(profile, module, region):
    return region not in excluded_regions(profile, module)


def scoped_required_items(profile, configuration=None):
    from .stage1_schema import required_items
    return [r for r in required_items(profile, configuration)
            if in_scope(profile, r['module'], r['region'])]


def annotate_payload_scope(payload):
    profile = payload['capture_profile']
    payload['assessment_scope'] = {'version': POLICY['version'], 'sha256': digest(POLICY),
                                   'policy': deepcopy(POLICY)}
    payload['assessment_coverage'] = [
        r for r in payload['coverage'] if in_scope(profile, r['module'], r['region'])]
    return payload


def retained_table_rows(model, module, rows):
    from .word_slots import REGIONS
    from .word_tables import table_kind
    if not model.get('assessment_scope') or table_kind(rows) != 'regional':
        return list(range(len(rows)))
    return [i for i, row in enumerate(rows)
            if i == 0 or in_scope(model['capture_profile'], module, REGIONS.get(row[0], ''))]


def apply_report_scope(model):
    result = deepcopy(model)
    profile = result['capture_profile']
    removed = []
    for mid, module in result['modules'].items():
        for region in sorted(excluded_regions(profile, mid)):
            for collection in ('regions', 'metrics'):
                module.get(collection, {}).pop(region, None)
            result.get('stage1_native_assessments', {}).get(mid, {}).pop(region, None)
            for key in list(result.get('score_trace', {})):
                if key.startswith(mid + '.') and key.endswith(':' + region):
                    result['score_trace'].pop(key)
            removed.append({'module': mid, 'region': region,
                            'reason': POLICY['profiles'][profile][mid]['reason']})
    result['stage1_coverage'] = [
        r for r in result.get('stage1_coverage', [])
        if in_scope(profile, r['module'], r['region'])]
    result['assessment_scope'] = {
        'version': POLICY['version'], 'sha256': digest(POLICY), 'policy': deepcopy(POLICY),
        'excluded': removed, 'required_count': len(scoped_required_items(
            profile, result.get('stage1_measurement_config'))),
        'detector_measurements_changed': False,
    }
    return result
