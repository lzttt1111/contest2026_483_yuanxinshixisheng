"""Bind missing institution report scores to the same subject's scored RGB route."""
from copy import deepcopy


def _fill(target, source, proof):
    if not isinstance(target, dict) or not isinstance(source, dict):
        return 0
    filled = 0
    if 'score' in target and target.get('score') is None and source.get('score') is not None:
        target['native_unscored_state'] = {key: deepcopy(target.get(key)) for key in
                                           ('value', 'unit', 'status', 'reason', 'no_score_reason')}
        for key in ('score_raw', 'score', 'grade'):
            target[key] = deepcopy(source[key])
        target.update(status='candidate', method='paired_same_subject_rgb_score',
                      assessment_basis='paired_consumer_rgb_reference',
                      paired_score_proof=deepcopy(proof))
        target.pop('no_score_reason', None)
        filled += 1
    for key, child in list(target.items()):
        if key in source and isinstance(child, dict) and isinstance(source[key], dict):
            filled += _fill(child, source[key], proof)
    return filled


def apply_paired_consumer_scores(institution_model, consumer_model, institution_stage, consumer_stage):
    if not institution_model.get('capture_profile') == institution_stage.get('capture_profile') == 'institution':
        raise ValueError('institution report/stage profile mismatch')
    if not consumer_model.get('capture_profile') == consumer_stage.get('capture_profile') == 'consumer':
        raise ValueError('consumer report/stage profile mismatch')
    if institution_model.get('subject_id') != consumer_model.get('subject_id'):
        raise ValueError('paired scoring requires the same subject')
    institution_rgb = institution_stage.get('input_sha256', {}).get('RGB_M')
    consumer_rgb = consumer_stage.get('input_sha256', {}).get('RGB_M')
    if not institution_rgb or institution_rgb != consumer_rgb:
        raise ValueError('paired scoring requires identical RGB input SHA256')
    proof = {'method': 'paired_same_subject_rgb_score', 'subject_id': consumer_model['subject_id'],
             'rgb_input_sha256': consumer_rgb, 'consumer_reference': consumer_stage['versions']['reference'],
             'consumer_scoring': consumer_stage['versions']['scoring'],
             'institution_measurements_preserved': True, 'numeric_measurement_imputed': False,
             'clinical_validation': 'not_performed'}
    result = deepcopy(institution_model)
    # Equal RGB bytes identify a capture, not equivalent CP/PP/UV measurements.
    # A validated per-metric cross-profile mapping is not available here.
    result['paired_consumer_scoring'] = {
        **proof, 'filled_score_nodes': 0, 'status': 'not_applied',
        'reason': 'validated_cross_profile_metric_mapping_required'}
    return result
