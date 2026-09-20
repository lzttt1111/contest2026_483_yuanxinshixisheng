"""Independent saved-instance parameter checks, not clinical validation."""
from collections import Counter
from copy import deepcopy
import math
from .stage1_config import digest
from .stage2_reference import _check_dataset, _read_sample
from .stage2_parameter_fit import fit_parameters, _observed_samples, _context, TARGETS

VERSION = 'v301-parameter-validation-1'


def validate_parameters(dataset, artifact, *, min_confirmation_subjects, version):
    if type(min_confirmation_subjects) is not int or min_confirmation_subjects < 1:
        raise ValueError('explicit positive confirmation support required')
    if not isinstance(version, str) or not version.strip():
        raise ValueError('explicit validation version required')
    _check_dataset(dataset)
    protocol = artifact['protocol']
    reproduced = fit_parameters(dataset, quantile=protocol['quantile'],
                                min_subjects=protocol['min_subjects'], version=artifact['version'])
    if reproduced != artifact:
        raise ValueError('parameter candidate does not reproduce from frozen development results')
    fitted = {key: report for key, report in artifact['roi_reports'].items()
              if report['status'] == 'candidate'}
    details = {key: {'observations': [], 'empty_subjects': [], 'rejections': Counter()}
               for key in fitted}
    for sample in dataset['samples']:
        if sample['split'] != 'confirmation':
            continue
        payload = _read_sample(sample)
        indexed = {r['metric_id'] + ':' + r['region']: r for r in payload['measurements']}
        for key, report in fitted.items():
            target = details[key]
            row = indexed.get(key)
            if row is None:
                target['rejections']['missing_required_measurement'] += 1
                continue
            try:
                values, identity = _observed_samples(row, indexed, payload['basis'], payload['versions'])
                if identity != report['identity']:
                    raise ValueError('confirmation_identity_differs_from_development')
            except ValueError as exc:
                target['rejections'][str(exc)] += 1
                continue
            context = _context(key, payload['basis'])
            cutoff = report['cutoff']
            count = sum(value >= cutoff for value in values)
            density = count * 100000 / context['effective_area_px']
            if not math.isfinite(density) or not 0 <= count <= len(values):
                raise ValueError('invalid independently reconstructed parameter statistic')
            # The same explicit inclusive boundary as the stage-one computation.
            observation = {k: sample[k] for k in
                           ('subject_id', 'input_sha256', 'result_sha256', 'manifest_sha256')}
            observation.update(instance_count=len(values), qualifying_instance_count=count,
                               effective_area_px=context['effective_area_px'], density=density,
                               retained_fraction=count / len(values) if values else None,
                               instance_values_sha256=digest(values), source_identity_sha256=digest(identity))
            target['observations'].append(observation)
            if not values:
                target['empty_subjects'].append(sample['subject_id'])
    results = {}
    for key, report in fitted.items():
        target = details[key]
        nonempty = [row for row in target['observations'] if row['instance_count'] > 0]
        fractions = [row['retained_fraction'] for row in nonempty]
        sufficient = len(nonempty) >= min_confirmation_subjects
        results[key] = {
            'status': 'passed' if sufficient else 'failed',
            'parameter_name': report['parameter'], 'region': report['roi'], 'cutoff': report['cutoff'],
            'source_identity_sha256': digest(report['identity']),
            'training_subjects': [row['subject_id'] for row in report['sources']],
            'confirmation_subjects': [row['subject_id'] for row in nonempty],
            'result': {
                'independent_array_reconstruction': True,
                'comparison': 'instance_value_greater_than_or_equal_to_cutoff',
                'confirmation_nonempty_count': len(nonempty),
                'confirmation_empty_count': len(target['empty_subjects']),
                'minimum_confirmation_count': min_confirmation_subjects,
                'retained_fraction_mean': sum(fractions) / len(fractions) if fractions else None,
                'retained_fraction_min': min(fractions) if fractions else None,
                'retained_fraction_max': max(fractions) if fractions else None,
                'distribution_acceptance_threshold': None,
                'reason': None if sufficient else 'insufficient_independent_nonempty_observations',
                'rejections': dict(target['rejections']),
                'observations': target['observations'],
                'observation_sha256': digest(target['observations']),
                'clinical_validation': 'not_performed',
                'statistical_generalization_validated': False}}
    samples = sorted(dataset['samples'], key=lambda row: (row['split'], row['subject_id']))
    inputs = [{k: row[k] for k in ('subject_id', 'split', 'input_sha256')} for row in samples]
    return {
        'schema_version': VERSION, 'version': version,
        'status': 'passed' if results and all(r['status'] == 'passed' for r in results.values()) else 'failed',
        'validation_scope': 'independent_parameter_application_engineering_only',
        'calibration_state': 'candidate', 'test_only': False,
        'parameter_artifact_sha256': digest(artifact), 'protocol_sha256': artifact['protocol_sha256'],
        'candidate_parameters_sha256': artifact['candidate_parameters_sha256'],
        'dataset_binding_sha256': dataset['binding_sha256'],
        'training_subjects': [r['subject_id'] for r in samples if r['split'] == 'train'],
        'confirmation_subjects': [r['subject_id'] for r in samples if r['split'] == 'confirmation'],
        'input_manifest': inputs, 'input_manifest_sha256': digest(inputs),
        'roi_results': results, 'min_confirmation_subjects': min_confirmation_subjects,
        'all_required_rois_supported': not any(r['status'] != 'candidate' for r in artifact['roi_reports'].values()),
        'candidate_parameters': deepcopy(artifact['candidate_parameters']),
        'clinical_validation': 'not_performed', 'statistical_generalization_validated': False,
        'patient_identity_verified': False,
        'confirmation_used_for_parameter_selection': False}
