"""Pure reference-evidence qualification, not a claim of clinical calibration.

Formal references require independent subject manifests and a content-bound
validation artifact. Hashes establish consistency, not third-party authorship
or medical validity. Test references must explicitly declare test_only=True.
"""
from .stage1_config import digest

QUALIFICATION_SCHEMA = 'doctor-v3-reference-qualification-1'
VALIDATION_SCHEMA = 'doctor-v3-reference-validation-1'
IDENTITY_FIELDS = ('metric_id', 'unit', 'definition_version', 'capture_profile',
                   'region', 'source_kind', 'direction', 'roi_version',
                   'formula_identity', 'threshold_identity', 'version')


class ReferenceQualificationError(ValueError):
    """Reference evidence is absent, inconsistent, or ineligible."""


def _require(condition, reason):
    if not condition:
        raise ReferenceQualificationError(reason)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _subjects(value, label):
    _require(isinstance(value, list) and bool(value), label + '_must_be_nonempty_list')
    _require(all(_text(v) and v == v.strip() for v in value), label + '_invalid_subject')
    _require(len(value) == len(set(value)), label + '_duplicate_subject')
    return set(value)


def reference_content_sha256(reference):
    """Qualification is excluded to avoid a self-referential digest."""
    return digest({key: value for key, value in reference.items() if key != 'qualification'})


def measurement_identity_sha256(metric):
    return digest({field: metric.get(field) for field in IDENTITY_FIELDS})


def _manifest(qualification, training, confirmation):
    rows = qualification.get('input_manifest')
    _require(isinstance(rows, list) and bool(rows), 'missing_input_manifest')
    subjects = {'train': set(), 'confirmation': set()}
    sources, pairs = {}, set()
    for row in rows:
        _require(isinstance(row, dict), 'invalid_input_manifest_row')
        subject, split, sha = row.get('subject_id'), row.get('split'), row.get('input_sha256')
        _require(split in subjects and _text(subject) and _sha(sha), 'invalid_input_source_identity')
        allowed = training if split == 'train' else confirmation
        _require(subject in allowed, 'input_manifest_subject_or_split_mismatch')
        _require((subject, sha) not in pairs, 'duplicate_input_manifest_source')
        _require(sha not in sources or sources[sha] == split, 'input_source_leaks_across_splits')
        pairs.add((subject, sha))
        sources[sha] = split
        subjects[split].add(subject)
    _require(subjects['train'] == training and subjects['confirmation'] == confirmation,
             'input_manifest_does_not_cover_subjects')
    actual = digest(rows)
    _require(qualification.get('input_manifest_sha256') == actual, 'input_manifest_sha256_mismatch')
    return actual


def _validate(reference):
    _require(isinstance(reference, dict), 'reference_must_be_object')
    _require(all(_text(reference.get(k)) for k in ('version', 'source', 'purpose')),
             'missing_reference_lineage')
    purpose = reference['purpose']
    _require(purpose in ('formal', 'test'), 'unsupported_reference_purpose')
    content_sha = reference_content_sha256(reference)
    if purpose == 'test':
        _require(reference.get('test_only') is True, 'test_reference_requires_test_only_true')
        return {'status': 'test_only', 'formal_report_eligible': False,
                'proof': {'reference_sha256': content_sha, 'test_only': True}}
    _require(reference.get('test_only') is False, 'formal_reference_requires_test_only_false')
    qualification = reference.get('qualification')
    _require(isinstance(qualification, dict), 'missing_formal_reference_qualification')
    _require(qualification.get('schema_version') == QUALIFICATION_SCHEMA, 'unsupported_qualification_schema')
    _require(_text(qualification.get('version')), 'missing_qualification_version')
    _require(qualification.get('calibration_state') in ('candidate', 'validated'), 'invalid_calibration_state')
    training = _subjects(qualification.get('training_subjects'), 'training_subjects')
    confirmation = _subjects(qualification.get('confirmation_subjects'), 'confirmation_subjects')
    _require(not training & confirmation, 'training_confirmation_subject_overlap')
    manifest_sha = _manifest(qualification, training, confirmation)
    _require(qualification.get('reference_sha256') == content_sha, 'reference_content_sha256_mismatch')
    artifact = qualification.get('validation_artifact')
    _require(isinstance(artifact, dict), 'missing_validation_artifact')
    artifact_sha = digest(artifact)
    _require(qualification.get('validation_artifact_sha256') == artifact_sha, 'validation_artifact_sha256_mismatch')
    _require(artifact.get('schema_version') == VALIDATION_SCHEMA, 'unsupported_validation_schema')
    _require(artifact.get('test_only') is False, 'formal_validation_artifact_is_test_only')
    _require(_text(artifact.get('method')), 'missing_validation_method')
    expected = {'reference_sha256': content_sha, 'input_manifest_sha256': manifest_sha,
                'training_subjects_sha256': digest(qualification['training_subjects']),
                'confirmation_subjects_sha256': digest(qualification['confirmation_subjects']),
                'qualification_version': qualification['version'],
                'calibration_state': qualification['calibration_state']}
    for key, value in expected.items():
        _require(artifact.get(key) == value, 'validation_binding_mismatch:' + key)
    metrics, results = reference.get('metrics'), artifact.get('metric_results')
    _require(isinstance(metrics, dict) and bool(metrics), 'missing_reference_metrics')
    _require(isinstance(results, dict) and set(results) == set(metrics), 'validation_metric_coverage_mismatch')
    for key, metric in metrics.items():
        _require(isinstance(metric, dict) and all(_text(metric.get(f)) for f in IDENTITY_FIELDS),
                 'invalid_measurement_identity:' + str(key))
        _require(metric['version'] == reference['version'], 'reference_metric_version_mismatch:' + str(key))
        _require(key == metric['metric_id'] + ':' + metric['region'], 'reference_metric_key_mismatch:' + str(key))
        result = results[key]
        _require(isinstance(result, dict) and result.get('status') == 'passed', 'validation_not_passed:' + key)
        _require(result.get('identity_sha256') == measurement_identity_sha256(metric), 'validation_identity_mismatch:' + key)
        local_training = _subjects(result.get('training_subjects'), 'metric_training_subjects')
        local_confirmation = _subjects(result.get('confirmation_subjects'), 'metric_confirmation_subjects')
        _require(local_training <= training and local_confirmation <= confirmation, 'validation_metric_subject_mismatch:' + key)
        _require(isinstance(result.get('result'), dict) and bool(result['result']), 'missing_validation_result:' + key)
    return {'status': 'qualified', 'formal_report_eligible': True,
            'qualification_version': qualification['version'],
            'calibration_state': qualification['calibration_state'],
            'proof': {'schema_version': QUALIFICATION_SCHEMA, 'reference_sha256': content_sha,
                      'qualification_sha256': digest(qualification), 'validation_artifact_sha256': artifact_sha,
                      'input_manifest_sha256': manifest_sha, 'training_subject_count': len(training),
                      'confirmation_subject_count': len(confirmation), 'validated_metric_count': len(metrics),
                      'test_only': False}}


def validate_reference_qualification(reference):
    """Validate embedded evidence; return qualification/proof or ValueError.

    No files are opened and no reference is fitted. Nonempty independent
    samples are an integrity requirement, not a medical sample-size threshold.
    A candidate may qualify for engineering use without claiming calibration.
    """
    try:
        return _validate(reference)
    except ReferenceQualificationError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ReferenceQualificationError('invalid_qualification_json') from exc
