"""Frozen-subject ECDF/direct-state engineering candidates; never clinical calibration."""
from bisect import bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from .stage1_config import digest, scoring_identity
from .stage1_storage import load_result, _target, _relative, _write, _bytes, WORKSPACE
from .stage1_schema import required_items
from .stage1_scoring import CONTRACT_FIELDS, score_measurements, _raw_measurement_score
from .stage1_reference import (QUALIFICATION_SCHEMA, VALIDATION_SCHEMA,
    reference_content_sha256, measurement_identity_sha256, validate_reference_qualification)
from .scoring import display_score, grade
from .stage2_measurement_support import jaw_measurement_support
from .stage2_direct_scores import (COUNT_IDS, METHODS, observed_zero_count,
    anatomical_range_state, observed_zero_high_density, build_direct_metric_reference, score_direct_state, validate_direct_reference)

VERSION = 'v301-reference-builder-1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _json(path):
    return json.loads(_target(path).read_bytes())


def _context(key, basis):
    local = basis.get(key, {})
    return {**basis.get(local.get('evidence_ref'), {}), **local}


def _terminal_failure(item, frozen_row):
    path = _target(item['terminal_record_file'])
    if _sha(path) != item.get('terminal_record_sha256'):
        raise ValueError('terminal record checksum mismatch')
    record = _json(path)
    if not isinstance(record,dict) or not isinstance(record.get('signature'),dict):
        raise ValueError('terminal record must be one object with a signature')
    signature = record.get('signature', {})
    if (record.get('status') not in ('failed','partial_success')
            or signature.get('subject_id') != frozen_row['source_subject_id']
            or signature.get('input_sha256') != {'RGB_M':frozen_row['input_sha256']}
            or signature.get('relative_path') != frozen_row['relative_path']
            or signature.get('capture_profile') != 'consumer'
            or not isinstance(signature.get('v3_versions'),dict) or not signature['v3_versions']):
        raise ValueError('terminal record is not the matching frozen V3 consumer failure')
    reasons = {key:record[key] for key in ('failure_category','error','message')
               if isinstance(record.get(key),str) and record[key].strip()}
    if not reasons: raise ValueError('terminal failure requires a concrete reason')
    return {'subject_id':frozen_row['leakage_group'],'source_subject_id':frozen_row['source_subject_id'],
            'split':'train' if frozen_row['split']=='development' else 'confirmation',
            'input_sha256':frozen_row['input_sha256'],'relative_path':frozen_row['relative_path'],
            'status':record['status'],'reasons':reasons,'versions':signature['v3_versions'],
            'terminal_record_file':str(path),'terminal_record_relative_path':path.relative_to(WORKSPACE).as_posix(),
            'terminal_record_sha256':_sha(path)}


def _excluded_binding(samples):
    return [{k:v for k,v in sample.items() if k!='terminal_record_file'} for sample in samples]


def load_dataset(frozen_root, results_index, *, expected_counts=(800, 200), allow_unscored_previous_scoring=False):
    """Read explicit scoring_root entries; no source/staging image is opened.

    Every frozen identity must have either scoring_root or the mutually
    exclusive terminal_record_file + terminal_record_sha256 failure proof.
    scoring_root is the existing v3_scoring directory, not a V2 result.
    """
    frozen = _target(frozen_root)
    receipt = _json(frozen/'FREEZE_COMPLETE.json')
    if receipt.get('schema_version') != 'v301_frozen_inputs_v1' or receipt.get('status') != 'complete':
        raise ValueError('completed V301 freeze required')
    manifest_path = frozen/'freeze_manifest.json'
    if receipt.get('files', {}).get('freeze_manifest.json') != _sha(manifest_path):
        raise ValueError('frozen manifest checksum mismatch')
    manifest = _json(manifest_path)
    rows = manifest.get('inputs', [])
    if (manifest.get('schema_version') != 'v301_frozen_inputs_v1' or manifest.get('status') != 'frozen'
            or manifest.get('capture_profile') != 'consumer'
            or manifest.get('identity_basis') != 'legacy_manifest_subject_and_sha'):
        raise ValueError('explicit consumer subject/SHA freeze required')
    if (manifest.get('request') != receipt.get('request')
            or receipt.get('request', {}).get('development_count') != expected_counts[0]
            or receipt.get('request', {}).get('independent_confirmation_count') != expected_counts[1]):
        raise ValueError('freeze receipt request does not match authoritative split')
    if Counter(r.get('split') for r in rows) != {'development': expected_counts[0], 'independent_confirmation': expected_counts[1]}:
        raise ValueError('frozen split counts mismatch')
    for field in ('source_subject_id', 'leakage_group', 'input_sha256', 'relative_path'):
        values = [r.get(field) for r in rows]
        if not all(isinstance(v, str) and v.strip() for v in values) or len(set(values)) != len(rows):
            raise ValueError('duplicate or missing frozen identity: ' + field)
    index = _json(results_index) if isinstance(results_index, (str, Path)) else results_index
    if not isinstance(index, list): raise ValueError('results index must be an explicit list')
    indexed = {}
    for item in index:
        if not isinstance(item,dict): raise ValueError('invalid result index entry')
        key = item.get('input_sha256')
        if key in indexed: raise ValueError('duplicate result input SHA')
        success = 'scoring_root' in item
        failure = 'terminal_record_file' in item and 'terminal_record_sha256' in item
        if success == failure or (success and any(k in item for k in ('terminal_record_file','terminal_record_sha256'))):
            raise ValueError('result index needs mutually exclusive success or terminal failure evidence')
        indexed[key] = item
    if set(indexed) != {r['input_sha256'] for r in rows}:
        raise ValueError('results index must cover the freeze exactly; partial batch is not a complete dataset')
    samples, excluded, versions, configuration = [], [], None, None
    for row in rows:
        entry = indexed[row['input_sha256']]
        if 'terminal_record_file' in entry:
            excluded.append(_terminal_failure(entry,row))
            continue
        root = _target(entry['scoring_root'])
        payload = load_result(root)
        if (payload['subject_id'] != row['source_subject_id'] or payload['capture_profile'] != 'consumer'
                or payload['input_sha256'] != {'RGB_M': row['input_sha256']}):
            raise ValueError('result identity differs from frozen subject/input')
        if 'measurement_config' not in payload:
            raise ValueError('result lacks effective measurement configuration')
        if versions is None:
            versions, configuration = payload['versions'], payload['measurement_config']
        if payload['versions'] != versions or payload['measurement_config'] != configuration:
            raise ValueError('mixed measurement/scoring versions or effective parameters')
        if versions.get('scoring') != scoring_identity():
            scores = payload.get('scores')
            unscored = (versions.get('reference') == 'none' and payload.get('reference_purpose') is None
                        and isinstance(scores,dict) and all(isinstance(s,dict)
                        and 'score' in s and 'score_raw' in s and s['score'] is None and s['score_raw'] is None
                        for s in scores.values()))
            if allow_unscored_previous_scoring is not True or not unscored:
                raise ValueError('saved scoring implementation differs from current validator')
        samples.append({'subject_id': row['leakage_group'], 'source_subject_id': row['source_subject_id'],
                        'split': 'train' if row['split'] == 'development' else 'confirmation',
                        'input_sha256': row['input_sha256'], 'scoring_root': str(root),
                        'result_sha256': _sha(root/'result.json'), 'manifest_sha256': _sha(root/'manifest.json')})
    samples.sort(key=lambda r: (r['split'], r['subject_id']))
    excluded.sort(key=lambda r: (r['split'],r['subject_id']))
    if versions is None: raise ValueError('no successful result establishes the effective measurement configuration')
    if any(sample['versions']!=versions for sample in excluded):
        raise ValueError('terminal failure has different V3 measurement/scoring versions')
    inputs = [{k: r[k] for k in ('subject_id', 'split', 'input_sha256')} for r in samples]
    cohort_inputs = [{'subject_id':r['leakage_group'],'split':'train' if r['split']=='development' else 'confirmation',
                      'input_sha256':r['input_sha256']} for r in rows]
    cohort_counts = {'train':expected_counts[0],'confirmation':expected_counts[1]}
    binding = {'frozen_manifest_sha256': _sha(manifest_path), 'input_manifest': inputs,
               'result_sources': [{k: r[k] for k in ('subject_id', 'source_subject_id', 'split', 'input_sha256', 'result_sha256', 'manifest_sha256')} for r in samples],
               'excluded_sources':_excluded_binding(excluded),'frozen_input_manifest':cohort_inputs,
               'cohort_counts':cohort_counts,'versions': versions, 'measurement_config': configuration}
    return {'samples': samples, 'input_manifest': inputs, 'versions': versions,
            'validator_scoring_identity':scoring_identity(),
            'excluded_samples':excluded,'cohort_counts':cohort_counts,
            'measurement_config': configuration, 'binding': binding, 'binding_sha256': digest(binding)}


def _check_dataset(dataset):
    fields = ('subject_id', 'source_subject_id', 'split', 'input_sha256', 'result_sha256', 'manifest_sha256')
    if (digest(dataset['binding']) != dataset['binding_sha256']
            or [{k: r[k] for k in fields} for r in dataset['samples']] != dataset['binding']['result_sources']
            or dataset['input_manifest'] != dataset['binding']['input_manifest']
            or dataset['versions'] != dataset['binding']['versions']
            or dataset['measurement_config'] != dataset['binding']['measurement_config']
            or _excluded_binding(dataset['excluded_samples']) != dataset['binding']['excluded_sources']
            or dataset['cohort_counts'] != dataset['binding']['cohort_counts']
            or dataset.get('validator_scoring_identity',dataset['versions'].get('scoring')) != scoring_identity()):
        raise ValueError('dataset split, source, configuration, or scorer binding changed')
    for sample in dataset['excluded_samples']:
        row={'source_subject_id':sample['source_subject_id'],'leakage_group':sample['subject_id'],
             'input_sha256':sample['input_sha256'],'relative_path':sample['relative_path'],
             'split':'development' if sample['split']=='train' else 'independent_confirmation'}
        checked = _terminal_failure(sample,row)
        if checked != sample or checked['versions'] != dataset['versions']:
            raise ValueError('terminal failure changed after dataset collection')


def _read_sample(sample):
    root = _target(sample['scoring_root'])
    if _sha(root/'manifest.json') != sample['manifest_sha256'] or _sha(root/'result.json') != sample['result_sha256']:
        raise ValueError('result changed after dataset collection')
    return load_result(root)


def _reason(row, basis, versions, *, direct=False):
    if not direct and (row.get('metric_id') == '09.extent' or row.get('source_kind') == 'anatomical_extent_state'):
        return 'anatomical_extent_requires_approved_range_mapping_not_ecdf'
    status = row.get('measurement_status', row.get('status'))
    if status == 'pending_parameter': return 'pending_parameter:' + str(row.get('reason'))
    if row.get('status') != 'measured' or status not in ('measured', 'zero_target') or not _finite(row.get('value')):
        return 'unavailable_measurement:' + str(row.get('reason') or status)
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in CONTRACT_FIELDS):
        return 'incomplete_measurement_identity'
    for field, version in (('roi_version', 'roi'), ('formula_identity', 'implementation'), ('threshold_identity', 'thresholds')):
        if row[field] != versions.get(version): return 'measurement_identity_disagrees_with_payload:' + field
    if row['direction'] not in ('higher_health', 'higher_burden'): return 'unsupported_direction'
    context = _context(row['metric_id'] + ':' + row['region'], basis)
    if context.get('quality_status') != 'PASS' or context.get('quality_reason'):
        return 'quality_not_pass_or_rejected_basis'
    if row['metric_id'] == '11.jaw_continuity':
        support = jaw_measurement_support(row,basis)
        return None if support['status'] == 'supported' else 'invalid_jaw_path_support:' + support['reason']
    area = context.get('effective_area_px', context.get('valid_area_px'))
    if not _finite(area) or area <= 0: return 'missing_valid_measurement_area'
    return None


def fit_reference(dataset, version, *, min_training_subjects, min_confirmation_subjects):
    """Use development values only. Minima are explicit engineering protocol inputs."""
    _check_dataset(dataset)
    if not isinstance(version, str) or not version.strip(): raise ValueError('reference version required')
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in (min_training_subjects, min_confirmation_subjects)):
        raise ValueError('explicit engineering sample minima of at least two are required')
    populations, exclusions = defaultdict(list), defaultdict(Counter)
    required = {r['metric_id']+':'+r['region']:r for r in required_items('consumer',dataset['measurement_config'])}
    for key in required:
        exclusions[key]  # Preserve the mother set even when every development result failed.
    for sample in dataset['samples']:
        if sample['split'] != 'train': continue
        payload = _read_sample(sample)
        present = {row['metric_id']+':'+row['region'] for row in payload['measurements']}
        for key in set(required)-present:
            exclusions[key]['missing_required_measurement'] += 1
        for row in payload['measurements']:
            key = row['metric_id'] + ':' + row['region']
            anatomy = row.get('metric_id') == '09.extent'
            reason = _reason(row, payload['basis'], payload['versions'], direct=anatomy)
            state = (anatomical_range_state(row, payload['basis']) if anatomy else
                     observed_zero_count(row, payload['basis']) if row.get('metric_id') in COUNT_IDS else
                     observed_zero_high_density(row, payload['basis'],measurement_config=dataset['measurement_config'])
                     if row.get('metric_id')=='07.high_area' else None)
            if anatomy and state['status'] != 'candidate':
                reason = 'anatomical_extent_requires_approved_range_mapping_not_ecdf'
            if reason:
                exclusions[key][reason] += 1
                continue
            populations[key].append({'subject_id': sample['subject_id'], 'input_sha256': sample['input_sha256'],
                                     'result_sha256': sample['result_sha256'], 'value': row['value'],
                                     'identity': {k: row[k] for k in CONTRACT_FIELDS},
                                     **({'direct_evidence': state} if state is not None else {})})
    metrics, evidence, skipped = {}, {}, {}
    for key in sorted(set(populations) | set(exclusions)):
        population = sorted(populations[key], key=lambda r: r['subject_id'])
        reasons = dict(exclusions[key])
        identities = {digest(row['identity']) for row in population}
        if len(identities) > 1: reasons['mixed_measurement_identity'] = len(identities)
        if len(population) < min_training_subjects: reasons['insufficient_development_subjects'] = len(population)
        method = (METHODS[1] if key.split(':',1)[0] == '09.extent' else
                  METHODS[0] if key.split(':',1)[0] in COUNT_IDS and population
                  and all(row['value'] == 0 for row in population) else
                  METHODS[2] if key.split(':',1)[0]=='07.high_area' and population
                  and all(row['value']==0 for row in population) else None)
        if method:
            observed = [r for r in population if r.get('direct_evidence', {}).get('status') == 'candidate']
            for row in population:
                if row not in observed:
                    cause = row.get('direct_evidence', {}).get('reason', 'missing_direct_observation')
                    reasons[cause] = reasons.get(cause, 0) + 1
            if len(identities) == 1 and len(observed) >= min_training_subjects:
                metrics[key] = build_direct_metric_reference(observed[0]['identity'], method,
                    version=version, source_data_sha256=digest(observed))
                evidence[key] = {'training_subjects': [r['subject_id'] for r in observed],
                    'population': observed, 'excluded_training_measurements': reasons}
                continue
            reasons['insufficient_direct_development_subjects'] = len(observed)
        if population and len({row['value'] for row in population}) < 2:
            name=key.split(':',1)[0].split('.',1)[1]
            zero = all(row['value']==0 for row in population)
            constant_reason=('constant_zero_count_population' if zero and (name.endswith('_count') or name in ('clusters','roi_count'))
                             else 'constant_zero_population' if zero else 'constant_development_population')
            reasons[constant_reason] = len(population)
        if len(identities) != 1 or len(population) < min_training_subjects or len({r['value'] for r in population}) < 2:
            skipped[key] = {'reasons': reasons, 'training_subject_count': len(population)}
            continue
        metrics[key] = {**population[0]['identity'], 'version': version,
                        'sorted_values': sorted(row['value'] for row in population),
                        'population_sha256': digest(population), 'scoring_method': 'ecdf',
                        'population_count': len(population), 'value_transform': 'identity_no_log'}
        evidence[key] = {'training_subjects': [r['subject_id'] for r in population],
                         'population': population, 'excluded_training_measurements': reasons}
    reference = {'schema_version': 'doctor_v3_scoring_reference_v1',
                 'capture_profile': 'consumer', 'status': 'candidate',
                 'version': version, 'source': 'frozen V301 consumer single-RGB development cohort',
                 'purpose': 'test', 'test_only': True, 'calibration_status': 'candidate',
                 'clinical_validation': 'not_performed', 'identity_basis': 'legacy_manifest_subject_and_sha',
                 'patient_identity_verified': False, 'metrics': metrics,
                 'thresholds': deepcopy(dataset['measurement_config']),
                 'imaging': deepcopy(dataset['measurement_config'].get('imaging', {})),
                 'training_measurement_versions': deepcopy(dataset['versions'])}
    return {'schema_version': VERSION, 'stage': 'fit_candidate_not_published',
            'dataset_binding_sha256': dataset['binding_sha256'], 'reference': reference,
            'cohort_counts':deepcopy(dataset['cohort_counts']),
            'eligible_result_counts':dict(Counter(r['split'] for r in dataset['samples'])),
            'excluded_samples':_excluded_binding(dataset['excluded_samples']),
            'required_metric_count':len(required),
            'protocol': {'min_training_subjects': min_training_subjects, 'min_confirmation_subjects': min_confirmation_subjects,
                         'validator_scoring_identity':scoring_identity(),
                         'quality_inclusion': 'PASS_without_quality_reason', 'calibration_state': 'candidate',
                         'acceptance_scope': 'engineering_contract_and_distribution_diagnostics_not_clinical_validation'},
            'fit_evidence': evidence, 'skipped_metrics': skipped}


def _distribution(values):
    values = sorted(values)
    def quantile(p):
        position = p * (len(values)-1); lo = int(position); hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(position-lo)
    return {'count': len(values), 'min': values[0], 'max': values[-1], 'mean': sum(values)/len(values),
            'p10': quantile(.1), 'p25': quantile(.25), 'p50': quantile(.5),
            'median': quantile(.5), 'p75': quantile(.75), 'p90': quantile(.9), 'p95': quantile(.95)}


def _ks(a, b):
    a, b = sorted(a), sorted(b)
    return max(abs(bisect_right(a, x)/len(a)-bisect_right(b, x)/len(b)) for x in set(a+b))


def validate_reference(dataset, draft):
    """Independent confirmation diagnostics; never refit or select cutoffs from holdout."""
    _check_dataset(dataset)
    if draft.get('schema_version') != VERSION or draft.get('dataset_binding_sha256') != dataset['binding_sha256']:
        raise ValueError('draft/dataset binding mismatch')
    reference = draft['reference']
    if reference.get('purpose') != 'test' or reference.get('test_only') is not True:
        raise ValueError('validation must use an explicitly non-publishable test candidate')
    # Recompute the fit from training only to reject a tampered/holdout-fitted draft.
    rebuilt = fit_reference(dataset, reference['version'], **{k: draft['protocol'][k] for k in ('min_training_subjects', 'min_confirmation_subjects')})
    if digest(rebuilt) != digest(draft): raise ValueError('draft is not the deterministic development-only fit')
    evaluation_reference_sha = reference_content_sha256(reference)
    confirmations, rejected, trace_errors = defaultdict(list), defaultdict(Counter), defaultdict(list)
    for sample in dataset['samples']:
        if sample['split'] != 'confirmation': continue
        payload = _read_sample(sample)
        indexed = {r['metric_id']+':'+r['region']: r for r in payload['measurements']}
        ecdf_rows = [r for r in payload['measurements']
                     if reference['metrics'].get(r['metric_id']+':'+r['region'], {}).get('scoring_method') not in METHODS]
        scores = score_measurements(ecdf_rows, reference, allow_test_reference=True, basis=payload['basis'])
        for key, metric in reference['metrics'].items():
            row = indexed.get(key)
            direct = metric.get('scoring_method') in METHODS
            reason = 'missing_confirmation_measurement' if row is None else _reason(row, payload['basis'], payload['versions'], direct=direct)
            if not reason and any(row.get(k) != metric[k] for k in CONTRACT_FIELDS): reason = 'confirmation_measurement_identity_mismatch'
            if reason:
                rejected[key][reason] += 1
                continue
            if direct:
                score = score_direct_state(row, payload['basis'], reference, metric['scoring_method'],
                                           allow_test_reference=True)
                if score['status'] != 'test_only':
                    rejected[key][score.get('reason') or 'direct_state_unavailable'] += 1
                    continue
                trace = score.get('trace', {})
                proof = score.get('reference_proof', {})
                raw = row['value'] if metric['scoring_method'] == METHODS[1] else 100.0
                if not (score.get('formal_report_eligible') is False
                        and trace.get('formal_report_eligible') is False
                        and proof.get('source_data_sha256') == metric['source_data_sha256']
                        and proof.get('independent_validation_status') == 'independent_validation_pending'
                        and proof.get('qualification', {}).get('status') == 'test_only'
                        and score.get('reference_sha256') == evaluation_reference_sha
                        and all(trace.get(k) == row[k] for k in CONTRACT_FIELDS)
                        and trace.get('value') == row['value']
                        and trace.get('population_sha256') == metric['population_sha256']
                        and trace.get('reference_version') == reference['version']
                        and score.get('score_raw') == raw and score.get('score') == display_score(raw)
                        and score.get('grade') == grade(raw)
                        and trace.get('state_proof_sha256') == score.get('proof_sha256')
                        and digest(score.get('proof')) == score.get('proof_sha256')):
                    trace_errors[key].append(sample['subject_id'])
                    continue
                confirmations[key].append({'subject_id': sample['subject_id'], 'input_sha256': sample['input_sha256'],
                    'result_sha256': sample['result_sha256'], 'value': row['value'],
                    'score_raw_statistical': raw, 'score_display': score['score'],
                    'trace_sha256': digest(trace), 'direct_evidence_sha256': score['proof_sha256']})
                continue
            score = scores[key]; trace = score.get('trace', {})
            raw = _raw_measurement_score(row, metric, True)['score_raw']
            checks = trace.get('reference_identity_validation', {}).get('checks', {})
            valid_trace = (score.get('score_status') == 'test_only' and trace.get('formal_report_eligible') is False
                and score.get('reference_status') == 'compatible_test' and score.get('value') == row['value']
                and trace.get('reference_identity_validation', {}).get('status') == 'matched'
                and all(checks.get(k) == {'measurement': row[k], 'reference': metric[k], 'matched': True} for k in CONTRACT_FIELDS)
                and trace.get('reference_qualification', {}).get('status') == 'test_only'
                and trace.get('reference_qualification', {}).get('proof', {}).get('reference_sha256') == evaluation_reference_sha
                and trace.get('reference_version') == reference['version'] and trace.get('population_sha256') == metric['population_sha256']
                and all(trace.get(k) == row[k] for k in ('metric_id', 'unit', 'definition_version', 'roi_version', 'formula_identity', 'threshold_identity'))
                and trace.get('value') == row['value'] and trace.get('raw_statistical_score') == raw
                and trace.get('severity_guard') and _finite(score.get('score_raw'))
                and score.get('score') == display_score(score['score_raw']) and score.get('grade') == grade(score['score_raw']))
            if not valid_trace:
                trace_errors[key].append(sample['subject_id'])
                continue
            confirmations[key].append({'subject_id': sample['subject_id'], 'input_sha256': sample['input_sha256'],
                                       'result_sha256': sample['result_sha256'], 'value': row['value'],
                                       'score_raw_statistical': raw, 'score_display': score['score'],
                                       'trace_sha256': digest(trace)})
    results, skipped = {}, deepcopy(draft['skipped_metrics'])
    for key, metric in reference['metrics'].items():
        rows = sorted(confirmations[key], key=lambda r: r['subject_id'])
        if len(rows) < draft['protocol']['min_confirmation_subjects'] or trace_errors[key]:
            reasons = ({**dict(rejected[key]), 'insufficient_confirmation_subjects': len(rows)} if not trace_errors[key]
                       else {'confirmation_trace_validation_failed': len(trace_errors[key])})
            skipped[key] = {'reasons': reasons,
                           'confirmation_subject_count': len(rows), 'trace_error_subjects': trace_errors[key]}
            continue
        if metric.get('scoring_method') in METHODS:
            results[key] = {'status': 'passed', 'identity_sha256': measurement_identity_sha256(metric),
                'training_subjects': draft['fit_evidence'][key]['training_subjects'],
                'confirmation_subjects': [r['subject_id'] for r in rows],
                'result': {'validation_scope': 'engineering_integrity_only', 'clinical_validation': 'not_performed',
                    'direct_score_method': metric['scoring_method'], 'source_data_sha256': metric['source_data_sha256'],
                    'ecdf_applied': False, 'positive_count_extrapolation': False,
                    'trace_checks_passed': len(rows), 'confirmation_evidence_sha256': digest(rows),
                    'excluded_confirmation_measurements': dict(rejected[key])}}
            continue
        training, heldout = metric['sorted_values'], [r['value'] for r in rows]
        probes = sorted(set(training+heldout))
        probe_row = {**metric, 'module': metric['metric_id'].split('.')[0], 'status': 'measured'}
        raw = [_raw_measurement_score(dict(probe_row, value=v), metric, True)['score_raw'] for v in probes]
        monotone = all(a >= b if metric['direction'] == 'higher_burden' else a <= b for a, b in zip(raw, raw[1:]))
        if not monotone:
            skipped[key] = {'reasons': {'raw_score_monotonicity_failed': 1}}
            continue
        below, above = sum(v < training[0] for v in heldout), sum(v > training[-1] for v in heldout)
        results[key] = {'status': 'passed', 'identity_sha256': measurement_identity_sha256(metric),
            'training_subjects': draft['fit_evidence'][key]['training_subjects'],
            'confirmation_subjects': [r['subject_id'] for r in rows],
            'result': {'validation_scope': 'engineering_integrity_only', 'clinical_validation': 'not_performed',
                'training_distribution': _distribution(training), 'confirmation_distribution': _distribution(heldout),
                'empirical_ks_distance': _ks(training, heldout),
                'out_of_range': {'below_training_min': below, 'above_training_max': above, 'fraction': (below+above)/len(heldout)},
                'distribution_acceptance_threshold': None, 'distribution_diagnostics_do_not_establish_clinical_validity': True,
                'raw_score_monotonicity_passed': True, 'monotonic_probe_count': len(probes),
                'trace_checks_passed': len(rows), 'confirmation_evidence_sha256': digest(rows),
                'excluded_confirmation_measurements': dict(rejected[key])}}
    return {'schema_version': VERSION, 'stage': 'independent_engineering_validation',
            'dataset_binding_sha256': dataset['binding_sha256'], 'draft_sha256': digest(draft),
            'metric_results': results, 'skipped_metrics': skipped,
            'calibration_state': 'candidate', 'clinical_validation': 'not_performed'}


def publish_reference(dataset, draft, validation, destination):
    """Publish only independently checked metrics; existing destinations are never replaced."""
    actual = validate_reference(dataset, draft)
    if digest(actual) != digest(validation): raise ValueError('validation artifact does not match recomputed confirmation checks')
    if not actual['metric_results']: raise ValueError('no metrics have sufficient independent confirmation evidence')
    reference = deepcopy(draft['reference'])
    reference.update(purpose='formal', test_only=False)
    reference['metrics'] = {k: v for k, v in reference['metrics'].items() if k in actual['metric_results']}
    reference_sha = reference_content_sha256(reference)
    training = sorted(r['subject_id'] for r in dataset['samples'] if r['split'] == 'train')
    confirmation = sorted(r['subject_id'] for r in dataset['samples'] if r['split'] == 'confirmation')
    inputs = dataset['input_manifest']
    qualification_version = VERSION + '/' + reference['version']
    artifact = {'schema_version': VALIDATION_SCHEMA, 'test_only': False,
                'method': 'independent_engineering_ecdf_and_direct_state_contract_diagnostics',
                'reference_sha256': reference_sha, 'input_manifest_sha256': digest(inputs),
                'training_subjects_sha256': digest(training), 'confirmation_subjects_sha256': digest(confirmation),
                'qualification_version': qualification_version, 'calibration_state': 'candidate',
                'metric_results': actual['metric_results'], 'clinical_validation': 'not_performed',
                'dataset_binding_sha256': dataset['binding_sha256'], 'evaluation_draft_sha256': digest(draft)}
    reference['qualification'] = {'schema_version': QUALIFICATION_SCHEMA, 'version': qualification_version,
        'calibration_state': 'candidate', 'training_subjects': training, 'confirmation_subjects': confirmation,
        'input_manifest': inputs, 'input_manifest_sha256': digest(inputs), 'reference_sha256': reference_sha,
        'validation_artifact': artifact, 'validation_artifact_sha256': digest(artifact),
        'identity_basis': 'legacy_manifest_subject_and_sha', 'patient_identity_verified': False}
    qualification = validate_reference_qualification(reference)
    for metric in reference['metrics'].values():
        if metric.get('scoring_method') in METHODS:
            validate_direct_reference(reference, metric, metric['scoring_method'])
    destination = _target(destination)
    if destination.exists(): raise FileExistsError('new reference version directory required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.reference-stage-', dir=destination.parent))
    try:
        documents = {'reference.json': reference, 'validation.json': artifact,
                     'fit_report.json': {'protocol': draft['protocol'], 'skipped_metrics': actual['skipped_metrics'],
                                        'cohort_counts':draft['cohort_counts'],'eligible_result_counts':draft['eligible_result_counts'],
                                        'excluded_samples':draft['excluded_samples'],'required_metric_count':draft['required_metric_count'],
                                        'published_metric_count': len(reference['metrics']), 'qualification': qualification,
                                        'clinical_validation': 'not_performed'},
                     'dataset_proof.json': dataset['binding']}
        for name, doc in documents.items(): _write(stage/name, _bytes(doc))
        _write(stage/'manifest.json', _bytes({'schema_version': VERSION, 'status': 'published_engineering_candidate',
                                             'files': {name: _sha(stage/name) for name in documents}}))
        stage.rename(destination)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return destination/'reference.json'
