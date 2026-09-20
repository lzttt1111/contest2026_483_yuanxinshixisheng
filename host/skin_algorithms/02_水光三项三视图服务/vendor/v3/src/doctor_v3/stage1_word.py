"""Bind one stage-one truth to the existing Word model without mixing score domains."""
from copy import deepcopy
import math

from .clinical_scoring import formatted
from .registry import MODULES, stage1_weights
from .scoring import aggregate, display_score, grade
from .stage1_schema import SCHEMA_VERSION, coverage
from .stage1_zero_state import verify_saved_states, state_score

EMPTY = {'score_raw': None, 'score': None, 'grade': '不可评估', 'status': 'unavailable'}
SCORE_FIELDS = ('score_raw', 'score', 'grade', 'raw_statistical_score', 'severity_guard',
                'method', 'assessment_basis', 'reference', 'reason', 'no_score_reason',
                'missing', 'weights', 'status', 'measurement_basis', 'compatibility_definition')
REFERENCE_METHODS = {'same_metric_reference', 'previous_same_definition_reference',
                     'held_out_validated_monotone_mapping', 'exact_additive_roi_reference',
                     'quantile_piecewise_large_density_estimate'}
ANCHORS = {'observed_zero_target_anchor', 'no_target_burden_in_valid_measured_scope',
           'valid_zero_v3_grid_domain', 'absolute_observed_class_count'}
REASON_LABELS = {
    'insufficient_remaining_valid_region': '排除局部干扰后，该区域有效皮肤面积不足',
    'insufficient_classification_domain': '该区域毛囊与炎症目标可共同评估的范围不足',
    'unknown_classification_targets': '该区域存在类别尚不能确定的目标',
    'missing_follicle_support': '缺少对应毛囊定位依据',
    'missing_qualified_acne_exclusion': '缺少可靠的痤疮目标排除依据',
    'insufficient_pigment_support_coverage': '该区域多图色素证据的有效对应范围不足',
    'insufficient_valid_anatomical_path': '该区域解剖路径的有效覆盖不足',
    'insufficient_valid_mesh_coverage': '该区域有效几何网格覆盖不足',
    'insufficient_valid_mesh_edges': '该区域有效几何网格连接不足',
    'insufficient_curvature_samples': '该区域曲率估计的有效采样不足',
    'insufficient_geometry_resolution': '现有图像几何分辨率不足以评价该项',
    'anatomical_extent_threshold_pending': '该类沟槽的解剖范围评分规则尚未确定',
    'large_pore_threshold_pending': '该分区大毛孔判定阈值尚未确定',
    'porphyrin_threshold_pending': '该图源卟啉高强度判定阈值尚未确定',
    'no_valid_targets': '该区域未检出目标，无法计算目标大小或强度分布',
    'glabella_targets_without_valid_aligned_correspondence': '眉间纹路与几何依据缺少有效对应',
    'missing_relative_z': '缺少该分区的相对几何依据',
    'missing_dense_geometry': '缺少该分区足够分辨率的图像几何依据',
    'missing_dense_depth': '缺少该分区足够分辨率的凹陷估计依据',
    'high_precision_not_confirmed': '该项所需成像条件未确认',
    'missing_rgb_geometry': '缺少本次图像对应的面部几何依据',
    'missing_image_geometry_correspondence': '缺少该分区图像与几何的对应依据',
    'insufficient_image_geometry_correspondence': '该分区图像与几何的有效对应范围不足',
    'insufficient_regional_surface_coverage': '该分区图像与几何的有效对应范围不足',
    'insufficient_regional_z_samples': '该分区图像几何采样依据不足',
    'missing_texture_correspondence': '缺少该分区表面纹理的对应依据',
    'missing_exclusion_evidence': '缺少区分表面起伏与其他皮肤表现的依据',
    'missing_wrinkle_evidence': '缺少该分区的纹路量化依据',
    'missing_classification_evidence': '缺少该分区目标分类的有效依据',
    'missing_anatomical_jaw_path': '缺少该侧下颌轮廓的完整依据',
    'missing_bilateral_jaw_evidence': '缺少双侧下颌轮廓的完整依据',
    'invalid_jaw_path': '该侧下颌轮廓依据未满足评价条件',
    'jaw_path_outside_image': '该侧下颌轮廓超出有效图像范围',
    'degenerate_jaw_path': '该侧下颌轮廓依据不足',
    'insufficient_valid_region': '该分区有效图像范围不足',
    'insufficient_region_for_scoring': '该分区有效图像范围不足',
    'rejected_quality': '本次图像质量未满足该项评价条件',
    'rejected_input_quality': '本次图像质量未满足该项评价条件',
    'no_target': '有效范围内未见可用于该项统计的目标',
    'pending_parameter': '该项测量参数尚未确定',
    'not_applicable': '该指标不适用于此分区',
}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _reject_test_scores(value):
    if isinstance(value, dict):
        if value.get('score') is not None or value.get('score_raw') is not None:
            trace = value.get('trace', {})
            for node in (value, trace):
                if (node.get('reference_purpose') == 'test' or node.get('score_status') == 'test_only'
                        or node.get('formal_report_eligible') is False):
                    raise ValueError('test/ineligible score cannot enter formal Word')
        for child in value.values():
            _reject_test_scores(child)
    elif isinstance(value, list):
        for child in value:
            _reject_test_scores(child)


def _historical(item, trace=None):
    if item.get('score') is None:
        return False
    trace = trace or {}
    for node in (item, trace):
        for key in ('assessment_basis', 'measurement_basis', 'compatibility_definition', 'method'):
            text = str(node.get(key, '')).lower()
            if any(token in text for token in ('historical', 'legacy', 'v2')):
                return True
        if node.get('source_v2') or node.get('method') in ANCHORS:
            return True
    return bool(trace.get('reference') and trace.get('method') in REFERENCE_METHODS)


def _pure_zero_anchor(item, trace):
    methods = (ANCHORS - {'absolute_observed_class_count'}) | {'valid_zero_historical_roi_state'}
    return (trace.get('method', item.get('method')) in methods
            and not trace.get('reference') and not trace.get('reference_version'))


def _reason(row, scored):
    why = row.get('reason') or scored.get('trace', {}).get('reason') or 'missing_compatible_reference'
    if row.get('metric_id')=='11.jaw_continuity' and ('jaw' in why or 'bilateral' in why):
        return '本次下颌轮廓依据未满足评价条件'
    if any(text in why for text in ('classification_observation','classification_samples','same_class_negative')):
        return '该区域目标分类依据不足'
    if any(text in why for text in ('anatomical_measurement','anatomical_classification','anatomical_configuration_or_evidence')):
        return '该区域解剖范围依据未满足评价条件'
    if why in REASON_LABELS:
        return REASON_LABELS[why]
    if row.get('value') is not None:
        return '该指标在本分区及本图源下的同定义参考区间尚未建立'
    if why == 'high_precision_not_confirmed':
        return '该项所需成像条件未确认'
    if row.get('measurement_status') == 'not_applicable':
        return '该指标不适用于此分区'
    if row.get('measurement_status') == 'pending_parameter':
        return '该项测量参数尚未确定'
    return REASON_LABELS.get(why, '缺少该项有效测量依据')


def _native(row, scored, basis):
    node = deepcopy(row)
    node.update(EMPTY, value=row.get('value'), status=row.get('status', 'unavailable'),
                measurement_status=row.get('measurement_status', row.get('status', 'unavailable')),
                reference_status=scored.get('reference_status', 'missing'),
                score_status=scored.get('score_status', 'unavailable'), basis=deepcopy(basis),
                measurement_basis='stage1_native_measurement')
    source = row.get('source_kind', '').lower()
    appearance = source == 'image_appearance_proxy' or basis.get('appearance_proxy') is True
    nonphysical = any(basis.get(key) is False for key in
                      ('physical_3d', 'physical_depth', 'is_physical_depth', 'is_physical_3d', 'physical_measurement'))
    proxy = '25d' in source or appearance or nonphysical
    node['source_label'] = ('图像估计（外观）' if appearance else '图像估计（2.5D）') if proxy else '本次图像量化'
    if proxy:
        node['name'] = node.get('name', '').replace('实测三维', '外观' if appearance else '2.5D').replace('三维', '外观' if appearance else '2.5D')
        if '图像估计' not in node['name']:
            node['name'] += '（图像估计）'
        node['physical_3d'] = False
    node['display'] = formatted(node['value'], node.get('unit', ''))
    if scored.get('score') is None:
        node['no_score_reason'] = _reason(row, scored)
        node['reason'] = row.get('reason') or scored.get('trace', {}).get('reason', 'missing_compatible_reference')
        return node
    trace = scored.get('trace', {})
    zero_target_score = (trace.get('method') == 'observed_zero_burden_reference_anchor'
                         and row.get('value') is None
                         and row.get('measurement_status') in ('no_target', 'no_targets')
                         and trace.get('score_input') == 0
                         and trace.get('numeric_statistic_imputed') is False)
    if (scored.get('score_status') != 'candidate' or scored.get('reference_status') != 'compatible'
            or trace.get('reference_purpose') != 'formal' or trace.get('formal_report_eligible') is not True
            or not trace.get('reference_version') or scored.get('value') != row.get('value')
            or (trace.get('value') != row.get('value') and not zero_target_score)
            or (row.get('status') != 'measured' and not zero_target_score)):
        raise ValueError('stage1 score is not a compatible formal measurement score')
    for key in ('metric_id', 'definition_version', 'unit'):
        if trace.get(key) != row.get(key):
            raise ValueError('stage1 score trace mismatch: ' + key)
    raw = scored.get('score_raw')
    if not _finite(raw) or not 0 <= raw <= 100 or scored['score'] != display_score(raw) or scored.get('grade') != grade(raw):
        raise ValueError('stage1 score display/grade mismatch')
    node.update({key: deepcopy(scored[key]) for key in ('score_raw', 'score', 'grade')},
                status='candidate', method='stage1_same_measurement_reference',
                trace=deepcopy(trace), stage1_bound=True)
    if zero_target_score:
        node.update(status='confirmed_zero', zero_state=True,
                    measurement_basis='valid_no_target_state_with_qualified_reference')
    return node


def _weighted(nodes, weights):
    result = aggregate({key: nodes.get(key, {}) for key in weights}, weights)
    lineages = set()
    for key in weights:
        node = nodes.get(key, {})
        trace = node.get('trace', {})
        if node.get('score') is not None and not node.get('zero_state'):
            lineage = node.get('reference_lineage') or (trace.get('reference_version'), trace.get('reference_source'))
            lineages.add(tuple(lineage))
    all_zero = bool(weights) and all(nodes.get(key, {}).get('zero_state') for key in weights)
    if result['score'] is not None and not all_zero and (len(lineages) != 1 or any(None in lineage for lineage in lineages)):
        result.update(EMPTY, reason='incompatible_reference_lineage', missing=list(weights))
    result['reference_lineage'] = list(next(iter(lineages))) if len(lineages) == 1 else None
    if all_zero:
        result['zero_state'] = True
    result.update(method='stage1_complete_same_definition_aggregation', stage1_bound=True)
    result['component_traces'] = {key: deepcopy(nodes.get(key, {}).get('trace', {})) for key in weights}
    if result['score'] is None:
        reasons = list(dict.fromkeys(nodes.get(key, {}).get('no_score_reason') for key in weights
                                    if nodes.get(key, {}).get('score_raw') is None))
        reasons = [reason for reason in reasons if reason]
        result['no_score_reason'] = '；'.join(reasons[:2]) if reasons else '部分指标缺少有效测量或参考数据'
    return result


def _project(target, native):
    if _historical(target) and native.get('score') is None and native.get('allow_historical', True):
        target['native_v3_assessment'] = deepcopy(native)
        return
    for key in SCORE_FIELDS:
        target.pop(key, None)
    for key in ('source_v2', 'source_interval', 'target_interval', 'assessment_title', 'proxy_definition'):
        if key in target:
            target.setdefault('previous_state_assessment', {})[key] = target.pop(key)
    target.update(deepcopy(native))



def attach_stage1(model, payload):
    """Return a copy bound to payload; only explicit historical scores survive gaps."""
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported stage1 Word schema')
    for field in ('subject_id', 'capture_profile'):
        if payload.get(field) != model.get(field):
            raise ValueError('stage1 Word identity mismatch: ' + field)
    _reject_test_scores(payload.get('scores', {}))
    _reject_test_scores(model)
    result = deepcopy(model)
    zero_states = verify_saved_states(payload)
    traces = result.setdefault('score_trace', {})
    current_rows = {r['metric_id'] + ':' + r['region']: r for r in payload['measurements']}
    # Some existing approved aggregates store lineage on their component traces.
    # Freeze that lineage before binding any new native rows.
    for mid, module in result['modules'].items():
        for region, metrics in module.get('metrics', {}).items():
            target = module if region == 'full_face' else module.get('regions', {}).get(region, {})
            weights = target.get('weights', {})
            contradicted_zero = any(
                _pure_zero_anchor(metrics.get(key, {}), traces.get(mid + '.' + key + ':' + region, {}))
                and _finite(current_rows.get(mid + '.' + key + ':' + region, {}).get('value'))
                and current_rows[mid + '.' + key + ':' + region]['value'] > 0 for key in weights)
            if contradicted_zero:
                if target.get('assessment_basis') == 'historical_compatible_component_aggregation':
                    target['previous_historical_assessment'] = {k: deepcopy(target.get(k)) for k in ('score', 'score_raw', 'grade', 'assessment_basis')}
                    target.pop('assessment_basis')
                # Genuine replayed V2 totals retain their own explicit lineage;
                # a zero-anchor-only parent cannot acquire compatibility status.
                continue
            if target.get('score') is None or _historical(target) or not weights:
                continue
            if all(_historical(metrics.get(key, {}), traces.get(mid + '.' + key + ':' + region, {})) for key in weights):
                target['assessment_basis'] = 'historical_compatible_component_aggregation'
                target['historical_component_traces'] = {
                    key: {'measurement': deepcopy(metrics[key]),
                          'trace': deepcopy(traces.get(mid + '.' + key + ':' + region, {}))}
                    for key in weights}
    native_rows = {}
    for row in payload['measurements']:
        address = row['metric_id'] + ':' + row['region']
        if address in native_rows or row.get('capture_profile') != payload['capture_profile']:
            raise ValueError('duplicate stage1 row or profile mismatch')
        scored = payload.get('scores', {}).get(address, {})
        native = _native(row, scored, payload.get('basis', {}).get(address, {}))
        native_rows[address] = native
        mid, key = row['metric_id'].split('.', 1)
        module = result['modules'].get(mid)
        if module is None:
            continue
        metrics = module.setdefault('metrics', {}).setdefault(row['region'], {})
        old = metrics.get(key, {})
        old_trace = traces.get(address, {})
        zero_anchor = _pure_zero_anchor(old, old_trace)
        positive_native = _finite(native.get('value')) and native['value'] > 0
        rejected_now = native.get('measurement_status') == 'insufficient_quality' or native.get('reason') in ('rejected_quality', 'rejected_input_quality', 'rejected_input')
        retain_old = _historical(old, old_trace) and not rejected_now and not (zero_anchor and (positive_native or old.get('score') != 100))
        if retain_old and native.get('score') is None:
            if zero_anchor and native.get('value') is None and _finite(old.get('value')) and old['value'] != 0:
                previous = {k: deepcopy(old.get(k)) for k in ('value', 'unit', 'display', 'score', 'score_raw', 'grade')}
                previous['trace'] = deepcopy(old_trace)
                old['previous_historical_measurement'] = previous
                old.update(value=None, unit=native.get('unit'), display='—', status='zero_target',
                           measurement_status=native.get('measurement_status'),
                           measurement_basis='valid_zero_target_state_no_statistic')
                old_trace['previous_historical_measurement'] = deepcopy(previous)
                old_trace.update(value=None, display_value=None)
                old_trace.pop('score_input', None)
            old['native_v3_measurement'] = deepcopy(native)
            old['source_label'] = '图像估计（零目标状态）' if zero_anchor else '图像估计（历史兼容）'
            old['display'] = formatted(old.get('value'), old.get('unit', ''))
            old.pop('no_score_reason', None)
            traces.setdefault(address, {})['native_v3_measurement'] = deepcopy(native)
        else:
            metrics[key] = native
            traces[address] = {**deepcopy(scored.get('trace', {})),
                               'method': 'stage1_same_measurement_reference' if native['score'] is not None else 'stage1_native_unscored',
                               'value': row.get('value'), 'unit': row.get('unit'),
                               'source_kind': row.get('source_kind'), 'reason': native.get('reason'),
                               'stage1_versions': deepcopy(payload['versions'])}
    # A current, proven negative phenotype supersedes positive historical
    # statistics of another definition. Do not invent a zero P50 or density.
    for proof in zero_states.values():
        for address in proof['supporting_fields']:
            if address not in native_rows:
                continue
            mid, key = address.split(':')[0].split('.', 1)
            region = address.split(':')[1]
            previous = result['modules'][mid]['metrics'].get(region, {}).get(key)
            node = deepcopy(native_rows[address])
            if previous:
                node['previous_compatible_measurement'] = deepcopy(previous)
            result['modules'][mid]['metrics'].setdefault(region, {})[key] = node
            old_trace = deepcopy(traces.get(address, {}))
            traces[address] = {**deepcopy(payload.get('scores', {}).get(address, {}).get('trace', {})),
                               'method': 'stage1_confirmed_negative_measurement',
                               'value': node.get('value'), 'unit': node.get('unit'),
                               'score': node.get('score'), 'score_raw': node.get('score_raw'),
                               'grade': node.get('grade'), 'state_proof': proof,
                               'previous_compatible_trace': old_trace,
                               'stage1_versions': deepcopy(payload['versions'])}
    definitions = {m.id: m for m in MODULES}
    native_assessments = {}
    for mid, module in result['modules'].items():
        if mid not in definitions:
            continue
        definition = definitions[mid]
        local_assessments = {}
        for region, metrics in module.get('metrics', {}).items():
            nodes = {address.split(':')[0].split('.', 1)[1]: value for address, value in native_rows.items()
                     if address.startswith(mid + '.') and address.endswith(':' + region)}
            weights = stage1_weights(mid)
            if mid == '03':
                for layer in ('spots', 'brown', 'uv'):
                    zero = zero_states.get('03.' + layer + ':' + region)
                    score = state_score(zero) if zero else _weighted({k: nodes.get(layer + '.' + k, {}) for k in ('area', 'p90')}, {'area': .55, 'p90': .45})
                    _project(metrics.setdefault(layer, {}), dict(score, display=score['grade']))
                    nodes[layer] = score
            elif mid == '02':
                for name, group in (('surface_gloss', {'gloss_area': .25/.6, 'gloss_high_area': .2/.6, 'gloss_mean': .15/.6}),
                                    ('porphyrin', {'porphyrin_high_density': .7, 'porphyrin_p90': .3})):
                    zero = zero_states.get('02.' + name + ':' + region)
                    score = state_score(zero) if zero else _weighted(nodes, group)
                    _project(metrics.setdefault(name, {}), dict(score, display=score['grade']))
                    nodes[name] = score
                weights = {'surface_gloss': .60, 'porphyrin': .40}
            elif mid == '06':
                weights = {'erythema_count': .2, 'papule_count': .4, 'pustule_count': .4}
            elif mid == '07':
                high_precision = payload.get('basis', {}).get('07.contrast_p50:' + region, {}).get('high_precision') is True
                if not high_precision:
                    weights = {'area': .35/.85, 'high_area': .25/.85, 'density': .25/.85}
            elif mid == '08':
                if region == 'full_face':
                    continue
                weights = ({'main_count': .30, 'length_burden': .35, 'depth': .35} if region in ('forehead', 'glabella')
                           else {'area': .30, 'density': .25, 'high_area': .25, 'contrast_p50': .20})
            elif mid == '11' and region != 'full_face':
                relevant = (['jaw_continuity'] if region.endswith('_jaw') else
                            ['smoothness', 'turning', 'jowl'] if region.endswith('_lower_face') else
                            ['smoothness', 'turning'] if region.endswith('_midface') else [])
                total = sum(weights[k] for k in relevant)
                weights = {k: weights[k]/total for k in relevant}
            score = (_weighted(nodes, weights) if weights else
                     {**EMPTY, 'reason': 'region_specific_dimensions_no_generic_total',
                      'no_score_reason': '轮廓各分区适用维度不同，不生成通用总分'})
            if mid + ':' + region in zero_states:
                score = state_score(zero_states[mid + ':' + region])
            if any(n.get('measurement_status') == 'insufficient_quality' or n.get('reason') in ('rejected_quality', 'rejected_input_quality') for n in nodes.values()):
                score['allow_historical'] = False
            local_assessments[region] = score
            target = module if region == 'full_face' else module.setdefault('regions', {}).setdefault(region, {})
            _project(target, score)
        if mid == '08':
            score = _weighted(local_assessments, definition.weights)
            local_assessments['full_face'] = score
            _project(module, score)
        values = [row.get('value') for row in payload['measurements'] if row['module'] == mid]
        module['has_measurements'] = any(_finite(v) for v in values)
        from .stage1_report_state import phenotype_state
        module['detected'] = phenotype_state(mid, payload['measurements'], module, zero_states)
        native_assessments[mid] = local_assessments
    result.update(stage1_versions=deepcopy(payload['versions']), stage1_scores=deepcopy(payload.get('scores', {})),
                  stage1_measurement_config=deepcopy(payload.get('measurement_config', {})),
                  stage1_coverage=coverage(payload['measurements'], payload.get('scores', {}), payload['capture_profile'], payload.get('measurement_config')),
                  stage1_native_assessments=native_assessments)
    from .evaluation_scope import apply_report_scope
    return apply_report_scope(result)
