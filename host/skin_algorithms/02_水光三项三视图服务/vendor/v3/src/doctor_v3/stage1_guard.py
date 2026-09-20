"""Pure same-region evidence protection and independently proven zero states."""
from copy import deepcopy
from .severity_guard import GROUPS, POLICY, apply_score, absolute_count_score, finite
from .scoring import display_score, grade
from .registry import GROOVE_REGIONS
from ._stage1_anatomical_extent import PARAMETERS as EXTENT_PARAMETERS

ZERO_STATE_SUPPORTED_MODULES = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
ZERO_STATE_UNSUPPORTED = {'11': 'zero_curvature_is_not_absence_of_laxity'}
ACNE_CLASSES = ('erythema', 'papule', 'pustule')


def _indexed(measurements):
    result = {}
    for row in measurements:
        key = row['metric_id'] + ':' + row['region']
        if key in result:
            raise ValueError('duplicate measurement: ' + key)
        result[key] = row
    return result


def _context(key, basis):
    item = basis.get(key, {})
    if not isinstance(item, dict):
        return {}
    shared = basis.get(item.get('evidence_ref'), {})
    return {**(shared if isinstance(shared, dict) else {}), **item}


def _quality(context, minimum):
    area = context.get('effective_area_px') if 'effective_area_px' in context else context.get('valid_area_px')
    quality = context.get('quality_status')
    ok = quality == 'PASS' and finite(area) and area >= minimum and not context.get('quality_reason')
    return {'quality_status': quality, 'valid_area_px': area,
            'minimum_valid_area_px': minimum, 'passed': bool(ok)}


def _measured(row):
    return row.get('status') == 'measured' and row.get('measurement_status', 'measured') in ('measured','zero_target') and finite(row.get('value'))


def _count_floor(row, indexed, basis):
    name = row['metric_id'].split('.', 1)[1]
    kind = next((kind for kind in ACNE_CLASSES if name in (kind+'_count', kind+'_density')), None)
    if kind is None:
        return {'status': 'not_applicable'}
    key = '06.'+kind+'_count:'+row['region']
    count_row, context = indexed.get(key, {}), _context(key, basis)
    evidence_valid = context.get('unknown_count') == 0 and context.get('reason') is None
    quality = _quality(context, 100)
    count = count_row.get('value') if _measured(count_row) else None
    result = absolute_count_score(count, quality['valid_area_px'], evidence_valid and quality['passed'])
    return {**result, 'count_source': key, 'class': kind, 'quality': quality,
            'unknown_count': context.get('unknown_count')}


def _unscore(item,reason):
    trace=item.setdefault('trace',{})
    trace['rejected_statistical_score']=item.get('score_raw')
    for name in ('score','score_raw','grade'):
        trace.pop(name,None)
    item.update(score_raw=None,score=None,grade='不可评估',score_status='unavailable')
    trace.update(reason=reason,formal_report_eligible=False)


def guard_measurement_scores(measurements, scores, basis):
    """Protect unguarded scored-item mappings, preserving input and lineage.

    Consensus never crosses module/region. Aliased extent measurements remain
    one GROUPS category; derived density cannot become an independent vote.
    """
    indexed, output = _indexed(measurements), deepcopy(scores)
    if any(key not in indexed and finite(item.get('score_raw')) for key,item in scores.items()):
        raise ValueError('scored item has no corresponding measurement')
    grouped = {}
    path_support = {}
    for key, row in indexed.items():
        item = scores.get(key, {})
        if not finite(item.get('score_raw')):
            continue
        trace = item.get('trace', {})
        observed_negative = (trace.get('method') == 'observed_zero_burden_reference_anchor'
                             and row.get('direction') == 'higher_burden'
                             and row.get('value') is None
                             and row.get('measurement_status') in ('no_target', 'no_targets')
                             and row.get('reason') in ('no_target', 'no_valid_targets')
                             and trace.get('score_input') == 0
                             and trace.get('numeric_statistic_imputed') is False
                             and isinstance(trace.get('observed_negative_evidence'), dict)
                             and finite(trace['observed_negative_evidence'].get('valid_area_px'))
                             and trace['observed_negative_evidence']['valid_area_px'] >= 100)
        if observed_negative:
            from .stage1_scoring import _observed_negative
            actual_negative = _observed_negative(row, basis)
            observed_negative = (actual_negative is not None
                                 and actual_negative == trace['observed_negative_evidence'])
        if not _measured(row) and not observed_negative:
            _unscore(output[key],'unavailable_measurement_cannot_score')
            continue
        if row['metric_id']=='11.jaw_continuity':
            from .stage2_measurement_support import jaw_measurement_support
            support=jaw_measurement_support(row,basis)
            if support['status']!='supported':
                _unscore(output[key],support['reason'])
                continue
            path_support[key]=support
        if item.get('trace', {}).get('severity_guard'):
            raise ValueError('score already protected: ' + key)
        module = row['metric_id'].split('.')[0]
        grouped.setdefault((module, row['region']), []).append((key,row))
    for (module, region), members in grouped.items():
        metrics, checks = {}, {}
        absolute = {}
        shared_counts = set()
        for key, row in members:
            name = row['metric_id'].split('.',1)[1]
            alias = 'length' if module == '08' and name == 'length_burden' else name
            # When a legacy alias and its replacement coexist, they are still
            # one measurement vote and never increase the group count.
            raw = scores[key]['score_raw']
            if alias not in metrics or raw < metrics[alias]['score_raw']:
                metrics[alias] = {'score_raw': raw}
            checks[key] = ({'passed':True,'support_kind':'2d_anatomical_path',
                            'support_proof':path_support[key]['proof'],
                            'support_proof_sha256':path_support[key]['proof_sha256'],
                            'skin_area_imputed':False} if key in path_support else
                           _quality(_context(key,basis), POLICY['minimum_region_valid_pixels']))
            if finite(row.get('value')):
                absolute[name] = row['value']
            count = _context(key,basis).get('count')
            if finite(count):
                shared_counts.add(count)
        if len(shared_counts) == 1:
            absolute['count'] = next(iter(shared_counts))
        quality_ok = bool(checks) and all(c['passed'] for c in checks.values())
        for key,row in members:
            item = output[key]
            trace = deepcopy(item.get('trace', {}))
            original = item['score_raw']
            if module not in GROUPS:
                item.update(score_raw=None,score=None,grade='不可评估',score_status='unavailable')
                trace.update(reason='unsupported_severity_guard_module',formal_report_eligible=False)
                item['trace'] = trace
                continue
            protected = apply_score({**trace, 'score_raw': original},module,metrics,quality_ok,absolute=absolute)
            count_guard = _count_floor(row,indexed,basis) if module == '06' else {'status':'not_applicable'}
            if finite(count_guard.get('score_raw')):
                final = max(protected['score_raw'], count_guard['score_raw'])
                protected.update(score_raw=final,score=display_score(final),grade=grade(final))
            protected['guard_context'] = {'module':module,'region':region,'quality_checks':checks,
                                          'absolute_measurements':absolute,
                                          'group_raw_scores':metrics,'pre_guard_score_raw':original,
                                          'absolute_count_guard':count_guard}
            item.update(score_raw=protected['score_raw'],score=protected['score'],grade=protected['grade'],trace=protected)
    return output


def _zero(value):
    return finite(value) and value == 0


def _eligible(rows, basis, names, module, region):
    selected = {}
    for name in names:
        key = module+'.'+name+':'+region
        if key not in rows:
            return None
        row, context = rows[key], _context(key,basis)
        if not _quality(context,100)['passed']:
            return None
        state = row.get('measurement_status', context.get('measurement_status', row.get('status')))
        reason = row.get('reason') or context.get('reason')
        allowed = state in ('measured','zero_target','no_target','no_targets')
        allowed = allowed or (name == 'contrast_p50' and module == '07' and context.get('high_precision') is False
                              and reason == 'high_precision_not_confirmed')
        if row.get('value') is not None and not _zero(row['value']):
            return None
        if row.get('value') is None and state not in ('no_target','no_targets','not_applicable'):
            return None
        if not allowed or (reason and reason not in ('no_target','no_valid_targets','high_precision_not_confirmed')):
            return None
        if context.get('parameter_status') in ('pending_parameter','pending','unresolved'):
            return None
        selected[name] = (row,context)
    return selected


def _groove_negative(selected, indexed, basis, region):
    """Require original regional samples and an approved no-target path state."""
    volume, context = selected['volume']
    if not _zero(volume.get('value')) or not _zero(context.get('volume')) or not _zero(context.get('image_line_pixels')):
        return False
    for name in ('mean_depth','p90_depth'):
        row, _ = selected[name]
        if row.get('value') is not None or row.get('reason') != 'no_target':
            return False
    ids, xy, z = context.get('sample_ids'), context.get('sample_xy'), context.get('relative_z')
    if (not isinstance(ids,list) or not ids or any(not finite(i) or int(i)!=i for i in ids)
            or len(set(ids)) != len(ids) or not isinstance(xy,list) or len(xy)!=len(ids)
            or any(not isinstance(p,list) or len(p)!=2 or not all(finite(v) for v in p) for p in xy)
            or not isinstance(z,list) or len(z)!=len(ids) or not all(finite(v) for v in z)):
        return False
    anatomy = context.get('anatomical_extent',{})
    path = anatomy.get('path_xy')
    if (anatomy.get('measurement_status') != 'measured' or anatomy.get('reason') is not None
            or anatomy.get('target_intervals') != [] or not anatomy.get('valid_intervals')
            or not finite(anatomy.get('path_length_px')) or anatomy['path_length_px'] <= 0
            or not isinstance(path,list) or len(path)<2
            or any(not isinstance(p,list) or len(p)!=2 or not all(finite(v) for v in p) for p in path)):
        return False
    intervals = anatomy['valid_intervals']
    if (not isinstance(intervals,list) or any(not isinstance(p,list) or len(p)!=2
            or not all(finite(v) for v in p) or not 0<=p[0]<p[1]<=1 for p in intervals)
            or any(a[1]>b[0] for a,b in zip(intervals,intervals[1:]))
            or sum(b-a for a,b in intervals)<EXTENT_PARAMETERS['minimum_valid_path_fraction']
            or any(a==b for a,b in zip(path,path[1:]))):
        return False
    state = context.get('extent_classification',{})
    key = '09.extent:'+region
    extent = indexed.get(key,{})
    threshold_independent_zero = (
        state.get('status') == 'pending_parameter'
        and state.get('reason') == 'anatomical_extent_threshold_pending'
        and extent.get('value') is None
        and extent.get('measurement_status') == 'pending_parameter'
        and extent.get('reason') == 'anatomical_extent_threshold_pending'
        and _quality(_context(key,basis),100)['passed'])
    if threshold_independent_zero:
        return True
    config = state.get('configuration',{})
    if (state.get('status') != 'approved_state' or state.get('extent_grade') != 'none'
            or state.get('reason') is not None or config.get('approved') is not True
            or config.get('test_only') is not False or config.get('purpose') != 'formal'
            or not isinstance(config.get('approval_id'),str) or not config['approval_id'].strip()
            or not isinstance(state.get('config_version'),str) or not state['config_version'].strip()
            or state['config_version'] != config.get('version')):
        return False
    return (_measured(extent) and _quality(_context(key,basis),100)['passed']
            and finite(state.get('extent_score')) and extent['value'] == state['extent_score'])


def _pigment_negative(selected, layer):
    area, p90 = selected[layer+'.area'], selected[layer+'.p90']
    if not _measured(area[0]) or not _zero(area[0].get('value')):
        return False
    if p90[0].get('value') is not None or p90[0].get('reason') != 'no_valid_targets':
        return False
    return all(_zero(ctx.get('affected_area_px')) and _zero(ctx.get('pixel_score_summary',{}).get('count'))
               and (layer != 'spots' or ctx.get('support_authorized') is True) for _,ctx in selected.values())


def _diffuse_negative(selected):
    for name in ('area','high_area'):
        row,_ = selected[name]
        if not _measured(row) or not _zero(row.get('value')):
            return False
    for name in ('mean','p90'):
        row,_ = selected[name]
        if row.get('value') is not None or row.get('reason') != 'no_valid_targets':
            return False
    domains = []
    for _,ctx in selected.values():
        original, excluded, remaining = (ctx.get(field) for field in
                                        ('original_valid_area_px','excluded_area_px','effective_area_px'))
        if (not all(finite(v) and int(v)==v for v in (original,excluded,remaining))
                or remaining < 100 or excluded < 0 or original != excluded+remaining
                or not _zero(ctx.get('pixel_score_summary',{}).get('count'))
                or not isinstance(ctx.get('exclusion_components'),list)
                or set(ctx['exclusion_components']) != {'focal_red','vascular','acne_candidates'}):
            return False
        domains.append((original,excluded,remaining))
    return len(set(domains)) == 1


def zero_state_candidates(measurements, basis):
    """Return evidence-only negative candidates; never replace measured nulls.

    09 requires an approved non-test anatomical state and original samples.
    11 is never inferred negative from zero geometric curvature.
    """
    indexed, output = _indexed(measurements), {}
    regions = sorted({(r['metric_id'].split('.')[0],r['region']) for r in measurements})
    for module, region in regions:
        def candidate(names, predicate, subgroup=None):
            selected = _eligible(indexed,basis,names,module,region)
            if selected is None or not predicate(selected):
                return
            key = module+('.'+subgroup if subgroup else '')+':'+region
            evidence = {module+'.'+name+':'+region: {'value':row.get('value'),'reason':row.get('reason'),
                        'measurement_status':row.get('measurement_status',ctx.get('measurement_status')),
                        'evidence':deepcopy(ctx)} for name,(row,ctx) in selected.items()}
            output[key] = {'module':module,'region':region,'subgroup':subgroup,'status':'confirmed_zero',
                           'reason':'explicit_evaluable_negative_evidence','quality_status':'PASS',
                           'valid_area_px':min(_quality(ctx,100)['valid_area_px'] for _,ctx in selected.values()),
                           'supporting_fields':list(evidence),'proof':evidence}
        value = lambda selected,name: selected[name][0].get('value')
        context = lambda selected,name: selected[name][1]
        if module == '01':
            candidate(('density','area_p50','large_density'), lambda s:
                      _zero(value(s,'density')) and _zero(context(s,'density').get('count'))
                      and context(s,'area_p50').get('instance_areas_px2') == []
                      and _zero(context(s,'density').get('unlocated_instance_count',0)))
        elif module == '02':
            candidate(('gloss_area','gloss_high_area','gloss_mean'), lambda s:
                      _zero(value(s,'gloss_area')) and _zero(value(s,'gloss_high_area'))
                      and _zero(context(s,'gloss_mean').get('pixel_score_summary',{}).get('count')), 'surface_gloss')
            candidate(('porphyrin_high_density','porphyrin_p90'), lambda s:
                      _zero(value(s,'porphyrin_high_density'))
                      and _zero(context(s,'porphyrin_p90').get('count'))
                      and context(s,'porphyrin_p90').get('instance_intensities') == [], 'porphyrin')
            keys = ['02.'+group+':'+region for group in ('surface_gloss','porphyrin')]
            if all(key in output for key in keys):
                output['02:'+region] = {'module':module,'region':region,'subgroup':None,'status':'confirmed_zero',
                    'reason':'both_oil_subgroups_confirmed_zero','quality_status':'PASS',
                    'valid_area_px':min(output[key]['valid_area_px'] for key in keys),
                    'supporting_fields':keys,'proof':{key:output[key] for key in keys}}
        elif module == '03':
            for layer in ('spots','brown','uv'):
                candidate((layer+'.area',layer+'.p90'),lambda s: _pigment_negative(s,layer),layer)
            keys = ['03.'+layer+':'+region for layer in ('spots','brown','uv')]
            if all(key in output for key in keys):
                output['03:'+region] = {'module':module,'region':region,'subgroup':None,'status':'confirmed_zero',
                    'reason':'all_three_imaging_layers_confirmed_zero','quality_status':'PASS',
                    'valid_area_px':min(output[key]['valid_area_px'] for key in keys),
                    'valid_area_aggregation':'minimum_layer_area_not_cross_channel_union',
                    'supporting_fields':keys,'proof':{key:deepcopy(output[key]) for key in keys}}
        elif module == '04':
            candidate(('area','high_area','mean','p90'),_diffuse_negative)
        elif module == '05':
            candidate(('clusters','roi_count','affected_area','local_density'), lambda s:
                      all(_zero(value(s,n)) for n in ('clusters','roi_count','affected_area'))
                      and _zero(context(s,'local_density').get('affected_area_px'))
                      and _zero(context(s,'local_density').get('skeleton_length_px')))
        elif module == '06':
            names = tuple(kind+'_'+suffix for kind in ACNE_CLASSES for suffix in ('count','density'))
            candidate(names, lambda s: all(_zero(value(s,n)) and _zero(context(s,n).get('unknown_count'))
                                           and context(s,n).get('classification_samples') == [] for n in names))
        elif module == '07':
            candidate(('area','high_area','density','contrast_p50'), lambda s:
                      _zero(value(s,'area')) and _zero(value(s,'high_area'))
                      and all(_zero(ctx.get('line_pixels')) for _,ctx in s.values()))
        elif module == '08':
            names = ('main_count','length_burden','depth') if region in ('forehead','glabella') else ('area','high_area','density','contrast_p50')
            candidate(names, lambda s: _zero(value(s,names[0]))
                      and all(_zero(ctx.get('line_pixels')) for _,ctx in s.values()))
        elif module == '09' and region in GROOVE_REGIONS:
            candidate(('mean_depth','p90_depth','volume'),
                      lambda s: _groove_negative(s,indexed,basis,region))
            key, extent_key = '09:'+region, '09.extent:'+region
            if key in output:
                output[key]['supporting_fields'].append(extent_key)
                output[key]['proof'][extent_key] = {'value':indexed[extent_key]['value'],
                                                   'evidence':deepcopy(_context(extent_key,basis))}
        elif module == '10':
            names = ('raised_area','raised_p90','depressed_area','depressed_p90')
            def surface_negative(s):
                if not all(_zero(value(s,kind+'_area')) for kind in ('raised','depressed')):
                    return False
                for _,ctx in s.values():
                    classes = basis.get(ctx.get('texture_classification_ref'),{})
                    if (ctx.get('missing_exclusions') != [] or ctx.get('unclassified_texture_area_px') != 0
                            or ctx.get('class_area_px') != 0 or not isinstance(classes,dict)
                            or classes.get('ambiguous_components') != [] or not isinstance(classes.get('instances'),list)):
                        return False
                return True
            candidate(names,surface_negative)
    groove_keys = ['09:'+region for region in GROOVE_REGIONS]
    if all(key in output for key in groove_keys):
        output['09:full_face'] = {'module':'09','region':'full_face','subgroup':None,'status':'confirmed_zero',
            'reason':'all_required_anatomical_regions_confirmed_zero','quality_status':'PASS',
            'valid_area_px':min(output[key]['valid_area_px'] for key in groove_keys),
            'valid_area_aggregation':'minimum_child_area_not_an_overlapping_sum',
            'supporting_fields':groove_keys,'proof':{key:deepcopy(output[key]) for key in groove_keys}}
    return output
