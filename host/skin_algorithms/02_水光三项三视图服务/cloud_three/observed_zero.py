"""Explicit regional absence policy for standalone water scoring."""
import hashlib,json,math
VERSION='water-observed-zero-20260920-2'
def finite(v):return type(v) in (int,float) and math.isfinite(v)

def apply_observed_zero(scores,payload):
    indexed={r['metric_id']+':'+r['region']:r for r in payload['measurements']}
    basis=payload['basis'];applied=[]
    for project,module,names in [('pores','01',('density','area_p50','large_density')),
                                  ('surface_gloss','02',('gloss_area','gloss_high_area','gloss_mean'))]:
        item=scores[project]
        for region,node in [('full_face',item),*((r['region'],r) for r in item['regional_scores']['items'])]:
            if node['score'] is not None:continue
            evidence={};valid=True
            for name in names:
                key=module+'.'+name+':'+region
                row=indexed.get(key,{});context=basis.get(key,{})
                area=context.get('effective_area_px')
                if (context.get('quality_status') not in ('PASS','WARNING')
                    or 'quality_reason' not in context or context['quality_reason'] is not None
                    or not finite(area) or area<100):
                    valid=False;break
                value=row.get('value');state=row.get('measurement_status')
                if value is None:
                    if (state!='no_targets' or row.get('reason')!='no_valid_targets'
                        or context.get('reason')!='no_valid_targets'):
                        valid=False;break
                elif (not finite(value) or value!=0 or row.get('status')!='measured'
                      or state!='measured' or context.get('reason') is not None):
                    valid=False;break
                evidence[name]={'value':value,'measurement_status':state,'basis':context}
            if not valid:continue
            if project=='pores':
                d=evidence['density'];a=evidence['area_p50']
                valid=(d['value']==0 and d['basis'].get('count')==0
                       and d['basis'].get('unlocated_instance_count')==0
                       and a['value'] is None and a['basis'].get('instance_areas_px2')==[]
                       and evidence['large_density']['value']==0)
            else:
                valid=(evidence['gloss_area']['value']==0 and evidence['gloss_high_area']['value']==0
                       and evidence['gloss_mean']['value'] is None
                       and evidence['gloss_mean']['basis'].get('pixel_score_summary',{}).get('count')==0)
            if not valid:continue
            proof={'project':project,'region':region,'rule':VERSION,'evidence':evidence,
                   'quality_statuses':sorted({e['basis']['quality_status'] for e in evidence.values()}),
                   'statistics_imputed':False}
            proof['sha256']=hashlib.sha256(json.dumps(proof,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            node.update(score=100,severity='未见明显',score_status='available',score_reason=None,
                        zero_state_proof=proof,score_method=VERSION)
            applied.append({'project':project,'region':region,'proof':proof})
    return {'version':VERSION,'applied':applied,'statistics_imputed':False}
