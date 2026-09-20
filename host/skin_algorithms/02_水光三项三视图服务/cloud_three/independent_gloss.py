"""Uniform surface-gloss coverage score using existing same-region references."""
import math
from .v3_rules.scoring import grade,display_score
VERSION='water-gloss-area-high-area-20260920-1'
WEIGHTS={'gloss_area':25/45,'gloss_high_area':20/45}
def finite(x):return type(x) in (int,float) and math.isfinite(x)

def score_gloss(scores,payload):
    item=scores['surface_gloss'];rows={r['metric_id']+':'+r['region']:r for r in payload['measurements']}
    output=[]
    for region,node in [('full_face',item),*((r['region'],r) for r in item['regional_scores']['items'])]:
        if node.get('zero_state_proof'):
            output.append({'region':region,'method':'observed_zero','score':node['score']});continue
        components={};reason=None;lineage=set()
        for name,weight in WEIGHTS.items():
            key='02.'+name+':'+region;row=rows.get(key,{});scored=payload['scores'].get(key,{})
            context=payload['basis'].get(key,{});trace=scored.get('trace',{})
            if (row.get('status')!='measured' or not finite(row.get('value')) or not 0<=row['value']<=1
                or context.get('quality_status') not in ('PASS','WARNING') or context.get('quality_reason')
                or not finite(context.get('effective_area_px')) or context['effective_area_px']<100):
                reason='gloss_coverage_measurement_unavailable';break
            if (scored.get('score_status')!='candidate' or scored.get('reference_status')!='compatible'
                or trace.get('reference_identity_validation',{}).get('status')!='matched'
                or trace.get('formal_report_eligible') is not True
                or not finite(scored.get('score_raw')) or not 0<=scored['score_raw']<=100):
                reason='gloss_coverage_reference_unavailable';break
            lineage.add((trace.get('reference_version'),trace.get('reference_source')))
            components[name]={'weight':weight,'measurement':row['value'],'score_raw':scored['score_raw'],
                              'reference_trace':trace,'valid_area_px':context['effective_area_px']}
        if len(components)==2 and (len(lineage)!=1 or any(None in x for x in lineage)):
            reason='gloss_coverage_reference_lineage_mismatch'
        if reason:
            node.update(score=None,severity=None,score_status='unavailable',score_reason=reason)
            output.append({'region':region,'reason':reason});continue
        raw=sum(c['score_raw']*c['weight'] for c in components.values())
        node.update(score=display_score(raw),severity=grade(raw),score_status='available',score_reason=None,score_method=VERSION)
        output.append({'region':region,'method':VERSION,'score':node['score'],'score_raw':raw,'components':components})
    item['score_basis']=VERSION;item['scoring_version']=VERSION
    item['regional_scores']['scoring_version']=VERSION
    return {'version':VERSION,'weights':WEIGHTS,'regions':output,'mean_intensity_used':False,'reference_fit_runs':0}
