"""Pure saved jaw-path support checks; no skin-area imputation or image loading."""
import math

from .stage1_config import digest

VERSION = 'v301-jaw-path-support-1'
JAW_IDS = {'left_jaw': [132,58,172,136,150,149,176,148,152],
           'right_jaw': [361,288,397,365,379,378,400,377,152]}


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def _context(basis, region):
    key = '11.jaw_continuity:' + region
    local = basis.get(key)
    if not isinstance(local,dict):
        raise ValueError('missing_jaw_context:' + region)
    shared = {}
    if 'evidence_ref' in local:
        ref = local['evidence_ref']
        if not isinstance(ref,str) or ref == key or not isinstance(basis.get(ref),dict):
            raise ValueError('invalid_jaw_evidence_ref:' + region)
        shared = basis[ref]
        if 'evidence_ref' in shared:
            raise ValueError('nested_jaw_evidence_ref:' + region)
    context = {**shared, **local}
    if (context.get('quality_status') != 'PASS' or context.get('quality_reason')
            or context.get('reason') is not None or context.get('measurement_status') != 'measured'
            or context.get('geometry_sampling') != '2d_anatomical_path'):
        raise ValueError('invalid_jaw_observation_status:' + region)
    return context


def _same(actual, expected):
    return (isinstance(actual,list) and len(actual) == len(expected)
            and all(_number(a) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
                    for a,b in zip(actual,expected)))


def _side(context, region):
    ids = context.get('sample_ids')
    points = context.get('path_xy')
    if (ids != JAW_IDS[region] or not all(type(i) is int for i in ids)
            or not isinstance(points,list) or len(points) != len(ids)
            or any(not isinstance(p,list) or len(p) != 2 or
                   any(not _number(v) or v < 0 for v in p) for p in points)):
        raise ValueError('invalid_jaw_path_samples:' + region)
    segments = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(points,points[1:])]
    lengths = [math.hypot(x,y) for x,y in segments]
    if any(v <= 0 or not math.isfinite(v) for v in lengths):
        raise ValueError('degenerate_jaw_path:' + region)
    turns = [abs(math.atan2(a[0]*b[1]-a[1]*b[0],a[0]*b[0]+a[1]*b[1]))
             for a,b in zip(segments,segments[1:])]
    if not (_same(context.get('segment_length_px'),lengths)
            and _same(context.get('turn_angles_radians'),turns)):
        raise ValueError('jaw_geometry_recomputation_mismatch:' + region)
    return turns, {'region':region,'sample_ids':ids,'path_sha256':digest(points),
                   'segment_count':len(lengths),'turn_count':len(turns),'basis_sha256':digest(context)}


def jaw_measurement_support(measurement, basis):
    """Verify saved per-side geometry and the original P90 formula, without pixels.

    The full-face observation must contain the exact left/right concatenation.
    Support proves arithmetic/source consistency, not clinical calibration.
    """
    try:
        row = measurement
        region = row.get('region')
        if (row.get('metric_id') != '11.jaw_continuity' or region not in (*JAW_IDS,'full_face')
                or row.get('direction') != 'higher_health'
                or row.get('source_kind') != 'regional_25d_proxy'
                or row.get('status') != 'measured'
                or row.get('measurement_status',row.get('status')) != 'measured'
                or row.get('reason') is not None or not _number(row.get('value'))
                or not isinstance(basis,dict)):
            raise ValueError('invalid_jaw_measurement')
        context = _context(basis,region)
        sides = list(JAW_IDS) if region == 'full_face' else [region]
        turns, paths = [], []
        for side in sides:
            side_context = _context(basis,side) if region == 'full_face' else context
            values, proof = _side(side_context,side)
            turns.extend(values)
            paths.append(proof)
        if region == 'full_face' and (context.get('bilateral_complete') is not True
                or not _same(context.get('turn_angles_radians'),turns)):
            raise ValueError('incomplete_or_inconsistent_bilateral_jaw')
        ordered = sorted(turns)
        position = .9*(len(ordered)-1)
        low = int(position)
        p90 = ordered[low] + (ordered[min(low+1,len(ordered)-1)]-ordered[low])*(position-low)
        value = min(1.,max(0.,1-p90/(math.pi/2)))
        if not math.isclose(row['value'],value,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('jaw_value_recomputation_mismatch')
        proof = {'version':VERSION,'support_kind':'2d_anatomical_path','region':region,
                 'paths':paths,'turn_angles_sha256':digest(turns),'recomputed_value':value,
                 'measurement_sha256':digest(row),'basis_sha256':digest(context),
                 'skin_area_imputed':False,'clinical_validation':'not_performed'}
        return {'status':'supported','reason':None,'proof':proof,'proof_sha256':digest(proof)}
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return {'status':'unavailable','reason':str(exc),'proof':None}
