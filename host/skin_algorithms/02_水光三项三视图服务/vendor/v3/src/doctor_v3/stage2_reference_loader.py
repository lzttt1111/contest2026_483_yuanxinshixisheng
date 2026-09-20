"""Validate a unified new scoring reference without mutating its signed content."""
from copy import deepcopy
import math
from .stage1_config import resolve_config, digest
from .stage1_schema import version_identity
from .stage1_reference import validate_reference_qualification, reference_content_sha256
from .stage1_scoring import CONTRACT_FIELDS
from .registry import MODULES, PORE_REGIONS, SKIN_REGIONS

SCHEMA='doctor_v3_scoring_reference_v1'
VERSION_FIELDS=('data','roi','formula','implementation','thresholds')


def _require(condition,reason):
    if not condition:raise ValueError(reason)


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _thresholds(config):
    for name,regions,maximum in (('large_pore_area',PORE_REGIONS,None),('porphyrin_high_intensity',SKIN_REGIONS,1)):
        if name not in config:continue
        value=config[name]
        entries=value if isinstance(value,dict) else {'full_face':value}
        _require(set(entries)<=set(regions)|{'full_face'},'unknown threshold ROI:'+name)
        for cutoff in entries.values():
            _require(_finite(cutoff) and cutoff>0 and (maximum is None or cutoff<=maximum),'invalid threshold:'+name)
    if 'vascular_grid_step_px' in config:
        step=config['vascular_grid_step_px']
        _require(_finite(step) and step>=1 and int(step)==step,'invalid vascular grid step')
    defaults=resolve_config({})
    for group in ('lines','geometry','imaging'):
        for name,default in defaults[group].items():
            value=config[group].get(name)
            if isinstance(default,bool):_require(type(value) is bool,'invalid boolean configuration:'+group+'.'+name)
            elif _finite(default):_require(_finite(value),'invalid numeric configuration:'+group+'.'+name)
    anatomy=config.get('anatomical_extent',{})
    _require(isinstance(anatomy,dict),'anatomical configuration must be an object')
    for value in anatomy.values():
        _require(isinstance(value,dict) and value.get('test_only') is False and value.get('approved') is True
                 and value.get('purpose')=='formal' and isinstance(value.get('approval_id'),str) and value['approval_id'].strip(),
                 'anatomical configuration lacks explicit non-test approval')


def validate_scoring_reference_config(doc,profile):
    """Return an independent context, never add sha256 or legacy references to doc.

    Qualification binds source/unit and all ten identity fields to independent
    validation. Actual future measurement matching remains the scorer's strict
    ten-field responsibility; this loader cannot infer a detector's unit.
    """
    _require(isinstance(doc,dict) and doc.get('schema_version')==SCHEMA,'new scoring reference schema required')
    _require(profile in ('consumer','institution') and doc.get('capture_profile')==profile,'scoring reference profile mismatch')
    _require(doc.get('status') in ('candidate','calibrated'),'explicit scoring reference status required')
    _require(doc.get('purpose')=='formal' and doc.get('test_only') is False,'test or unvalidated scoring reference forbidden')
    _require('references' not in doc,'legacy references mapping is not a new scoring reference')
    qualification=validate_reference_qualification(doc)
    _require(qualification['status']=='qualified' and qualification['formal_report_eligible'],'formal qualification required')
    config=doc.get('thresholds')
    _require(isinstance(config,dict),'complete measurement configuration required')
    _require(config==resolve_config(config),'measurement configuration omits effective defaults')
    _require(doc.get('imaging')==config.get('imaging'),'top-level imaging disagrees with measurement configuration')
    _thresholds(config)
    trained=doc.get('training_measurement_versions')
    _require(isinstance(trained,dict),'training measurement versions required')
    expected=version_identity(config,None)
    for field in VERSION_FIELDS:
        _require(isinstance(trained.get(field),str) and trained[field]==expected[field],
                 'training configuration/version mismatch:'+field)
    definitions={module.id:module for module in MODULES}
    metrics=doc.get('metrics')
    _require(isinstance(metrics,dict) and bool(metrics),'nonempty metric declarations required')
    for key,row in metrics.items():
        _require(isinstance(row,dict) and all(isinstance(row.get(f),str) and row[f].strip() for f in CONTRACT_FIELDS),
                 'incomplete metric identity:'+key)
        module=row['metric_id'].split('.',1)[0]
        _require(module in definitions and row['region'] in ('full_face',*definitions[module].regions),'unknown module/ROI:'+key)
        _require(row['capture_profile']==profile and key==row['metric_id']+':'+row['region'], 'metric profile/key mismatch:'+key)
        _require(row.get('version')==doc.get('version'),'metric reference version mismatch:'+key)
        for field,version in (('roi_version','roi'),('formula_identity','implementation'),('threshold_identity','thresholds')):
            _require(row[field]==trained[version],'metric measurement identity mismatch:'+field+':'+key)
        _require(row['definition_version'].startswith(trained['formula']+'/'+row['source_kind']+'/'),
                 'metric definition/source does not match training formula:'+key)
        _require(row['direction'] in ('higher_health','higher_burden'),'invalid metric direction:'+key)
    return {'schema_version':SCHEMA,'capture_profile':profile,'reference':deepcopy(doc),
            'measurement_config':deepcopy(config),'training_measurement_versions':deepcopy(trained),
            'qualification':qualification,'reference_content_sha256':reference_content_sha256(doc),
            'document_sha256':digest(doc),'clinical_validation':doc.get('clinical_validation','not_performed')}
