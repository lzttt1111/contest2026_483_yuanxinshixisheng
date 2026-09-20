"""Read-only binding of a stage-two execution review to current runtime code."""
import hashlib
import json
from pathlib import Path, PurePosixPath

SCHEMA = 'dermavision_v301_execution_review_v1'
REQUIRED_NEW_RUNTIME = {'scripts/run_v301_monitored_batch.py','src/doctor_v3/stage2_execution_gate.py'}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _file(base, relative):
    _require(isinstance(relative,str) and relative and '\\' not in relative and ':' not in relative,
             'invalid relative source path')
    name = PurePosixPath(relative)
    _require(relative==name.as_posix() and not name.is_absolute() and '..' not in name.parts and '.' not in name.parts,
             'escaped relative source path')
    path = base
    for part in name.parts:
        path = path/part
        _require(not path.is_symlink(),'symlink source path')
    _require(path.is_file() and path.resolve().is_relative_to(base.resolve()),'missing or escaped source file')
    return path


def _read_plain(path, root):
    path = Path(path).absolute()
    _require(path.is_relative_to(root.absolute()),'review outside V3')
    return _file(root,path.relative_to(root.absolute()).as_posix())


def validate_execution_review(execution_review, phase1_review, code):
    """Re-read the two reviews, both manifests and each current source file."""
    code = Path(code).absolute(); workspace = code.parent
    _require(execution_review is not None,'stage-two --execution-review is required')
    phase1 = _read_plain(phase1_review,workspace)
    _require(phase1.name=='REVIEW.json' and phase1.parent.name=='astra_xhigh_final_20260910'
             and phase1.parents[1].name=='review' and phase1.parents[2].name=='candidate_r7',
             'phase-one review must be the original candidate_r7 review')
    original = json.loads(phase1.read_bytes())
    _require(original.get('candidate')=='candidate_r7','wrong phase-one candidate identity')
    baseline = _file(phase1.parents[2],'definitions/FINAL_SOURCE_SHA256.json')
    _require(original.get('source_manifest_sha256')==_sha(baseline),'phase-one source manifest binding mismatch')
    baseline_rows = json.loads(baseline.read_bytes())
    _require(isinstance(baseline_rows,list) and bool(baseline_rows),'missing phase-one runtime manifest')
    required = set(REQUIRED_NEW_RUNTIME)
    baseline_paths = set()
    for row in baseline_rows:
        _require(isinstance(row,dict),'invalid baseline source row')
        name = row.get('path')
        _require(isinstance(name,str) and name not in baseline_paths,'invalid baseline source identity')
        baseline_paths.add(name)
        # Validate baseline paths syntactically, without requiring their old
        # hashes to match approved stage-two edits.
        _file(code,name)
        if name.endswith('.py') and PurePosixPath(name).parts[0] not in ('tests','docs'):
            required.add(name)
    review = _read_plain(execution_review,workspace)
    document = json.loads(review.read_bytes())
    reviewer = str(document.get('reviewer','')).lower()
    _require(document.get('schema_version')==SCHEMA and document.get('passed') is True
             and 'astra' in reviewer and 'xhigh' in reviewer,'execution review must explicitly pass Astra xhigh')
    _require(not any(document.get(k) for k in ('blockers','blocking_findings','unresolved_blockers')),
             'execution review contains blockers')
    _require(document.get('verdict',document.get('status')) not in ('BLOCK','FAIL','FAILED'),
             'contradictory execution review verdict')
    findings=document.get('findings',[])
    for finding in findings:
        _require(isinstance(finding,dict) and (str(finding.get('status','')).lower() in ('resolved','fixed','closed')
                 or (finding.get('before_batch') is False and finding.get('blocking') is False)),
                 'execution review has unresolved findings')
    _require(bool(findings) or not any(document.get('finding_counts',{}).values()),
             'execution review finding counts contradict empty findings')
    _require(document.get('phase1_review_sha256')==_sha(phase1),'execution review phase-one binding mismatch')
    manifest = _file(review.parent,document.get('source_manifest'))
    _require(document.get('source_manifest_sha256')==_sha(manifest),'execution source manifest hash mismatch')
    entries = json.loads(manifest.read_bytes()).get('files')
    _require(isinstance(entries,list) and bool(entries),'missing execution source files')
    paths = set()
    for item in entries:
        _require(isinstance(item,dict) and isinstance(item.get('path'),str),'invalid execution source row')
        name = item.get('path')
        _require(name not in paths,'duplicate execution source path')
        path = _file(code,name)
        _require(_sha(path)==item.get('sha256'),'current runtime source drift: '+name)
        paths.add(name)
    _require(required<=paths,'execution manifest omits required runtime files: '+','.join(sorted(required-paths)))
    return {'execution_review':str(review),'execution_review_sha256':_sha(review),
            'phase1_review_sha256':_sha(phase1),'phase1_source_manifest_sha256':_sha(baseline),
            'execution_source_manifest':str(manifest),'execution_source_manifest_sha256':_sha(manifest),
            'verified_source_count':len(paths),'required_runtime_count':len(required)}
