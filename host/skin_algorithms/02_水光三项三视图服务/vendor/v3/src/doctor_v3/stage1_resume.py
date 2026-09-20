"""Completion receipts written only after the requested V3 delivery is complete."""
import json
import os
from pathlib import Path
import tempfile
from xml.etree import ElementTree
import zipfile

from .stage1_storage import _bytes, _sha, _target, _relative, _lock, _sync_dir, load_result

RECEIPT = 'STAGE1_COMPLETE.json'
SCHEMA = 'doctor_v3_stage1_completion_v1'


def _payload(root, signature):
    if not isinstance(signature, dict) or not signature.get('v3_versions'):
        raise ValueError('explicit completion signature and versions required')
    payload = load_result(root / 'v3_scoring')
    if _bytes(payload['versions']) != _bytes(signature['v3_versions']):
        raise ValueError('completion versions mismatch')
    if signature.get('capture_profile') != payload['capture_profile']:
        raise ValueError('completion capture profile mismatch')
    for key in ('subject_id', 'input_sha256'):
        if key in signature and _bytes(signature[key]) != _bytes(payload[key]):
            raise ValueError('completion identity mismatch: ' + key)
    if signature.get('v3_output_profile') == 'scoring' and signature.get('v3_pdf_requested'):
        raise ValueError('scoring-only mode cannot request Word')
    return payload


def _word(root, relative):
    relative = str(_relative(relative))
    path = _target(root / relative)
    if root not in path.parents or path.suffix.lower() != '.docx' or not path.is_file():
        raise ValueError('missing or escaped Word report')
    with zipfile.ZipFile(path) as package:
        required = {'[Content_Types].xml', '_rels/.rels', 'word/document.xml'}
        if not required.issubset(package.namelist()) or package.testzip() is not None:
            raise ValueError('incomplete Word package')
        for name in required:
            ElementTree.fromstring(package.read(name))
        document = ElementTree.fromstring(package.read('word/document.xml'))
        if document.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body') is None:
            raise ValueError('Word document has no body')
    return {'path': relative, 'sha256': _sha(path.read_bytes()), 'bytes': path.stat().st_size}


def _reports(root, signature, report_files):
    if isinstance(report_files, (str, bytes)):
        raise ValueError('report_files must be a sequence of relative paths')
    names = [str(name) for name in report_files]
    if len(names) != len(set(names)):
        raise ValueError('duplicate Word reports')
    requested = signature.get('v3_pdf_requested') is True
    if (requested and len(names) != 2) or (not requested and names):
        raise ValueError('Word request requires exactly two complete reports')
    result = [_word(root, name) for name in names]
    if len({entry['path'] for entry in result}) != len(result):
        raise ValueError('duplicate normalized Word reports')
    return sorted(result, key=lambda entry: entry['path'])


def mark_complete(root, signature, report_files=()):
    """Atomically write completion after SHA-valid quantification and both Word files.

    signature.v3_versions must exactly match payload.versions. A true
    v3_pdf_requested requires two relative .docx paths; scoring mode passes none.
    Returns the completion receipt Path. Existing incomplete deliveries are never
    marked complete merely because their scoring manifest exists.
    """
    root = _target(root)
    signature = json.loads(_bytes(signature))
    with _lock(root / '.stage1-complete.lock'):
        payload = _payload(root, signature)
        reports = _reports(root, signature, report_files)
        receipt = {'schema_version': SCHEMA, 'status': 'success', 'signature': signature,
                   'versions': payload['versions'], 'subject_id': payload['subject_id'],
                   'input_sha256': payload['input_sha256'],
                   'scoring_manifest_sha256': _sha((root / 'v3_scoring/manifest.json').read_bytes()),
                   'report_files': reports}
        target = _target(root / RECEIPT)
        descriptor, temporary_name = tempfile.mkstemp(prefix='.stage1-complete-', suffix='.tmp', dir=root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(_bytes(receipt))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _sync_dir(root)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            # A failed final directory flush must not leave this new success marker.
            if target.exists() and target.read_bytes() == _bytes(receipt):
                target.unlink()
            raise
    return target


def matches(root, signature):
    """False for any missing/corrupt payload, incomplete Word, or changed signature."""
    try:
        root = _target(root)
        target = _target(root / RECEIPT)
        receipt = json.loads(target.read_bytes())
        if receipt.get('schema_version') != SCHEMA or receipt.get('status') != 'success':
            return False
        if _bytes(receipt.get('signature')) != _bytes(signature):
            return False
        payload = _payload(root, signature)
        if any(_bytes(receipt.get(key)) != _bytes(payload[key]) for key in ('versions', 'subject_id', 'input_sha256')):
            return False
        if receipt.get('scoring_manifest_sha256') != _sha((root / 'v3_scoring/manifest.json').read_bytes()):
            return False
        reports = receipt.get('report_files')
        if not isinstance(reports, list):
            return False
        actual = _reports(root, signature, [entry['path'] for entry in reports])
        return _bytes(actual) == _bytes(reports)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, zipfile.BadZipFile, ElementTree.ParseError, RuntimeError):
        return False
