"""Atomic, integrity-checked V3 results and locked idempotent sample collection."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
from .stage1_schema import SCHEMA_VERSION

WORKSPACE = Path(os.environ.get("DERMAVISION_V3_WORKSPACE", Path(__file__).resolve().parents[3])).resolve()
FIELDS = {'schema_version', 'subject_id', 'capture_profile', 'input_sha256',
          'versions', 'measurements', 'basis', 'scores'}


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _target(path):
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('symlink target forbidden')
    resolved = path.resolve()
    if resolved == WORKSPACE or WORKSPACE not in resolved.parents:
        raise ValueError('write target must remain inside V3 workspace')
    return resolved


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('invalid relative asset path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '.' == name:
        raise ValueError('asset path escape')
    return p


def _portable(value):
    if isinstance(value, dict):
        for key, child in value.items():
            _portable(key)
            _portable(child)
    elif isinstance(value, list):
        for child in value:
            _portable(child)
    elif isinstance(value, str):
        if PureWindowsPath(value).is_absolute() or value.startswith('/') or '..' in value.replace('\\', '/').split('/'):
            raise ValueError('absolute or escaping path in result basis')


def _validate(payload):
    if not isinstance(payload, dict) or not FIELDS.issubset(payload):
        raise ValueError('incomplete result payload')
    if payload.get('status', 'success') != 'success':
        raise ValueError('failed result cannot be published as success')
    if payload['schema_version'] != SCHEMA_VERSION or payload['capture_profile'] not in ('consumer', 'institution'):
        raise ValueError('unsupported schema or capture profile')
    for field in ('schema_version', 'subject_id', 'capture_profile'):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError('missing identity: ' + field)
    versions = payload['versions']
    if not isinstance(versions, dict) or not versions or any(not isinstance(v, str) or not v for v in versions.values()):
        raise ValueError('explicit nonempty versions required')
    hashes = payload['input_sha256']
    hashes = hashes.values() if isinstance(hashes, dict) and hashes else [hashes]
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
        raise ValueError('invalid input SHA256')
    if not isinstance(payload['measurements'], list) or not all(isinstance(payload[k], dict) for k in ('basis', 'scores')):
        raise ValueError('invalid measurement/basis/score shape')
    seen = set()
    for row in payload['measurements']:
        if not isinstance(row, dict) or not all(row.get(k) for k in ('metric_id', 'region', 'definition_version', 'capture_profile', 'unit', 'source_kind', 'direction', 'status')):
            raise ValueError('invalid measurement row')
        key = row['metric_id'] + ':' + row['region']
        if row['status'] not in ('measured', 'unavailable'):
            raise ValueError('unknown measurement status')
        if key in seen or row['capture_profile'] != payload['capture_profile']:
            raise ValueError('duplicate metric or profile mismatch')
        seen.add(key)
        val = row.get('value')
        if row['status'] == 'measured' and (isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val)):
            raise ValueError('measured value must be finite')
    _portable(payload['basis'])
    _bytes(payload)
    return payload


def resume_key(identity, versions):
    """Stable identity hash; all supplied versions participate in isolation."""
    if not isinstance(identity, dict) or not identity or not isinstance(versions, dict) or not versions:
        raise ValueError('identity and versions required')
    return _sha(_bytes({'identity': identity, 'versions': versions}))


def _payload_key(payload):
    return resume_key({k: payload[k] for k in ('schema_version', 'subject_id', 'capture_profile', 'input_sha256')}, payload['versions'])


@contextmanager
def _lock(path):
    path = _target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_dir(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_result(destination, payload, assets=None):
    """Publish a new immutable directory. Assets map relative names to bytes/Paths."""
    payload = json.loads(_bytes(_validate(payload)))
    destination = _target(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _lock(destination.parent / ('.' + destination.name + '.publish.lock')):
        if destination.exists():
            raise FileExistsError(destination)
        staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.staging-', dir=destination.parent))
        published = False
        try:
            files = {'result.json': _bytes(payload)}
            for name, value in (assets or {}).items():
                path = str(_relative(name))
                if path in ('result.json', 'manifest.json') or path in files:
                    raise ValueError('reserved or duplicate asset name')
                files[path] = value if isinstance(value, bytes) else Path(value).read_bytes()
            for name, data in files.items():
                _write(staging / name, data)
            manifest = {'status': 'success', 'resume_key': _payload_key(payload),
                        'files': {name: _sha(data) for name, data in files.items()}}
            _write(staging / 'manifest.json', _bytes(manifest))
            load_result(staging)
            for directory in sorted((p for p in staging.rglob('*') if p.is_dir()), reverse=True):
                _sync_dir(directory)
            _sync_dir(staging)
            staging.rename(destination)
            published = True
            _sync_dir(destination.parent)
        except BaseException:
            if published:
                shutil.rmtree(destination)
            if staging.exists():
                shutil.rmtree(staging)
            raise
    return destination


def load_result(root):
    """Return payload only after every published file and identity verifies."""
    root = _target(root)
    if not root.is_dir():
        raise ValueError('result is not a directory')
    if (root / 'manifest.json').is_symlink():
        raise ValueError('symlink manifest')
    manifest = json.loads((root / 'manifest.json').read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError('invalid manifest')
    if manifest.get('status') != 'success' or not isinstance(manifest.get('files'), dict) or 'result.json' not in manifest['files']:
        raise ValueError('result has no successful manifest')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in result')
        if path.is_file() and path != root / 'manifest.json':
            actual.add(path.relative_to(root).as_posix())
    if actual != set(manifest['files']):
        raise ValueError('result file set mismatch')
    for name, expected in manifest['files'].items():
        _relative(name)
        if _sha((root / name).read_bytes()) != expected:
            raise ValueError('result checksum mismatch: ' + name)
    payload = _validate(json.loads((root / 'result.json').read_bytes()))
    if _payload_key(payload) != manifest.get('resume_key'):
        raise ValueError('result identity mismatch')
    return payload


def append_sample(jsonl_path, payload):
    """Append once using a durable SQLite accelerator; JSONL stays authoritative."""
    from .sample_collection_index import append_indexed
    return append_indexed(jsonl_path, payload)
