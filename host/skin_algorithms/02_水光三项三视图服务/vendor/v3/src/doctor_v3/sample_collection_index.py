"""SQLite accelerator for authoritative JSONL, with bounded streaming recovery.

Normal append reads only its new byte range and never parses committed history.
An fsynced post-write receipt proves safe tail-only recovery when SQLite lags.
Unrecognized file changes (including the fsync-to-receipt crash window) trigger
a full streaming rebuild. Memory is O(largest JSONL record + incoming payload),
plus a bounded SQLite page cache; fingerprints are not adversarial authenticity.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

VERSION = 1


@dataclass
class IOStats:
    bytes_read: int = 0
    records_parsed: int = 0
    mode: str = ''


def _fingerprint(stream):
    stat = os.fstat(stream.fileno())
    return {key: int(getattr(stat, field)) for key, field in
            (('device', 'st_dev'), ('inode', 'st_ino'), ('size', 'st_size'),
             ('mtime_ns', 'st_mtime_ns'), ('ctime_ns', 'st_ctime_ns'))}


def _path_matches(path, fingerprint):
    stat = path.stat()
    actual = {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size,
              'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}
    if actual != fingerprint:
        raise ValueError('sample collection changed outside its owned file lock')


def _put(connection, key, payload_sha, offset, length):
    previous = connection.execute('SELECT payload_sha FROM records WHERE identity_key=?', (key,)).fetchone()
    if previous is not None:
        if previous[0] != payload_sha:
            raise ValueError('conflicting sample for identity/version')
        return
    connection.execute('INSERT INTO records VALUES (?,?,?,?)', (key, payload_sha, offset, length))


def _scan(stream, start, connection, stats, codec):
    validate, canonical, identity = codec
    stream.seek(start)
    end, newline, last_offset = start, True, None
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line: break
        stats.bytes_read += len(line)
        stats.records_parsed += 1
        try:
            decoded = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if not line.endswith(b'\n'):
                return end, newline, last_offset
            raise ValueError('corrupt sample collection') from exc
        old = validate(decoded)  # A valid JSON object with invalid schema is never discarded.
        raw = canonical(old)
        _put(connection, identity(old), hashlib.sha256(raw).hexdigest(), offset, len(line))
        end, newline, last_offset = stream.tell(), line.endswith(b'\n'), offset
    return end, newline, last_offset


def _range_sha(stream, start, end, stats):
    stream.seek(start)
    remaining, digest = end-start, hashlib.sha256()
    while remaining:
        block = stream.read(min(1024*1024, remaining))
        if not block: raise ValueError('sample collection shortened during verification')
        stats.bytes_read += len(block)
        digest.update(block); remaining -= len(block)
    return digest.hexdigest()


def _pending(path):
    try:
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, ValueError, OSError):
        return None


def _write_pending(path, value, canonical, sync_dir):
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        if temporary.exists(): temporary.unlink()


def _commit_index(connection):
    """Separate seam for testing the durable-data / index-commit crash boundary."""
    connection.commit()


def append_indexed(jsonl_path, payload, *, stats=None):
    from .stage1_storage import _target, _lock, _validate, _bytes, _payload_key, _sync_dir
    stats = stats if stats is not None else IOStats()
    payload = json.loads(_bytes(_validate(payload)))
    encoded, key = _bytes(payload), _payload_key(payload)
    payload_sha = hashlib.sha256(encoded).hexdigest()
    target = _target(jsonl_path)
    db = _target(target.with_name(target.name + '.index.sqlite3'))
    pending_path = _target(target.with_name(target.name + '.index.pending.json'))
    with _lock(target.with_name(target.name + '.lock')):
        for suffix in ('-journal', '-wal', '-shm'):
            _target(db.with_name(db.name + suffix))
        # SQLite temporary work is in-memory; all persistent journals are adjacent.
        connection = sqlite3.connect(str(db), timeout=30, isolation_level=None)
        try:
            connection.execute('PRAGMA journal_mode=DELETE')
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('PRAGMA temp_store=MEMORY')
            connection.execute('PRAGMA cache_size=-1024')
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, VERSION): raise ValueError('unsupported sample index schema')
            connection.execute('CREATE TABLE IF NOT EXISTS metadata (singleton INTEGER PRIMARY KEY CHECK(singleton=1), document TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS records (identity_key TEXT PRIMARY KEY, payload_sha TEXT NOT NULL, byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL)')
            if version == 0: connection.execute('PRAGMA user_version=1')
            connection.execute('BEGIN IMMEDIATE')
            stored = connection.execute('SELECT document FROM metadata WHERE singleton=1').fetchone()
            metadata = json.loads(stored[0]) if stored else None
            with target.open('r+b' if target.exists() else 'w+b') as stream:
                before = _fingerprint(stream)
                same = bool(metadata and metadata.get('target') == str(target) and metadata.get('file') == before
                            and metadata.get('indexed_end') == before['size'])
                receipt = _pending(pending_path)
                if same:
                    stats.mode = 'incremental'
                    end, newline, last_offset = before['size'], True, None
                    start = end
                else:
                    trusted_tail = bool(metadata and receipt and receipt.get('version') == VERSION
                        and receipt.get('target') == str(target) and metadata.get('target') == str(target)
                        and receipt.get('before') == metadata.get('file') and receipt.get('after') == before
                        and receipt.get('start') == metadata.get('indexed_end')
                        and receipt.get('end') == before['size'] and receipt['start'] <= receipt['end'])
                    if trusted_tail:
                        start = metadata['indexed_end']
                        if _range_sha(stream, start, before['size'], stats) != receipt.get('sha256'):
                            raise ValueError('pending sample tail checksum mismatch')
                        stats.mode = 'tail_recovery'
                    else:
                        start = 0
                        stats.mode = 'streaming_rebuild'
                        connection.execute('DELETE FROM records')
                    end, newline, last_offset = _scan(stream, start, connection, stats, (_validate, _bytes, _payload_key))
                _path_matches(target, before)
                existing = connection.execute('SELECT payload_sha FROM records WHERE identity_key=?', (key,)).fetchone()
                if existing is not None and existing[0] != payload_sha:
                    raise ValueError('conflicting sample for identity/version')
                added = existing is None
                if same and not added:
                    connection.rollback()
                    if pending_path.exists(): pending_path.unlink()
                    return False
                # No file mutation happens until parsing and identity conflicts pass.
                stream.seek(end)
                if end != before['size']: stream.truncate(end)
                if end and not newline:
                    stream.write(b'\n'); end += 1
                    connection.execute('UPDATE records SET byte_length=byte_length+1 WHERE byte_offset=?', (last_offset,))
                if added:
                    _put(connection, key, payload_sha, end, len(encoded)+1)
                    stream.write(encoded+b'\n'); end += len(encoded)+1
                stream.flush(); os.fsync(stream.fileno())
                after = _fingerprint(stream)
                _path_matches(target, after)
                # Bind the fsynced tail before the index transaction is committed.
                pending = {'version': VERSION, 'target': str(target),
                           'before': metadata.get('file') if metadata else None,
                           'after': after, 'start': start, 'end': end,
                           'sha256': _range_sha(stream, start, end, stats)}
                _write_pending(pending_path, pending, _bytes, _sync_dir)
                _path_matches(target, after)
                state = {'version': VERSION, 'target': str(target), 'file': after, 'indexed_end': end}
                connection.execute('INSERT OR REPLACE INTO metadata VALUES (1,?)', (_bytes(state).decode('utf-8'),))
                _commit_index(connection)  # The authoritative JSONL is already fsynced.
                if pending_path.exists(): pending_path.unlink()
                return added
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
