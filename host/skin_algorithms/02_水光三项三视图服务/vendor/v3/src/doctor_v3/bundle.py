"""Portable, hash-bound V3 report bundle. No model imports or source-path dependency."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from . import BUNDLE_VERSION

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def contained(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
        raise ValueError("unsafe bundle path")
    root = Path(root).resolve()
    candidate = root.joinpath(*p.parts)
    cursor = root
    for part in p.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("symlink in bundle")
    if not candidate.resolve().is_relative_to(root) or not candidate.is_file():
        raise ValueError("missing or escaped bundle member")
    return candidate

def validate_payload(payload):
    if payload.get("schema_version") != BUNDLE_VERSION:
        raise ValueError("unsupported bundle schema")
    if payload.get("capture_profile") not in ("consumer", "institution"):
        raise ValueError("invalid capture profile")
    if not payload.get("subject_id"):
        raise ValueError("missing subject ID")
    if [m["id"] for m in payload.get("modules", [])] != [f"{i:02d}" for i in range(1, 12)]:
        raise ValueError("eleven ordered modules required")
    seen = set()
    for m in payload.get("measurements", []):
        key = (m["metric_id"], m["region"])
        if key in seen:
            raise ValueError("duplicate measurement")
        seen.add(key)
        if m["status"] == "measured" and (m.get("value") is None or isinstance(m["value"], bool)):
            raise ValueError("measured value missing")
        if m["status"] == "unavailable" and m.get("value") is not None:
            raise ValueError("unavailable value must be null")
    json.dumps(payload, allow_nan=False)

def publish(destination, payload, assets):
    """Publish to a new directory only; failed staging never becomes a valid bundle."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("bundle destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".doctor-v3-", dir=destination.parent))
    try:
        validate_payload(payload)
        entries = []
        for relative, source in sorted(assets.items()):
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts or ":" in relative or "\\" in relative:
                raise ValueError("invalid asset path")
            source = Path(source)
            if source.is_symlink() or not source.is_file():
                raise ValueError("asset must be a regular file")
            target = stage.joinpath(*parts.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entries.append({"path": relative, "sha256": sha256(target), "size": target.stat().st_size})
        write_json(stage / "report.json", payload)
        entries.append({"path": "report.json", "sha256": sha256(stage / "report.json"), "size": (stage / "report.json").stat().st_size})
        write_json(stage / "manifest.json", {"schema_version": BUNDLE_VERSION, "files": entries})
        load(stage)
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination

def load(root):
    root = Path(root)
    manifest = json.loads(contained(root, "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_VERSION:
        raise ValueError("unsupported manifest")
    seen = set()
    for entry in manifest["files"]:
        relative = entry["path"]
        if relative in seen:
            raise ValueError("duplicate manifest member")
        seen.add(relative)
        path = contained(root, relative)
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError("bundle hash mismatch: " + relative)
    if "report.json" not in seen:
        raise ValueError("report is not hash-bound")
    payload = json.loads(contained(root, "report.json").read_text(encoding="utf-8"))
    validate_payload(payload)
    for media in payload.get("media", []):
        if media["path"] not in seen:
            raise ValueError("unbound report media")
    return payload
