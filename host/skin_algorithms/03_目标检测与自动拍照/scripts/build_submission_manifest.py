#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
SUMS = ROOT / "SHA256SUMS"
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "captures"}


def included(path: Path) -> bool:
    return path.is_file() and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    MANIFEST.touch(exist_ok=True)
    SUMS.touch(exist_ok=True)
    files = sorted((path for path in ROOT.rglob("*") if included(path)), key=lambda p: p.as_posix())
    rows = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path in {MANIFEST, SUMS}:
            rows.append({"path": relative, "size": None, "sha256": None, "note": "self/cross-reference; verified by SHA256SUMS where possible"})
        else:
            rows.append({"path": relative, "size": path.stat().st_size, "sha256": digest(path)})
    payload = {
        "schema": "aisia_submission_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "0.0.1",
        "status": "PC_FUNCTIONAL_BASELINE_RK3576_NOT_DEPLOYED",
        "excluded_runtime_paths": sorted(SKIP_PARTS),
        "files": rows,
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    files = sorted((path for path in ROOT.rglob("*") if included(path) and path != SUMS), key=lambda p: p.as_posix())
    SUMS.write_text("".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest_files": len(rows), "sha256_entries": len(files)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
