#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
SUMS = ROOT / "SHA256SUMS"
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "captures"}
REQUIRED = [
    "README.md", "目标检测与自动拍照技术及部署说明_20260915.md",
    "pyproject.toml", "uv.lock", "UPSTREAM_SOURCES.json", "MODEL_MANIFEST.json",
    "QA_RECEIPT.json", "pc_demo/server.py", "pc_demo/lab/state.py",
    "pc_demo/lab/regions.py", "pc_demo/vendor/target_detection/runtime.py",
    "pc_demo/web/index.html", "pc_demo/models/device/model.onnx",
    "pc_demo/models/contact/model.onnx", "pc_demo/models/face_detection_yunet.onnx",
    "pc_demo/models/head_pose_fsanet_1x1.onnx", "pc_demo/models/face_landmarker.task",
]


def included(path: Path) -> bool:
    return path.is_file() and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if included(path)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = {row["path"]: row for row in manifest["files"]}
    if set(rows) != actual:
        fail(f"manifest path set differs: missing={sorted(actual-set(rows))}, extra={sorted(set(rows)-actual)}")
    for relative, row in rows.items():
        if row.get("sha256") is None:
            continue
        path = ROOT / relative
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            fail("manifest mismatch: " + relative)
    sum_rows = {}
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line)
        if not match:
            fail("invalid SHA256SUMS line: " + line)
        sum_rows[match.group(2)] = match.group(1)
    expected_sum_paths = actual - {SUMS.relative_to(ROOT).as_posix()}
    if set(sum_rows) != expected_sum_paths:
        fail("SHA256SUMS path set differs")
    for relative, expected in sum_rows.items():
        if digest(ROOT / relative) != expected:
            fail("SHA256 mismatch: " + relative)
    models = json.loads((ROOT / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    for row in models["models"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            fail("model mismatch: " + row["path"])
    print(json.dumps({"status": "ok", "files": len(actual), "sha256_verified": len(sum_rows), "models": len(models["models"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
