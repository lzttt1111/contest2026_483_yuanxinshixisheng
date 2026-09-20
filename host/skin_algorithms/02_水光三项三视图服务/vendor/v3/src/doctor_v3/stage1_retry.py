"""Preserve an incomplete stage-one delivery before retry, never overwrite it."""
import json
import os
from pathlib import Path
import uuid
from .stage1_runtime import require_output
from .stage1_schema import SCHEMA_VERSION


def prepare_retry(root):
    root=require_output(root)
    if not root.exists():return None
    payload=root/"v3_scoring/result.json"
    try:
        row=json.loads(payload.read_text(encoding="utf8"))
    except (OSError,ValueError):
        raise ValueError("existing output has no recognized V3 stage-one provenance; preserve it and choose a new output")
    if row.get("schema_version")!=SCHEMA_VERSION:
        raise ValueError("not a stage-one output; refusing to move it")
    archive=require_output(root.parent/"_incomplete_stage1"/(root.name+"_"+uuid.uuid4().hex[:12]))
    archive.parent.mkdir(parents=True,exist_ok=True)
    os.rename(root,archive)
    return archive
