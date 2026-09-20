from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detection_runtime.cloud_input import parse_capture_images
from src.detection_runtime.cloud_task import CaptureTaskContext, _download_images, close_capture_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh preparation output")
    args.output.mkdir(parents=True)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    rows = fixture["images"]
    results = []
    try:
        for alias in ("clinic28-25", "clinic28-09", "clinic28-23"):
            selected = [r for r in rows if r["capture_alias"] == alias]
            sources = [
                {"filename": f"{alias}_{r['role']}_M.jpg", "oss_key": r["relative_path"]}
                for r in selected
            ]
            root = args.output / alias
            root.mkdir()
            context = CaptureTaskContext(root, parse_capture_images(sources))
            start = time.perf_counter()
            _download_images(context, lambda key: (args.fixture.parent / key).read_bytes(),
                             lambda _: (_ for _ in ()).throw(AssertionError("unexpected URL")), 32*1024*1024)
            record = {
                "alias": alias, "seconds": time.perf_counter()-start,
                "channels": {c.role: {"sha256":c.sha256, "registration":c.registration_status}
                             for c in context.capture.channels},
            }
            expected = {("365_M" if r["role"]=="365" else r["role"]+"_M"):r for r in selected}
            for c in context.capture.channels:
                assert c.sha256 == expected[c.role]["sha256"]
                assert c.registration_status == expected[c.role]["registration_status"], (alias,c.role,c.registration_status)
            results.append(record)
            print(json.dumps(record), flush=True)
    finally:
        close_capture_runtime()
        (args.output/"PREPARATION_RECEIPT.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
