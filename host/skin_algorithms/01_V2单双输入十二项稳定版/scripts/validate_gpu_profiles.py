#!/usr/bin/env python3
"""本地验证一个 DermaVision GPU profile 的耗时和重复确定性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = os.getenv("DERMAVISION_GPU_PROFILE", "compat").strip().lower()
    inputs = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not inputs:
        raise SystemExit(f"没有测试图片: {args.input_dir}")

    from src.pipeline import DermaVisionPipeline

    pipeline = DermaVisionPipeline()
    cold_started = time.perf_counter()
    warmup = pipeline.warmup()
    cold_seconds = time.perf_counter() - cold_started
    records: list[dict[str, object]] = []
    previous_hashes: dict[str, dict[str, str]] = {}
    algorithms = ["redness", "spots", "brown", "texture", "pores"]

    try:
        for repeat in range(1, args.repeat + 1):
            for source in inputs:
                started = time.perf_counter()
                result = pipeline.process_single(str(source), algorithms)
                wall = time.perf_counter() - started
                if result.get("status") != "success":
                    raise RuntimeError(f"{source.name}: {result}")
                files = {
                    name: Path(path)
                    for name, path in result["results"].items()
                    if Path(path).is_file()
                }
                hashes = {name: sha256(path) for name, path in files.items()}
                deterministic = (
                    True
                    if repeat == 1
                    else hashes == previous_hashes[source.name]
                )
                if repeat == 1:
                    previous_hashes[source.name] = hashes
                timing = result.get("metadata", {}).get("timing_seconds", {})
                records.append(
                    {
                        "profile": profile,
                        "repeat": repeat,
                        "source": source.name,
                        "wall_seconds": round(wall, 4),
                        "timing_seconds": timing,
                        "deterministic_formal_outputs": deterministic,
                        "hashes": hashes,
                    }
                )
                if repeat == args.repeat:
                    destination = args.output_dir / profile / source.stem
                    destination.mkdir(parents=True, exist_ok=True)
                    for path in files.values():
                        shutil.copy2(path, destination / path.name)
    finally:
        pipeline.close()

    summary = {
        "profile": profile,
        "cold_warmup_seconds": round(cold_seconds, 4),
        "cuda": warmup["cuda"],
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{profile}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
