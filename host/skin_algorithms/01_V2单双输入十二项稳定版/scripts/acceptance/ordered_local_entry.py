#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly:
#      uv run scripts/acceptance/ordered_local_entry.py --help
# 3. Or make executable and run:
#      chmod +x scripts/acceptance/ordered_local_entry.py && ./scripts/acceptance/ordered_local_entry.py --help
# ─────────────────

"""Run merged run.py with manifest-pinned 25,09,23 iteration order."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

TOOL_ROOT = Path(__file__).resolve().parents[2]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from scripts.acceptance.acceptance_io import load_fixed_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    return parser.parse_args()


def _load_run(project_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("acceptance_target_run", project_root / "run.py")
    if spec is None or spec.loader is None:
        raise ImportError("unable to load target run.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    loaded = False
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        loaded = True
    finally:
        if not loaded:
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
    return module


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    inputs = load_fixed_inputs(args.inputs_manifest.resolve())
    sys.path.insert(0, str(project_root))
    target = _load_run(project_root)

    def ordered_images(*_positional, **_keywords):
        return iter(asset.path for asset in inputs)

    target.iter_images = ordered_images
    return int(
        target.main(
            [
                "--input-dir",
                str(args.inputs_manifest.resolve().parent),
                "--output-dir",
                str(args.output.resolve()),
                "--runtime-dir",
                str(args.runtime.resolve()),
                "--output-profile",
                "review",
                "--algorithms",
                "all",
                "--precount",
                "--no-resume",
                "--stop-on-error",
                "--images-per-group",
                "3",
                "--worker-id",
                "final-acceptance",
                "--job-name",
                "final-acceptance",
                "--generate-medical-report",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
