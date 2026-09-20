from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_bridge.hybrid_profile import (  # noqa: E402
    verify_hybrid_profile_document,
)
from src.scoring_bridge.hybrid_promotion import (  # noqa: E402
    ScoringApprovalRecord,
    promote_hybrid_profile,
)


def _candidate(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("hybrid candidate root must be an object")
    verify_hybrid_profile_document(value)
    return value


def _approvals(path: Path) -> tuple[ScoringApprovalRecord, ...]:
    return tuple(
        ScoringApprovalRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证批准记录并生成promoted hybrid profile；失败返回非零",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--approvals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidate = _candidate(args.candidate.resolve(strict=True))
    approvals = _approvals(args.approvals.resolve(strict=True))
    promoted = promote_hybrid_profile(candidate, approvals=approvals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(promoted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "promoted",
        "profile": args.output.name,
        "profile_sha256": promoted["profile_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
