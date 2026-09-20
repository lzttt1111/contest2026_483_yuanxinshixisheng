from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import Pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="target-detection")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="run one local image")
    infer.add_argument("image", type=Path)
    serve = commands.add_parser("serve", help="start the local HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "infer":
        result = Pipeline().infer_bytes(args.image.read_bytes())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    import uvicorn

    uvicorn.run("target_detection.service:app", host=args.host, port=args.port, reload=False)
    return 0
