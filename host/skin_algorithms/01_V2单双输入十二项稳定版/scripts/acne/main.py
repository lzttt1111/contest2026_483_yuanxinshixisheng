"""Local single-image smoke entrypoint without Celery or OSS.

Usage:
  uv run python main.py <image_path> [algorithms...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acne.pipeline import Pipeline


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python main.py <图片路径> [algorithms...]")
        return 1
    result = Pipeline().process_single(sys.argv[1], sys.argv[2:] or None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
