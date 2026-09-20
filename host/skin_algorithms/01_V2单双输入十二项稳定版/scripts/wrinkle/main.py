"""本地测试入口（不经过 Celery / OSS）。

用法:
  uv run python main.py <图片路径> [algorithms...]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wrinkle.pipeline import Pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <图片路径> [algorithms...]")
        sys.exit(1)

    image_path = sys.argv[1]
    algorithms = sys.argv[2:] or None

    result = Pipeline().process_single(image_path, algorithms)
    print(f"\n返回结果: {result}")
