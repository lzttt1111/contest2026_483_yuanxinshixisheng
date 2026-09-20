from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import cv2
import mediapipe
import numpy
import torch
import torchvision


EXPECTED = {
    "python": "3.10.20",
    "torch": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "cuda_runtime": "12.8",
    "numpy": "1.26.4",
    "mediapipe": "0.10.14",
    "opencv-contrib-python": "4.11.0.86",
    "opencv-python": "4.11.0.86",
    "ultralytics": "8.4.90",
}


def main() -> int:
    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": numpy.__version__,
        "mediapipe": mediapipe.__version__,
        "opencv-contrib-python": version("opencv-contrib-python"),
        "opencv-python": version("opencv-python"),
        "ultralytics": version("ultralytics"),
    }
    errors = [f"{key}: expected={value}, actual={actual.get(key)}" for key, value in EXPECTED.items() if actual.get(key) != value]
    root_python = (Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != root_python:
        errors.append(f"未使用根环境: {sys.executable}")
    if not torch.cuda.is_available():
        errors.append("CUDA不可用")
    if not hasattr(cv2, "ximgproc"):
        errors.append("OpenCV ximgproc不可用")
    result = {
        "状态": "passed" if not errors else "failed",
        "Python可执行文件": sys.executable,
        "实际版本": actual,
        "CUDA可用": torch.cuda.is_available(),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "计算能力": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "OpenCV_ximgproc": hasattr(cv2, "ximgproc"),
        "错误": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
