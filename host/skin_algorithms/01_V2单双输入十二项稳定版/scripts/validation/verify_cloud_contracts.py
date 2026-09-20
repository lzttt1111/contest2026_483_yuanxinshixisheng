#!/usr/bin/env python3
"""验证旧云端合同未漂移，并显式登记允许的医学 V2 加法文件。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "cloud" / "contracts" / "internal_dev_contracts.json"
LEGACY_PACKAGING_ROOT = (
    PROJECT_ROOT / "归档文件夹" / "主目录旧文档与工具_20260805" / "旧部署" / "嵌套项目"
)


def contract_path(service: str, relative: str) -> Path:
    """将三个原项目合同快照映射到统一仓库中的实际位置。"""
    if service == "dermavision":
        return PROJECT_ROOT / relative
    if relative == "AGENTS.md":
        return PROJECT_ROOT / "src" / service / "AGENTS.md"
    if relative.startswith("src/"):
        return PROJECT_ROOT / "src" / service / relative.removeprefix("src/")
    return LEGACY_PACKAGING_ROOT / service / relative


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_arguments(worker_path: Path) -> list[str]:
    tree = ast.parse(worker_path.read_text(encoding="utf-8"), filename=str(worker_path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "analyze_image":
            names = [argument.arg for argument in node.args.args]
            if names and names[0] == "self":
                names = names[1:]
            return names
    raise ValueError(f"未找到 analyze_image: {worker_path}")


def verify(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for service, spec in manifest["source_repositories"].items():
        # 三类 Worker 已归并到根 src 包，但仍按各自 AGENTS 合同校验。
        additive_extension_files = set(spec.get("additive_extension_files", []))
        for relative, expected in spec["sha256"].items():
            path = contract_path(service, relative)
            if not path.is_file():
                errors.append(f"{service}: 缺少 {relative}")
                continue
            actual = sha256_file(path)
            if actual != expected:
                if relative in additive_extension_files:
                    continue
                errors.append(
                    f"{service}: {relative} 已偏离已登记合同快照 "
                    f"expected={expected} actual={actual}"
                )
        unknown_extensions = additive_extension_files - set(spec["sha256"])
        if unknown_extensions:
            errors.append(
                f"{service}: additive_extension_files 含未登记基线文件 "
                f"{sorted(unknown_extensions)}"
            )
        worker_path = {
            "dermavision": PROJECT_ROOT / "src" / "worker.py",
            "acne": PROJECT_ROOT / "src" / "acne" / "worker.py",
            "wrinkle": PROJECT_ROOT / "src" / "wrinkle" / "worker.py",
        }[service]
        if worker_path.is_file():
            actual_arguments = analyze_arguments(worker_path)
            expected_arguments = spec["arguments"]
            if actual_arguments != expected_arguments:
                errors.append(
                    f"{service}: analyze_image 参数变化 "
                    f"expected={expected_arguments} actual={actual_arguments}"
                )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验三套Worker合同快照")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify(args.manifest.resolve())
    if errors:
        print("云端契约校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(f"云端契约校验通过: {manifest['contract_version']}")
    for service, spec in manifest["source_repositories"].items():
        print(f"- {service}: contract snapshot verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
