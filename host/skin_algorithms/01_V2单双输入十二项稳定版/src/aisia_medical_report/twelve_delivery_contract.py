from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Final


EXPECTED_ITEM_IDS: Final = frozenset({
    "redness", "spots", "brown", "texture", "pores", "uv_spots",
    "porphyrin", "wrinkle", "acne", "surface_gloss", "vascular",
    "contour_firmness",
})
EXPECTED_MEDIA_COUNTS: Final = {
    "redness": 2,
    "spots": 1,
    "brown": 2,
    "texture": 1,
    "pores": 1,
    "uv_spots": 2,
    "porphyrin": 2,
    "wrinkle": 3,
    "acne": 1,
    "surface_gloss": 1,
    "vascular": 1,
    "contour_firmness": 1,
}


class WordDeliveryContractError(RuntimeError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(result_root: Path, relative: str) -> Path:
    candidate = result_root / relative
    if candidate.is_symlink():
        raise WordDeliveryContractError(f"十二项产物不允许使用符号链接: {relative}")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(result_root.resolve()):
        raise WordDeliveryContractError(f"十二项产物越界: {relative}")
    return path


def validate_index_media(index: dict[str, Any], result_root: Path) -> None:
    items = index.get("items") or index.get("十二项结果")
    consumer_index = (
        index.get("schema_version") == "single_rgb_twelve_index_v3"
        and index.get("status") == "success"
    )
    institution_index = (
        index.get("路线") == "clinic_four_light"
        and index.get("状态") == "success"
        and index.get("成功项目数") == 12
        and index.get("项目总数") == 12
    )
    if (
        not (consumer_index or institution_index)
        or not isinstance(items, dict)
        or set(items) != EXPECTED_ITEM_IDS
    ):
        raise WordDeliveryContractError("正式双版只接受十二项全部成功的公开结果")

    indexed_roles_by_hash: dict[str, list[tuple[str, int]]] = {}
    for item_id in sorted(EXPECTED_ITEM_IDS):
        item = items[item_id]
        if not isinstance(item, dict) or item.get("状态") != "success":
            raise WordDeliveryContractError(f"十二项索引项状态不可用: {item_id}")
        main = item.get("主结果图")
        extras = item.get("附加结果图") or []
        images = item.get("images")
        if (
            not isinstance(main, str)
            or not isinstance(extras, list)
            or not all(isinstance(value, str) for value in extras)
            or not isinstance(images, list)
        ):
            raise WordDeliveryContractError(f"十二项索引媒体结构无效: {item_id}")

        indexed_paths: list[str] = []
        indexed_hashes: list[str] = []
        for image in images:
            if not isinstance(image, dict):
                raise WordDeliveryContractError(f"十二项索引媒体条目无效: {item_id}")
            path = image.get("path")
            expected_sha256 = image.get("sha256")
            if (
                not isinstance(path, str)
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
            ):
                raise WordDeliveryContractError(f"十二项索引媒体SHA256缺失或无效: {item_id}")
            indexed_paths.append(path)
            indexed_hashes.append(expected_sha256)

        declared_paths = [main, *extras]
        if (
            len(declared_paths) != EXPECTED_MEDIA_COUNTS[item_id]
            or indexed_paths != declared_paths
            or len(set(indexed_paths)) != len(indexed_paths)
        ):
            raise WordDeliveryContractError(f"十二项索引媒体映射不一致: {item_id}")
        for media_index, (relative, expected_sha256) in enumerate(
            zip(indexed_paths, indexed_hashes)
        ):
            if sha256(resolve_artifact(result_root, relative)) != expected_sha256:
                raise WordDeliveryContractError(
                    f"十二项索引媒体SHA256不匹配: {item_id} {relative}"
                )
            indexed_roles_by_hash.setdefault(expected_sha256, []).append(
                (item_id, media_index)
            )

    shared_365_roles = {("uv_spots", 1), ("porphyrin", 1)}
    duplicate_groups = [
        set(roles)
        for roles in indexed_roles_by_hash.values()
        if len(roles) > 1
    ]
    if any(roles != shared_365_roles for roles in duplicate_groups):
        raise WordDeliveryContractError("十二项索引媒体SHA256存在非授权重复")


def require_fresh_report_root(report_root: Path) -> None:
    if report_root.exists():
        raise WordDeliveryContractError(
            f"正式报告目录已存在，拒绝保留或覆盖旧DOCX: {report_root}"
        )


def portable_payload(payload: dict[str, Any], report_root: Path) -> dict[str, Any]:
    output = copy.deepcopy(payload)

    def convert(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"path", "结果图"}:
                    if isinstance(child, str) and Path(child).is_absolute():
                        value[key] = Path(child).relative_to(report_root).as_posix()
                    elif isinstance(child, list):
                        value[key] = [
                            Path(item).relative_to(report_root).as_posix()
                            if isinstance(item, str) and Path(item).is_absolute()
                            else item
                            for item in child
                        ]
                else:
                    convert(child)
            return
        if isinstance(value, list):
            for child in value:
                convert(child)

    convert(output)
    return output


__all__ = [
    "EXPECTED_ITEM_IDS",
    "EXPECTED_MEDIA_COUNTS",
    "WordDeliveryContractError",
    "portable_payload",
    "require_fresh_report_root",
    "resolve_artifact",
    "sha256",
    "validate_index_media",
]
