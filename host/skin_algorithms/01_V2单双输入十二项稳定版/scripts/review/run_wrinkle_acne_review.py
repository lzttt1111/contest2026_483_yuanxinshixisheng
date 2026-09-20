from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from src.nine_analysis.metrics import extract_item
from src.nine_analysis.orchestrator import PROJECT_ROOT, ResidentClient
from src.nine_analysis.review_output import ReviewOutputExporter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(directory: Path) -> list[Path]:
    images = sorted(path.resolve() for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"未找到测试图片: {directory}")
    return images


def _service_items(responses: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    acne = responses["acne"]["result"]
    acne_files = acne.get("results") or {}
    wrinkle = responses["wrinkle"]["result"]
    wrinkle_files = wrinkle.get("results") or {}
    return {
        "acne": {
            "项目": "痤疮", "状态": "success",
            "主结果图": acne_files.get("acne_circles"),
            "量化JSON": acne_files.get("acne_summary") or acne.get("summary_path"),
            "量化CSV": acne_files.get("acne_detections"),
        },
        "wrinkle": {
            "项目": "皱纹", "状态": "success",
            "主结果图": wrinkle_files.get("final_result") or wrinkle_files.get("region_overlay"),
            "量化JSON": wrinkle_files.get("summary_json") or str(Path(wrinkle.get("output_dir", "")) / "summary.json"),
            "量化CSV": wrinkle_files.get("region_metrics_csv"),
        },
    }


def _update_indexes(target: Path, entries: dict[str, dict[str, Any]]) -> None:
    index_path = target / "九项检测结果索引.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    index["九项结果"].update(entries)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = target / "九项核心量化指标.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    for key in ("wrinkle", "acne"):
        entry = entries[key]
        raw = json.loads((target / entry["量化JSON"]).read_text(encoding="utf-8-sig"))
        summary, features = extract_item(key, raw)
        metrics_item = metrics["九项"][key]
        metrics_item.update(entry)
        metrics_item["核心总体指标"] = summary
        metrics_item["评分输入"] = features
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    ReviewOutputExporter.write_core_metrics_csv(metrics, target / "九项核心量化指标.csv")


def _update_timing(target: Path, responses: dict[str, dict[str, Any]], wall_seconds: float) -> None:
    json_path = target / "运行耗时.json"
    timing = json.loads(json_path.read_text(encoding="utf-8-sig"))
    timing["最近一次皱纹痤疮复测"] = {
        "并行墙钟时间秒": round(wall_seconds, 4),
        "服务": {
            key: {
                "状态": value.get("status"),
                "总耗时秒": value.get("seconds"),
                "GPU": value.get("gpu_after"),
            }
            for key, value in responses.items()
        },
    }
    json_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    with (target / "运行耗时.csv").open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        for key, value in responses.items():
            writer.writerow((f"{key}_refresh", value.get("status"), value.get("seconds"), value.get("gpu_after", {}).get("peak_allocated_mib", "")))
        writer.writerow(("wrinkle_acne_parallel_wall", "success", round(wall_seconds, 4), ""))


def _atomic_refresh(
    output_root: Path,
    image_stem: str,
    items: dict[str, dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    wall_seconds: float,
) -> Path:
    destination = output_root / image_stem
    if not destination.is_dir():
        raise FileNotFoundError(f"缺少已有九项验收目录，无法只更新两项: {destination}")
    temporary = output_root / f".{image_stem}.refresh-{uuid.uuid4().hex[:8]}"
    backup = output_root / f".{image_stem}.backup-{uuid.uuid4().hex[:8]}"
    shutil.copytree(destination, temporary)
    try:
        shutil.rmtree(temporary / "皱纹")
        shutil.rmtree(temporary / "痤疮")
        exporter = ReviewOutputExporter(output_root)
        entries = exporter.export_wrinkle_acne(items, temporary)
        _update_indexes(temporary, entries)
        _update_timing(temporary, responses, wall_seconds)
        os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            os.replace(backup, destination)
            raise
        shutil.rmtree(backup)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="只复测皱纹和痤疮并刷新人工验收报告")
    parser.add_argument("--input-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("output-test"))
    parser.add_argument("--cuda-device", default="0")
    args = parser.parse_args()
    images = collect_images(args.input_dir.resolve())
    output_root = args.output.resolve()
    runtime_root = output_root / ".runtime"
    run_root = runtime_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True)
    env = {
        "CUDA_VISIBLE_DEVICES": args.cuda_device,
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "DETECTOR_DEVICE": "cuda:0",
        "WRINKLE_DEVICE": "cuda:0",
    }
    clients: dict[str, ResidentClient] = {}
    all_success = False
    try:
        cold_started = time.perf_counter()
        for name in ("acne", "wrinkle"):
            clients[name] = ResidentClient(name, PROJECT_ROOT / "services" / name, log_dir, env)
        cold: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(client.wait_event, "ready", 600): name for name, client in clients.items()}
            for future in as_completed(futures):
                cold[futures[future]] = future.result()
        cold["并行冷启动墙钟时间秒"] = round(time.perf_counter() - cold_started, 4)
        (run_root / "cold_start.json").write_text(json.dumps(cold, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"冷启动": cold["并行冷启动墙钟时间秒"]}, ensure_ascii=False))

        for image in images:
            sample = run_root / image.stem
            input_dir = sample / "00_input"
            input_dir.mkdir(parents=True)
            copied = input_dir / image.name
            shutil.copy2(image, copied)
            started = time.perf_counter()
            responses: dict[str, dict[str, Any]] = {}
            request_id = f"{image.stem}-{uuid.uuid4().hex[:8]}"
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(client.analyze, request_id, copied, sample): name
                    for name, client in clients.items()
                }
                for future in as_completed(futures):
                    responses[futures[future]] = future.result()
            wall = time.perf_counter() - started
            if any(value.get("status") != "success" for value in responses.values()):
                raise RuntimeError(f"{image.name} 两项复测失败: {responses}")
            target = _atomic_refresh(output_root, image.stem, _service_items(responses), responses, wall)
            print(f"{image.name}: 皱纹+痤疮 success，并行墙钟 {wall:.4f}s -> {target}")
        all_success = True
        return 0
    finally:
        for client in clients.values():
            client.close()
        if all_success:
            shutil.rmtree(run_root, ignore_errors=True)
            if runtime_root.is_dir() and not any(runtime_root.iterdir()):
                runtime_root.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
