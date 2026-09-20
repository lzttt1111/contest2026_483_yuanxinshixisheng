from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.capture_profile import CaptureProfile

from .json_utils import json_safe
from .metrics import extract_item
from .resident_worker import PROTOCOL_PREFIX
from .service_paths import SERVICE_NAMES, service_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NINE_ITEMS = {
    "redness": "红区",
    "spots": "可见斑点",
    "brown": "棕区",
    "texture": "纹理",
    "pores": "毛孔",
    "uv_spots": "紫外线色斑",
    "porphyrin": "紫质",
    "wrinkle": "皱纹",
    "acne": "痤疮",
}
ADDED_ITEMS = {
    "surface_gloss": "油光",
    "vascular": "血管样结构",
    "contour_firmness": "轮廓紧致度",
}
TWELVE_ITEMS = {**NINE_ITEMS, **ADDED_ITEMS}


@dataclass(frozen=True, slots=True)
class InputStagingError(Exception):
    source: Path
    attempts: int

    def __str__(self) -> str:
        return f"NFS输入复制不完整: {self.source}，已重试{self.attempts}次"


def runtime_python() -> Path:
    """Use the checkout venv, shared dev venv, or the active launcher Python."""
    candidates = (
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT.parent.parent / ".venv" / "bin" / "python",
        Path(sys.executable).absolute(),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"九项统一Python不存在: {checked}；请使用已安装依赖的Python启动"
    )


class ResidentClient:
    def __init__(self, name: str, service_root: Path, log_dir: Path, env: dict[str, str]) -> None:
        self.name = name
        self.service_root = service_root
        self.log_path = log_dir / f"{name}.log"
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        python = runtime_python()
        command = [
            str(python), "-u", str(PROJECT_ROOT / "src" / "nine_analysis" / "resident_worker.py"),
            "--service", name,
            "--service-root", str(service_root),
        ]
        child_env = os.environ.copy()
        child_env.update(env)
        child_env["PYTHONPATH"] = str(PROJECT_ROOT)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
            cwd=service_root,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._log_handle.write(line)
            self._log_handle.flush()
            if line.startswith(PROTOCOL_PREFIX):
                try:
                    self.responses.put(json.loads(line[len(PROTOCOL_PREFIX):]))
                except json.JSONDecodeError:
                    continue

    def wait_event(self, expected: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None and self.responses.empty():
                raise RuntimeError(f"{self.name} 已退出，日志: {self.log_path}")
            try:
                payload = self.responses.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if payload.get("event") == "fatal":
                raise RuntimeError(f"{self.name} 启动失败: {payload.get('message')}")
            if payload.get("event") == expected:
                return payload
        raise TimeoutError(f"等待 {self.name}:{expected} 超时，日志: {self.log_path}")

    def analyze(
        self,
        request_id: str,
        image_path: Path,
        output_root: Path,
        algorithms: list[str],
        timeout: float = 1800,
    ) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({
            "command": "analyze",
            "id": request_id,
            "image_path": str(image_path),
            "output_root": str(output_root),
            "algorithms": algorithms,
        }, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        return self.wait_event("result", timeout)

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps({"command": "close", "id": "close"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except Exception:
                self.process.terminate()
        self._log_handle.close()


class NineAnalysisOrchestrator:
    def __init__(
        self,
        output_root: Path,
        cuda_device: str = "0",
        *,
        stage_input: bool = True,
        algorithms: tuple[str, ...] | None = None,
        capture_profile: CaptureProfile = CaptureProfile.INSTITUTION,
    ) -> None:
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cuda_device = cuda_device
        # 人工验收模式默认保留输入快照；NFS 大批量模式可关闭，令三个
        # 常驻服务直接读取只读挂载路径，避免把源图片复制到本地输出盘。
        self.stage_input = stage_input
        self.algorithms = algorithms or tuple(TWELVE_ITEMS)
        self.capture_profile = capture_profile
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.run_root = self.output_root / self.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.run_root / "logs"
        self.clients: dict[str, ResidentClient] = {}
        self.cold_start: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        env = {
            "CUDA_VISIBLE_DEVICES": self.cuda_device,
            "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            "DERMAVISION_DEVICE": "cuda:0",
            "DETECTOR_DEVICE": "cuda:0",
            "WRINKLE_DEVICE": "cuda:0",
            "DERMAVISION_CAPTURE_PROFILE": self.capture_profile.value,
        }
        started = time.perf_counter()
        requested_services = {
            *(('dermavision',) if set(self.algorithms) - {"acne", "wrinkle"} else ()),
            *(('acne',) if "acne" in self.algorithms else ()),
            *(('wrinkle',) if "wrinkle" in self.algorithms else ()),
        }
        for name in SERVICE_NAMES:
            if name not in requested_services:
                continue
            self.clients[name] = ResidentClient(name, service_root(name), self.log_dir, env)
        with ThreadPoolExecutor(max_workers=max(len(self.clients), 1)) as pool:
            futures = {pool.submit(client.wait_event, "ready", 600): name for name, client in self.clients.items()}
            for future in as_completed(futures):
                name = futures[future]
                self.cold_start[name] = future.result()
        self.cold_start["orchestrator_wall_seconds"] = round(time.perf_counter() - started, 4)
        (self.run_root / "cold_start.json").write_text(
            json.dumps(json_safe(self.cold_start), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.cold_start

    @staticmethod
    def _copy_input(image_path: Path, sample_root: Path) -> Path:
        target_dir = sample_root / "00_input"
        target_dir.mkdir(parents=True, exist_ok=True)
        temporary = target_dir / (image_path.name + ".part")
        expected_size = image_path.stat().st_size
        attempts = 3
        for _attempt in range(attempts):
            temporary.unlink(missing_ok=True)
            try:
                shutil.copyfile(image_path, temporary)
                if temporary.stat().st_size != expected_size:
                    continue
                with temporary.open("rb+") as handle:
                    header = handle.read(16)
                    if header.startswith(b"\xff\xd8"):
                        detected_suffix = ".jpg"
                        handle.seek(-2, os.SEEK_END)
                        if handle.read(2) != b"\xff\xd9":
                            handle.seek(0, os.SEEK_END)
                            handle.write(b"\xff\xd9")
                    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
                        detected_suffix = ".png"
                    elif header.startswith(b"BM"):
                        detected_suffix = ".bmp"
                    elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                        detected_suffix = ".webp"
                    else:
                        detected_suffix = image_path.suffix.lower()
                target = target_dir / f"{image_path.stem}{detected_suffix}"
                os.replace(temporary, target)
                return target
            except OSError:
                continue
        temporary.unlink(missing_ok=True)
        raise InputStagingError(image_path, attempts)

    @staticmethod
    def _load_json(path: str | None) -> Any:
        if not path:
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except Exception:
            return None

    def _canonical_items(self, responses: dict[str, Any]) -> dict[str, dict[str, Any]]:
        items = {key: {"项目": TWELVE_ITEMS[key], "状态": "failed"} for key in self.algorithms}
        derma = responses.get("dermavision", {})
        derma_result = (
            derma.get("result", {}).get("results", {})
            if derma.get("status") in {"success", "partial_success"}
            else {}
        )
        mapping = {
            "redness": ("redness", "redness_metrics", "redness_report"),
            "spots": ("spots", "spots_metrics", "spots_report"),
            "brown": ("brown", "brown_metrics", "brown_report"),
            "texture": ("texture", "texture_metrics", "texture_report"),
            "pores": ("pores", "pores_metrics", "pores_report"),
            "uv_spots": ("purple_uv_spots_overlay", "purple_metrics", "purple_report"),
            "porphyrin": ("purple_porphyrin_overlay", "purple_metrics", "purple_report"),
            "surface_gloss": ("surface_gloss", "surface_gloss_metrics", "surface_gloss_report"),
            "vascular": ("vascular", "vascular_metrics", "vascular_report"),
            "contour_firmness": ("contour_firmness", "contour_firmness_metrics", "contour_firmness_report"),
        }
        full_metrics_mapping = {
            "uv_spots": "purple_full_metrics",
            "porphyrin": "purple_full_metrics",
            "surface_gloss": "surface_gloss_full_metrics",
            "vascular": "vascular_full_metrics",
            "contour_firmness": "contour_firmness_full_metrics",
        }
        for key, (overlay_key, metrics_key, report_key) in mapping.items():
            if key not in items:
                continue
            overlay = derma_result.get(overlay_key)
            if overlay and Path(overlay).is_file():
                items[key].update({
                    "状态": "success",
                    "主结果图": overlay,
                    "量化JSON": derma_result.get(metrics_key),
                    "量化CSV": derma_result.get(report_key),
                })
                full_metrics = derma_result.get(full_metrics_mapping.get(key, ""))
                if full_metrics and Path(full_metrics).is_file():
                    items[key]["完整量化JSON"] = full_metrics
        acne = responses.get("acne", {})
        acne_result = acne.get("result", {})
        if "acne" in items and acne.get("status") == "success":
            files = acne_result.get("results", {})
            items["acne"].update({
                "状态": "success",
                "主结果图": files.get("acne_circles"),
                "量化JSON": files.get("acne_summary") or acne_result.get("summary_path"),
                "量化CSV": files.get("acne_detections"),
            })
        wrinkle = responses.get("wrinkle", {})
        wrinkle_result = wrinkle.get("result", {})
        if "wrinkle" in items and wrinkle.get("status") == "success":
            files = wrinkle_result.get("results", {})
            summary_path = files.get("summary_json") or str(
                Path(wrinkle_result.get("output_dir", "")) / "summary.json"
            )
            summary = self._load_json(summary_path) or {}
            grouped = summary.get("report_group_overlays") or {}
            report_images = [
                grouped[module_id]["path"]
                for module_id in ("07", "08", "09")
                if isinstance(grouped.get(module_id), dict)
                and Path(str(grouped[module_id].get("path", ""))).is_file()
            ]
            if len(report_images) == 3:
                items["wrinkle"].update({
                    "状态": "success",
                    "主结果图": report_images[0],
                    "附加结果图": report_images[1:],
                    "量化JSON": summary_path,
                    "量化CSV": files.get("region_metrics_csv"),
                })
        return items

    def _write_indexes(self, sample_root: Path, image_path: Path, responses: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
        items = self._canonical_items(responses)
        succeeded = sum(item["状态"] == "success" for item in items.values())
        total = len(items)
        status = "success" if succeeded == total else ("partial_success" if succeeded else "failed")
        manifest = {
            "运行ID": self.run_id,
            "输入图片": str(image_path),
            "状态": status,
            "成功项目数": succeeded,
            "项目总数": total,
            "九项结果": {key: value for key, value in items.items() if key in NINE_ITEMS},
            "十二项结果": items,
            "服务原始响应": responses,
            "评分状态": "uncalibrated",
            "采集Profile": self.capture_profile.value,
        }
        (sample_root / "manifest.json").write_text(
            json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        compact_metrics: dict[str, Any] = {}
        scoring_features: dict[str, Any] = {}
        for key, item in items.items():
            summary, features = extract_item(key, self._load_json(item.get("量化JSON")))
            compact_metrics[key] = {**item, "核心总体指标": summary, "评分输入": features}
            scoring_features[key] = features
        metrics = {
            "指标版本": "nine_metrics_compact_v2",
            "评分状态": "uncalibrated",
            "说明": "当前保存九项原始量化结果；参考人群分布生成前不输出正式评分。",
            "九项": compact_metrics,
        }
        (sample_root / "nine_metrics.json").write_text(
            json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (sample_root / "scoring_features.json").write_text(
            json.dumps(json_safe({
                "指标版本": "scoring_features_v1",
                "评分状态": "uncalibrated",
                "九项评分输入": scoring_features,
            }), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (sample_root / "nine_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("检测项目", "状态", "主结果图", "量化JSON", "量化CSV"))
            writer.writeheader()
            for item in items.values():
                writer.writerow({
                    "检测项目": item["项目"],
                    "状态": item["状态"],
                    "主结果图": item.get("主结果图", ""),
                    "量化JSON": item.get("量化JSON", ""),
                    "量化CSV": item.get("量化CSV", ""),
                })
        timing = {
            "运行ID": self.run_id,
            "图片": str(image_path),
            "并行服务墙钟时间秒": round(wall_seconds, 4),
            "服务": {
                key: {
                    "状态": value.get("status"),
                    "总耗时秒": value.get("seconds"),
                    "算法耗时": value.get("result", {}).get("metadata", {}).get("timing_seconds"),
                    "GPU": value.get("gpu_after"),
                }
                for key, value in responses.items()
            },
        }
        (sample_root / "timing.json").write_text(
            json.dumps(json_safe(timing), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (sample_root / "timing.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("阶段", "状态", "耗时秒", "峰值显存MiB"))
            for key, value in responses.items():
                writer.writerow((key, value.get("status"), value.get("seconds"), value.get("gpu_after", {}).get("peak_allocated_mib", "")))
            writer.writerow(("parallel_wall", status, round(wall_seconds, 4), ""))
        return manifest

    def analyze(
        self,
        image_path: Path,
        repetition: int = 1,
        sample_id: str | None = None,
        timeout_seconds: float = 1800,
    ) -> dict[str, Any]:
        image_path = image_path.resolve()
        sample_name = sample_id or image_path.stem
        if not sample_name or sample_name in {".", ".."} or Path(sample_name).name != sample_name:
            raise ValueError(f"非法 sample_id: {sample_name!r}")
        sample_root = self.run_root / f"repeat_{repetition}" / sample_name
        sample_root.mkdir(parents=True, exist_ok=True)
        analysis_input = (
            self._copy_input(image_path, sample_root)
            if self.stage_input
            else image_path
        )
        request_id = f"{sample_name}-{repetition}-{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        responses: dict[str, Any] = {}
        derma_algorithms = list(dict.fromkeys(
            "purple" if name in {"uv_spots", "porphyrin"} else name
            for name in self.algorithms
            if name not in {"acne", "wrinkle"}
        ))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(
                    client.analyze,
                    request_id,
                    analysis_input,
                    sample_root,
                    (
                        derma_algorithms
                        if name == "dermavision"
                        else [name]
                    ),
                    timeout_seconds,
                ): name
                for name, client in self.clients.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    responses[name] = future.result()
                except Exception as exc:
                    responses[name] = {"service": name, "status": "failed", "message": f"{type(exc).__name__}: {exc}"}
        return self._write_indexes(
            sample_root,
            analysis_input,
            responses,
            time.perf_counter() - started,
        )

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
