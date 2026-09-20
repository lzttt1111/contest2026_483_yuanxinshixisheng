#!/usr/bin/env python3
"""用本地 OSS 替身执行11个 Worker任务，并生成验收网页。"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_contracts import (
    public_metric_documentation_rows,
    worker_field_documentation_rows,
)
from cloud.cloud_word_report import generate_cloud_review_reports
from cloud.review_sanitize import sanitize_html_response, sanitize_review_bundle
from src.nine_analysis.service_paths import service_root


HELPER = PROJECT_ROOT / "cloud" / "simulate_service_task.py"
SERVICES = ("dermavision", "acne", "wrinkle")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_generate_medical_reports = generate_cloud_review_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模拟单RGB十二项正式云端 Worker 请求")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--generate-medical-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认关闭；开启后用consumer十二项聚合出口生成两份Word",
    )
    parser.add_argument(
        "--parallel-services",
        action="store_true",
        help="并行启动三个来源服务；默认串行以控制本地验收峰值",
    )
    parser.add_argument(
        "--report-subject-id",
        default=None,
        help="Word受检者编号；仅在显式开启Word时使用",
    )
    return parser.parse_args()


def _service_environment(service: str, service_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["DERMAVISION_CAPTURE_PROFILE"] = "consumer"
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    if service == "acne":
        env["SERVICE_NAME"] = "acne-detection-worker"
    if service == "wrinkle":
        env["SERVICE_NAME"] = "wrinkle"
    return env


def _run_service(service: str, input_path: Path, output_root: Path) -> subprocess.Popen:
    root = service_root(service)
    response_path = output_root / "responses" / f"{service}.json"
    log_path = output_root / "logs" / f"{service}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(HELPER),
        "--service", service,
        "--input", str(input_path),
        "--oss-root", str(output_root / "simulated_oss"),
        "--response", str(response_path),
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=root,
        env=_service_environment(service, root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._cloud_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def _execute_services(
    input_path: Path,
    output_root: Path,
    *,
    parallel: bool,
) -> list[str]:
    """Run each source service once; sequential mode bounds peak local memory."""
    failures: list[str] = []
    if parallel:
        processes = {
            service: _run_service(service, input_path, output_root)
            for service in SERVICES
        }
        service_processes = processes.items()
    else:
        service_processes = (
            (service, _run_service(service, input_path, output_root))
            for service in SERVICES
        )
    for service, process in service_processes:
        return_code = process.wait()
        process._cloud_log_handle.close()  # type: ignore[attr-defined]
        if return_code != 0:
            failures.append(f"{service}: exit={return_code}")
    return failures


def _collect_payload(output_root: Path) -> dict:
    services: dict[str, dict] = {}
    tasks: dict[str, dict] = {}
    for service in SERVICES:
        payload = json.loads(
            (output_root / "responses" / f"{service}.json").read_text(encoding="utf-8")
        )
        services[service] = payload
        tasks.update(payload["tasks"])
    return {"services": services, "tasks": tasks}


def _resolve_oss(output_root: Path, value: object) -> Path | None:
    if not isinstance(value, str):
        return None
    candidate = output_root / "simulated_oss" / value
    return candidate if candidate.is_file() else None


def _web_path(index_dir: Path, path: Path) -> str:
    return os.path.relpath(path, index_dir).replace(os.sep, "/")


def _collect_display_files(
    value: object,
    output_root: Path,
    label: str = "display_result",
) -> list[tuple[str, Path]]:
    """递归提取 acne/wrinkle 的 debug_info.display_result OSS 文件。"""
    collected: list[tuple[str, Path]] = []
    if isinstance(value, str):
        path = _resolve_oss(output_root, value)
        if path is not None:
            collected.append((label, path))
        return collected
    if isinstance(value, list):
        for index, item in enumerate(value, start=1):
            collected.extend(
                _collect_display_files(item, output_root, f"{label}[{index}]")
            )
        return collected
    if not isinstance(value, dict):
        return collected

    # wrinkle 的固定展示项使用 oss_key，各分区使用 image_oss_key。
    direct_keys = ("oss_key", "image_oss_key")
    for key in direct_keys:
        if key in value:
            display_name = (
                value.get("region_name")
                or value.get("file_name")
                or label
            )
            collected.extend(
                _collect_display_files(value[key], output_root, str(display_name))
            )
            return collected
    for key, item in value.items():
        collected.extend(_collect_display_files(item, output_root, str(key)))
    return collected


def _render_file_gallery(
    files_with_labels: list[tuple[str, Path]],
    output_root: Path,
) -> tuple[str, str]:
    media: list[str] = []
    files: list[str] = []
    seen: set[Path] = set()
    for label_value, path in files_with_labels:
        if path in seen:
            continue
        seen.add(path)
        rel = html.escape(_web_path(output_root, path))
        label = html.escape(str(label_value))
        files.append(
            f'<li><a href="{rel}" target="_blank">{label}: '
            f'{html.escape(path.name)}</a></li>'
        )
        if path.suffix.lower() in IMAGE_SUFFIXES:
            media.append(
                f'<figure><a href="{rel}" target="_blank"><img src="{rel}" loading="lazy"></a>'
                f'<figcaption>{label}<br><small>{html.escape(path.name)}</small></figcaption></figure>'
            )
    return "".join(media), "".join(files)


def _bundle_counts(tasks: dict[str, object]) -> tuple[int, int]:
    task_count = len(tasks)
    return task_count, task_count + (1 if "purple" in tasks else 0)


def _render_task(name: str, item: dict, output_root: Path) -> str:
    response = sanitize_html_response(item["response"])
    raw = response.get("raw_result", {})
    raw_files: list[tuple[str, Path]] = []
    for key, value in raw.items():
        path = _resolve_oss(output_root, value)
        if path is None:
            continue
        raw_files.append((str(key), path))
    debug = response.get("debug_info", {})
    report_path = _resolve_oss(output_root, debug.get("report_csv"))
    if report_path:
        raw_files.append(("CSV量化报告", report_path))

    raw_media, raw_links = _render_file_gallery(raw_files, output_root)
    metrics_value = raw.get("metrics")
    if name == "acne":
        metrics_value = raw.get("量化结果")
    elif name == "wrinkle":
        metrics_value = raw.get("region_metrics")
    metrics_json = html.escape(
        json.dumps(metrics_value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    def render_documentation_rows(rows: list[dict]) -> str:
        rendered: list[str] = []
        for row in rows:
            value = row.get("value")
            if isinstance(value, (dict, list)):
                value_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if len(value_text) > 180:
                    value_text = value_text[:177] + "..."
            elif value is None:
                value_text = "null"
            else:
                value_text = str(value)
            rendered.append(
                "<tr>"
                f"<td><code>{html.escape(str(row['json_path']))}</code></td>"
                f"<td>{html.escape(str(row['title']))}</td>"
                f"<td><code>{html.escape(value_text)}</code></td>"
                f"<td>{html.escape(str(row['value_type']))}</td>"
                f"<td>{html.escape(str(row['unit']))}</td>"
                f"<td>{html.escape(str(row['description']))}</td>"
                f"<td>{'是' if row.get('required') else '否'}</td>"
                "</tr>"
            )
        return "".join(rendered)

    metric_documentation = render_documentation_rows(
        public_metric_documentation_rows(name, raw)
    )
    worker_documentation = render_documentation_rows(
        worker_field_documentation_rows(name, response)
    )
    response_json = html.escape(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
    return f"""
    <section class="task" id="{html.escape(name)}">
      <h2>{html.escape(name)}</h2>
      <div class="meta">Queue: <code>{html.escape(str(item['queue']))}</code> · Task:
      <code>{html.escape(str(item['task']))}</code> · 耗时 {item['elapsed_seconds']}s ·
      状态 <strong>{html.escape(str(response.get('status')))}</strong> ·
      Pydantic校验 <strong>{html.escape(str(item.get('pydantic_validation', 'unknown')))}</strong></div>
      <h3>前端量化指标</h3>
      <pre>{metrics_json}</pre>
      <details open><summary>查看前端量化指标逐项中文说明</summary>
      <table><thead><tr><th>JSON路径</th><th>中文名称</th><th>当前值</th><th>类型</th><th>单位</th><th>详细说明</th><th>必填</th></tr></thead>
      <tbody>{metric_documentation}</tbody></table></details>
      <h3>Worker raw_result 文件</h3>
      <div class="gallery raw-gallery">{raw_media or '<p>本任务无图片字段</p>'}</div>
      <ul>{raw_links or '<li>无</li>'}</ul>
      <details><summary>查看本次Worker实际返回字段说明</summary>
      <p class="hint">仅展示本次响应实际出现的字段；Pydantic内部模型类名不属于云端JSON字段。</p>
      <table><thead><tr><th>JSON路径</th><th>中文名称</th><th>当前值</th><th>类型</th><th>单位</th><th>详细说明</th><th>必填</th></tr></thead>
      <tbody>{worker_documentation}</tbody></table></details>
      <details><summary>查看完整六字段 Worker 返回</summary><pre>{response_json}</pre></details>
    </section>
    """


def _write_html(input_path: Path, output_root: Path, bundle: dict, elapsed: float) -> None:
    sections = "".join(
        _render_task(name, item, output_root) for name, item in bundle["tasks"].items()
    )
    task_count = int(bundle.get("task_count", len(bundle["tasks"])))
    result_count = int(bundle.get("projected_result_count", task_count))
    report_links = "".join(
        f' · <a href="{html.escape(name)}" target="_blank">{html.escape(name)}</a>'
        for name in bundle.get("medical_report_files", [])
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AISIA 单RGB{result_count}项云端返回模拟</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f4f7fb;color:#1f2937;margin:0}}
main{{max-width:1500px;margin:auto;padding:24px}} h1{{margin-bottom:8px}} .summary,.task{{background:white;border:1px solid #dbe3ee;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 2px 8px #0001}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}} figure{{margin:0}} img{{width:100%;max-height:520px;object-fit:contain;background:#111;border-radius:8px}} figcaption{{padding:8px;text-align:center;font-weight:600}}
code,pre{{font-family:Consolas,monospace}} pre{{white-space:pre-wrap;word-break:break-all;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;max-height:600px;overflow:auto}}
table{{width:100%;border-collapse:collapse;margin:12px 0}} th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left;vertical-align:top}} th{{background:#eaf3f8}}
.meta{{color:#475569;margin-bottom:16px}} .display-result{{border:2px solid #22a6b3;background:#f0fdff;padding:16px;border-radius:10px;margin:14px 0 22px}}
.display-result h3{{margin-top:0;color:#08788d}} .hint{{color:#475569}} .raw-gallery{{opacity:.88}}
a{{color:#087ea4}} details summary{{cursor:pointer;font-weight:700;padding:10px 0}}
</style></head><body><main>
<h1>AISIA 单RGB{result_count}项云端返回模拟</h1>
<div class="summary"><p>输入：<strong>{html.escape(input_path.name)}</strong></p>
<p>覆盖 {task_count} 个独立云端任务、{result_count} 项检测；总墙钟耗时：{elapsed:.3f}s。</p>
<p><a href="cloud_response_bundle.json" target="_blank">完整云端返回集合 JSON</a> ·
<a href="/docs" target="_blank">Pydantic Swagger接口文档</a> ·
<a href="00_输入图片{html.escape(input_path.suffix)}" target="_blank">查看输入图</a> ·
<a href="#acne">跳到痤疮结果</a> ·
<a href="#wrinkle">跳到皱纹结果</a>{report_links}</p></div>
{sections}
</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(input_path, staging / f"00_输入图片{input_path.suffix}")

    started = time.perf_counter()
    failures = _execute_services(
        input_path,
        staging,
        parallel=args.parallel_services,
    )
    if failures:
        raise RuntimeError("云端模拟服务失败: " + ", ".join(failures))

    elapsed = time.perf_counter() - started
    bundle = _collect_payload(staging)
    task_count, projected_result_count = _bundle_counts(bundle["tasks"])
    bundle["task_count"] = task_count
    bundle["projected_result_count"] = projected_result_count
    bundle["input"] = str(input_path)
    bundle["wall_clock_seconds"] = round(elapsed, 6)
    bundle["capture_profile"] = "consumer"
    bundle["simulation_note"] = "正式 Worker 与 Pipeline；仅 Redis/OSS 替换为本地文件存储"
    bundle["medical_report_requested"] = bool(args.generate_medical_report)
    reports = (
        _generate_medical_reports(input_path, staging, args.report_subject_id)
        if args.generate_medical_report
        else ()
    )
    bundle["medical_report_generated"] = bool(reports)
    bundle["medical_report_files"] = [path.name for path in reports]
    bundle["medical_report_note"] = (
        "单项Celery Worker不改六字段；两份Word由consumer全十二项验收聚合出口生成"
        if reports
        else "Word默认关闭；可用--generate-medical-report显式开启"
    )
    internal_bundle = staging / "logs/cloud_response_bundle.internal.json"
    internal_bundle.parent.mkdir(parents=True, exist_ok=True)
    internal_bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    public_bundle = sanitize_review_bundle(bundle)
    (staging / "cloud_response_bundle.json").write_text(
        json.dumps(public_bundle, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_html(input_path, staging, public_bundle, elapsed)

    if output_root.exists():
        shutil.rmtree(output_root)
    staging.rename(output_root)
    print(json.dumps({
        "status": "success",
        "output": str(output_root),
        "html": str(output_root / "index.html"),
        "tasks": list(public_bundle["tasks"]),
        "medical_report_files": public_bundle["medical_report_files"],
        "wall_clock_seconds": round(elapsed, 6),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
