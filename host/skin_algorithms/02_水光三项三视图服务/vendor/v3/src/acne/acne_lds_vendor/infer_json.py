#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acne-LDS single image JSON inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-type", default="model_ld_smoothing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from predict_on_img import ModelInit

        if args.device != "cpu":
            raise ValueError("Only CPU is supported in this project")
        checkpoint = Path(args.checkpoint)
        image = Image.open(args.image).convert("RGB")
        model = ModelInit(model_type=args.model_type, path_checkpoint=str(checkpoint), device="cpu")
        cls, cou, cou2cls = model.predict_on_img(image)
        probs = (0.5 * (cls + cou2cls)).detach().cpu()[0]
        probs = probs / probs.sum()
        severity_level = int(torch.argmax(probs).item() + 1)
        predicted_count = float(torch.argmax(cou.detach().cpu()[0]).item() + 1)
        payload = {
            "status": "ok",
            "predicted_count": predicted_count,
            "severity_level": severity_level,
            "severity_probabilities": [float(x) for x in probs.tolist()],
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "model": "acne-lds",
            "model_type": args.model_type,
            "device": "cpu",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {
            "status": "unavailable",
            "reason": type(exc).__name__,
            "message": str(exc),
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "model": "acne-lds",
            "device": "cpu",
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
