"""ONNX Runtime wrapper for FSA-Net head-pose estimation."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import onnxruntime as ort

from .preprocess import NormalizedCrop, preprocess_bgr, read_image_bgr

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "miniprogram_demo"
    / "models"
    / "head_pose_fsanet_1x1.onnx"
)


@dataclass(frozen=True)
class PoseResult:
    yaw: float
    pitch: float
    roll: float
    valid: bool
    timestamp: int

    def to_dict(self) -> Dict[str, Union[float, bool, int]]:
        return asdict(self)


class HeadPoseEstimator:
    """Load the fixed-shape model once and reuse it for multiple images."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        yaw_sign: float = 1.0,
        intra_op_threads: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        if yaw_sign not in (-1, 1, -1.0, 1.0):
            raise ValueError("yaw_sign must be 1 or -1")
        self.yaw_sign = float(yaw_sign)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(intra_op_threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._validate_contract()

    def _validate_contract(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "input":
            raise ValueError("model must expose exactly one input named 'input'")
        if list(inputs[0].shape) != [1, 3, 64, 64]:
            raise ValueError(f"unexpected input shape: {inputs[0].shape}")
        if inputs[0].type != "tensor(float)":
            raise ValueError(f"unexpected input type: {inputs[0].type}")
        if len(outputs) != 1 or outputs[0].name != "output":
            raise ValueError("model must expose exactly one output named 'output'")
        if list(outputs[0].shape) != [1, 3]:
            raise ValueError(f"unexpected output shape: {outputs[0].shape}")
        if outputs[0].type != "tensor(float)":
            raise ValueError(f"unexpected output type: {outputs[0].type}")

    def infer_bgr(
        self,
        image_bgr: np.ndarray,
        crop: Optional[NormalizedCrop] = None,
        center_square: bool = False,
    ) -> PoseResult:
        tensor = preprocess_bgr(
            image_bgr=image_bgr, crop=crop, center_square=center_square
        )
        output = self.session.run(["output"], {"input": tensor})[0]
        if output.shape != (1, 3):
            raise RuntimeError(f"runtime returned unexpected output shape: {output.shape}")

        raw_yaw, pitch, roll = (float(value) for value in output[0])
        yaw = raw_yaw * self.yaw_sign
        angles = (yaw, pitch, roll)
        valid = all(math.isfinite(value) and abs(value) <= 99.0 for value in angles)
        return PoseResult(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            valid=valid,
            timestamp=int(time.time() * 1000),
        )

    def infer_path(
        self,
        image_path: Union[str, Path],
        crop: Optional[NormalizedCrop] = None,
        center_square: bool = False,
    ) -> PoseResult:
        return self.infer_bgr(
            read_image_bgr(image_path), crop=crop, center_square=center_square
        )

