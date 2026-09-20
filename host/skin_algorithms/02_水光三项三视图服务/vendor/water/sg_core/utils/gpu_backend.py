"""Shared CUDA primitives for the five DermaVision analysis engines.

Only dense numeric kernels live here.  MediaPipe, connected components,
watershed, peak extraction and report generation deliberately remain on CPU.
The production default is strict CUDA: a missing GPU is an explicit error and
never changes the algorithm silently to a CPU implementation.
"""
from __future__ import annotations
import math
import os
import threading
import time
from dataclasses import dataclass, field
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def _truthy(value: str) -> bool:
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}

@dataclass
class GPUProfile:
    execution_profile: str
    device: str
    torch_version: str
    cuda_version: str
    gpu_name: str
    capability: tuple[int, int]
    memory_fraction: float
    kernel_seconds: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_timing(self, seconds: float) -> None:
        with self._lock:
            self.kernel_seconds += float(seconds)
            self.calls += 1

    def as_dict(self) -> dict[str, object]:
        return {'device': self.device, 'execution_profile': self.execution_profile, 'torch_version': self.torch_version, 'cuda_version': self.cuda_version, 'gpu_name': self.gpu_name, 'compute_capability': list(self.capability), 'memory_fraction': self.memory_fraction, 'gpu_kernel_seconds': round(self.kernel_seconds, 6), 'gpu_kernel_calls': self.calls}

class CUDABackend:

    def __init__(self) -> None:
        requested = os.getenv('SHUIGUANG_DEVICE', 'cpu').strip()
        if requested == 'cpu':
            self.device = torch.device('cpu')
            self.execution_profile = 'compat'
            self.profile = GPUProfile('compat', 'cpu', torch.__version__, '', 'CPU', (0, 0), 0.0)
            self._kernel_cache = {}
            self._lock = threading.Lock()
            return
        required = _truthy(os.getenv('DERMAVISION_GPU_REQUIRED', 'true'))
        if not requested.startswith('cuda'):
            if required:
                raise RuntimeError('DermaVision production mode requires DERMAVISION_DEVICE=cuda:0')
            raise RuntimeError('CPU backend is not provided by gpu_backend')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required but unavailable; CPU fallback is intentionally disabled')
        self.device = torch.device(requested)
        self.execution_profile = os.getenv('DERMAVISION_GPU_PROFILE', 'compat').strip().lower()
        if self.execution_profile not in {'compat', 'fast'}:
            raise RuntimeError("DERMAVISION_GPU_PROFILE must be 'compat' or 'fast'")
        index = self.device.index if self.device.index is not None else 0
        torch.cuda.set_device(index)
        fraction = float(os.getenv('DERMAVISION_GPU_MEMORY_FRACTION', '0.18'))
        fraction = float(np.clip(fraction, 0.05, 0.95))
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.profile = GPUProfile(execution_profile=self.execution_profile, device=str(self.device), torch_version=torch.__version__, cuda_version=str(torch.version.cuda), gpu_name=torch.cuda.get_device_name(index), capability=tuple(torch.cuda.get_device_capability(index)), memory_fraction=fraction)
        self._kernel_cache: dict[float, torch.Tensor] = {}
        self._lock = threading.Lock()

    def begin_task_profile(self) -> tuple[float, int]:
        """Start a process-local task timing window without changing results."""
        torch.cuda.reset_peak_memory_stats(self.device)
        return (self.profile.kernel_seconds, self.profile.calls)

    def finish_task_profile(self, baseline: tuple[float, int]) -> dict[str, object]:
        """Return CUDA work and peak memory consumed since ``baseline``."""
        (start_seconds, start_calls) = baseline
        index = self.device.index if self.device.index is not None else 0
        return {**self.profile.as_dict(), 'task_gpu_kernel_seconds': round(self.profile.kernel_seconds - float(start_seconds), 6), 'task_gpu_kernel_calls': self.profile.calls - int(start_calls), 'task_peak_memory_mib': round(torch.cuda.max_memory_allocated(index) / (1024.0 * 1024.0), 2)}

    @staticmethod
    def _opencv_kernel_size(sigma: float) -> int:
        return max(3, int(round(float(sigma) * 8.0 + 1.0)) | 1)

    def _kernel_size(self, sigma: float) -> int:
        if self.execution_profile == 'compat':
            return self._opencv_kernel_size(sigma)
        return max(3, int(round(float(sigma) * 6.0 + 1.0)) | 1)

    def _kernel(self, sigma: float) -> torch.Tensor:
        key = float(sigma)
        with self._lock:
            cached = self._kernel_cache.get(key)
            if cached is not None:
                return cached
            size = self._kernel_size(key)
            values = cv2.getGaussianKernel(size, key, cv2.CV_32F).reshape(-1)
            kernel = torch.from_numpy(values).to(self.device).reshape(1, 1, -1)
            self._kernel_cache[key] = kernel
            return kernel

    def _gaussian(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel = self._kernel(sigma)
        radius = kernel.shape[-1] // 2
        horizontal = kernel.reshape(1, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1)
        width = tensor.shape[-1]
        x = torch.arange(-radius, width + radius, device=self.device)
        x_mod = torch.remainder(x, 2 * width)
        x_index = torch.where(x_mod < width, x_mod, 2 * width - 1 - x_mod)
        work = tensor.index_select(-1, x_index)
        if kernel.shape[-1] >= 65:
            work = self._fft_valid_1d(work, kernel.reshape(-1), dim=-1)
        else:
            work = F.conv2d(work, horizontal)
        height = work.shape[-2]
        y = torch.arange(-radius, height + radius, device=self.device)
        y_mod = torch.remainder(y, 2 * height)
        y_index = torch.where(y_mod < height, y_mod, 2 * height - 1 - y_mod)
        work = work.index_select(-2, y_index)
        if kernel.shape[-1] >= 65:
            return self._fft_valid_1d(work, kernel.reshape(-1), dim=-2)
        return F.conv2d(work, vertical)

    @staticmethod
    def _fft_valid_1d(signal: torch.Tensor, kernel: torch.Tensor, dim: int) -> torch.Tensor:
        """Exact valid 1-D convolution using cuFFT for broad Gaussian kernels."""
        dim = dim if dim >= 0 else signal.ndim + dim
        signal_length = signal.shape[dim]
        kernel_length = kernel.numel()
        output_length = signal_length - kernel_length + 1
        full_length = signal_length + kernel_length - 1
        fft_length = 1 << int(math.ceil(math.log2(full_length)))
        spectrum = torch.fft.rfft(signal, n=fft_length, dim=dim)
        kernel_shape = [1] * signal.ndim
        kernel_shape[dim] = kernel_length
        kernel_spectrum = torch.fft.rfft(kernel.reshape(kernel_shape), n=fft_length, dim=dim)
        convolved = torch.fft.irfft(spectrum * kernel_spectrum, n=fft_length, dim=dim)
        return convolved.narrow(dim, kernel_length - 1, output_length)

    def masked_gaussian_batch(self, channels: tuple[np.ndarray, ...], mask: np.ndarray, sigmas: tuple[float, ...], *, zero_outside: bool=True) -> dict[float, tuple[np.ndarray, ...]]:
        if not channels or not sigmas:
            return {}
        started = time.perf_counter()
        mask_np = (mask > 0).astype(np.float32, copy=False)
        channel_np = np.stack([np.asarray(channel, dtype=np.float32) for channel in channels], axis=0)
        values = torch.from_numpy(channel_np).to(self.device).unsqueeze(1)
        weights = torch.from_numpy(mask_np).to(self.device)[None, None]
        weighted = values * weights
        outputs: dict[float, tuple[np.ndarray, ...]] = {}
        with torch.inference_mode():
            for sigma in sigmas:
                denominator = self._gaussian(weights, float(sigma)).clamp_min_(1e-05)
                numerator = self._gaussian(weighted, float(sigma))
                background = numerator / denominator
                if zero_outside:
                    background = background * weights
                array = background[:, 0].cpu().numpy().astype(np.float32, copy=False)
                outputs[float(sigma)] = tuple((array[index] for index in range(array.shape[0])))
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        self.profile.add_timing(time.perf_counter() - started)
        return outputs

    def masked_gaussian(self, channel: np.ndarray, mask: np.ndarray, sigma: float, *, zero_outside: bool=True) -> np.ndarray:
        return self.masked_gaussian_batch((channel,), mask, (sigma,), zero_outside=zero_outside)[float(sigma)][0]

    @staticmethod
    def _ellipse_footprint(size: int, device: torch.device) -> torch.Tensor:
        footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return torch.from_numpy(footprint.astype(bool).reshape(-1)).to(device)

    def _morphology(self, image: torch.Tensor, size: int, operation: str) -> torch.Tensor:
        radius = size // 2
        footprint = self._ellipse_footprint(size, self.device)
        pad_value = -float('inf') if operation == 'dilate' else float('inf')
        padded = F.pad(image, (radius, radius, radius, radius), value=pad_value)
        patches = F.unfold(padded, kernel_size=size)
        invalid = ~footprint
        patches[:, invalid, :] = pad_value
        if operation == 'dilate':
            reduced = patches.amax(dim=1)
        else:
            reduced = patches.amin(dim=1)
        return reduced.reshape(image.shape)

    def elliptical_blackhat_stack(self, image: np.ndarray, kernel_sizes: tuple[int, ...]) -> np.ndarray:
        started = time.perf_counter()
        source = torch.from_numpy(np.asarray(image, dtype=np.float32)).to(self.device)
        source = source[None, None]
        results: list[np.ndarray] = []
        with torch.inference_mode():
            for size in kernel_sizes:
                dilated = self._morphology(source, int(size), 'dilate')
                closed = self._morphology(dilated, int(size), 'erode')
                response = (closed - source).clamp_min_(0.0)
                results.append(response[0, 0].cpu().numpy().astype(np.float32))
                del dilated, closed, response
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        self.profile.add_timing(time.perf_counter() - started)
        return np.stack(results, axis=0)

    def warmup(self) -> dict[str, object]:
        """在 Celery 子进程内建立 CUDA 上下文并预热常用算子。"""
        sample = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)
        mask = np.full((128, 128), 255, dtype=np.uint8)
        self.masked_gaussian_batch((sample, sample * sample), mask, (1.2, 3.0, 6.0, 12.0))
        self.elliptical_blackhat_stack(sample, (3, 5))
        return self.profile.as_dict()
_BACKEND: CUDABackend | None = None
_BACKEND_LOCK = threading.Lock()

def get_cuda_backend() -> CUDABackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = CUDABackend()
        return _BACKEND

def cuda_runtime_metadata() -> dict[str, object]:
    return get_cuda_backend().profile.as_dict()
