"""Process-local write isolation for the September V3 workspace."""
import os
from pathlib import Path
import sys


def configure_runtime(root=None):
    workspace = Path(os.environ.get("DERMAVISION_V3_WORKSPACE", Path(__file__).resolve().parents[3])).resolve()
    destination = Path(root).resolve() if root else workspace / "07_运行依赖" / "stage1_runtime" / str(os.getpid())
    if not destination.is_relative_to(workspace):
        raise ValueError("V3 runtime must stay within September workspace")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TEMP", "TMP", "MPLCONFIGDIR", "XDG_CACHE_HOME", "YOLO_CONFIG_DIR", "HF_HOME", "TORCH_HOME"):
        folder = destination / name.lower()
        folder.mkdir(exist_ok=True)
        os.environ[name] = str(folder)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import tempfile
    tempfile.tempdir = os.environ["TMPDIR"]
    return destination


def require_output(path):
    workspace = Path(os.environ.get("DERMAVISION_V3_WORKSPACE", Path(__file__).resolve().parents[3])).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError("V3 writes must remain within September workspace")
    return path
