import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/"vendor"
RUNTIME=Path(os.environ.get("SHUIGUANG_RUNTIME_ROOT",str(ROOT/"runtime"))).resolve()

def preprocessing_mode():
    # New workers must never silently fall back to the rejected legacy mask.
    mode=os.environ.setdefault('SHUIGUANG_PREPROCESS_MODE','bisenet_rgb_diagnostic_v1')
    if mode not in {'bisenet_rgb_diagnostic_v1','legacy'}:
        raise RuntimeError('Unsupported service preprocessing mode; select a version explicitly')
    return mode
def bootstrap():
    water=VENDOR/"water"
    if not (water/"shuiguang/config.py").is_file():
        raise RuntimeError("Vendored water module is missing")
    if str(water) not in sys.path:sys.path.insert(0,str(water))
    import shuiguang.config as config
    assert Path(config.ROOT).resolve()==water.resolve()
def device():
    return os.environ.get("SHUIGUANG_CLOUD_DEVICE","cpu")
def input_root():
    return Path(os.environ.get("SHUIGUANG_INPUT_ROOT",str(RUNTIME/"inputs"))).resolve()
