import os
from pathlib import Path
from .config import ROOT

def configure(device):
 if device!="cpu" and not device.startswith("cuda:"): raise ValueError("设备必须为cpu或cuda:N")
 # Explicitly scoped caches; no parent repository or user-global model downloads.
 cache=ROOT/"runtime/cache"
 for key,folder in {"MPLCONFIGDIR":"matplotlib","YOLO_CONFIG_DIR":"ultralytics","XDG_CACHE_HOME":"xdg","TORCH_HOME":"torch"}.items():
  path=cache/folder; path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
 os.environ["SHUIGUANG_DEVICE"]=device
 os.environ["PYTHONDONTWRITEBYTECODE"]="1"
 os.environ["YOLO_OFFLINE"]="true"
 import torch
 if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA不可用，不会静默切换CPU")
 torch.set_num_threads(min(8,os.cpu_count() or 1))

