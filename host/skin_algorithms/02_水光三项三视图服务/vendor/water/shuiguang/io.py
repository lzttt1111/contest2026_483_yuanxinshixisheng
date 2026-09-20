import hashlib, json, os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

def sha256(path):
 with Path(path).open("rb") as stream:
  digest=hashlib.sha256()
  for block in iter(lambda: stream.read(1048576), b""): digest.update(block)
 return digest.hexdigest()

def read_image(path, mirrored=False):
 with Image.open(path) as raw:
  if raw.width*raw.height > 40_000_000: raise ValueError("图片像素过大")
  rgb=ImageOps.exif_transpose(raw).convert("RGB")
  if mirrored: rgb=ImageOps.mirror(rgb)
  return np.asarray(rgb)[:,:,::-1].copy()

def write_image(path, image):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 ok, encoded=cv2.imencode(path.suffix, image)
 if not ok: raise ValueError("图像编码失败")
 encoded.tofile(str(path))

def plain(value):
 if isinstance(value,np.ndarray): return plain(value.tolist())
 if isinstance(value,np.generic): return plain(value.item())
 if isinstance(value,float) and not np.isfinite(value): return None
 if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)): return [plain(v) for v in value]
 return value

def write_json(path,data):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+".tmp")
 temp.write_text(json.dumps(plain(data),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 os.replace(temp,path)

def contained(root,relative):
 relative=Path(relative)
 if relative.is_absolute() or ".." in relative.parts: raise ValueError("结果路径必须为包内相对路径")
 resolved=(Path(root)/relative).resolve()
 if not resolved.is_relative_to(Path(root).resolve()) or not resolved.is_file(): raise ValueError("结果文件缺失或越界")
 return resolved

