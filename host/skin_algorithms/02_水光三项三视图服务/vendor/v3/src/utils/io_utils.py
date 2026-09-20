# -*- coding: utf-8 -*-
"""公共 IO 工具库 — 支持中文路径的图像读写"""

import os
import cv2
import numpy as np


def cv_imread(file_path):
    """支持中文路径的安全读取"""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"⚠️ 读取异常 {file_path}: {e}")
        return None


def cv_imwrite(file_path, img):
    """支持中文路径的安全保存"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            ext = '.jpg'
        cv2.imencode(ext, img)[1].tofile(file_path)
    except Exception as e:
        print(f"⚠️ 保存异常 {file_path}: {e}")