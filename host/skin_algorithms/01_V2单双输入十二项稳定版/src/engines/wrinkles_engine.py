# -*- coding: utf-8 -*-
"""
皱纹检测引擎 V2 (Wrinkles) — 五官遮挡 + Frangi 多尺度滤波 + 骨架化 + 绿色线条渲染
═══════════════════════════════════════════════════════════════
V2 修复：复用色斑引擎 V4.0 的五官遮挡逻辑，彻底消除眼睛/鼻子/眉毛误判

算法路线：
  1. 读取预处理底图 → MediaPipe 提取面部关键点
  2. 五官标定 + 虚拟粉底抹平（眼睛、眉毛、鼻孔、嘴唇用肤色填充）
  3. 双边滤波去噪 + CLAHE 对比度增强
  4. scikit-image frangi() 多尺度滤波（在纯净皮肤区域检测线状纹理）
  5. 阈值二值化 + 面部掩膜裁切
  6. 形态学骨架化 + 连通域过滤
  7. 原图叠加绿色线条渲染
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from skimage.filters import frangi
from skimage.morphology import skeletonize

from src.utils.io_utils import cv_imread, cv_imwrite
from src.utils.model_loader import load_face_landmarker

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    """⚙️ 皱纹检测参数配置中心 (V2)"""

    # 🎨 --- 渲染色彩 ---
    COLOR_WRINKLE = (0, 255, 100)     # 皱纹标记绿色 (BGR)
    COLOR_BG = (255, 255, 255)        # 非面部背景纯白 (BGR)

    # 🛡️ --- 背景分离 ---
    BG_THRESH = 1
    BG_BLUR_SIZE = 3

    # 🧹 --- 五官遮挡参数 (复用色斑引擎 V4.0) ---
    HAIR_DARKNESS_PREPROCESS = 110    # 碎发识别阈值
    FEATURE_DILATE_KERNEL = 21        # 五官遮挡膨胀核

    # 🔬 --- Frangi 滤波参数 ---
    FRANGI_SIGMAS = [1, 2, 3, 4]      # 多尺度（1=细纹, 4=深纹）
    FRANGI_THRESHOLD = 0.02           # Frangi 响应阈值

    # 🎯 --- 后处理过滤 ---
    MIN_LINE_AREA = 8                 # 最小连通域面积
    CLAHE_CLIP_LIMIT = 2.0            # CLAHE 对比度增强倍率

    # 🖌️ --- 渲染参数 ---
    WRINKLE_THICKNESS = 1             # 皱纹线条粗细


class WrinklesEngine:
    def __init__(self):
        self.face_landmarker = load_face_landmarker()

    def get_face_mask(self, image, landmarks):
        """提取面部凸包掩膜"""
        h, w = image.shape[:2]
        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        face_area = cv2.countNonZero(mask)
        erode_size = max(3, int(np.sqrt(face_area) * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask, face_area

    def erase_features(self, img, l_channel, landmarks, face_hull_mask, h, w):
        """
        五官遮挡 + 碎发清除（复用色斑引擎 V4.0 逻辑）
        将眼睛、眉毛、鼻孔、嘴唇、碎发区域用平均肤色填充
        """
        to_erase_mask = np.zeros((h, w), dtype=np.uint8)

        # 标定五官（MediaPipe 478点关键点索引）
        exclusions = [
            [33, 133, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155],  # 左眼
            [362, 263, 387, 386, 385, 384, 398, 388, 390, 373, 374, 380, 381, 382],  # 右眼
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],  # 嘴唇
            [79, 74, 304, 309, 2],  # 鼻孔
            [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # 左眉毛
            [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]  # 右眉毛
        ]
        for indices in exclusions:
            pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
            cv2.fillConvexPoly(to_erase_mask, cv2.convexHull(pts), 255)

        # 膨胀五官遮挡区
        to_erase_mask = cv2.dilate(to_erase_mask,
                                   np.ones((Config.FEATURE_DILATE_KERNEL, Config.FEATURE_DILATE_KERNEL), np.uint8),
                                   iterations=1)

        # 标定碎发
        _, dark_pixels = cv2.threshold(l_channel, Config.HAIR_DARKNESS_PREPROCESS, 255, cv2.THRESH_BINARY_INV)
        hair_on_face = cv2.bitwise_and(dark_pixels, face_hull_mask)
        hair_on_face = cv2.dilate(hair_on_face, np.ones((5, 5), np.uint8), iterations=1)

        to_erase_mask = cv2.bitwise_or(to_erase_mask, hair_on_face)

        # 计算纯净肤色均值
        pure_skin_mask = cv2.subtract(face_hull_mask, to_erase_mask)
        mean_l = cv2.mean(l_channel, mask=pure_skin_mask)[0]

        # 用肤色填充五官区域
        cleaned_l_channel = l_channel.copy()
        cleaned_l_channel[to_erase_mask > 0] = int(mean_l)

        return cleaned_l_channel

    def process_image(self, img_path, output_dir):
        filename = os.path.basename(img_path)
        img = cv_imread(img_path)
        if img is None:
            return None

        print(f"\n📐 [皱纹引擎 V2] 正在分析: {filename} ...")

        # 1. AI 锁定面部区域
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            print(f"⚠️ 跳过 {filename}: 未检测到有效人脸。")
            return None

        landmarks = results.face_landmarks[0]
        h, w = img.shape[:2]

        face_mask, face_area = self.get_face_mask(img, landmarks)

        # 2. 背景蒙版
        raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(raw_gray, Config.BG_THRESH, 255, cv2.THRESH_BINARY)
        bg_blur = Config.BG_BLUR_SIZE if Config.BG_BLUR_SIZE % 2 != 0 else Config.BG_BLUR_SIZE + 1
        fg_mask_float = cv2.GaussianBlur(fg_mask.astype(np.float32), (bg_blur, bg_blur), 0) / 255.0
        fg_mask_3d = np.expand_dims(fg_mask_float, axis=2)

        # 3. LAB 空间 L 通道 + 五官遮挡（核心修复！）
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # 面部凸包掩膜（未腐蚀版，用于五官遮挡）
        face_hull_mask = np.zeros((h, w), dtype=np.uint8)
        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks], dtype=np.int32)
        cv2.fillConvexPoly(face_hull_mask, cv2.convexHull(points), 255)

        # 五官遮挡 + 碎发清除 → 纯净 L 通道
        cleaned_l_channel = self.erase_features(img, l_channel, landmarks, face_hull_mask, h, w)

        # 4. 双边滤波去噪 + CLAHE 对比度增强
        l_filtered = cv2.bilateralFilter(cleaned_l_channel, d=7, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_filtered)

        # 5. Frangi 多尺度滤波（在五官已抹平的纯净皮肤上检测）
        l_float = l_enhanced.astype(np.float64) / 255.0
        frangi_response = frangi(l_float, sigmas=Config.FRANGI_SIGMAS, black_ridges=True)

        # 6. 阈值二值化 + 面部掩膜裁切
        wrinkle_mask = (frangi_response > Config.FRANGI_THRESHOLD).astype(np.uint8) * 255
        wrinkle_mask = cv2.bitwise_and(wrinkle_mask, face_mask)

        # 7. 形态学骨架化
        wrinkle_skeleton = skeletonize(wrinkle_mask > 0).astype(np.uint8) * 255

        # 8. 连通域过滤
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wrinkle_skeleton, connectivity=8)
        clean_wrinkles = np.zeros_like(wrinkle_skeleton)
        wrinkle_count = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= Config.MIN_LINE_AREA:
                clean_wrinkles[labels == i] = 255
                wrinkle_count += 1

        print(f"   📊 检测到 {wrinkle_count} 条有效皱纹线段（五官已遮挡）")

        # 9. 渲染：原图上叠加绿色线条
        final_render = img.copy()
        contours, _ = cv2.findContours(clean_wrinkles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(final_render, [cnt], -1, Config.COLOR_WRINKLE, Config.WRINKLE_THICKNESS, cv2.LINE_AA)

        # 替换纯白背景
        final_render_float = final_render.astype(np.float32)
        pure_bg = np.array(Config.COLOR_BG, dtype=np.float32)
        final_render = (final_render_float * fg_mask_3d + pure_bg * (1 - fg_mask_3d)).astype(np.uint8)

        # 输出保存
        base_name = os.path.splitext(filename)[0]
        if base_name.endswith("_01_Preprocessed"):
            base_name = base_name.replace("_01_Preprocessed", "")
        output_path = os.path.join(output_dir, f"{base_name}_04_Wrinkles_Analysis.jpg")
        cv_imwrite(output_path, final_render)

        print(f"✅ 生成完毕 -> {base_name}_04_Wrinkles_Analysis.jpg")
        return output_path


if __name__ == "__main__":
    input_dir = os.path.join(PROJECT_ROOT, "output", "preprocessed")
    output_dir = os.path.join(PROJECT_ROOT, "output", "wrinkles")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(" 📐 皱纹检测引擎 V2 (五官遮挡 + Frangi 多尺度滤波)")
    print("=" * 60)

    engine = WrinklesEngine()
    valid_files = [f for f in os.listdir(input_dir) if f.endswith("_01_Preprocessed.jpg")]
    if not valid_files:
        print(f"⚠️ 找不到输入目录！请确保 {input_dir} 存在。")
    else:
        for file in valid_files:
            file_path = os.path.join(input_dir, file)
            engine.process_image(file_path, output_dir)
        print(f"\n🎉 皱纹检测 V2 全部完成！五官误判已消除。")