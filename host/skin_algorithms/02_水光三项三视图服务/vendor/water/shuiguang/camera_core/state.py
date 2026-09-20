"""Frame-time capture gate and conservative spatial tracking. No identity recognition."""
import math
import time
from collections import deque
from statistics import median


class PoseFilter:
    """Three-frame median plus time-based EMA; never carries across missing frames."""
    def __init__(self):self.clear()
    def clear(self):
        self.history=deque(maxlen=3);self.last=None;self.value=None
    def step(self,pose,now):
        if not pose or not pose.get('valid') or not all(math.isfinite(pose[k]) for k in ('yaw','pitch','roll')):
            self.clear();return None
        if self.last is None or not 0<now-self.last<=350:self.clear()
        self.history.append(pose)
        med={k:median(p[k] for p in self.history) for k in ('yaw','pitch','roll')}
        alpha=1 if self.last is None else 1-math.exp(-(now-self.last)/120.)
        self.value={k:med[k] if self.value is None else self.value[k]+alpha*(med[k]-self.value[k]) for k in med}
        self.last=now
        return {**pose,**self.value}


def iou(a, b):
    x = max(0, min(a.x+a.width, b.x+b.width)-max(a.x, b.x))
    y = max(0, min(a.y+a.height, b.y+b.height)-max(a.y, b.y))
    intersection = x*y
    return intersection / max(a.area+b.area-intersection, 1)


class FaceLock:
    def __init__(self):
        self.face = None
        self.lost = False
        self.missing_since = None

    def select(self, faces, detector, width, height, now_ms=None):
        now_ms = time.monotonic()*1000 if now_ms is None else now_ms
        if self.lost:
            return None, '请重置本轮以重新锁定人脸'
        if self.face is None:
            self.face = detector.select_primary(faces, width, height)
            return self.face, '未检测到人脸' if self.face is None else ''
        if self.missing_since is not None and now_ms-self.missing_since > 350:
            self.lost = True
            return None, '主脸丢失超过350ms，请重置本轮'
        if not faces:
            if self.missing_since is None:
                self.missing_since = now_ms
            return None, '人脸短暂漏检，暂停拍照等待恢复'
        candidates = sorted(((iou(self.face, f), f) for f in faces), key=lambda x: x[0], reverse=True)
        if not candidates or candidates[0][0] < .30:
            self.lost = True
            return None, '主脸丢失，请重置本轮重新确认'
        if len(candidates) > 1 and candidates[1][0] > candidates[0][0]-.15:
            self.lost = True
            return None, '主脸存在歧义，请重置本轮'
        self.face = candidates[0][1]
        self.missing_since = None
        return self.face, ''


class CaptureGate:
    def __init__(self, config):
        self.config = config
        self.done = set()
        self.clear()

    def clear(self):
        self.start = self.last = None
        self.side = None
        self.anchor = None
        self.samples = 0

    def step(self, pose, now_ms, sharpness, face_size, age_ms, raw_pose=None):
        c = self.config
        reason = ''
        if not pose or not pose.get('valid'):
            reason = '等待有效人脸'
        elif not all(math.isfinite(pose[k]) for k in ('yaw', 'pitch', 'roll')):
            reason = '角度无效'
        elif age_ms > c['max_frame_age_ms']:
            reason = f'画面结果过期：{age_ms:.0f}ms > {c["max_frame_age_ms"]}ms'
        elif face_size < c['min_face_px']:
            reason = f'人脸过小：{face_size:.0f}px < {c["min_face_px"]}px'
        elif sharpness < c['blur_threshold']:
            reason = f'画面模糊：清晰度 {sharpness:.0f} < {c["blur_threshold"]}'
        elif abs(pose['pitch']) > c['pitch_limit'] or abs(pose['roll']) > c['roll_limit']:
            reason = f'抬头/歪头超限：Pitch {pose["pitch"]:.1f}°，Roll {pose["roll"]:.1f}°'
        if reason:
            self.clear()
            return {'progress': 0, 'reason': reason, 'trigger': None}
        targets = {'front':0., 'left':c['yaw_target']*c['left_yaw_sign'], 'right':-c['yaw_target']*c['left_yaw_sign']}
        side = next((v for v in ('front','left','right') if v not in self.done), None)
        if side is None:
            self.clear()
            return {'progress':1,'reason':'三视图已完成，请预览确认','trigger':None}
        if abs(pose['yaw']-targets[side]) > c['yaw_tolerance']:
            self.clear()
            label={'front':'正面','left':'左侧','right':'右侧'}[side]
            return {'progress': 0, 'reason': f'请调整至{label} {targets[side]:g}°（±{c["yaw_tolerance"]:g}°）', 'trigger': None}
        if side in self.done:
            self.clear()
            return {'progress': 0, 'reason': '本侧已完成，请拍另一侧' if len(self.done)<2 else '左右照片已完成', 'trigger': None}
        if (self.start is None or self.side != side or now_ms-self.last > c['max_gap_ms']
                or now_ms <= self.last or abs(pose['yaw']-self.anchor) > c['yaw_jitter']):
            self.start = now_ms
            self.anchor = pose['yaw']
            self.samples = 0
        self.side, self.last = side, now_ms
        self.samples += 1
        progress = min(1, (now_ms-self.start)/c['stable_ms'])
        trigger = side if progress >= 1 and self.samples >= 3 else None
        if trigger and raw_pose is not None:
            if (not raw_pose.get('valid') or not all(math.isfinite(raw_pose[k]) for k in ('yaw','pitch','roll'))
                or abs(raw_pose['yaw']-targets[side])>c['yaw_tolerance']
                or abs(raw_pose['pitch'])>c['pitch_limit'] or abs(raw_pose['roll'])>c['roll_limit']):
                return {'progress':progress,'reason':'稳定角度已达标，等待当前原始帧也达标','trigger':None}
        return {'progress': progress, 'reason': '保持稳定' if not trigger else '已拍摄', 'trigger': trigger}

    def commit(self, side):
        self.done.add(side)
        self.clear()
