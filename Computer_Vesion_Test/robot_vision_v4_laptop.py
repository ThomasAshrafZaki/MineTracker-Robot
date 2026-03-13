#!/usr/bin/env python3
"""
Robot Vision Unified System V4 - Enhanced Edition
==================================================
الكود الأصلي V3 + إضافات جديدة للصحراء والواقع:

  [جديد 1] Watchdog Thread        — لو الكود وقف أي سبب، يبعت STOP فوراً
  [جديد 2] Dynamic ROI            — ROI يتعدل تلقائياً لو العربية اهتزت
  [جديد 3] Multi-Obstacle Avoidance — يشوف كل العوائق مش بس الأقرب
  [جديد 4] Adaptive Confidence    — يعدل حساسية الكشف تلقائياً حسب الإضاءة
  [جديد 5] Ultrasonic Multi-Zone  — 3 مناطق للـ ultrasonic (خطر/تحذير/آمن)

تشغيل على اللاب (CPU):
  python3 robot_vision_v4.py --config config_v4.json

تشغيل على Jetson/Pi (GPU):
  python3 robot_vision_v4.py --config config_v4.json --set yolo.device=cuda

Install:
  pip install ultralytics opencv-python numpy
  pip install pyserial  (للأردوينو)
"""
import os
import sys
import time
import json
import csv
import signal
import argparse
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
try:
    from ultralytics import YOLO
except Exception:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    raise
try:
    import serial
    SERIAL_OK = True
except Exception:
    SERIAL_OK = False

Box = Tuple[int, int, int, int]
Det = Tuple[int, int, int, int, str, float]

# ------------------------------------------------------------------ #
#  Config helpers  (أصلي — لم يتغير)
# ------------------------------------------------------------------ #
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def parse_value(s: str) -> Any:
    s2 = s.strip()
    if s2.lower() in ("true", "false"):
        return s2.lower() == "true"
    if s2.lower() in ("null", "none"):
        return None
    try:
        if s2.isdigit() or (s2.startswith("-") and s2[1:].isdigit()):
            return int(s2)
    except Exception:
        pass
    try:
        return float(s2)
    except Exception:
        return s2

def now_ms() -> int:
    return int(time.time() * 1000)

# ------------------------------------------------------------------ #
#  Math utils  (أصلي — لم يتغير)
# ------------------------------------------------------------------ #
def clamp_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return (x1, y1, x2, y2)

def point_in_poly(poly: np.ndarray, p: Tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(poly, p, False) >= 0

def inside_roi_by_3points(roi_contour: np.ndarray, box: Box) -> bool:
    x1, y1, x2, y2 = box
    pts = [(float(x1 + 2), float(y2)),
           (float((x1 + x2) // 2), float(y2)),
           (float(x2 - 2), float(y2))]
    return any(point_in_poly(roi_contour, p) for p in pts)

# ------------------------------------------------------------------ #
#  NCNN helper  (أصلي — لم يتغير)
# ------------------------------------------------------------------ #
def resolve_model_path(model_path: str) -> str:
    """على اللاب: نستخدم .pt مباشرة على CPU."""
    print(f"[Model] Using: {model_path}")
    return model_path

def export_tensorrt(model_path: str):
    print("[TensorRT] للـ Jetson فقط — شغّل على الجيتسون مباشرة.")

# ------------------------------------------------------------------ #
#  [جديد 1] Watchdog Thread
#  لو المين لوب وقف لأكتر من timeout ثانية → يبعت STOP تلقائياً
#  يحمي من: crash، freeze، أي خطأ غير متوقع
# ------------------------------------------------------------------ #
class Watchdog:
    """
    Safety watchdog: يراقب المين لوب.
    كل frame لازم يعمل ping().
    لو وقف → يبعت STOP للأردوينو.
    """
    def __init__(self, serial_ref, timeout_s: float = 1.0):
        self._serial = serial_ref
        self._timeout = timeout_s
        self._last_ping = time.time()
        self._running = True
        self._triggered = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Watchdog] Started — timeout={timeout_s}s")

    def ping(self):
        """استدعيها كل frame من المين لوب."""
        with self._lock:
            self._last_ping = time.time()
            self._triggered = False

    def _loop(self):
        while self._running:
            time.sleep(0.1)
            with self._lock:
                age = time.time() - self._last_ping
                if age > self._timeout and not self._triggered:
                    print(f"[Watchdog] ⚠️  TRIGGERED — no ping for {age:.1f}s → sending STOP")
                    self._serial.send("S")
                    self._triggered = True

    def stop(self):
        self._running = False

# ------------------------------------------------------------------ #
#  [جديد 2] Dynamic ROI
#  لو العربية اهتزت أو انحرفت، الـ ROI يتعدل تلقائياً
#  بيستخدم Optical Flow يحسب ميل الصورة ويعوض عنه
# ------------------------------------------------------------------ #
class DynamicROI:
    """
    يراقب حركة الكاميرا بالـ Optical Flow.
    لو الكاميرا اتحركت جنب (اهتزاز)، يزحزح الـ ROI عشان يعوّض.
    """
    def __init__(self, enabled: bool, max_shift_px: int = 40):
        self.enabled = enabled
        self.max_shift = max_shift_px
        self._prev_gray = None
        self._shift_x = 0
        self._shift_y = 0
        self._alpha = 0.3  # smoothing factor
        print(f"[DynamicROI] enabled={enabled}")

    def update(self, frame_bgr: np.ndarray):
        """استدعيها كل frame — تحسب shift الكاميرا."""
        if not self.enabled:
            return
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return
        # Optical flow على نقاط Shi-Tomasi
        pts = cv2.goodFeaturesToTrack(self._prev_gray, maxCorners=50,
                                      qualityLevel=0.3, minDistance=10)
        if pts is not None and len(pts) >= 5:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts, None)
            good_old = pts[status.flatten() == 1]
            good_new = next_pts[status.flatten() == 1]
            if len(good_old) >= 5:
                dx = float(np.median(good_new[:, 0, 0] - good_old[:, 0, 0]))
                dy = float(np.median(good_new[:, 0, 1] - good_old[:, 0, 1]))
                # smoothing
                self._shift_x = self._alpha * dx + (1 - self._alpha) * self._shift_x
                self._shift_y = self._alpha * dy + (1 - self._alpha) * self._shift_y
                # clamp
                self._shift_x = max(-self.max_shift, min(self.max_shift, self._shift_x))
                self._shift_y = max(-self.max_shift, min(self.max_shift, self._shift_y))
        self._prev_gray = gray

    def apply(self, roi_pts: np.ndarray) -> np.ndarray:
        """طبّق الـ shift على الـ ROI — يعوّض حركة الكاميرا."""
        if not self.enabled:
            return roi_pts
        shifted = roi_pts.copy().astype(np.float32)
        # نعوّض عكس الحركة (لو الكاميرا راحت يمين، ROI يمشي شمال)
        shifted[0, :, 0] -= self._shift_x
        shifted[0, :, 1] -= self._shift_y
        return shifted.astype(np.int32)

    def get_shift(self) -> Tuple[float, float]:
        return self._shift_x, self._shift_y

# ------------------------------------------------------------------ #
#  [جديد 3] VFH — Vector Field Histogram Planner
#  أقوى من Potential Fields — بيبني histogram للاتجاهات المسدودة
#  ويختار أوسع فراغ متاح تعدي منه العربية
#
#  فكرة VFH:
#  1. قسّم الأمام لـ N sector (زوايا)
#  2. كل عائق يرفع "كثافة" الـ sector اللي فيه
#  3. اختار أوسع sector فارغ متاح
#  4. لو مفيش فراغ → STOP
# ------------------------------------------------------------------ #
class MultiObstaclePlanner:
    """
    VFH (Vector Field Histogram) Planner:
    - يقسّم الـ frame لـ sectors أفقية
    - كل عائق يملأ الـ sector بتاعه بنسبة خطورته
    - يختار أوسع sector فارغ → يبعت L / F / R
    - لو كل الـ sectors مسدودة → S
    - نفس الـ interface بالظبط زي الكلاس القديم (مش محتاج تغيير في App)
    """
    def __init__(self, prefer: str = "AUTO", num_sectors: int = 9,
                 density_threshold: float = 0.35):
        self.prefer = prefer
        self.num_sectors = num_sectors          # عدد الـ sectors (홀 عشان يكون في sector وسط)
        self.density_threshold = density_threshold  # فوق الرقم ده → sector مسدود

    def _build_histogram(self, hazards: List[Dict], w: int, h: int) -> List[float]:
        """
        يبني histogram: قائمة density لكل sector.
        كل عائق بيضيف density للـ sector بتاعه بناءً على:
        - مساحته (كبير = أخطر)
        - قربه (y2 أكبر = أقرب = أخطر)
        """
        histogram = [0.0] * self.num_sectors
        sector_w = w / self.num_sectors

        for hz in hazards:
            x1, y1, x2, y2 = hz["box"]
            cx = (x1 + x2) / 2.0

            # خطورة العائق: قربه × مساحته النسبية
            proximity  = (y2 / float(h)) if h > 0 else 0.5
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
            danger     = proximity * (1.0 + area_ratio * 3.0)
            danger     = min(1.0, danger)

            # الـ sector الأساسي
            s = int(cx / sector_w)
            s = max(0, min(self.num_sectors - 1, s))
            histogram[s] = min(1.0, histogram[s] + danger)

            # spread للـ sectors المجاورة (العائق ممتد مش نقطة)
            half_spread = max(1, int((x2 - x1) / sector_w / 2))
            for ds in range(1, half_spread + 1):
                fade = danger * (0.5 ** ds)
                for ns in (s - ds, s + ds):
                    if 0 <= ns < self.num_sectors:
                        histogram[ns] = min(1.0, histogram[ns] + fade)

        return histogram

    def _find_best_gap(self, histogram: List[float]) -> Tuple[int, int, float]:
        """
        يدور على أوسع فراغ متصل في الـ histogram.
        إرجاع: (gap_start, gap_end, gap_center_sector)
        """
        n   = self.num_sectors
        thr = self.density_threshold

        # ابني قائمة الـ sectors الفارغة
        free = [i for i in range(n) if histogram[i] < thr]
        if not free:
            return -1, -1, -1.0  # كل حاجة مسدودة

        # دور على أوسع run متصل
        best_start = best_end = free[0]
        cur_start  = free[0]
        for i in range(1, len(free)):
            if free[i] == free[i - 1] + 1:
                best_end = free[i]
            else:
                # run انتهى — قارن مع الأحسن
                if (best_end - best_start) < (free[i] - cur_start):
                    best_start = cur_start
                    best_end   = free[i - 1]
                cur_start  = free[i]
                best_end   = free[i]
        # آخر run
        if (best_end - best_start) < (free[-1] - cur_start):
            best_start = cur_start
            best_end   = free[-1]

        center = (best_start + best_end) / 2.0
        return best_start, best_end, center

    def decide(self, hazards: List[Dict], w: int, h: int,
               tri_left: float, tri_center: float, tri_right: float) -> str:
        """
        نفس الـ interface القديم بالظبط.
        إرجاع: F / L / R / S
        """
        if not hazards:
            return "F"

        histogram = self._build_histogram(hazards, w, h)
        gap_start, gap_end, gap_center = self._find_best_gap(histogram)

        # كل الـ sectors مسدودة → STOP
        if gap_center < 0:
            return "S"

        # الـ sector الأوسط هو النص
        mid = (self.num_sectors - 1) / 2.0

        # هامش حول النص = ربع عدد الـ sectors
        margin = self.num_sectors * 0.25

        if gap_center < mid - margin:
            return "L"   # الفراغ على اليسار
        elif gap_center > mid + margin:
            return "R"   # الفراغ على اليمين
        else:
            return "F"   # الفراغ في النص → امشي قدام

# ------------------------------------------------------------------ #
#  [جديد 4] Adaptive Confidence
#  في الصحراء الإضاءة بتتغير كتير (شمس حادة، ظل، غبار)
#  بيرفع الـ confidence لو الصورة واضحة جداً (noise كتير)
#  بيخفضه لو الصورة مظلمة (مش شايف كويس)
# ------------------------------------------------------------------ #
class AdaptiveConfidence:
    """
    يحسب brightness متوسط الفريم ويعدل confidence تلقائياً.
    - صورة فاتحة جداً (شمس حادة) → confidence أعلى (أقل false alarms)
    - صورة داكنة (غبار/ظل) → confidence أقل (أحسس بأكتر)
    """
    def __init__(self, enabled: bool,
                 base_conf: float = 0.60,
                 min_conf: float = 0.35,
                 max_conf: float = 0.80,
                 update_every: int = 30):
        self.enabled = enabled
        self.base_conf = base_conf
        self.min_conf = min_conf
        self.max_conf = max_conf
        self.update_every = update_every
        self._frame_count = 0
        self._current_conf = base_conf
        print(f"[AdaptiveConf] enabled={enabled}, base={base_conf}")

    def update(self, frame_bgr: np.ndarray) -> float:
        """استدعيها كل frame — إرجاع الـ conf المناسب."""
        if not self.enabled:
            return self._current_conf

        self._frame_count += 1
        # حساب كل update_every frame بس عشان نوفر CPU
        if self._frame_count % self.update_every != 0:
            return self._current_conf

        # متوسط brightness الفريم
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))  # 0-255

        # تطبيع 0-255 → 0-1
        norm = brightness / 255.0

        # Curve:
        # داكن جداً (norm<0.2) → conf منخفض (أكثر حساسية)
        # متوسط (norm≈0.5) → base_conf
        # فاتح جداً (norm>0.8) → conf مرتفع (أقل false alarms)
        if norm < 0.2:
            new_conf = self.min_conf
        elif norm > 0.8:
            new_conf = self.max_conf
        else:
            # linear interpolation
            ratio = (norm - 0.2) / 0.6
            new_conf = self.min_conf + ratio * (self.max_conf - self.min_conf)

        # Smooth التغيير عشان ميتقفزش
        self._current_conf = 0.7 * self._current_conf + 0.3 * new_conf
        self._current_conf = max(self.min_conf, min(self.max_conf, self._current_conf))
        return self._current_conf

    def get(self) -> float:
        return self._current_conf

# ------------------------------------------------------------------ #
#  [جديد 5] Ultrasonic Multi-Zone
#  بدل zone واحدة (emergency stop فقط)، 3 مناطق:
#  DANGER  → وقف فوراً
#  WARNING → خفف السرعة + ابعت تحذير
#  SAFE    → كمل عادي
# ------------------------------------------------------------------ #
class UltrasonicZone:
    """
    3 مناطق للـ ultrasonic:
    DANGER  (< danger_cm)  → STOP + S command
    WARNING (< warning_cm) → slow down + W command
    SAFE    (else)         → proceed normally
    """
    def __init__(self, enabled: bool,
                 danger_cm: float = 25.0,
                 warning_cm: float = 60.0,
                 max_age_ms: int = 600,
                 fail_safe: bool = False):
        self.enabled = enabled
        self.danger_cm = danger_cm
        self.warning_cm = warning_cm
        self.max_age_ms = max_age_ms
        self.fail_safe = fail_safe
        print(f"[UltraZone] enabled={enabled}, danger={danger_cm}cm, warning={warning_cm}cm")

    def evaluate(self, ultra_cm: Optional[float], ultra_age_ms: int,
                 vision_state: str) -> Tuple[str, str, str]:
        """
        إرجاع: (final_state, cause, zone)
        zone: DANGER / WARNING / SAFE / STALE / DISABLED
        """
        if not self.enabled:
            return vision_state, "NONE" if vision_state != "STOP" else "VISION", "DISABLED"

        # Sensor stale (قديم)
        if ultra_cm is None or ultra_age_ms > self.max_age_ms:
            if self.fail_safe:
                return "STOP", "ULTRA_STALE", "STALE"
            return vision_state, "NONE" if vision_state != "STOP" else "VISION", "STALE"

        # Zone classification
        if ultra_cm <= self.danger_cm:
            return "STOP", "ULTRA", "DANGER"
        elif ultra_cm <= self.warning_cm:
            # Warning zone: vision تقرر بس نسجّل التحذير
            # لو vision STOP كمان → STOP
            # لو vision PROCEED → PROCEED بس بسرعة أقل (App بتتعامل مع ده)
            cause = "VISION" if vision_state == "STOP" else "NONE"
            return vision_state, cause, "WARNING"
        else:
            cause = "VISION" if vision_state == "STOP" else "NONE"
            return vision_state, cause, "SAFE"

# ------------------------------------------------------------------ #
#  كشف لافتات التحذير
#  بيكتشف:
#  1. الألوان التحذيرية (أحمر / أصفر / برتقالي)
#  2. الأشكال المثلثة والدائرية (علامات التحذير)
#  3. بيرسم مربع حول اللافتة ويكتب WARNING
# ------------------------------------------------------------------ #
class SignDetector:
    """
    يكتشف لافتات التحذير بالألوان والأشكال:
    - أحمر → خطر / قف
    - أصفر/برتقالي → تحذير
    - مثلث أو دائرة → شكل علامة تحذير
    """
    def __init__(self, enabled: bool,
                 min_sign_area: int = 800):
        self.enabled = enabled
        self.min_sign_area = min_sign_area
        print(f"[SignDetector] enabled={enabled}")

    def detect(self, frame_bgr: np.ndarray) -> Tuple[bool, List[Dict]]:
        """
        إرجاع: (sign_found, list of signs)
        كل sign: {"box": (x1,y1,x2,y2), "color": str, "shape": str}
        """
        if not self.enabled:
            return False, []

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        signs = []

        # ── كشف الأحمر (لونين عشان الأحمر في HSV في الطرفين) ──
        red_mask1 = cv2.inRange(hsv, np.array([0,   120, 70]),  np.array([10,  255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 120, 70]),  np.array([180, 255, 255]))
        red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

        # ── كشف الأصفر ──
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

        # ── كشف البرتقالي ──
        orange_mask = cv2.inRange(hsv, np.array([10, 120, 100]), np.array([20, 255, 255]))

        color_masks = [
            (red_mask,    "RED_DANGER"),
            (yellow_mask, "YELLOW_WARNING"),
            (orange_mask, "ORANGE_WARNING"),
        ]

        for mask, color_name in color_masks:
            # نظّف الـ mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_sign_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                # حدد الشكل
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 3:
                    shape = "TRIANGLE"
                elif len(approx) >= 8:
                    shape = "CIRCLE"
                else:
                    shape = "RECT"

                signs.append({
                    "box":   (x, y, x + bw, y + bh),
                    "color": color_name,
                    "shape": shape,
                    "area":  area
                })

        return len(signs) > 0, signs


# ------------------------------------------------------------------ #
#  كشف الحفرة
#  الفكرة: الحفرة = منطقة داكنة جداً في الجزء السفلي من الـ ROI
#  لو في منطقة داكنة كبيرة قدام العربية → خطر حفرة
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
#  كشف الحفرة — النسخة المحترفة (3 طرق مع بعض)
#
#  طريقة 1: Texture Variance
#    الأرض المستوية → texture منتظم → variance منخفض
#    الحفرة → texture بيختفي أو يتغير → variance مختلف جداً
#
#  طريقة 2: Edge Density Drop
#    الأرض الطبيعية فيها edges من الحجارة والرمل
#    الحفرة = منطقة فارغة → edges بتقل فجأة
#
#  طريقة 3: Brightness Gradient
#    مش بس داكن — لازم يكون في تغيير مفاجئ في السطوع
#    ده بيميّز الحفرة عن الظل العادي
#
#  النتيجة: الـ 3 طرق لازم توافق مع بعض عشان يقول PIT
#  ده بيقلل الـ false alarms بشكل كبير
# ------------------------------------------------------------------ #
class PitDetector:
    """
    كشف الحفر بـ 3 طرق مجتمعة:
    1. Texture Variance  — الحفرة بتغيّر الـ texture فجأة
    2. Edge Density Drop — الحفرة فيها edges أقل من الأرض العادية
    3. Brightness Gradient — تغيير مفاجئ في السطوع (مش ظل عادي)

    لازم اتنين من التلاتة يوافقوا عشان يحكم بحفرة.
    """
    def __init__(self, enabled: bool,
                 dark_threshold: int = 40,
                 min_pit_area_pct: float = 0.08):
        self.enabled       = enabled
        # نحتفظ بالـ params القديمة للتوافق مع الـ config
        self.dark_threshold    = dark_threshold
        self.min_pit_area_pct  = min_pit_area_pct

        # ── إعدادات الطرق الثلاث ──
        # طريقة 1: Texture
        self._tex_diff_thresh  = 0.35   # نسبة الاختلاف في variance اللازمة للحكم
        # طريقة 2: Edge Density
        self._edge_drop_thresh = 0.40   # لو edges الحفرة أقل من 40% من edges الخلفية
        # طريقة 3: Gradient
        self._grad_thresh      = 25.0   # قيمة gradient المفاجئ بين منطقة وتانية

        # Temporal smoothing — عشان نتجنب false alarm من frame واحد
        self._pit_score_smooth = 0.0
        self._smooth_alpha     = 0.4

        print(f"[PitDetector] enabled={enabled} — Professional 3-method mode")

    # ── الطريقة الأولى: Texture Variance ──
    def _texture_score(self, upper: np.ndarray, lower: np.ndarray) -> float:
        """
        يقارن الـ texture variance بين الجزء العلوي (أرض عادية)
        والجزء السفلي (المنطقة قدام العربية).
        لو الفرق كبير → احتمال حفرة.
        """
        def local_variance(gray: np.ndarray) -> float:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = gray.astype(np.float32) - blur.astype(np.float32)
            return float(np.mean(np.abs(diff)))

        var_upper = local_variance(upper) + 1e-5
        var_lower = local_variance(lower) + 1e-5
        ratio = abs(var_upper - var_lower) / max(var_upper, var_lower)
        return ratio  # 0→1: كلما أكبر = اختلاف أكبر = احتمال حفرة أكبر

    # ── الطريقة الثانية: Edge Density ──
    def _edge_density_score(self, upper: np.ndarray, lower: np.ndarray) -> float:
        """
        يحسب كثافة الـ edges في المنطقتين.
        الحفرة = edges أقل بكتير من الأرض العادية.
        """
        def edge_density(gray: np.ndarray) -> float:
            edges = cv2.Canny(gray, 30, 80)
            return float(np.mean(edges > 0))

        density_upper = edge_density(upper) + 1e-5
        density_lower = edge_density(lower) + 1e-5
        drop = (density_upper - density_lower) / density_upper
        return max(0.0, drop)  # 0→1: كلما أكبر = edges أقل في الأسفل = احتمال حفرة

    # ── الطريقة الثالثة: Brightness Gradient ──
    def _gradient_score(self, lower: np.ndarray) -> float:
        """
        يحسب الـ gradient الرأسي في الجزء السفلي.
        الظل العادي = تغيير ناعم.
        الحفرة = تغيير مفاجئ وحاد.
        """
        if lower.shape[0] < 4:
            return 0.0
        # قسّم المنطقة لنصين رأسيًا
        mid = lower.shape[0] // 2
        top_half = float(np.mean(lower[:mid]))
        bot_half = float(np.mean(lower[mid:]))
        gradient = abs(top_half - bot_half)
        # نطبّع على 0→1
        return min(1.0, gradient / 80.0)

    def detect(self, frame_bgr: np.ndarray, roi_contour: np.ndarray) -> Tuple[bool, float]:
        """
        إرجاع: (pit_detected, confidence_score 0→1)
        """
        if not self.enabled:
            return False, 0.0

        h, w = frame_bgr.shape[:2]

        # المنطقة اللي بنحللها:
        # Upper = الجزء من 50% → 70%  (أرض عادية كمرجع)
        # Lower = الجزء من 75% → 100% (قدام العربية)
        upper_bgr = frame_bgr[int(h * 0.50): int(h * 0.70), :]
        lower_bgr = frame_bgr[int(h * 0.75):, :]

        if upper_bgr.shape[0] < 2 or lower_bgr.shape[0] < 2:
            return False, 0.0

        upper_gray = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2GRAY)
        lower_gray = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2GRAY)

        # ── تشغيل الـ 3 طرق ──
        tex_score   = self._texture_score(upper_gray, lower_gray)
        edge_score  = self._edge_density_score(upper_gray, lower_gray)
        grad_score  = self._gradient_score(lower_gray)

        # ── تحويل لـ boolean لكل طريقة ──
        tex_flag  = tex_score  >= self._tex_diff_thresh
        edge_flag = edge_score >= self._edge_drop_thresh
        grad_flag = grad_score >= (self._grad_thresh / 80.0)

        # ── لازم اتنين من التلاتة يوافقوا ──
        votes = int(tex_flag) + int(edge_flag) + int(grad_flag)
        raw_confidence = (tex_score + edge_score + grad_score) / 3.0

        # Temporal smoothing — يمنع الـ false alarms من frame واحد
        self._pit_score_smooth = (self._smooth_alpha * raw_confidence +
                                  (1 - self._smooth_alpha) * self._pit_score_smooth)

        pit_detected = (votes >= 2) and (self._pit_score_smooth >= 0.30)
        return pit_detected, self._pit_score_smooth



class ArduinoSerial:
    def __init__(self, enabled: bool, port: str, baud: int,
                 throttle_ms: int, send_proceed: bool, rx_enabled: bool):
        self.enabled = bool(enabled)
        self.port = port
        self.baud = int(baud)
        self.throttle_ms = int(throttle_ms)
        self.send_proceed = bool(send_proceed)
        self.rx_enabled = bool(rx_enabled)
        self._ser = None
        self._last_cmd = None
        self._last_ts = 0
        self._lock = threading.Lock()
        self._rx_thread = None
        self._rx_running = False
        self._ultra_cm: Optional[float] = None
        self._ultra_ts_ms: int = 0
        if self.enabled:
            if not SERIAL_OK:
                raise RuntimeError("pyserial not installed. Run: pip install pyserial")
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(1.5)
            if self.rx_enabled:
                self._rx_running = True
                self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
                self._rx_thread.start()

    def send(self, cmd: str):
        if not self.enabled or self._ser is None:
            return
        t = now_ms()
        with self._lock:
            if self._last_cmd == cmd and (t - self._last_ts) < self.throttle_ms:
                return
            try:
                self._ser.write((cmd + "\n").encode("utf-8"))
                self._last_cmd = cmd
                self._last_ts = t
            except Exception:
                pass

    def _rx_loop(self):
        while self._rx_running:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("D:"):
                    try:
                        val = float(line[2:].strip())
                        if 0.0 < val < 500.0:
                            self._ultra_cm = val
                            self._ultra_ts_ms = now_ms()
                    except Exception:
                        pass
            except Exception:
                pass

    def get_ultrasonic_cm(self) -> Tuple[Optional[float], int]:
        if self._ultra_cm is None:
            return None, 10_000_000
        return self._ultra_cm, max(0, now_ms() - self._ultra_ts_ms)

    def close(self):
        self._rx_running = False
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass

# ------------------------------------------------------------------ #
#  Logging  (أصلي — لم يتغير)
# ------------------------------------------------------------------ #
class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._writer = None

    def log(self, row: Dict[str, Any]):
        if self._writer is None:
            self._writer = csv.DictWriter(self._f, fieldnames=list(row.keys()))
            if self._f.tell() == 0:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

# ------------------------------------------------------------------ #
#  Metrics  (أصلي — لم يتغير)
# ------------------------------------------------------------------ #
@dataclass
class MetricsState:
    frames: int = 0
    stop_events: int = 0
    stop_ultra_events: int = 0
    stop_vision_events: int = 0
    false_stops: int = 0
    last_state: str = "PROCEED"
    last_stop_cause: str = "NONE"
    obstacle_near_ts: Optional[float] = None
    reaction_samples: List[float] = field(default_factory=list)

class MetricsEngine:
    def __init__(self, false_stop_ultra_far_cm: float):
        self.m = MetricsState()
        self.false_stop_ultra_far_cm = float(false_stop_ultra_far_cm)

    def update(self, final_state: str, stop_cause: str, ultra_cm: Optional[float]):
        self.m.frames += 1
        if final_state == "STOP" and self.m.last_state != "STOP":
            self.m.stop_events += 1
            self.m.last_stop_cause = stop_cause
            if stop_cause == "ULTRA":
                self.m.stop_ultra_events += 1
            elif stop_cause == "VISION":
                self.m.stop_vision_events += 1
        if final_state == "STOP" and ultra_cm is not None and ultra_cm > self.false_stop_ultra_far_cm:
            self.m.false_stops += 1
        if ultra_cm is not None and ultra_cm < 60:
            if self.m.obstacle_near_ts is None:
                self.m.obstacle_near_ts = time.time()
        if final_state == "STOP" and self.m.obstacle_near_ts is not None:
            dt = time.time() - self.m.obstacle_near_ts
            if 0.0 < dt < 10.0:
                self.m.reaction_samples.append(dt)
            self.m.obstacle_near_ts = None
        self.m.last_state = final_state

    def report(self) -> Dict[str, Any]:
        avg_reaction = 0.0
        if self.m.reaction_samples:
            avg_reaction = sum(self.m.reaction_samples) / len(self.m.reaction_samples)
        return {
            "frames": self.m.frames,
            "stop_events": self.m.stop_events,
            "stop_ultra_events": self.m.stop_ultra_events,
            "stop_vision_events": self.m.stop_vision_events,
            "false_stops": self.m.false_stops,
            "false_stop_rate": self.m.false_stops / max(1, self.m.frames),
            "avg_reaction_s": round(avg_reaction, 3),
            "reaction_samples": len(self.m.reaction_samples),
        }

# ------------------------------------------------------------------ #
#  Vision Engine  (أصلي + تعديلات صغيرة للـ Dynamic ROI والـ Adaptive Conf)
# ------------------------------------------------------------------ #
class VisionEngine:
    """
    Pi4/Jetson optimization: YOLO runs in a background thread.
    Main loop reads latest results without ever blocking on inference.
    [جديد] يستقبل dynamic_roi و adaptive_conf
    """
    def __init__(self, cfg: Dict[str, Any],
                 dynamic_roi: "DynamicROI" = None,
                 adaptive_conf: "AdaptiveConfidence" = None):
        self.cfg = cfg
        self.dynamic_roi = dynamic_roi
        self.adaptive_conf = adaptive_conf

        y = cfg["yolo"]
        raw_path = y["model_path"]
        self.model_path = resolve_model_path(raw_path)
        self.conf = float(y["conf"])
        self.imgsz = int(y["imgsz"])
        # device من الـ config (cpu للاب، cuda للجيتسون)
        self.device = str(y.get("device", "cpu"))
        self.every_n = max(1, int(y["every_n_frames"]))

        f = cfg["frame"]
        self.resize_w = int(f["resize_w"])
        self.resize_h = int(f["resize_h"])

        r = cfg["roi"]
        self.lane_width_pct = int(r["lane_width_pct"])
        self.top_narrow_pct = int(r["top_narrow_pct"])
        self.y_top_pct = int(r["y_top_pct"])
        self.stop_line_pct = int(r["stop_line_pct"])

        flt = cfg["filters"]
        self.min_stop_box_area = int(flt["min_stop_box_area"])
        self.max_stop_box_area_ratio = float(flt["max_stop_box_area_ratio"])
        self.max_stop_box_height_ratio = float(flt["max_stop_box_height_ratio"])

        t = cfg["temporal"]
        self.stop_confirm_frames = int(t["stop_confirm_frames"])
        self.clear_frames_to_reset = int(t["clear_frames_to_reset"])

        sc = cfg["scoring"]
        self.center_weight_enabled = bool(sc["center_weight_enabled"])
        self.center_weight_power = float(sc["center_weight_power"])

        tri = cfg.get("tri_roi", {})
        self.tri_enabled = bool(tri.get("enabled", True))
        self.left_right_split_pct = int(tri.get("left_right_split_pct", 33))

        print(f"[Vision] Loading model: {self.model_path} on device: {self.device}")
        self.model = YOLO(self.model_path)
        print(f"[Vision] Model loaded.")

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._frame_idx = 0

        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._cached: List[Det] = []
        self._infer_running = True
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        self.state = "PROCEED"
        self.reason = "INIT"
        self._stop_hits = 0
        self._clear_hits = 0
        self.fps = 0.0
        self._fps_cnt = 0
        self._fps_t0 = time.time()

    def _infer_loop(self):
        while self._infer_running:
            try:
                img = self._infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # [جديد] استخدام الـ adaptive confidence لو مفعّل
            conf = self.adaptive_conf.get() if self.adaptive_conf else self.conf
            dets = self._yolo_detect(img, conf)
            with self._result_lock:
                self._cached = dets

    def _yolo_detect(self, img: np.ndarray, conf: float = None) -> List[Det]:
        if conf is None:
            conf = self.conf
        dets: List[Det] = []
        try:
            results = self.model(img, conf=conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    c = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                    cid = int(b.cls[0]) if hasattr(b, "cls") else -1
                    cname = self.model.names.get(cid, str(cid)) if hasattr(self.model, "names") else str(cid)
                    dets.append((x1, y1, x2, y2, cname, c))
        except Exception:
            pass
        return dets

    def stop(self):
        self._infer_running = False

    def get_roi_trapezoid(self, h: int, w: int) -> np.ndarray:
        lane_w = int(w * (self.lane_width_pct / 100.0))
        lane_w = max(60, min(w - 10, lane_w))
        x_left = (w - lane_w) // 2
        x_right = x_left + lane_w
        y_top = int(h * (self.y_top_pct / 100.0))
        y_top = max(10, min(h - 10, y_top))
        shrink = int(lane_w * (self.top_narrow_pct / 100.0))
        shrink = max(0, min(lane_w // 2 - 5, shrink))
        return np.array([[
            (x_left, h - 1),
            (x_left + shrink, y_top),
            (x_right - shrink, y_top),
            (x_right, h - 1)
        ]], dtype=np.int32)

    def stop_line_y(self, h: int) -> int:
        y = int(h * (self.stop_line_pct / 100.0))
        return max(10, min(h - 5, y))

    def passes_stop_filters(self, box: Box, w: int, h: int) -> Tuple[bool, str]:
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        if area < self.min_stop_box_area:
            return False, "SMALL"
        if area > int(w * h * self.max_stop_box_area_ratio):
            return False, "HUGE"
        if bh > int(h * self.max_stop_box_height_ratio):
            return False, "TALL"
        return True, "OK"

    def danger_score(self, box: Box, w: int, h: int) -> float:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        dist = abs(cx - (w / 2.0)) / (w / 2.0)
        dist = min(1.0, max(0.0, dist))
        y_score = (y2 / float(h)) if h > 0 else 0.0
        y_score = min(1.0, max(0.0, y_score))
        if not self.center_weight_enabled:
            return y_score
        center_factor = (1.0 - dist) ** self.center_weight_power
        return y_score * center_factor

    def update_fps(self):
        self._fps_cnt += 1
        dt = time.time() - self._fps_t0
        if dt >= 1.0:
            self.fps = self._fps_cnt / dt
            self._fps_cnt = 0
            self._fps_t0 = time.time()

    def tri_roi_scores(self, w: int, chosen_box: Optional[Box]) -> Dict[str, float]:
        if not self.tri_enabled:
            return {"left": 0.0, "center": 0.0, "right": 0.0}
        split_pct = max(10, min(45, self.left_right_split_pct))
        left_x2 = int(w * (split_pct / 100.0))
        right_x1 = int(w * (1.0 - split_pct / 100.0))
        scores = {"left": 0.0, "center": 0.0, "right": 0.0}
        if chosen_box is None:
            return scores
        x1, y1, x2, y2 = chosen_box
        cx = (x1 + x2) // 2
        if cx < left_x2:
            scores["left"] = 1.0
        elif cx > right_x1:
            scores["right"] = 1.0
        else:
            scores["center"] = 1.0
        return scores

    def process(self, frame_bgr: np.ndarray, flip: bool, with_view: bool) -> Dict[str, Any]:
        self._frame_idx += 1
        frame = cv2.resize(frame_bgr, (self.resize_w, self.resize_h))
        if flip:
            frame = cv2.flip(frame, 1)

        # [جديد] Adaptive confidence: حدّث قبل YOLO
        if self.adaptive_conf:
            self.adaptive_conf.update(frame)

        # [جديد] Dynamic ROI: حدّث الـ optical flow
        if self.dynamic_roi:
            self.dynamic_roi.update(frame)

        # CLAHE contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        h, w = frame.shape[:2]

        # [جديد] Dynamic ROI: طبّق الـ shift
        roi_pts = self.get_roi_trapezoid(h, w)
        if self.dynamic_roi:
            roi_pts = self.dynamic_roi.apply(roi_pts)

        roi_contour = roi_pts[0].astype(np.int32)
        stop_y = self.stop_line_y(h)

        if (self._frame_idx % self.every_n) == 0:
            try:
                self._infer_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        with self._result_lock:
            cached = list(self._cached)

        hazards = []
        for (x1, y1, x2, y2, cls, conf) in cached:
            box = clamp_box((x1, y1, x2, y2), w, h)
            hazards.append({"box": box, "cls": cls, "conf": conf})

        found_stop = False
        chosen = None
        chosen_score = -1.0
        chosen_filter = "NONE"

        for hz in hazards:
            box = hz["box"]
            if not inside_roi_by_3points(roi_contour, box):
                continue
            x1, y1, x2, y2 = box
            if y2 < stop_y:
                continue
            ok, why = self.passes_stop_filters(box, w, h)
            if not ok:
                continue
            score = self.danger_score(box, w, h)
            found_stop = True
            if chosen is None or score > chosen_score:
                chosen = (x1, y1, x2, y2, hz["cls"], hz["conf"])
                chosen_score = score
                chosen_filter = why

        if found_stop:
            self._stop_hits += 1
            self._clear_hits = 0
            self.reason = "STOP_CANDIDATE"
        else:
            self._clear_hits += 1
            self._stop_hits = max(0, self._stop_hits - 1)
            self.reason = "CLEAR"

        if self._stop_hits >= self.stop_confirm_frames:
            self.state = "STOP"
        elif self.state == "STOP":
            if self._clear_hits >= self.clear_frames_to_reset:
                self.state = "PROCEED"
        else:
            if self._clear_hits >= self.clear_frames_to_reset:
                self.state = "PROCEED"

        self.update_fps()

        chosen_box = None
        chosen_cls = ""
        chosen_conf = 0.0
        if chosen is not None:
            x1, y1, x2, y2, cls, conf = chosen
            chosen_box = (x1, y1, x2, y2)
            chosen_cls = cls
            chosen_conf = float(conf)

        tri_scores = self.tri_roi_scores(w, chosen_box)

        out = {
            "state": self.state,
            "reason": self.reason,
            "stop_hits": self._stop_hits,
            "clear_hits": self._clear_hits,
            "fps": self.fps,
            "hazards": len(hazards),
            "hazards_list": hazards,      # [جديد] للـ MultiObstaclePlanner
            "stop_line_y": stop_y,
            "roi_pts": roi_pts,
            "chosen_box": chosen_box,
            "chosen_cls": chosen_cls,
            "chosen_conf": chosen_conf,
            "chosen_score": chosen_score,
            "chosen_filter": chosen_filter,
            "tri_left": tri_scores["left"],
            "tri_center": tri_scores["center"],
            "tri_right": tri_scores["right"],
            "adaptive_conf": self.adaptive_conf.get() if self.adaptive_conf else self.conf,
        }

        out["frame_idx"] = self._frame_idx

        if with_view:
            view = frame.copy()
            cv2.polylines(view, roi_pts, True, (255, 255, 255), 2)
            cv2.line(view, (0, stop_y), (w, stop_y), (255, 255, 255), 1)
            for hz in hazards:
                box = hz["box"]
                if not inside_roi_by_3points(roi_contour, box):
                    continue
                x1, y1, x2, y2 = box
                cv2.rectangle(view, (x1, y1), (x2, y2), (160, 160, 160), 1)
                cv2.putText(view, f"{hz['cls'][:10]} {hz['conf']:.2f}",
                            (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 2)
            if chosen_box is not None:
                x1, y1, x2, y2 = chosen_box
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(view, f"CHOSEN {chosen_cls} {chosen_conf:.2f}",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(view, (0, 0), (w, 175), (30, 30, 30), -1)
            col = (0, 0, 255) if self.state == "STOP" else (0, 255, 0)
            cv2.putText(view, f"STATE: {self.state}", (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)
            cv2.putText(view, f"hits={self._stop_hits}/{self.stop_confirm_frames} clear={self._clear_hits}/{self.clear_frames_to_reset}",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(view, f"hazards={len(hazards)} score={chosen_score:.3f} filter={chosen_filter}",
                        (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(view, f"tri(L/C/R)={tri_scores['left']:.0f}/{tri_scores['center']:.0f}/{tri_scores['right']:.0f}",
                        (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            # [جديد] عرض الـ adaptive confidence والـ ROI shift
            if self.adaptive_conf:
                cv2.putText(view, f"conf={self.adaptive_conf.get():.2f}",
                            (12, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 0), 1)
            if self.dynamic_roi:
                sx, sy = self.dynamic_roi.get_shift()
                cv2.putText(view, f"roi_shift=({sx:.1f},{sy:.1f})",
                            (12, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 0), 1)
            cv2.putText(view, f"FPS: {self.fps:.1f}", (w - 120, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            out["view"] = view
        return out

# ------------------------------------------------------------------ #
#  Video source  (أصلي — لم يتغير)
# ------------------------------------------------------------------ #
def open_capture(source: str) -> cv2.VideoCapture:
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)

# ------------------------------------------------------------------ #
#  App  (أصلي + تكامل الإضافات الجديدة)
# ------------------------------------------------------------------ #
class App:
    def __init__(self, cfg: Dict[str, Any], headless: bool, gui: bool):
        self.cfg = cfg
        self.headless = bool(headless)
        self.gui = bool(gui) and (not self.headless)
        self.running = True
        self.flip = bool(cfg.get("flip", False))

        # ── [جديد] إنشاء الإضافات الجديدة أولاً ──
        dyn_cfg = cfg.get("dynamic_roi", {})
        self.dynamic_roi = DynamicROI(
            enabled=bool(dyn_cfg.get("enabled", False)),  # مغلول افتراضياً — ثقيل على CPU
            max_shift_px=int(dyn_cfg.get("max_shift_px", 40))
        )

        ac_cfg = cfg.get("adaptive_conf", {})
        self.adaptive_conf = AdaptiveConfidence(
            enabled=bool(ac_cfg.get("enabled", True)),
            base_conf=float(cfg["yolo"]["conf"]),
            min_conf=float(ac_cfg.get("min_conf", 0.35)),
            max_conf=float(ac_cfg.get("max_conf", 0.80)),
            update_every=int(ac_cfg.get("update_every_frames", 30))
        )

        # ── Vision Engine (مع تمرير الإضافات) ──
        self.engine = VisionEngine(cfg,
                                   dynamic_roi=self.dynamic_roi,
                                   adaptive_conf=self.adaptive_conf)

        # ── Serial ──
        s = cfg.get("serial", {})
        self.serial = ArduinoSerial(
            enabled=s.get("enabled", False),
            port=s.get("port", "/dev/ttyACM0"),
            baud=s.get("baud", 115200),
            throttle_ms=s.get("throttle_ms", 60),
            send_proceed=s.get("send_proceed", True),
            rx_enabled=s.get("rx_enabled", True),
        )

        # ── [جديد] Watchdog (بعد Serial عشان يقدر يبعت STOP) ──
        wd_cfg = cfg.get("watchdog", {})
        self.watchdog = Watchdog(
            serial_ref=self.serial,
            timeout_s=float(wd_cfg.get("timeout_s", 1.0))
        )

        # ── [جديد] Sign Detector ──
        sign_cfg = cfg.get("sign_detector", {})
        self.sign_detector = SignDetector(
            enabled=bool(sign_cfg.get("enabled", True)),
            min_sign_area=int(sign_cfg.get("min_sign_area", 800))
        )

        # ── [جديد] Pit Detector ──
        pit_cfg = cfg.get("pit_detector", {})
        self.pit_detector = PitDetector(
            enabled=bool(pit_cfg.get("enabled", True)),
            dark_threshold=int(pit_cfg.get("dark_threshold", 40)),
            min_pit_area_pct=float(pit_cfg.get("min_pit_area_pct", 0.08))
        )

        # ── Control ──
        ctrl = cfg.get("control", {})
        self.auto_enabled = bool(ctrl.get("auto_enabled", True))
        self.manual_mode_passthrough = bool(ctrl.get("manual_mode_passthrough", False))
        self.default_speed = int(ctrl.get("default_speed", 120))

        # ── [جديد] Ultrasonic Multi-Zone ──
        ultra = cfg.get("ultrasonic", {})
        self.ultra_zone = UltrasonicZone(
            enabled=bool(ultra.get("enabled", False)),
            danger_cm=float(ultra.get("emergency_stop_cm", 25.0)),
            warning_cm=float(ultra.get("warning_cm", 60.0)),
            max_age_ms=int(ultra.get("max_age_ms", 600)),
            fail_safe=bool(ultra.get("fail_safe_stop_if_stale", False))
        )
        # للتوافق مع الكود الأصلي
        self.ultra_enabled = bool(ultra.get("enabled", False))
        self.ultra_emergency_cm = float(ultra.get("emergency_stop_cm", 25.0))

        # ── [جديد] Multi-Obstacle Planner ──
        av = cfg.get("avoidance", {})
        self.avoid_enabled = bool(av.get("enabled", True))
        self.avoid_planner = MultiObstaclePlanner(prefer=str(av.get("prefer", "AUTO")))
        self.avoid_cooldown_ms = int(av.get("cooldown_ms", 250))
        self._avoid_last_ts = 0

        # ── Metrics ──
        met = cfg.get("metrics", {})
        self.metrics = MetricsEngine(
            false_stop_ultra_far_cm=float(met.get("false_stop_ultra_far_cm", 200.0)))

        # ── Logging ──
        lg = cfg.get("logging", {})
        self.logger = None
        self.save_snaps = bool(lg.get("save_snapshots", False))
        self.snap_dir = str(lg.get("snapshots_dir", "logs/snaps"))
        self.snap_every_n = int(lg.get("snapshot_every_n", 30))
        self._snap_i = 0
        if bool(lg.get("enabled", True)):
            self.logger = CSVLogger(str(lg.get("csv_path", "logs/run_log.csv")))

        self._sent_stop_before = False
        self._last_snap_was_stop = False
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        self.running = False

    def maybe_snapshot(self, view: np.ndarray, state: str, cause: str):
        """
        [محسّن] يحفظ صورة في حالتين:
        1. لما يكتشف عائق (STOP) — دليل بصري
        2. كل snap_every_n frame — تصوير المسار كامل
        """
        os.makedirs(self.snap_dir, exist_ok=True)
        ts = int(time.time() * 1000)

        # دايماً احفظ لما يكتشف عائق جديد
        if state == "STOP" and cause != "NONE" and not self._last_snap_was_stop:
            path = os.path.join(self.snap_dir, f"{ts}_OBSTACLE_{cause}.jpg")
            cv2.imwrite(path, view)
            self._last_snap_was_stop = True
            print(f"[Snapshot] عائق محفوظ: {path}")
            return

        if state != "STOP":
            self._last_snap_was_stop = False

        # تصوير المسار كل snap_every_n frame
        if not self.save_snaps:
            return
        self._snap_i += 1
        if self._snap_i % self.snap_every_n != 0:
            return
        path = os.path.join(self.snap_dir, f"{ts}_PATH_{state}.jpg")
        cv2.imwrite(path, view)

    def detect_environment(self, frame: np.ndarray) -> str:
        """
        كشف البيئة من الألوان:
        - صحراء / خارجي / داخلي
        بيحسب نسبة الألوان الدافئة (رمال/تراب) vs الباردة (جدران/سماء)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # صحراء: ألوان دافئة (أصفر/بني/برتقالي)
        desert_lower = np.array([10, 30, 80])
        desert_upper = np.array([35, 180, 255])
        desert_mask = cv2.inRange(hsv, desert_lower, desert_upper)

        # داخلي: ألوان محايدة (رمادي/أبيض)
        indoor_lower = np.array([0, 0, 100])
        indoor_upper = np.array([180, 40, 255])
        indoor_mask = cv2.inRange(hsv, indoor_lower, indoor_upper)

        # سماء/خارجي: ألوان زرقاء
        outdoor_lower = np.array([90, 40, 80])
        outdoor_upper = np.array([130, 255, 255])
        outdoor_mask = cv2.inRange(hsv, outdoor_lower, outdoor_upper)

        total = frame.shape[0] * frame.shape[1]
        desert_pct  = cv2.countNonZero(desert_mask)  / total
        indoor_pct  = cv2.countNonZero(indoor_mask)  / total
        outdoor_pct = cv2.countNonZero(outdoor_mask) / total

        if desert_pct > 0.25:
            return "DESERT"
        elif outdoor_pct > 0.20:
            return "OUTDOOR"
        elif indoor_pct > 0.30:
            return "INDOOR"
        else:
            return "UNKNOWN"

    def send_state_control(self, final_state: str, stop_cause: str = "NONE",
                           motion_cmd: str = "F", vision_out: Dict = None):
        """
        [محدّث] بروتوكول كامل مع الأردوينو:
          STOP + PIT   → 'P'
          STOP + SIGN  → 'G'
          STOP + يمين → 'OR'
          STOP + شمال → 'OL'
          STOP عادي   → 'O'
          PROCEED      → 'C'
        """
        if final_state == "STOP":
            if stop_cause == "PIT":
                self.serial.send("P")
            elif stop_cause == "SIGN":
                self.serial.send("G")
            elif motion_cmd == "R":
                self.serial.send("OR")  # عائق على اليمين
            elif motion_cmd == "L":
                self.serial.send("OL")  # عائق على اليسار
            else:
                self.serial.send("O")   # عائق عادي
            self._sent_stop_before = True
        else:
            if self._sent_stop_before and self.serial.send_proceed:
                self.serial.send("C")   # كمّل Zigzag
                self._sent_stop_before = False

    def send_motion_cmd(self, cmd: str):
        self.serial.send(cmd)

    def fusion_and_plan(self, vision_out: Dict[str, Any]) -> Tuple[str, str, Optional[float], int, str]:
        """
        [جديد] يستخدم UltrasonicZone بدل الـ fusion البسيط
        إرجاع: (final_state, cause, ultra_cm, ultra_age, ultra_zone)
        """
        vision_state = vision_out["state"]
        ultra_cm, ultra_age = self.serial.get_ultrasonic_cm()

        final_state, cause, zone = self.ultra_zone.evaluate(
            ultra_cm, ultra_age, vision_state)

        return final_state, cause, ultra_cm, ultra_age, zone

    def decide_motion(self, final_state: str, vision_out: Dict[str, Any],
                      ultra_zone: str = "SAFE") -> str:
        """
        [جديد] يستخدم MultiObstaclePlanner
        + يخفف السرعة في WARNING zone
        """
        if final_state == "STOP":
            return "S"
        if not self.auto_enabled:
            return "F"
        if not self.avoid_enabled:
            return "F"

        t = now_ms()
        if (t - self._avoid_last_ts) < self.avoid_cooldown_ms:
            return "F"

        # [جديد] في WARNING zone → ابعت أمر تبطيء للأردوينو
        if ultra_zone == "WARNING":
            self.serial.send("W")  # W = warning/slow down

        hazards = vision_out.get("hazards_list", [])
        w = self.engine.resize_w
        h = self.engine.resize_h

        cmd = self.avoid_planner.decide(
            hazards=hazards,
            w=w, h=h,
            tri_left=float(vision_out.get("tri_left", 0.0)),
            tri_center=float(vision_out.get("tri_center", 0.0)),
            tri_right=float(vision_out.get("tri_right", 0.0))
        )

        if cmd in ("L", "R"):
            self._avoid_last_ts = t
        return cmd

    def run(self) -> int:
        source = str(self.cfg.get("source", "0"))
        cap = open_capture(source)
        if not cap.isOpened():
            print("ERROR: Cannot open source:", source)
            self.serial.send("S")
            return 2

        window = "Robot Vision V4 — Enhanced"
        self._win_w, self._win_h = 1280, 720
        if self.gui:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            try:
                import tkinter as _tk
                _root = _tk.Tk(); _root.withdraw()
                self._win_w = _root.winfo_screenwidth()
                self._win_h = _root.winfo_screenheight()
                _root.destroy()
                cv2.resizeWindow(window, self._win_w, self._win_h)
                cv2.moveWindow(window, 0, 0)
            except Exception:
                cv2.resizeWindow(window, self._win_w, self._win_h)
            print("GUI keys: Q/Esc=quit | F=flip | +/-=conf")

        last_ok = time.time()

        print("\n=== RUN START (V4 Enhanced) ===")
        print(f"Device: {self.engine.device} | Source: {source} | Flip: {self.flip}")
        print(f"Serial: {self.serial.enabled} | Ultrasonic: {self.ultra_zone.enabled}")
        print(f"Watchdog: ON (timeout={self.cfg.get('watchdog',{}).get('timeout_s',1.0)}s)")
        print(f"DynamicROI: {self.dynamic_roi.enabled} | AdaptiveConf: {self.adaptive_conf.enabled}")
        print(f"MultiObstacle planner: ON | UltraZone: danger={self.ultra_zone.danger_cm}cm warning={self.ultra_zone.warning_cm}cm")
        print("=================================\n")

        self.serial.send(f"V{self.default_speed}")
        self.serial.send("M1" if self.auto_enabled else "M0")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.serial.send("S")
                time.sleep(0.05)
                if not source.isdigit():
                    break
                continue

            last_ok = time.time()

            # [جديد] Watchdog ping — كل frame
            self.watchdog.ping()

            try:
                out = self.engine.process(frame, flip=self.flip, with_view=self.gui)
            except Exception as e:
                print("Vision error:", e)
                self.serial.send("S")
                continue

            # [جديد] fusion مع UltrasonicZone
            final_state, stop_cause, ultra_cm, ultra_age, ultra_zone = self.fusion_and_plan(out)

            # [جديد] كشف الحفرة — يـ override الـ state لو في حفرة
            pit_detected, pit_ratio = self.pit_detector.detect(
                frame, out.get("roi_pts", np.array([[[0,0]]])))
            if pit_detected and final_state != "STOP":
                final_state = "STOP"
                stop_cause = "PIT"
                print(f"[PitDetector] ⚠️ حفرة مكتشفة! نسبة={pit_ratio:.2f}")

            # [جديد] كشف لافتات التحذير
            sign_found, signs = self.sign_detector.detect(frame)
            if sign_found and final_state != "STOP":
                final_state = "STOP"
                stop_cause = "SIGN"
                for s in signs:
                    print(f"[SignDetector] ⚠️ لافتة: {s['color']} {s['shape']}")
            self.metrics.update(final_state, stop_cause, ultra_cm)

            # [جديد] decide_motion مع MultiObstaclePlanner
            motion_cmd = self.decide_motion(final_state, out, ultra_zone)

            self.send_state_control(final_state, stop_cause, motion_cmd, out)
            self.send_motion_cmd(motion_cmd)

            if self.logger:
                row = {
                    "t": time.time(),
                    "vision_state": out["state"],
                    "final_state": final_state,
                    "stop_cause": stop_cause,
                    "motion_cmd": motion_cmd,
                    "ultra_cm": "" if ultra_cm is None else round(float(ultra_cm), 2),
                    "ultra_age_ms": ultra_age,
                    "ultra_zone": ultra_zone,           # [جديد]
                    "adaptive_conf": round(out.get("adaptive_conf", self.engine.conf), 3),  # [جديد]
                    "reason": out["reason"],
                    "fps": round(out["fps"], 2),
                    "hazards": out["hazards"],
                    "stop_hits": out["stop_hits"],
                    "clear_hits": out["clear_hits"],
                    "stop_line_y": out["stop_line_y"],
                    "chosen_score": round(out["chosen_score"], 4),
                    "chosen_filter": out["chosen_filter"],
                    "chosen_cls": out["chosen_cls"],
                    "chosen_conf": round(out["chosen_conf"], 3),
                    "tri_left": out["tri_left"],
                    "tri_center": out["tri_center"],
                    "tri_right": out["tri_right"],
                }
                cb = out["chosen_box"]
                if cb is None:
                    row.update({"x1": "", "y1": "", "x2": "", "y2": ""})
                else:
                    x1, y1, x2, y2 = cb
                    row.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                self.logger.log(row)

            if self.gui:
                view = out.get("view")
                if view is not None:
                    h_v, w_v = view.shape[:2]
                    # [جديد] لون الـ zone في الـ ultra display
                    zone_color = (0,0,255) if ultra_zone=="DANGER" else \
                                 (0,165,255) if ultra_zone=="WARNING" else (200,200,200)
                    cv2.putText(view,
                                f"FINAL: {final_state}  cause={stop_cause}  cmd={motion_cmd}",
                                (12, h_v - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

                    # كشف البيئة كل 30 frame
                    if out.get("frame_idx", 0) % 30 == 0:
                        env = self.detect_environment(view)
                        self._last_env = env
                    env_colors = {
                        "DESERT":  (0, 165, 255),
                        "OUTDOOR": (0, 255, 100),
                        "INDOOR":  (255, 200, 0),
                        "UNKNOWN": (150, 150, 150)
                    }
                    env_label = getattr(self, "_last_env", "UNKNOWN")
                    # ENV تحت FPS (كان على نفس السطر — نزّلناها)
                    cv2.putText(view, f"ENV: {env_label}",
                                (w_v - 160, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, env_colors.get(env_label, (200,200,200)), 2)
                    if ultra_cm is not None:
                        cv2.putText(view,
                                    f"ULTRA: {ultra_cm:.1f}cm  zone={ultra_zone}  age={ultra_age}ms",
                                    (12, h_v - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 1)
                    else:
                        cv2.putText(view, f"ULTRA: None  zone={ultra_zone}",
                                    (12, h_v - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 1)

                    # [جديد] عرض كشف الحفرة
                    if pit_detected:
                        cv2.putText(view, f"⚠️ PIT DETECTED! ratio={pit_ratio:.2f}",
                                    (w_v//2 - 150, h_v//2),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

                    # لافتات التحذير معطلة (sign_detector.enabled=false في الكونفيج)

                    # كبّر الصورة عشان تملأ النافذة كاملة
                    try:
                        disp = cv2.resize(view, (self._win_w, self._win_h),
                                          interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        disp = view
                    cv2.imshow(window, disp)
                    self.maybe_snapshot(view, final_state, stop_cause)

                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("f"):
                    self.flip = not self.flip
                elif k == ord("+"):
                    self.engine.conf = min(0.95, self.engine.conf + 0.05)
                elif k == ord("-"):
                    self.engine.conf = max(0.10, self.engine.conf - 0.05)

            if source.isdigit() and (time.time() - last_ok) > 2.0:
                self.serial.send("S")
                time.sleep(0.1)

        # Cleanup
        cap.release()
        self.engine.stop()
        self.watchdog.stop()
        if self.gui:
            cv2.destroyAllWindows()
        self.serial.send("S")
        if self.logger:
            self.logger.close()
        self.serial.close()
        print("=== RUN END ===")
        print("Metrics:", self.metrics.report())
        return 0

# ------------------------------------------------------------------ #
#  CLI  (أصلي — لم يتغير)
# ------------------------------------------------------------------ #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config json")
    ap.add_argument("--headless", type=int, default=0,
                    help="0=show window, 1=headless (للجيتسون بدون شاشة)")
    ap.add_argument("--gui", type=int, default=1,
                    help="1=show window (افتراضي على اللاب)")
    ap.add_argument("--set", type=str, action="append", default=[],
                    help="Override config key=value (dotted paths)")
    ap.add_argument("--export-trt", action="store_true",
                    help="تحويل النموذج لـ TensorRT (مرة واحدة على Jetson)")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_json(args.config)
    for s in args.set:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        set_by_path(cfg, k.strip(), parse_value(v))
    if args.export_trt:
        model_path = cfg["yolo"]["model_path"]
        export_tensorrt(model_path)
        return 0
    if args.headless:
        args.gui = 0
    app = App(cfg, headless=bool(args.headless), gui=bool(args.gui))
    return app.run()

if __name__ == "__main__":
    sys.exit(main())
