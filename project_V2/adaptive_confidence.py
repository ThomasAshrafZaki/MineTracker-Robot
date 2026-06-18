"""
================================================================================
  adaptive_confidence.py  —  Dynamic YOLO Confidence Adjustment
================================================================================
  - بيرصد brightness الصورة كل N frame
  - صورة داكنة (غبار/ظل/ليل) → confidence أقل = أكثر حساسية
  - صورة فاتحة جداً (شمس حادة) → confidence أعلى = أقل false alarms
  - Smoothing عشان ما يتقفزش فجأة

  Author  : Robot Team
  Version : 1.0
================================================================================
"""

import cv2
import numpy as np
import logging

log = logging.getLogger("AdaptiveConf")


class AdaptiveConfidence:
    """
    يعدل YOLO confidence تلقائياً حسب إضاءة الصورة.

    المنطق:
        brightness < 0.20 (داكن جداً)  → min_conf (أكثر حساسية)
        brightness ≈ 0.50 (طبيعي)      → ~base_conf
        brightness > 0.80 (فاتح جداً)  → max_conf (أقل false alarms)

    الاستخدام:
        ac = AdaptiveConfidence(base_conf=0.45)
        # في كل frame:
        current_conf = ac.update(frame)
        # أو اقرأ القيمة الحالية بدون تحديث:
        current_conf = ac.get()
    """

    def __init__(
        self,
        base_conf:    float = 0.45,
        min_conf:     float = 0.30,
        max_conf:     float = 0.75,
        update_every: int   = 30,     # حدّث كل N frame (توفير CPU)
        smooth_alpha: float = 0.25,   # كلما صغر = أبطأ وأثبت
        enabled:      bool  = True,
    ):
        self._base         = base_conf
        self._min          = min_conf
        self._max          = max_conf
        self._update_every = update_every
        self._alpha        = smooth_alpha
        self._enabled      = enabled

        self._current         = base_conf
        self._frame_count     = 0
        self._last_brightness = 0.5

        log.info(
            f"AdaptiveConfidence enabled={enabled} "
            f"base={base_conf} min={min_conf} max={max_conf}"
        )

    # ── PUBLIC API ──────────────────────────────

    def update(self, frame_bgr: np.ndarray) -> float:
        """
        أعطيها الـ frame — بترجع الـ confidence الحالي.
        التحديث بيحصل كل update_every frame بس (توفير CPU).
        """
        if not self._enabled:
            return self._current

        self._frame_count += 1
        if self._frame_count % self._update_every != 0:
            return self._current

        # حساب brightness
        gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        self._last_brightness = brightness

        # linear interpolation بين min و max
        if brightness < 0.20:
            target = self._min
        elif brightness > 0.80:
            target = self._max
        else:
            ratio  = (brightness - 0.20) / 0.60
            target = self._min + ratio * (self._max - self._min)

        # exponential smoothing
        self._current = (
            self._alpha * target
            + (1.0 - self._alpha) * self._current
        )
        self._current = max(self._min, min(self._max, self._current))

        log.debug(
            f"brightness={brightness:.2f} "
            f"target={target:.2f} conf={self._current:.2f}"
        )
        return self._current

    def get(self) -> float:
        """الـ confidence الحالي بدون تحديث."""
        return self._current

    def reset(self):
        """ارجع للـ base confidence."""
        self._current = self._base

    @property
    def brightness(self) -> float:
        """آخر brightness محسوب."""
        return self._last_brightness

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __repr__(self):
        return (
            f"<AdaptiveConf conf={self._current:.2f} "
            f"brightness={self._last_brightness:.2f} "
            f"enabled={self._enabled}>"
        )


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    )

    print("TEST 1: Dark frame → low confidence")
    ac = AdaptiveConfidence(base_conf=0.45, min_conf=0.30, max_conf=0.75,
                            update_every=1, smooth_alpha=1.0)
    dark = np.zeros((480, 640, 3), dtype=np.uint8)
    dark[:] = 30
    for _ in range(3):
        ac.update(dark)
    assert ac.get() <= 0.32, f"Expected ~0.30 got {ac.get():.3f}"
    print(f"  ✅ dark conf={ac.get():.3f}")

    print("TEST 2: Bright frame → high confidence")
    ac2 = AdaptiveConfidence(base_conf=0.45, min_conf=0.30, max_conf=0.75,
                             update_every=1, smooth_alpha=1.0)
    bright = np.zeros((480, 640, 3), dtype=np.uint8)
    bright[:] = 230
    for _ in range(3):
        ac2.update(bright)
    assert ac2.get() >= 0.73, f"Expected ~0.75 got {ac2.get():.3f}"
    print(f"  ✅ bright conf={ac2.get():.3f}")

    print("TEST 3: Normal frame → mid confidence")
    ac3 = AdaptiveConfidence(base_conf=0.45, min_conf=0.30, max_conf=0.75,
                             update_every=1, smooth_alpha=1.0)
    normal = np.zeros((480, 640, 3), dtype=np.uint8)
    normal[:] = 128
    for _ in range(3):
        ac3.update(normal)
    assert 0.40 <= ac3.get() <= 0.60, f"Expected ~0.525 got {ac3.get():.3f}"
    print(f"  ✅ normal conf={ac3.get():.3f}")

    print("TEST 4: disabled → always returns base")
    ac4 = AdaptiveConfidence(base_conf=0.45, enabled=False)
    dark2 = np.zeros((480, 640, 3), dtype=np.uint8)
    ac4.update(dark2)
    assert ac4.get() == 0.45, f"Expected 0.45 got {ac4.get()}"
    print(f"  ✅ disabled conf={ac4.get():.3f}")

    print("TEST 5: smoothing — gradual change")
    ac5 = AdaptiveConfidence(base_conf=0.45, min_conf=0.30, max_conf=0.75,
                             update_every=1, smooth_alpha=0.3)
    bright5 = np.zeros((480, 640, 3), dtype=np.uint8)
    bright5[:] = 230
    prev = ac5.get()
    for i in range(10):
        ac5.update(bright5)
        assert ac5.get() >= prev, "Confidence should monotonically increase"
        prev = ac5.get()
    print(f"  ✅ smooth increase: {ac5.get():.3f}")

    print("\n✅ All AdaptiveConfidence tests passed")
