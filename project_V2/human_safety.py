"""
================================================================================
  human_safety.py  —  Human Detection Safety Monitor
================================================================================
  المهمة الوحيدة لهذا الملف:
      - يراقب الـ CV باستمرار
      - لو YOLO اكتشف إنسان → يوقف الروبوت فوراً + يحفظ صورة
      - يفضل واقف لحد ما الإنسان يمشي
      - بعد HUMAN_CLEAR_DELAY_S ثواني من اختفاء الإنسان → يكمل

  مستقل تماماً عن decision_engine.py
  مش بيتدخل في الاتجاهات خالص

  Author  : Robot Team
  Version : 1.0
================================================================================
"""

import os
import threading
import time
import logging

from dataclasses import dataclass, field
from typing import Optional, Callable

from vision_processor import VisionProcessor, DetectionResult
from arduino_bridge import ArduinoBridge

log = logging.getLogger("HumanSafety")

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────

# الكلاس بتاع الإنسان في YOLO
HUMAN_CLASS = "person"

# بعد كام ثانية من اختفاء الإنسان نكمل
HUMAN_CLEAR_DELAY_S = 3.0

# minimum confidence عشان نعتبره إنسان حقيقي
HUMAN_CONF_MIN = 0.75


# كام مرة في الثانية نفحص
MONITOR_HZ = 10


# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class HumanSafetyState:
    CLEAR   = "CLEAR"    # مفيش إنسان، الروبوت يكمل
    BLOCKED = "BLOCKED"  # في إنسان، الروبوت واقف
    WAITING = "WAITING"  # الإنسان اختفى، بنستنى HUMAN_CLEAR_DELAY_S


@dataclass
class HumanEvent:
    """تسجيل كل حدث كشف إنسان."""
    timestamp:   float = field(default_factory=time.time)
    confidence:  float = 0.0
    duration_s:  float = 0.0   # كام ثانية وقف الروبوت


# ──────────────────────────────────────────────
#  MAIN CLASS
# ──────────────────────────────────────────────

class HumanSafetyMonitor:
    """
    يراقب الكاميرا باستمرار.
    لو شاف إنسان → يوقف الأردوينو + يحفظ صورة.
    لو الإنسان مشي → يستنى 3 ثواني → يكمل.
    """

    def __init__(
        self,
        vision:  VisionProcessor,
        bridge:  ArduinoBridge,
        on_human_detected: Optional[Callable[[HumanEvent], None]] = None,
        on_human_cleared:  Optional[Callable[[HumanEvent], None]] = None,
    ):
        self.vision  = vision
        self.bridge  = bridge

        self._on_detected = on_human_detected
        self._on_cleared  = on_human_cleared

        self._state         = HumanSafetyState.CLEAR
        self._state_lock    = threading.Lock()

        self._running       = False
        self._thread: Optional[threading.Thread] = None

        # وقت آخر مرة شفنا إنسان
        self._last_human_time: float = 0.0

        # وقت ما وقفنا بسبب إنسان (للـ duration)
        self._blocked_since:   float = 0.0

        # آخر event لو محتاجين callback
        self._current_event: Optional[HumanEvent] = None

        # إحصائيات
        self._total_detections = 0
        self._total_stops      = 0



    # ──────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="HumanSafety"
        )
        self._thread.start()
        log.info("HumanSafetyMonitor started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("HumanSafetyMonitor stopped")

    def get_state(self) -> str:
        with self._state_lock:
            return self._state

    def is_safe(self) -> bool:
        """True لو المنطقة آمنة والروبوت يقدر يكمل."""
        return self.get_state() == HumanSafetyState.CLEAR

    def stats(self) -> dict:
        return {
            "state":             self.get_state(),
            "total_detections":  self._total_detections,
            "total_stops":       self._total_stops,
        }

    # ──────────────────────────────────────────
    #  MONITOR LOOP
    # ──────────────────────────────────────────

    def _monitor_loop(self):
        interval = 1.0 / MONITOR_HZ

        while self._running:
            t0 = time.time()

            try:
                self._tick()
            except Exception as e:
                log.error(f"HumanSafety error: {e}", exc_info=True)

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _tick(self):
        result = self.vision.get_latest()
        human_now = self._is_human(result)

        with self._state_lock:
            state = self._state

        if state == HumanSafetyState.CLEAR:
            if human_now:
                self._on_human_appeared(result)

        elif state == HumanSafetyState.BLOCKED:
            if human_now:
                # لسه موجود، نفضل واقفين
                self._last_human_time = time.time()
                self._send_stop()
            else:
                # اختفى، نبدأ نعد
                log.info("Human left frame → starting CLEAR countdown")
                with self._state_lock:
                    self._state = HumanSafetyState.WAITING

        elif state == HumanSafetyState.WAITING:
            if human_now:
                # رجع تاني، نرجع BLOCKED
                log.info("Human reappeared → back to BLOCKED")
                self._last_human_time = time.time()
                with self._state_lock:
                    self._state = HumanSafetyState.BLOCKED
                self._send_stop()
            else:
                # نفحص لو عدى HUMAN_CLEAR_DELAY_S
                waited = time.time() - self._last_human_time
                if waited >= HUMAN_CLEAR_DELAY_S:
                    self._on_area_cleared()

    # ──────────────────────────────────────────
    #  DETECTION LOGIC
    # ──────────────────────────────────────────

    def _is_human(self, result: DetectionResult) -> bool:
        """
        True لو الـ detection ده إنسان بـ confidence كافي.
        """
        if not result.obstacle_found():
            return False
        if result.label.lower() != HUMAN_CLASS:
            return False
        if result.confidence < HUMAN_CONF_MIN:
            return False
        # نتأكد إن الـ detection مش قديم
        if not result.is_valid(max_age=0.5):
            return False
        return True

    # ──────────────────────────────────────────
    #  STATE TRANSITIONS
    # ──────────────────────────────────────────

    def _on_human_appeared(self, result: DetectionResult):
        log.warning(
            f"HUMAN DETECTED | conf={result.confidence:.0%} "
            f"label={result.label}"
        )

        self._total_detections += 1
        self._total_stops      += 1
        self._last_human_time   = time.time()
        self._blocked_since     = time.time()

        with self._state_lock:
            self._state = HumanSafetyState.BLOCKED

        # وقف الروبوت فوراً
        self._send_stop()

        self._current_event = HumanEvent(confidence=result.confidence)

        if self._on_detected and self._current_event:
            try:
                self._on_detected(self._current_event)
            except Exception as e:
                log.error(f"on_human_detected callback error: {e}")

    def _on_area_cleared(self):
        duration = time.time() - self._blocked_since

        log.info(f"Area CLEAR | robot was stopped for {duration:.1f}s")

        if self._current_event:
            self._current_event.duration_s = duration

        with self._state_lock:
            self._state = HumanSafetyState.CLEAR

        if self._on_cleared and self._current_event:
            try:
                self._on_cleared(self._current_event)
            except Exception as e:
                log.error(f"on_human_cleared callback error: {e}")

        self._current_event = None

    # ──────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────

    def _send_stop(self):
        """يبعت STOP للأردوينو."""
        try:
            self.bridge.send_stop()
        except Exception as e:
            log.error(f"Failed to send STOP: {e}")




# ──────────────────────────────────────────────
#  INTEGRATION مع decision_engine
# ──────────────────────────────────────────────
#
#  في main.py بتاعك:
#
#  from human_safety import HumanSafetyMonitor
#
#  safety = HumanSafetyMonitor(vision, bridge)
#  safety.start()
#
#  وفي loop بتاعت الـ decision_engine، قبل ما تبعت FORWARD:
#
#  if not safety.is_safe():
#      bridge.send_stop()
#      continue
#
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    )

    from arduino_bridge import create_bridge

    print("=" * 60)
    print("  HumanSafetyMonitor — Simulation Test")
    print("=" * 60)

    bridge = create_bridge(simulate=True)
    vision = VisionProcessor(simulate=True, use_yolo=False)

    bridge.start()
    vision.start()

    def on_detected(event: HumanEvent):
        print(f"\n  ⚠️  HUMAN DETECTED | conf={event.confidence:.0%}")

    def on_cleared(event: HumanEvent):
        print(f"\n  ✅ AREA CLEAR | stopped for {event.duration_s:.1f}s")

    safety = HumanSafetyMonitor(
        vision, bridge,
        on_human_detected=on_detected,
        on_human_cleared=on_cleared,
    )
    safety.start()

    # Scenario 1: مفيش إنسان
    print("\n[SIM] Scenario 1: No human (5s)")
    time.sleep(5)
    print(f"  State: {safety.get_state()}")
    assert safety.is_safe(), "Should be CLEAR"
    print("  ✅ CLEAR — robot can move")

    # Scenario 2: إنسان اتشاف
    print("\n[SIM] Scenario 2: Human detected")
    from vision_processor import DetectionResult
    vision._latest_result = DetectionResult(
        position="FORWARD",
        label="person",
        confidence=0.88,
        timestamp=time.time(),
        threat_level="HIGH",
    )
    time.sleep(1)
    print(f"  State: {safety.get_state()}")
    assert safety.get_state() == "BLOCKED", "Should be BLOCKED"
    print("  ✅ BLOCKED — robot stopped")

    # Scenario 3: الإنسان مشي
    print(f"\n[SIM] Scenario 3: Human left → waiting {HUMAN_CLEAR_DELAY_S}s...")
    vision._latest_result = DetectionResult()  # reset
    time.sleep(HUMAN_CLEAR_DELAY_S + 1)
    print(f"  State: {safety.get_state()}")
    assert safety.is_safe(), "Should be CLEAR again"
    print("  ✅ CLEAR — robot resumed")

    print(f"\n  Stats: {safety.stats()}")
    print("=" * 60)

    safety.stop()
    vision.stop()
    bridge.stop()
    print("\nDone ✅")