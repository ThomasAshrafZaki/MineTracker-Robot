"""
================================================================================
  mine_detector.py  —  Surface Landmine Detector (Secondary YOLO Model)
================================================================================
  موديل ثانوي اختياري بالكامل — مستقل عن موديل الـ YOLO الأساسي بتاع
  vision_processor.py (yolov8n.pt اللي بيكشف عوائق/إنسان).

  الموديل ده (best.pt) مدرّب خصيصاً على لغم سطحي واحد بس، وهدفه:
      "ما يعملش Detection لأي حاجة عادية ممكن يغلط فيها (علبة تونة، حجر، ...)
       — يشتغل ويأكّد بس لو فعلاً شايف حاجة قريبة من اللي اتدرب عليها."

  ─────────────────────────────────────────────────────────────
  إزاي بنقلل الـ False Positives (مش بنضمن صفر، بس بنقلل جداً):
  ─────────────────────────────────────────────────────────────
    1) MINE_CONF_MIN عالي جداً عمداً (0.65 افتراضي) — أي كشف بثقة أقل
       بيترفض فوراً قبل حتى ما يوصل لأي منطق تاني.
    2) Temporal persistence (Sliding window زي sign_detector.py):
       لازم MINE_PERSIST_HITS من أصل MINE_PERSIST_WINDOW فريم متتالي
       يكونوا positive قبل ما الـ result يطلع "found=True" للـ caller.
       فريم واحد عشوائي (نور غريب، ظل، انعكاس) مش هيفعّل أي حاجة.
    3) Async thread منفصل تماماً (Latest-frame-wins) — زي الـ sign
       detector بالظبط — مش بيأثر على سرعة الـ capture loop خالص.
    4) لو best.pt مش موجود أو فشل تحميله → load() بترجع False ببساطة
       ولوج تحذير واحد بس، وكل باقي VisionProcessor بيكمل طبيعي
       (نفس الفلسفة اللي main.py و vision_processor.py مبنيين عليها
       أصلاً لـ use_mine).

  ─────────────────────────────────────────────────────────────
  مكان ملف الموديل:
  ─────────────────────────────────────────────────────────────
    افتراضياً بيدور على "best.pt" في نفس مجلد التشغيل (جنب main.py).
    لو عايز تحطه في مسار تاني، مرّر model_path وقت الإنشاء، أو
    غيّر MINE_MODEL_PATH تحت.

  Author  : Robot Team
  Version : 1.0
================================================================================
"""

import os
import time
import logging
import threading

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

log = logging.getLogger("MineDetector")


# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────

# اسم/مسار ملف الموديل المدرّب على كولاب — حطه جنب باقي ملفات المشروع
MINE_MODEL_PATH = "best.pt"

# عتبة الثقة وقت الـ inference نفسه — عالية جداً عمداً.
# الهدف: لو الموديل مش متأكد بدرجة عالية، نعتبره "مفيش لغم" بالكامل،
# مش بس نعرضه بشفافية أقل.
MINE_CONF_MIN = 0.65

# حجم الصورة وقت الـ inference (زي باقي موديلات YOLO في المشروع)
MINE_IMG_SIZE = 320

# ── Temporal persistence (دفاع إضافي ضد false positive لحظي) ──
# 5 فريمات نافذة، لازم 3 منهم على الأقل يكونوا positive قبل ما
# نأكد للـ caller إن في لغم. مش بديل عن عتبة الثقة فوق — ده طبقة
# تانية مستقلة.
MINE_PERSIST_WINDOW = 5
MINE_PERSIST_HITS   = 3

# لو آخر نتيجة "found" أقدم من كده، get_latest().is_valid() هترجع False
MINE_RESULT_MAX_AGE_DEFAULT = 1.0


# ──────────────────────────────────────────────
#  DATA MODEL
# ──────────────────────────────────────────────

@dataclass
class MineDetectionResult:
    """
    نتيجة واحدة من الـ MineDetector.
    boxes: list[dict] فيها x1,y1,x2,y2,label,conf — فاضية لو مفيش لغم مؤكد.
    """
    timestamp: float = field(default_factory=time.time)
    boxes:     list  = field(default_factory=list)

    def found(self) -> bool:
        """True لو في صندوق لغم واحد على الأقل اتأكد (بعد كل الفلاتر)."""
        return len(self.boxes) > 0

    def is_valid(self, max_age: float = MINE_RESULT_MAX_AGE_DEFAULT) -> bool:
        return (time.time() - self.timestamp) < max_age


# ──────────────────────────────────────────────
#  MAIN CLASS
# ──────────────────────────────────────────────

class MineDetector:
    """
    استخدامه (نفس الباترن اللي vision_processor.py بيستخدمه بالفعل):

        md = MineDetector(device="cpu")
        ok = md.load()
        if ok:
            md.start()
            ...
            md.submit_frame(frame)   # من capture loop، كل فريم أو اللي تحب
            result = md.get_latest() # MineDetectionResult
            if result.found() and result.is_valid(max_age=1.0):
                ...
            md.stop()
    """

    def __init__(
        self,
        device:     str   = "cpu",
        model_path: str   = MINE_MODEL_PATH,
        conf_min:   float = MINE_CONF_MIN,
    ):
        self.device     = device
        self.model_path = model_path
        self.conf_min   = conf_min

        self._model     = None
        self._available = False

        self._lock   = threading.Lock()
        self._latest = MineDetectionResult()

        self._pending_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._ready       = threading.Event()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # ── Temporal persistence state ──────────────
        self._history: deque = deque(maxlen=MINE_PERSIST_WINDOW)

        # إحصائيات بسيطة للـ debug
        self._frames_processed = 0
        self._raw_hits         = 0   # كشوفات قبل الـ persistence gate
        self._confirmed_hits   = 0   # كشوفات بعد الـ persistence gate

    # ──────────────────────────────────────────
    #  LOAD
    # ──────────────────────────────────────────

    def load(self) -> bool:
        """
        بيحمّل best.pt. لو الملف مش موجود أو ultralytics مش متثبتة أو
        أي مشكلة تانية → بترجع False ببساطة (مفيش exception طالعة برا).
        """
        try:
            if not os.path.exists(self.model_path):
                log.warning(
                    f"[MineDetector] ملف الموديل مش موجود: '{self.model_path}' — "
                    f"تأكد إنك حاطط best.pt في نفس مجلد main.py، أو مرّر "
                    f"model_path الصح. MineDetector هيفضل معطّل."
                )
                return False

            from ultralytics import YOLO

            log.info(f"[MineDetector] تحميل الموديل: {self.model_path} على {self.device}")
            self._model = YOLO(self.model_path)

            # warm-up — أول inference بطيء عادةً
            dummy = np.zeros((MINE_IMG_SIZE, MINE_IMG_SIZE, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)

            self._available = True
            log.info(
                f"[MineDetector] الموديل اتحمّل وجهز ✓ "
                f"(conf_min={self.conf_min}, classes={list(self._model.names.values())})"
            )
            return True

        except ImportError:
            log.warning("[MineDetector] مكتبة ultralytics مش متثبتة — mine detection معطّل.")
            return False
        except Exception as e:
            log.error(f"[MineDetector] فشل تحميل الموديل: {e}")
            return False

    # ──────────────────────────────────────────
    #  START / STOP
    # ──────────────────────────────────────────

    def start(self):
        if not self._available:
            log.debug("[MineDetector] start() اتنادت بس الموديل مش محمّل — تجاهل.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="MineDetector"
        )
        self._thread.start()
        log.info("[MineDetector] worker thread اشتغل")

    def stop(self):
        self._running = False
        self._ready.set()   # نصحي الـ thread لو واقف بينتظر، عشان يخرج فوراً
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info(
            f"[MineDetector] stopped — frames={self._frames_processed} "
            f"raw_hits={self._raw_hits} confirmed={self._confirmed_hits}"
        )

    @property
    def available(self) -> bool:
        return self._available

    # ──────────────────────────────────────────
    #  PUBLIC API — submit / get
    # ──────────────────────────────────────────

    def submit_frame(self, frame: np.ndarray):
        """
        بتستقبل فريم من الـ capture loop (latest-frame-wins، زي
        sign_detector.py بالظبط) — لو الـ worker مشغول، الفريم اللي
        فات بيتبدل من غير ما يعمل قطع/تأخير على الـ caller.
        """
        if not self._available or not self._running:
            return
        with self._frame_lock:
            self._pending_frame = frame
        if not self._ready.is_set():
            self._ready.set()

    def get_latest(self) -> MineDetectionResult:
        with self._lock:
            return self._latest

    def stats(self) -> dict:
        return {
            "available":         self._available,
            "frames_processed":  self._frames_processed,
            "raw_hits":          self._raw_hits,
            "confirmed_hits":    self._confirmed_hits,
            "conf_min":          self.conf_min,
        }

    # ──────────────────────────────────────────
    #  WORKER LOOP
    # ──────────────────────────────────────────

    def _loop(self):
        while self._running:
            self._ready.wait(timeout=1.0)
            if not self._running:
                break

            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None
            self._ready.clear()

            if frame is None:
                continue

            try:
                self._infer(frame)
            except Exception as e:
                log.error(f"[MineDetector] inference error: {e}", exc_info=True)

    # ──────────────────────────────────────────
    #  INFERENCE + GATES
    # ──────────────────────────────────────────

    def _infer(self, frame: np.ndarray):
        self._frames_processed += 1

        # ── Stage 1: YOLO inference بعتبة ثقة عالية من الأول ──
        # تمرير conf هنا معناها إن أي كشف أقل من المطلوب أصلاً مش
        # هيرجع في النتائج — مش بنرفضه بعدين، بنمنعه من الأساس.
        results = self._model(
            frame, imgsz=MINE_IMG_SIZE, conf=self.conf_min,
            device=self.device, verbose=False,
        )

        raw_boxes = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_min:
                    # حماية إضافية احتياطية — حتى لو الموديل رجّع
                    # حاجة تحت العتبة لأي سبب
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls   = int(box.cls[0])
                label = self._model.names[cls]
                raw_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label, "conf": conf,
                })

        detected_now = len(raw_boxes) > 0
        if detected_now:
            self._raw_hits += 1

        # ── Stage 2: Temporal persistence gate ──────────────────
        # مش بنأكد "في لغم" إلا لو ثبت على عدة فريمات متتالية —
        # ده اللي بيمنع فلاش لحظي (انعكاس ضوء، ظل غريب) من إنه
        # يوصل لحد ما يوقف الروبوت أو يتسجل في mission_logger.
        self._history.append(detected_now)
        persistent_enough = sum(self._history) >= MINE_PERSIST_HITS

        if detected_now and persistent_enough:
            final_boxes = raw_boxes
            self._confirmed_hits += 1
        else:
            final_boxes = []

        result = MineDetectionResult(boxes=final_boxes)
        with self._lock:
            self._latest = result

        if final_boxes:
            best = max(final_boxes, key=lambda b: b["conf"])
            log.warning(
                f"[MineDetector] *** POSSIBLE MINE *** "
                f"label={best['label']} conf={best['conf']:.0%} "
                f"(persist={sum(self._history)}/{MINE_PERSIST_WINDOW})"
            )


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )

    print("=" * 60)
    print("  MineDetector — Standalone Test")
    print("=" * 60)
    print(f"  بيدور على الموديل في: {os.path.abspath(MINE_MODEL_PATH)}")
    print()

    md = MineDetector(device="cpu")
    ok = md.load()

    if not ok:
        print(
            "  ❌ الموديل مش اتحمّل — تأكد إن ملف best.pt موجود في\n"
            "     نفس المجلد اللي بتشغّل منه السكريبت ده، أو إن\n"
            "     ultralytics متثبتة (pip install ultralytics)."
        )
        raise SystemExit(1)

    md.start()
    print("  ✅ الموديل شغال — هيختبر على كاميرا الجهاز (لو موجودة)\n")

    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠️  مفيش كاميرا متاحة — جرّب على صورة ثابتة بدل كده.")
        md.stop()
        raise SystemExit(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            md.submit_frame(frame)
            result = md.get_latest()

            display = frame.copy()
            if result.found() and result.is_valid(max_age=1.0):
                for b in result.boxes:
                    cv2.rectangle(display, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                                  (255, 0, 200), 2)
                    cv2.putText(display, f"{b['label']} {b['conf']:.0%}",
                                (b["x1"], max(15, b["y1"] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 2)
                cv2.putText(display, "MINE DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, "CLEAR", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            cv2.imshow("MineDetector Test", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        md.stop()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n  Stats: {md.stats()}")
        print("\nDone ✅")
