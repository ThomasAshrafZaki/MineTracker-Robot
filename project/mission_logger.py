"""
================================================================================
  mission_logger.py  —  Mission Data Logger
================================================================================
  لوجين منفصلين تماماً:

  1) MINE LOG — كل ما الأردوينو يكتشف لغم
       logs/mines/MINE_YYYYMMDD_HHMMSS.jpg
       logs/mines/mines_log.csv

  2) OBSTACLE LOG — كل ما YOLO يكتشف عائق جديد
       logs/obstacles/OBS_YYYYMMDD_HHMMSS_label.jpg
       logs/obstacles/obstacles_log.csv

  مشكلة الاهتزاز محلولة بـ:
       - Cooldown per unique obstacle label (5 ثواني)
       - Min persistence frames قبل الحفظ

  نظام المسح اليدوي:
       logger.clear_mines()
       logger.clear_obstacles()
       logger.clear_all()
       — كل أمر بيطبعلك تقرير قبل المسح

  Author  : Robot Team
  Version : 1.0
================================================================================
"""

import cv2
import csv
import os
import threading
import time
import logging
import shutil

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from vision_processor import VisionProcessor, DetectionResult
from arduino_bridge import ArduinoBridge, Telemetry

log = logging.getLogger("MissionLogger")

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────

MINE_DIR      = "logs/mines"
OBSTACLE_DIR  = "logs/obstacles"

MINE_CSV      = os.path.join(MINE_DIR,     "mines_log.csv")
OBSTACLE_CSV  = os.path.join(OBSTACLE_DIR, "obstacles_log.csv")

# cooldown — بعد ما يحفظ صورة لعائق معين، يستنى كام ثانية
OBSTACLE_COOLDOWN_S = 5.0

# minimum persistence frames قبل ما نحفظ
MIN_SAVE_PERSISTENCE = 4

# monitor frequency
MONITOR_HZ = 10

# CSV headers
MINE_CSV_HEADERS     = ["timestamp", "datetime", "confidence", "image_path"]
OBSTACLE_CSV_HEADERS = ["timestamp", "datetime", "label", "confidence",
                        "threat_level", "approx_dist", "image_path"]


# ──────────────────────────────────────────────
#  DATA MODELS
# ──────────────────────────────────────────────

@dataclass
class MineRecord:
    timestamp:   float = field(default_factory=time.time)
    confidence:  float = 0.0
    image_path:  str   = ""

    def to_csv_row(self) -> list:
        return [
            f"{self.timestamp:.3f}",
            datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            f"{self.confidence:.2f}",
            self.image_path,
        ]


@dataclass
class ObstacleRecord:
    timestamp:   float = field(default_factory=time.time)
    label:       str   = ""
    confidence:  float = 0.0
    threat_level: str  = ""
    approx_dist: str   = ""
    image_path:  str   = ""

    def to_csv_row(self) -> list:
        return [
            f"{self.timestamp:.3f}",
            datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            self.label,
            f"{self.confidence:.2f}",
            self.threat_level,
            self.approx_dist,
            self.image_path,
        ]


# ──────────────────────────────────────────────
#  MAIN CLASS
# ──────────────────────────────────────────────

class MissionLogger:
    """
    يراقب:
      - الأردوينو: لو metal_detected → يحفظ صورة + CSV row
      - YOLO: لو عائق جديد ومش cooldown → يحفظ صورة + CSV row

    الـ cooldown بيمنع الاهتزاز من تسجيل نفس العائق أكتر من مرة.
    """

    def __init__(
        self,
        vision: VisionProcessor,
        bridge: ArduinoBridge,
    ):
        self.vision = vision
        self.bridge = bridge

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # cooldown tracker: label → last save time
        self._last_obstacle_save: dict = {}

        # منع تسجيل نفس اللغم أكتر من مرة في ثانية واحدة
        self._last_mine_time: float = 0.0
        self._mine_cooldown_s: float = 2.0

        # stats
        self._mine_count     = 0
        self._obstacle_count = 0

        self._setup_dirs()

    # ──────────────────────────────────────────
    #  SETUP
    # ──────────────────────────────────────────

    def _setup_dirs(self):
        os.makedirs(MINE_DIR,     exist_ok=True)
        os.makedirs(OBSTACLE_DIR, exist_ok=True)
        self._init_csv(MINE_CSV,     MINE_CSV_HEADERS)
        self._init_csv(OBSTACLE_CSV, OBSTACLE_CSV_HEADERS)

    @staticmethod
    def _init_csv(path: str, headers: list):
        """ينشئ الـ CSV لو مش موجود ويكتب الـ headers."""
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    # ──────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MissionLogger"
        )
        self._thread.start()
        log.info("MissionLogger started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("MissionLogger stopped")

    def stats(self) -> dict:
        return {
            "mines_logged":     self._mine_count,
            "obstacles_logged": self._obstacle_count,
            "mine_dir_size":    self._dir_size_mb(MINE_DIR),
            "obstacle_dir_size": self._dir_size_mb(OBSTACLE_DIR),
        }

    # ──────────────────────────────────────────
    #  CLEAR COMMANDS — يدوي بس
    # ──────────────────────────────────────────

    def clear_mines(self):
        """يطبع تقرير ثم يمسح مجلد الألغام."""
        self._print_clear_report("MINES", MINE_DIR)
        self._clear_dir(MINE_DIR)
        self._init_csv(MINE_CSV, MINE_CSV_HEADERS)
        self._mine_count = 0
        print("✅ Mines log cleared.")

    def clear_obstacles(self):
        """يطبع تقرير ثم يمسح مجلد العوائق."""
        self._print_clear_report("OBSTACLES", OBSTACLE_DIR)
        self._clear_dir(OBSTACLE_DIR)
        self._init_csv(OBSTACLE_CSV, OBSTACLE_CSV_HEADERS)
        self._obstacle_count = 0
        self._last_obstacle_save.clear()
        print("✅ Obstacles log cleared.")

    def clear_all(self):
        """يمسح كل اللوجز."""
        self.clear_mines()
        self.clear_obstacles()
        print("✅ All logs cleared.")

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
                log.error(f"MissionLogger error: {e}", exc_info=True)

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _tick(self):
        # ── فحص اللغم ────────────────────────
        tel = self.bridge.get_telemetry()
        if tel.metal_detected:
            self._handle_mine(tel)

        # ── فحص العائق ───────────────────────
        result = self.vision.get_latest()
        if self._should_log_obstacle(result):
            self._handle_obstacle(result)

    # ──────────────────────────────────────────
    #  MINE LOGGING
    # ──────────────────────────────────────────

    def _handle_mine(self, tel: Telemetry):
        now = time.time()

        # cooldown عشان ما يسجلش نفس اللغم أكتر من مرة
        if now - self._last_mine_time < self._mine_cooldown_s:
            return

        self._last_mine_time = now
        self._mine_count    += 1

        log.info(f"MINE DETECTED #{self._mine_count}")

        frame = self.vision.get_raw_frame()
        record = MineRecord(timestamp=now)

        # احفظ الصورة في thread منفصل
        threading.Thread(
            target=self._save_mine_image,
            args=(frame, record),
            daemon=True,
            name="MineSave"
        ).start()

    def _save_mine_image(self, frame, record: MineRecord):
        try:
            ts   = datetime.fromtimestamp(record.timestamp).strftime("%Y%m%d_%H%M%S")
            name = f"MINE_{ts}.jpg"
            path = os.path.join(MINE_DIR, name)

            if frame is not None:
                cv2.imwrite(path, frame)
                record.image_path = path
                log.info(f"Mine image saved: {path}")
            else:
                log.warning("Mine detected but no frame available")
                record.image_path = "NO_FRAME"

            self._append_csv(MINE_CSV, record.to_csv_row())

        except Exception as e:
            log.error(f"Failed to save mine image: {e}")

    # ──────────────────────────────────────────
    #  OBSTACLE LOGGING
    # ──────────────────────────────────────────

    def _should_log_obstacle(self, result: DetectionResult) -> bool:
        """
        يحدد هل نحفظ صورة للعائق ده ولا لأ.

        الشروط:
          1. في عائق فعلاً
          2. الـ detection مش قديم
          3. الـ persistence كافي (مش اهتزاز)
          4. مش في cooldown لنفس الـ label
        """
        if not result.obstacle_found():
            return False

        if not result.is_valid(max_age=0.5):
            return False

        # persistence check — لازم يكون stable مش اهتزاز
        if not result.persistent:
            return False

        # cooldown check per label
        now = time.time()
        last_save = self._last_obstacle_save.get(result.label, 0.0)
        if now - last_save < OBSTACLE_COOLDOWN_S:
            return False

        return True

    def _handle_obstacle(self, result: DetectionResult):
        now = time.time()

        # سجل الـ cooldown فوراً عشان thread التاني ما يحفظش نفس الحاجة
        with self._lock:
            self._last_obstacle_save[result.label] = now

        self._obstacle_count += 1
        log.info(
            f"OBSTACLE LOG #{self._obstacle_count} | "
            f"label={result.label} conf={result.confidence:.0%} "
            f"threat={result.threat_level}"
        )

        frame  = self.vision.get_raw_frame()
        record = ObstacleRecord(
            timestamp    = now,
            label        = result.label,
            confidence   = result.confidence,
            threat_level = result.threat_level,
            approx_dist  = result.approx_dist,
        )

        threading.Thread(
            target=self._save_obstacle_image,
            args=(frame, record),
            daemon=True,
            name="ObstacleSave"
        ).start()

    def _save_obstacle_image(self, frame, record: ObstacleRecord):
        try:
            ts   = datetime.fromtimestamp(record.timestamp).strftime("%Y%m%d_%H%M%S")
            name = f"OBS_{ts}_{record.label}_{record.threat_level}.jpg"
            path = os.path.join(OBSTACLE_DIR, name)

            if frame is not None:
                cv2.imwrite(path, frame)
                record.image_path = path
                log.info(f"Obstacle image saved: {path}")
            else:
                record.image_path = "NO_FRAME"

            self._append_csv(OBSTACLE_CSV, record.to_csv_row())

        except Exception as e:
            log.error(f"Failed to save obstacle image: {e}")

    # ──────────────────────────────────────────
    #  CSV HELPER
    # ──────────────────────────────────────────

    def _append_csv(self, path: str, row: list):
        try:
            with self._lock:
                with open(path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
        except Exception as e:
            log.error(f"CSV write error: {e}")

    # ──────────────────────────────────────────
    #  CLEAR HELPERS
    # ──────────────────────────────────────────

    @staticmethod
    def _print_clear_report(name: str, directory: str):
        images = [
            f for f in os.listdir(directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ] if os.path.exists(directory) else []

        size_mb = MissionLogger._dir_size_mb(directory)
        print(f"\n{'='*40}")
        print(f"  {name} CLEAR REPORT")
        print(f"{'='*40}")
        print(f"  Images  : {len(images)}")
        print(f"  Size    : {size_mb:.1f} MB")
        print(f"  Dir     : {directory}")
        print(f"{'='*40}\n")

    @staticmethod
    def _clear_dir(directory: str):
        if not os.path.exists(directory):
            return
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                log.warning(f"Could not delete {path}: {e}")

    @staticmethod
    def _dir_size_mb(directory: str) -> float:
        if not os.path.exists(directory):
            return 0.0
        total = 0
        for f in os.listdir(directory):
            fp = os.path.join(directory, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
        return total / (1024 * 1024)


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    )

    from arduino_bridge import create_bridge

    print("=" * 60)
    print("  MissionLogger — Simulation Test")
    print("=" * 60)

    bridge = create_bridge(simulate=True)
    vision = VisionProcessor(simulate=True, use_yolo=False)

    bridge.start()
    vision.start()

    logger = MissionLogger(vision, bridge)
    logger.start()

    # Scenario 1: اكتشاف لغم
    print("\n[SIM] Scenario 1: Mine detected")
    bridge._latest_tel.metal_detected = True
    time.sleep(1)
    bridge._latest_tel.metal_detected = False
    time.sleep(1)
    print(f"  Mines logged: {logger._mine_count}")

    # Scenario 2: عائق اتشاف
    print("\n[SIM] Scenario 2: Obstacle detected")
    from vision_processor import DetectionResult
    vision._latest_result = DetectionResult(
        position     = "FORWARD",
        label        = "chair",
        confidence   = 0.87,
        threat_level = "MEDIUM",
        approx_dist  = "CLOSE",
        persistent   = True,
        timestamp    = time.time(),
    )
    time.sleep(1)
    vision._latest_result = DetectionResult()
    time.sleep(1)
    print(f"  Obstacles logged: {logger._obstacle_count}")

    # Scenario 3: اهتزاز — نفس العائق كتير ورا بعض
    print("\n[SIM] Scenario 3: Vibration — same obstacle repeated fast")
    for _ in range(5):
        vision._latest_result = DetectionResult(
            position="FORWARD", label="bottle",
            confidence=0.75, threat_level="LOW",
            approx_dist="MEDIUM", persistent=True,
            timestamp=time.time(),
        )
        time.sleep(0.3)
    time.sleep(1)
    print(f"  Should be 1 save only (cooldown): obstacles={logger._obstacle_count}")

    # Stats
    print(f"\n  Stats: {logger.stats()}")

    # Clear report
    print("\n[SIM] Testing clear report:")
    logger.clear_all()

    print("=" * 60)
    logger.stop()
    vision.stop()
    bridge.stop()
    print("\nDone ✅")
