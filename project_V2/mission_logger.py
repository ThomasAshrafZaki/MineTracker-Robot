"""
================================================================================
  mission_logger.py  —  Mission Data Logger
================================================================================
  Version 2.0 — Environment Logging Added

  لوجين منفصلين تماماً:

  1) MINE LOG — كل ما الأردوينو يكتشف لغم
       logs/mines/MINE_YYYYMMDD_HHMMSS.jpg
       logs/mines/mines_log.csv

  2) OBSTACLE LOG — كل ما YOLO يكتشف عائق جديد
       logs/obstacles/OBS_YYYYMMDD_HHMMSS_label.jpg
       logs/obstacles/obstacles_log.csv

  3) ENVIRONMENT LOG  ← جديد
       logs/environment/environment_log.csv
       — بيسجل كل تغيير في تصنيف البيئة
       — cooldown 15 ثانية عشان ما يكررش نفس البيئة كتير

  Author  : Robot Team
  Version : 2.0
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

from vision_processor import VisionProcessor, DetectionResult, EnvironmentResult
from arduino_bridge import ArduinoBridge, Telemetry

log = logging.getLogger("MissionLogger")

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────

MINE_DIR        = "logs/mines"
OBSTACLE_DIR    = "logs/obstacles"
ENVIRONMENT_DIR = "logs/environment"      # ← جديد

MINE_CSV        = os.path.join(MINE_DIR,        "mines_log.csv")
OBSTACLE_CSV    = os.path.join(OBSTACLE_DIR,    "obstacles_log.csv")
ENVIRONMENT_CSV = os.path.join(ENVIRONMENT_DIR, "environment_log.csv")  # ← جديد

# cooldown — بعد ما يحفظ صورة لعائق معين، يستنى كام ثانية
OBSTACLE_COOLDOWN_S = 5.0

# cooldown للبيئة — ما يسجلش نفس البيئة كتير
ENV_COOLDOWN_S = 15.0

# minimum persistence frames قبل ما نحفظ
MIN_SAVE_PERSISTENCE = 4

# monitor frequency
MONITOR_HZ = 10

# CSV headers
MINE_CSV_HEADERS        = ["timestamp", "datetime", "confidence", "image_path"]
OBSTACLE_CSV_HEADERS    = ["timestamp", "datetime", "label", "confidence",
                           "threat_level", "approx_dist", "image_path"]
ENVIRONMENT_CSV_HEADERS = ["timestamp", "datetime", "environment", "confidence"]  # ← جديد


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
    timestamp:    float = field(default_factory=time.time)
    label:        str   = ""
    confidence:   float = 0.0
    threat_level: str   = ""
    approx_dist:  str   = ""
    image_path:   str   = ""

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


@dataclass
class EnvironmentRecord:
    """سجل تصنيف البيئة — جديد في v2.0."""
    timestamp:   float = field(default_factory=time.time)
    environment: str   = ""
    confidence:  float = 0.0

    def to_csv_row(self) -> list:
        return [
            f"{self.timestamp:.3f}",
            datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            self.environment,
            f"{self.confidence:.2f}",
        ]


# ──────────────────────────────────────────────
#  MAIN CLASS
# ──────────────────────────────────────────────

class MissionLogger:
    """
    يراقب:
      - الأردوينو: لو metal_detected → يحفظ صورة + CSV row
      - YOLO: لو عائق جديد ومش cooldown → يحفظ صورة + CSV row
      - البيئة: لو تغيرت البيئة ومش cooldown → يسجل في CSV  ← جديد

    الـ cooldown بيمنع الاهتزاز من تسجيل نفس العائق/البيئة أكتر من مرة.
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

        # ── Environment tracking ─────────────────
        self._last_env_label:    str   = ""       # آخر بيئة اتسجلت
        self._last_env_log_time: float = 0.0      # آخر وقت سجلنا فيه

        # stats
        self._mine_count     = 0
        self._obstacle_count = 0
        self._env_count      = 0   # ← جديد

        self._setup_dirs()

    # ──────────────────────────────────────────
    #  SETUP
    # ──────────────────────────────────────────

    def _setup_dirs(self):
        os.makedirs(MINE_DIR,        exist_ok=True)
        os.makedirs(OBSTACLE_DIR,    exist_ok=True)
        os.makedirs(ENVIRONMENT_DIR, exist_ok=True)   # ← جديد
        self._init_csv(MINE_CSV,        MINE_CSV_HEADERS)
        self._init_csv(OBSTACLE_CSV,    OBSTACLE_CSV_HEADERS)
        self._init_csv(ENVIRONMENT_CSV, ENVIRONMENT_CSV_HEADERS)  # ← جديد

    @staticmethod
    def _init_csv(path: str, headers: list):
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
            "mines_logged":        self._mine_count,
            "obstacles_logged":    self._obstacle_count,
            "environments_logged": self._env_count,          # ← جديد
            "mine_dir_size":       self._dir_size_mb(MINE_DIR),
            "obstacle_dir_size":   self._dir_size_mb(OBSTACLE_DIR),
            "env_dir_size":        self._dir_size_mb(ENVIRONMENT_DIR),  # ← جديد
        }

    # ──────────────────────────────────────────
    #  CLEAR COMMANDS — يدوي بس
    # ──────────────────────────────────────────

    def clear_mines(self):
        self._print_clear_report("MINES", MINE_DIR)
        self._clear_dir(MINE_DIR)
        self._init_csv(MINE_CSV, MINE_CSV_HEADERS)
        self._mine_count = 0
        print("✅ Mines log cleared.")

    def clear_obstacles(self):
        self._print_clear_report("OBSTACLES", OBSTACLE_DIR)
        self._clear_dir(OBSTACLE_DIR)
        self._init_csv(OBSTACLE_CSV, OBSTACLE_CSV_HEADERS)
        self._obstacle_count = 0
        self._last_obstacle_save.clear()
        print("✅ Obstacles log cleared.")

    def clear_environment(self):
        """يمسح لوج البيئة — جديد في v2.0."""
        self._print_clear_report("ENVIRONMENT", ENVIRONMENT_DIR)
        self._clear_dir(ENVIRONMENT_DIR)
        self._init_csv(ENVIRONMENT_CSV, ENVIRONMENT_CSV_HEADERS)
        self._env_count      = 0
        self._last_env_label = ""
        print("✅ Environment log cleared.")

    def clear_all(self):
        self.clear_mines()
        self.clear_obstacles()
        self.clear_environment()   # ← جديد
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

        # ── فحص البيئة ───────────────────────   ← جديد
        env = self.vision.get_environment()
        if self._should_log_environment(env):
            self._handle_environment(env)

    # ──────────────────────────────────────────
    #  MINE LOGGING
    # ──────────────────────────────────────────

    def _handle_mine(self, tel: Telemetry):
        now = time.time()
        if now - self._last_mine_time < self._mine_cooldown_s:
            return

        self._last_mine_time = now
        self._mine_count    += 1
        log.info(f"MINE DETECTED #{self._mine_count}")

        frame  = self.vision.get_raw_frame()
        record = MineRecord(timestamp=now)

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
        if not result.obstacle_found():
            return False
        if not result.is_valid(max_age=0.5):
            return False
        if not result.persistent:
            return False

        now = time.time()
        last_save = self._last_obstacle_save.get(result.label, 0.0)
        if now - last_save < OBSTACLE_COOLDOWN_S:
            return False

        return True

    def _handle_obstacle(self, result: DetectionResult):
        now = time.time()
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
    #  ENVIRONMENT LOGGING  ← جديد في v2.0
    # ──────────────────────────────────────────

    def _should_log_environment(self, env: EnvironmentResult) -> bool:
        """
        يسجل البيئة لو:
          1. الـ label مش Unknown
          2. الـ result مش قديم (أقل من 10 ثواني)
          3. البيئة اتغيرت عن آخر مسجل  OR  فات cooldown
        """
        if env.label in ("Unknown", ""):
            return False

        if not env.is_valid(max_age=10.0):
            return False

        now = time.time()

        # تغيرت البيئة → سجل فوراً (لو فاتت ثانية على الأقل من آخر تسجيل)
        if env.label != self._last_env_label:
            if now - self._last_env_log_time >= 1.0:
                return True

        # نفس البيئة لكن فات الـ cooldown → سجل تأكيداً دورياً
        if now - self._last_env_log_time >= ENV_COOLDOWN_S:
            return True

        return False

    def _handle_environment(self, env: EnvironmentResult):
        now = time.time()

        changed = (env.label != self._last_env_label)

        with self._lock:
            self._last_env_label    = env.label
            self._last_env_log_time = now

        self._env_count += 1

        if changed:
            log.info(
                f"ENV CHANGE #{self._env_count} | "
                f"{env.label}  conf={env.confidence:.0%}"
            )
        else:
            log.debug(
                f"ENV PERIODIC #{self._env_count} | "
                f"{env.label}  conf={env.confidence:.0%}"
            )

        record = EnvironmentRecord(
            timestamp   = now,
            environment = env.label,
            confidence  = env.confidence,
        )
        # CSV فقط — مفيش صورة للبيئة
        self._append_csv(ENVIRONMENT_CSV, record.to_csv_row())

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
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".csv"))
        ] if os.path.exists(directory) else []

        size_mb = MissionLogger._dir_size_mb(directory)
        print(f"\n{'='*40}")
        print(f"  {name} CLEAR REPORT")
        print(f"{'='*40}")
        print(f"  Files   : {len(images)}")
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
    print("  MissionLogger v2.0 — Simulation Test")
    print("=" * 60)

    bridge = create_bridge(simulate=True)
    vision = VisionProcessor(simulate=True, use_yolo=False, use_env=False)

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

    # Scenario 3: تغيير البيئة  ← جديد
    print("\n[SIM] Scenario 3: Environment change")
    from vision_processor import EnvironmentResult
    vision._env_classifier._latest_env = EnvironmentResult(
        label="Open Field", confidence=0.82
    )
    time.sleep(2)
    vision._env_classifier._latest_env = EnvironmentResult(
        label="Desert", confidence=0.75
    )
    time.sleep(2)
    print(f"  Environments logged: {logger._env_count}")

    # Scenario 4: اهتزاز — نفس العائق كتير ورا بعض
    print("\n[SIM] Scenario 4: Vibration — same obstacle repeated fast")
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