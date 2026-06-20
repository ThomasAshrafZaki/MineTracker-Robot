"""
================================================================================
  main.py  —  Robot System Entry Point
================================================================================
  يشغل كل المكونات مع بعض:
      1. ArduinoBridge      — اتصال الأردوينو
      2. VisionProcessor    — كاميرا + YOLO + تصنيف البيئة
      3. HumanSafetyMonitor — كشف الإنسان + إيقاف أمان
      4. MissionLogger      — تسجيل الألغام والعوائق والبيئة
      5. Display loop       — عرض الفريم على الشاشة

  تشغيل:
      python main.py                      # هاردوير حقيقي
      python main.py --simulate           # simulation كاملة
      python main.py --no-display         # بدون شاشة (headless)
      python main.py --port /dev/ttyUSB0  # port محدد
      python main.py --no-yolo            # بدون YOLO
      python main.py --no-env             # بدون تصنيف البيئة
      python main.py --device cuda        # GPU على الجيتسون

  مفاتيح الشاشة:
      Q  — إيقاف
      P  — pause / resume
      R  — reset الأردوينو
      C  — clear all logs (بيطبع تقرير الحجم الأول)

  Author  : Robot Team
  Version : 4.1 — Environment Classification Added
================================================================================
"""

import argparse
import logging
import signal
import sys
import time
import cv2
import numpy as np
import os

from arduino_bridge    import create_bridge, UltrasonicData
from vision_processor  import VisionProcessor, DetectionResult
from human_safety      import HumanSafetyMonitor, HumanEvent
from mission_logger    import MissionLogger

# ──────────────────────────────────────────────
#  LOGGING SETUP
# ──────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# Unicode fix للـ Windows console (cp1252 مش بتعرف → ✓ إلخ)
import io as _io
if sys.platform == "win32":
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-18s] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/robot.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("Main")


# ──────────────────────────────────────────────
#  HUD
# ──────────────────────────────────────────────

_C_GREEN   = (0, 220,   0)
_C_RED     = (0,   0, 255)
_C_ORANGE  = (0, 140, 255)
_C_YELLOW  = (0, 220, 220)
_C_WHITE   = (220, 220, 220)
_C_GRAY    = (120, 120, 120)
_C_BLACK   = (0,   0,   0)
_C_CYAN    = (220, 220,   0)
_C_ENV     = (180, 255, 180)   # لون البيئة — أخضر فاتح


def draw_hud(
    frame:        np.ndarray,
    us:           UltrasonicData,
    vision:       DetectionResult,
    logger_stats: dict,
    safety_state: str,
    paused:       bool,
    env_label:    str   = "",
    env_conf:     float = 0.0,
    sign_danger:  bool  = False,
    sign_reason:  str   = "",
) -> np.ndarray:

    h, w = frame.shape[:2]

    # ملحوظة: شلنا من هنا نهائيًا (بناءً على طلبك):
    #   - bottom bar (F/L/R ultrasonic + ARDUINO/STATE + SAFETY)
    #   - stats top-right (MINES/OBS/ENV counts)
    #   - LOG disk usage (mines/obs MB)
    #   - الـ ENV label اللي كانت بتتكرر هنا (vision_processor.py
    #     عنده بار ENV خاص بيه دلوقتي بقى واضح وموجود لوحده تحت)
    #
    # لو احتجت أي واحدة منهم ترجع تاني قولي وأنا أرجعها.

    # ── Sign danger banner ─────────────────────────────────
    if sign_danger:
        cv2.rectangle(frame, (0, h // 2 - 28), (w, h // 2 + 28), (0, 0, 140), -1)
        cv2.putText(
            frame,
            f"!! DANGER SIGN  [{sign_reason}] !!",
            (w // 2 - 170, h // 2 + 9),
            cv2.FONT_HERSHEY_DUPLEX, 0.80, _C_RED, 2,
        )

    return frame


# ──────────────────────────────────────────────
#  ARGS
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Landmine Detection Robot")
    p.add_argument("--port",       default="/dev/ttyUSB0", help="Arduino serial port")
    p.add_argument("--camera",     type=int, default=0,    help="Camera index")
    p.add_argument("--simulate",   action="store_true",    help="Full simulation mode")
    p.add_argument("--no-display", action="store_true",    help="Headless mode")
    p.add_argument("--no-yolo",    action="store_true",    help="Disable YOLO")
    p.add_argument("--no-env",     action="store_true",    help="Disable environment classifier")  # ← جديد
    p.add_argument("--no-sign",    action="store_true",    help="Disable sign detector")
    p.add_argument("--device",     default="cpu",          help="YOLO device: cpu or cuda")
    return p.parse_args()


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  Landmine Detection Robot — Starting")
    log.info(f"  simulate={args.simulate}  port={args.port}")
    log.info(f"  yolo={not args.no_yolo}  device={args.device}")
    log.info(f"  env_classifier={not args.no_env}  sign_detector={not args.no_sign}")
    log.info("=" * 60)

    # ── callback: عائق أمامي مكتشف بالـ Ultrasonic (حرف 'O' من الأردوينو) ──
    #
    # ⚠️ ملحوظة ترتيب مهمة: المتغير `logger` لسه مش معرّف وقت تعريف
    # الدالة دي (هيتعرّف بعد كذا سطر تحت). ده آمن 100% بسبب late binding
    # في بايثون: closures بتدور على الاسم `logger` في الـ enclosing scope
    # وقت ما الدالة *بتتنفذ فعلياً*، مش وقت ما بتتعرّف. والدالة دي مش
    # بتتنفذ إلا من جوه RX thread بتاع الـ bridge، واللي مش بيبدأ إلا
    # عند `bridge.start()` تحت — وده بيحصل بعد تعريف `logger` بسطور.
    # لو غيّرت ترتيب الكود وحركت `bridge.start()` لفوق تعريف `logger`،
    # هتاخد NameError وقت أول إشارة 'O' — خليك واخد بالك من الترتيب ده.
    #
    # ملحوظة تانية: في --simulate، SimulatedArduinoBridge بيستخدم
    # _sim_loop مش _rx_loop، فمفيش حرف 'O' حقيقي جاي من سيريال. عشان
    # تقدر تتأكد إن باقي السلسلة (MissionLogger → صورة + GPS + بيئة +
    # صف إكسل) شغالة صح من غير هاردوير، SimulatedArduinoBridge بقى فيه
    # تريجر صناعي دوري لنفس الـ callback (راجع arduino_bridge.py).
    def on_obstacle_front():
        log.warning("OBSTACLE FRONT — Arduino detected obstacle ahead, triggering camera capture")
        logger.trigger_ultrasonic_obstacle(label="obstacle")

    # ── build components ─────────────────────
    bridge = create_bridge(
        port=args.port,
        simulate=args.simulate,
        auto_detect=not args.simulate,
        on_obstacle_front=on_obstacle_front,
    )

    vision = VisionProcessor(
        camera_index=args.camera,
        use_yolo=not args.no_yolo,
        use_env=not args.no_env,          # ← جديد: تصنيف البيئة
        device=args.device,
        simulate=args.simulate,
    )

    _use_sign = not args.no_sign   # فلاق التحكم في كشف اللوحات

    # callbacks للـ human safety
    def on_human_detected(event: HumanEvent):
        log.warning(f"HUMAN DETECTED | conf={event.confidence:.0%}")

    def on_human_cleared(event: HumanEvent):
        log.info(f"AREA CLEAR | robot was stopped for {event.duration_s:.1f}s")

    safety = HumanSafetyMonitor(
        vision=vision,
        bridge=bridge,
        on_human_detected=on_human_detected,
        on_human_cleared=on_human_cleared,
    )

    logger = MissionLogger(vision=vision, bridge=bridge)

    # ── graceful shutdown ─────────────────────
    running = [True]
    paused  = [False]

    def shutdown(sig=None, frame=None):
        log.info("Shutdown signal received")
        running[0] = False

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── start all ────────────────────────────
    log.info("Starting ArduinoBridge...")
    bridge_ok = bridge.start()
    if not bridge_ok and not args.simulate:
        log.warning("Arduino not connected — vision only mode")

    log.info("Starting VisionProcessor...")
    vision_ok = vision.start()
    if not vision_ok and not args.simulate:
        log.warning("Camera not available")

    log.info("Starting HumanSafetyMonitor...")
    safety.start()

    log.info("Starting MissionLogger...")
    logger.start()

    log.info("All components started.")
    if not args.no_display:
        log.info("Keys: Q=quit  P=pause/resume  R=reset  C=clear logs")

    # ── main loop ────────────────────────────
    try:
        while running[0]:

            # ── safety gate ──────────────────
            # مهم: send_human_stop() ('H') مش send_stop() ('s') —
            # عشان تتفعّل بس في AUTO mode على مستوى الأردوينو، ومتتعارضش
            # مع أوامر الموقع (a/m/f/b/l/r) ولا مع 's' العام.
            if not safety.is_safe():
                bridge.send_human_stop()

            # ── sign detector gate ────────────
            # bridge.send_sign_stop() بتبعت '!' — حرف مستقل عن 'H' بتاع
            # human_safety.py، مفيش أي تشارك بينهم. مش bridge.send_stop()
            # ('s') لأنها بترجع Stop() لحظي بس من غير return من loop() في
            # الأردوينو، يعني الزقزاق كان بيكمّل حركته فوقها في AUTO.
            # تفعيل/تعطيل تأثيرها على العربية بيتحكم فيه من الأردوينو
            # نفسه (#define ENABLE_SIGN_STOP) — مفيش أي فلاج هنا.
            if _use_sign:
                sign = vision.get_latest_sign()
                if sign is not None and sign.danger_confirmed:
                    bridge.send_sign_stop()
                    if hasattr(logger, "log_sign_danger"):
                        logger.log_sign_danger(sign.reason)
            else:
                sign = None


            # ── headless mode ────────────────
            if args.no_display:
                time.sleep(0.02)
                continue

            # ── display mode ─────────────────
            frame = vision.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            us     = bridge.get_ultrasonic()
            vis    = vision.get_latest()
            lstats = logger.stats()
            sstate = safety.get_state()
            env    = vision.get_environment()          # ← جديد

            annotated = draw_hud(
                frame, us, vis,
                lstats, sstate,
                paused=paused[0],
                env_label=env.label,                   # ← جديد
                env_conf=env.confidence,               # ← جديد
            )

            cv2.imshow("Landmine Robot — Vision", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                log.info("Q pressed — quitting")
                break

            elif key == ord('p'):
                paused[0] = not paused[0]
                if paused[0]:
                    bridge.send_stop()
                    log.info("Paused")
                else:
                    log.info("Resumed")

            elif key == ord('r'):
                bridge.send_reset()
                log.info("Reset sent to Arduino")

            elif key == ord('c'):
                log.info("Clearing all logs...")
                logger.clear_all()

    except KeyboardInterrupt:
        pass

    finally:
        log.info("Stopping all components...")
        logger.stop()
        safety.stop()
        vision.stop()
        bridge.stop()
        cv2.destroyAllWindows()
        log.info("System stopped cleanly.")
        log.info(f"Logger stats: {logger.stats()}")


if __name__ == "__main__":
    main()