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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-18s] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/robot.log", mode="a"),
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
    env_label:    str  = "",      # ← جديد
    env_conf:     float = 0.0,    # ← جديد
    sign_danger:  bool  = False,  # ← جديد: sign detector result
    sign_reason:  str   = "",     # ← جديد
) -> np.ndarray:

    h, w = frame.shape[:2]

    # ── bottom bar (70px) ─────────────────────
    cv2.rectangle(frame, (0, h - 72), (w, h), _C_BLACK, -1)

    # ultrasonic readings
    f_color = _C_RED if us.front < 50 else _C_GREEN
    cv2.putText(frame, f"F:{us.front:5.1f}cm",
                (10, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, f_color, 2)
    cv2.putText(frame, f"L:{us.left:5.1f}cm",
                (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, _C_WHITE, 1)
    cv2.putText(frame, f"R:{us.right:5.1f}cm",
                (185, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, _C_WHITE, 1)

    # robot / safety state
    safety_color = _C_RED if safety_state != "CLEAR" else _C_GREEN
    state_label  = "STATE: PAUSED" if paused else "ARDUINO: AUTO"
    sc           = _C_GRAY if paused else _C_GREEN
    cv2.putText(frame, state_label,
                (w // 2 - 85, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.68, sc, 2)

    # human safety state
    cv2.putText(frame, f"SAFETY: {safety_state}",
                (w // 2 - 85, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, safety_color, 1)

    # stats top-right — أضفنا ENV count
    cv2.putText(
        frame,
        f"MINES:{logger_stats.get('mines_logged', 0)}"
        f"  OBS:{logger_stats.get('obstacles_logged', 0)}"
        f"  ENV:{logger_stats.get('environments_logged', 0)}",
        (w - 310, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, _C_GRAY, 1
    )

    # logger disk usage
    mine_mb = logger_stats.get("mine_dir_size", 0.0)
    obs_mb  = logger_stats.get("obstacle_dir_size", 0.0)
    cv2.putText(
        frame,
        f"LOG: mines={mine_mb:.1f}MB  obs={obs_mb:.1f}MB",
        (w - 310, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_GRAY, 1
    )

    # ── Environment label — أسفل يسار فوق الـ bottom bar ──
    if env_label and env_label != "Unknown":
        env_text  = f"ENV: {env_label}  ({env_conf:.0%})"
        env_color = _C_ENV
    else:
        env_text  = "ENV: Scanning..."
        env_color = _C_GRAY

    cv2.putText(
        frame, env_text,
        (10, h - 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, env_color, 1
    )

    # ── Sign danger banner — وسط الشاشة لو فيه لوحة خطر متأكدة ──
    if sign_danger:
        banner_y1 = h // 2 - 28
        banner_y2 = h // 2 + 28
        cv2.rectangle(frame, (0, banner_y1), (w, banner_y2), (0, 0, 160), -1)
        cv2.putText(
            frame,
            f"!! DANGER SIGN [{sign_reason}] !!",
            (w // 2 - 165, h // 2 + 9),
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

    # ── build components ─────────────────────
    bridge = create_bridge(
        port=args.port,
        simulate=args.simulate,
        auto_detect=not args.simulate,
    )

    vision = VisionProcessor(
        camera_index=args.camera,
        use_yolo=not args.no_yolo,
        use_env=not args.no_env,          # ← جديد: تصنيف البيئة
        device=args.device,
        simulate=args.simulate,
    )

    # نحتفظ بفلاق الـ sign هنا في main لأن VisionProcessor بيشغله تلقائياً
    # لو --no-sign اتعمل، هنتجاهل نتيجة الـ sign في الـ main loop
    _use_sign = not args.no_sign

    # callbacks للـ human safety
    def on_human_detected(event: HumanEvent):
        log.warning(
            f"HUMAN DETECTED | conf={event.confidence:.0%} | "
            f"image={event.saved_path}"
        )

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
            if not safety.is_safe():
                bridge.send_stop()

            # ── sign detector gate ────────────
            sign = vision.get_latest_sign() if _use_sign else None
            if sign is not None and sign.danger_confirmed:
                bridge.send_stop()
                if hasattr(logger, "log_sign_danger"):
                    logger.log_sign_danger(sign.reason)


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
            sign   = vision.get_latest_sign() if _use_sign else None

            annotated = draw_hud(
                frame, us, vis,
                lstats, sstate,
                paused=paused[0],
                env_label=env.label,                   # ← جديد
                env_conf=env.confidence,               # ← جديد
                sign_danger=sign.danger_confirmed if sign else False,
                sign_reason=sign.reason        if sign else "",
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