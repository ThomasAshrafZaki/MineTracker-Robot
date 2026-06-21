"""
================================================================================
  arduino_bridge.py  —  Jetson Nano ↔ Arduino Mega Serial Bridge
================================================================================
  Protocol (Arduino → Jetson):
      "angle*xPos*yPos*metalDetected\n"
      Example: "45.23*1.50*0.80*0\n"

  Protocol (Jetson → Arduino):
      Single-char commands defined in Arduino updateState():
        'f' = Forward          'b' = Backward
        'l' = RotateLeft       'r' = RotateRight
        's' = Stop             'a' = Auto mode ON
        'm' = Manual mode      'R' = Reset all
        'u' = Speed up         'd' = Speed down
        'p' = Pause/Resume
        'H' = Human Safety Stop (من human_safety.py فقط — فعّالة في AUTO mode
              بس على مستوى الأردوينو؛ منفصلة عن 's' العادي ومتعارضتش مع
              أوامر الموقع a/m/f/b/l/r خالص)
      Parameter commands:
        'w <val>' = sweepWidth     'h <val>' = sweepHeight
        'W <val>' = carWidth       'L <val>' = lineWidth

  Ultrasonic reading:
      Arduino sends front/left/right via optional line:
      "U:xx.xx,L:yy.yy,R:zz.zz\n"

  Author  : Robot Team
  Version : 3.1 (bug fixes applied)
================================================================================
"""

import serial
import serial.tools.list_ports
import threading
import time
import logging
import queue
from dataclasses import dataclass, field
from typing import Optional, Callable

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────
BAUD_RATE          = 9600
READ_TIMEOUT_S     = 0.1
RECONNECT_DELAY_S  = 2.0
QUEUE_MAX          = 64
TELEMETRY_HISTORY  = 100

FRONT_STOP_CM      = 50
SIDE_MIN_CM        = 20

STATE_STRAIGHT = "straight"
STATE_TURNING  = "turning"
STATE_FINISHED = "finished"
STATE_PAUSED   = "paused"
STATE_BYPASS   = "bypassTarget"

log = logging.getLogger("ArduinoBridge")


# ──────────────────────────────────────────────
#  DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class Telemetry:
    """One frame of data received from the Arduino."""
    timestamp:      float = field(default_factory=time.time)
    angle:          float = 0.0
    x_pos:          float = 0.0
    y_pos:          float = 0.0
    metal_detected: bool  = False
    raw:            str   = ""


@dataclass
class UltrasonicData:
    """Latest ultrasonic readings (cm)."""
    timestamp: float = 0.0
    front:     float = 999.0
    left:      float = 999.0
    right:     float = 999.0

    def is_fresh(self, max_age_s: float = 0.5) -> bool:
        return (time.time() - self.timestamp) < max_age_s

    def obstacle_ahead(self) -> bool:
        return self.front < FRONT_STOP_CM

    def best_side(self) -> str:
        return "LEFT" if self.left >= self.right else "RIGHT"


# ──────────────────────────────────────────────
#  ARDUINO BRIDGE
# ──────────────────────────────────────────────

class ArduinoBridge:

    def __init__(
        self,
        port:              str  = "/dev/ttyUSB0",
        baud:              int  = BAUD_RATE,
        auto_detect:       bool = True,
        on_telemetry:      Optional[Callable[[Telemetry], None]] = None,
        on_metal_detect:   Optional[Callable[[], None]]         = None,
        on_disconnect:     Optional[Callable[[], None]]         = None,
        on_obstacle_front: Optional[Callable[[], None]]         = None,
    ):
        self.port        = port
        self.baud        = baud
        self.auto_detect = auto_detect

        self._on_telemetry  = on_telemetry
        self._on_metal      = on_metal_detect
        self._on_disconnect = on_disconnect
        self._on_obstacle_front = on_obstacle_front

        self._serial: Optional[serial.Serial] = None
        self._lock       = threading.Lock()
        self._running    = False
        self._connected  = False

        self._latest_tel = Telemetry()
        self._latest_us  = UltrasonicData()
        self._tel_history: list = []
        self._cmd_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)

        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None

        self._rx_count     = 0
        self._tx_count     = 0
        self._parse_errors = 0

        # ── Log-spam guards ──────────────────────
        # بتمنع تكرار رسائل "مش متصل" كل دورة retry (كل 2 ثانية) أو كل أمر TX.
        # بترجع False تاني أول ما الاتصال يرجع، عشان لو الانقطاع يتكرر تاني نتنبه.
        self._disconnect_logged = False
        self._tx_dropped_logged = False

        # ── Obstacle-front signal debounce ──────────
        # حماية دفاعية ضد تكرار حرف 'O' بسبب نويز/تكرار في الترانسميشن.
        # مش بديل عن cooldown الـ MissionLogger (10 ثواني) — ده أقصر
        # بكتير وهدفه بس يمنع استدعاءين للـ callback لنفس الحدث الفعلي
        # الواحد لو وصل سطر "O" مكرر بالغلط على مستوى السيريال.
        self._last_obstacle_signal_time = 0.0
        self._OBSTACLE_SIGNAL_MIN_GAP_S = 0.5

    # ── PUBLIC API ──────────────────────────────

    def start(self) -> bool:
        self._running = True
        connected = self._connect()

        self._rx_thread = threading.Thread(
            target=self._rx_loop, daemon=True, name="Arduino-RX"
        )
        self._tx_thread = threading.Thread(
            target=self._tx_loop, daemon=True, name="Arduino-TX"
        )
        self._rx_thread.start()
        self._tx_thread.start()

        log.info(f"ArduinoBridge started — port={self.port} connected={connected}")
        return connected

    def stop(self):
        log.info("ArduinoBridge stopping...")
        self._running = False

        try:
            while not self._cmd_queue.empty():
                self._cmd_queue.get_nowait()
        except Exception:
            pass

        if self._serial and self._serial.is_open:
            try:
                self.send_stop()
                time.sleep(0.1)
                self._serial.close()
            except Exception:
                pass

        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
        if self._tx_thread:
            self._tx_thread.join(timeout=2.0)

        log.info("ArduinoBridge stopped.")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── TELEMETRY ────────────────────────────────

    def get_telemetry(self) -> Telemetry:
        with self._lock:
            return self._latest_tel

    def get_telemetry_history(self, n: int = 10) -> list:
        """Return last n telemetry frames."""
        with self._lock:
            # FIX: was checking _latest_tel_history but real variable is _tel_history
            return list(self._tel_history[-n:]) if self._tel_history else []

    def get_ultrasonic(self) -> UltrasonicData:
        with self._lock:
            return self._latest_us

    def get_angle(self) -> float:
        return self._latest_tel.angle

    def get_position(self) -> tuple:
        return (self._latest_tel.x_pos, self._latest_tel.y_pos)

    def is_metal_detected(self) -> bool:
        return self._latest_tel.metal_detected

    # ── COMMANDS ─────────────────────────────────

    def send_forward(self):      self._enqueue('f')
    def send_backward(self):     self._enqueue('b')
    def send_rotate_left(self):  self._enqueue('l')
    def send_rotate_right(self): self._enqueue('r')
    def send_stop(self):         self._enqueue('s')

    def send_human_stop(self):
        """
        إشارة إيقاف خاصة بكشف الإنسان (Computer Vision / human_safety.py).
        منفصلة تماماً عن send_stop() العادي:
          - send_stop() ('s')  → إيقاف عام، بيشتغل في AUTO و MANUAL (زر/أمر يدوي).
          - send_human_stop() ('H') → بتتفعل في الأردوينو في وضع AUTO فقط،
            ومش بتتعارض مع أوامر الموقع (a/m/f/b/l/r) خالص.
        ينفع تتنادى بشكل متكرر (مثلاً كل tick من HumanSafetyMonitor) من غير قلق،
        لأن الـ throttling الخاص باللوج موجود جوه الـ bridge مش هنا.
        """
        self._enqueue('H')

    def send_sign_stop(self):
        """
        إشارة إيقاف مستقلة خاصة بكشف لوحات الخطر (sign_detector.py).
        بتبعت حرف '!' — مستقل تماماً عن send_human_stop() ('H')، بفلاج
        ومنطق Watchdog خاص بيه في الأردوينو (zegzag_code_h.ino: signStopFlag /
        ENABLE_SIGN_STOP). تفعيل/تعطيل تأثيرها على العربية بيتحكم فيه من
        الأردوينو نفسه بسطر واحد (#define ENABLE_SIGN_STOP) — مفيش أي
        تحكم أو فلاج جوه ملفات البايثون.
        """
        self._enqueue('!')

    def send_auto(self):         self._enqueue('a')
    def send_manual(self):       self._enqueue('m')
    def send_reset(self):        self._enqueue('R')
    def send_pause(self):        self._enqueue('p')
    def send_speed_up(self):     self._enqueue('u')
    def send_speed_down(self):   self._enqueue('d')

    def send_sweep_width(self,  value: float): self._enqueue(f"w {value:.2f}")
    def send_sweep_height(self, value: float): self._enqueue(f"h {value:.2f}")
    def send_car_width(self,    value: float): self._enqueue(f"W {value:.2f}")
    def send_line_width(self,   value: float): self._enqueue(f"L {value:.2f}")

    # ── HIGH-LEVEL NAVIGATION ────────────────────

    def navigate(self, direction: str):
        mapping = {
            "FORWARD": self.send_forward,
            "LEFT":    self.send_rotate_left,
            "RIGHT":   self.send_rotate_right,
            "STOP":    self.send_stop,
            "BACK":    self.send_backward,
        }
        fn = mapping.get(direction.upper())
        if fn:
            fn()
            log.debug(f"navigate → {direction}")
        else:
            log.warning(f"Unknown direction: {direction}")

    def update_ultrasonic(self, front: float, left: float, right: float):
        """Update ultrasonic data from external source (e.g. separate parser)."""
        with self._lock:
            self._latest_us = UltrasonicData(
                timestamp=time.time(),
                front=float(front),
                left=float(left),
                right=float(right),
            )

    # ── INTERNAL ─────────────────────────────────

    def _connect(self) -> bool:
        port = self.port

        if self.auto_detect and not self._port_exists(port):
            detected = self._detect_arduino_port()
            if detected:
                port = detected
                self.port = port
                log.info(f"Auto-detected Arduino on {port}")
            else:
                if not self._disconnect_logged:
                    log.warning("Arduino not connected — will keep retrying silently in the background.")
                    self._disconnect_logged = True
                self._connected = False
                return False

        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self.baud,
                timeout=READ_TIMEOUT_S,
                write_timeout=1.0,
            )
            time.sleep(2.0)
            self._serial.reset_input_buffer()
            self._connected = True
            log.info(f"Arduino connected ✓ — port={port} @ {self.baud} baud")
            # نصفّر الـ guards عشان لو الاتصال اتقطع تاني نتنبه من جديد
            self._disconnect_logged = False
            self._tx_dropped_logged = False
            return True
        except serial.SerialException as e:
            if not self._disconnect_logged:
                log.warning(f"Arduino not connected ({port}) — will keep retrying silently in the background.")
                self._disconnect_logged = True
            else:
                log.debug(f"Retry failed to open {port}: {e}")
            self._connected = False
            return False

    def _detect_arduino_port(self) -> Optional[str]:
        candidates = []
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            mfr  = (p.manufacturer or "").lower()
            if any(k in desc + mfr for k in ["arduino", "mega", "ch340", "cp210", "ftdi", "usb serial"]):
                candidates.append(p.device)

        if not candidates:
            import os
            for path in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]:
                if os.path.exists(path):
                    candidates.append(path)

        log.debug(f"Arduino port candidates: {candidates}")
        return candidates[0] if candidates else None

    @staticmethod
    def _port_exists(port: str) -> bool:
        import os
        return os.path.exists(port) if port.startswith("/dev/") else True

    # ── RX THREAD ────────────────────────────────

    def _rx_loop(self):
        while self._running:
            if not self._connected or not self._serial or not self._serial.is_open:
                self._try_reconnect()
                continue

            try:
                raw = self._serial.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # ── إشارة عائق أمامي من الأردوينو (Serial1.println("O")) ──
                # لازم نمسكها هنا *قبل* _parse_telemetry، وإلا هتدخل
                # _try_parse_ultrasonic (مش بادئة بـ "U:" فهترجع None
                # بهدوء) أو تتسجل كـ parse error في _parse_telemetry لو
                # كانت أقصر من 4 أجزاء — في الحالتين كانت هتتسجل غلط أو
                # تتجاهل بصمت من غير ما تشغّل الـ callback المطلوب.
                #
                # دفاع إضافي (مستقل عن الـ obstacleSignalSent flag اللي
                # في الأردوينو نفسه): لو وصلت أكتر من 'O' خلال أقل من
                # نص ثانية، نتجاهل التكرار. ده مش بديل عن الـ cooldown
                # الحقيقي بتاع MissionLogger (10 ثواني، بيمنع تسجيل نفس
                # العائق مرتين في الإكسل) — ده بس حماية من نويز/تكرار
                # على مستوى السيريال نفسه (سطر اتكرر بالغلط في الترانسميشن).
                if line == "O":
                    now = time.time()
                    if (now - self._last_obstacle_signal_time) > self._OBSTACLE_SIGNAL_MIN_GAP_S:
                        self._last_obstacle_signal_time = now
                        log.info("Obstacle-front signal received from Arduino ('O')")
                        if self._on_obstacle_front:
                            try:
                                self._on_obstacle_front()
                            except Exception as cb_err:
                                log.error(f"on_obstacle_front callback error: {cb_err}")
                    else:
                        log.debug("Duplicate 'O' signal within debounce window — ignored")
                    continue

                tel = self._parse_telemetry(line)
                if tel:
                    with self._lock:
                        self._latest_tel = tel
                        self._tel_history.append(tel)
                        if len(self._tel_history) > TELEMETRY_HISTORY:
                            self._tel_history.pop(0)
                    self._rx_count += 1

                    if self._on_telemetry:
                        try:
                            self._on_telemetry(tel)
                        except Exception as cb_err:
                            log.error(f"on_telemetry callback error: {cb_err}")

                    if tel.metal_detected and self._on_metal:
                        try:
                            self._on_metal()
                        except Exception:
                            pass

            except serial.SerialException as e:
                log.error(f"RX serial error: {e}")
                self._connected = False
                if self._on_disconnect:
                    try:
                        self._on_disconnect()
                    except Exception:
                        pass
            except Exception as e:
                log.error(f"RX unexpected error: {e}")
                time.sleep(0.01)

    def _parse_telemetry(self, line: str) -> Optional[Telemetry]:
        try:
            parts = line.split("*")
            if len(parts) < 4:
                self._try_parse_ultrasonic(line)
                return None

            angle = float(parts[0])
            x_pos = float(parts[1])
            y_pos = float(parts[2])
            metal = int(parts[3]) != 0

            return Telemetry(
                timestamp=time.time(),
                angle=angle,
                x_pos=x_pos,
                y_pos=y_pos,
                metal_detected=metal,
                raw=line,
            )
        except (ValueError, IndexError) as e:
            self._parse_errors += 1
            log.debug(f"Parse error on '{line}': {e}")
            return None

    def _try_parse_ultrasonic(self, line: str):
        """
        Robust parsing for ultrasonic string.
        Expected format: "U:xx.xx,L:yy.yy,R:zz.zz"

        FIX: added with self._lock to protect against race condition
             with the decision engine reading _latest_us simultaneously.
        FIX: uses split(":", 1) to handle edge cases in token parsing.
        """
        if not line.startswith("U:"):
            return
        try:
            parts = {}
            for token in line.split(","):
                if ":" in token:
                    k, v = token.split(":", 1)
                    parts[k.strip()] = float(v.strip())

            with self._lock:
                self._latest_us = UltrasonicData(
                    timestamp=time.time(),
                    front=parts.get("U", self._latest_us.front),
                    left=parts.get("L",  self._latest_us.left),
                    right=parts.get("R", self._latest_us.right),
                )
        except Exception as e:
            log.debug(f"Failed to parse ultrasonic line '{line}': {e}")

    # ── TX THREAD ────────────────────────────────

    def _tx_loop(self):
        while self._running:
            try:
                cmd = self._cmd_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if not self._connected or not self._serial or not self._serial.is_open:
                if not self._tx_dropped_logged:
                    log.warning(
                        f"Sending commands (e.g. {repr(cmd)}) — "
                        "Arduino not connected, commands will be dropped silently until it reconnects."
                    )
                    self._tx_dropped_logged = True
                continue

            try:
                payload = (cmd + "\n").encode("utf-8")
                self._serial.write(payload)
                self._serial.flush()
                self._tx_count += 1
                log.debug(f"TX → {repr(cmd)}")
            except serial.SerialException as e:
                log.error(f"TX serial error: {e}")
                self._connected = False

    def _enqueue(self, cmd: str):
        try:
            self._cmd_queue.put_nowait(cmd)
        except queue.Full:
            log.warning(f"TX queue full — dropping: {repr(cmd)}")

    def _try_reconnect(self):
        # debug مش info — عشان متتكررش الرسالة دي كل RECONNECT_DELAY_S للأبد
        # في اللوج. رسالة "مش متصل" الموحّدة بتتطبع مرة واحدة من _connect().
        log.debug(f"Retrying connection to {self.port}...")
        time.sleep(RECONNECT_DELAY_S)
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
        self._connect()

    # ── DIAGNOSTICS ──────────────────────────────

    def stats(self) -> dict:
        return {
            "connected":    self._connected,
            "port":         self.port,
            "rx_frames":    self._rx_count,
            "tx_commands":  self._tx_count,
            "parse_errors": self._parse_errors,
            "queue_depth":  self._cmd_queue.qsize(),
            "latest_angle": self._latest_tel.angle,
            "latest_pos":   (self._latest_tel.x_pos, self._latest_tel.y_pos),
            "metal":        self._latest_tel.metal_detected,
            "us_front":     self._latest_us.front,
            "us_left":      self._latest_us.left,
            "us_right":     self._latest_us.right,
        }

    def __repr__(self):
        s = self.stats()
        return (
            f"<ArduinoBridge port={s['port']} connected={s['connected']} "
            f"rx={s['rx_frames']} tx={s['tx_commands']} "
            f"angle={s['latest_angle']:.1f}° "
            f"US F={s['us_front']:.0f} L={s['us_left']:.0f} R={s['us_right']:.0f}>"
        )


# ──────────────────────────────────────────────
#  SIMULATION BRIDGE
# ──────────────────────────────────────────────

class SimulatedArduinoBridge(ArduinoBridge):
    """
    Drop-in replacement for ArduinoBridge when Arduino is not connected.
    Generates realistic simulated telemetry for development/testing.
    """

    def __init__(self, **kwargs):
        super().__init__(port="SIM", auto_detect=False, **kwargs)
        self._sim_angle    = 0.0
        self._sim_x        = 0.0
        self._sim_y        = 0.0
        self._last_cmd     = ""

        # ── Synthetic obstacle-front trigger (للاختبار بدون هاردوير) ──
        # بيتفعل كل SIM_OBSTACLE_EVERY_S تقريباً، عشان تقدر تتأكد إن
        # سلسلة MissionLogger كاملة (صورة + GPS placeholder + بيئة +
        # صف إكسل) شغالة صح وانت بتشغّل main.py --simulate من غير
        # الأردوينو ولا الـ Ultrasonic الحقيقي متوصلين.
        # ده منفصل تماماً عن أي منطق bypass حقيقي — هنا بس بيستدعي نفس
        # الـ on_obstacle_front callback اللي كان هيتنادى من سطر 'O'
        # الحقيقي، فمفيش فرق من وجهة نظر MissionLogger.
        self._sim_obstacle_elapsed = 0.0

    SIM_OBSTACLE_EVERY_S = 15.0   # كل كام ثانية simulation-time نبعت عائق وهمي

    def start(self) -> bool:
        self._running   = True
        self._connected = True
        self._rx_thread = threading.Thread(
            target=self._sim_loop, daemon=True, name="SimArduino"
        )
        self._tx_thread = threading.Thread(
            target=self._tx_loop, daemon=True, name="SimArduino-TX"
        )
        self._rx_thread.start()
        self._tx_thread.start()
        log.info("SimulatedArduinoBridge started")
        return True

    def _sim_loop(self):
        """
        Generate fake telemetry at ~10 Hz.

        FIX: ultrasonic now updates every tick unconditionally.
             Old code checked us_age > 0.5 which caused data to freeze
             after the first frame because timestamp kept getting updated
             by external calls (update_ultrasonic / vision thread).
        """
        import math
        import random

        t = 0.0
        while self._running:
            time.sleep(0.1)
            t += 0.1

            self._sim_angle = (self._sim_angle + random.gauss(0, 0.2)) % 360
            self._sim_x    += 0.001 * math.cos(math.radians(self._sim_angle))
            self._sim_y    += 0.001 * math.sin(math.radians(self._sim_angle))

            # ── Synthetic obstacle-front event (اختبار فقط) ──
            self._sim_obstacle_elapsed += 0.1
            if self._sim_obstacle_elapsed >= self.SIM_OBSTACLE_EVERY_S:
                self._sim_obstacle_elapsed = 0.0
                if self._on_obstacle_front:
                    try:
                        log.info("[SIM] Synthetic obstacle-front event")
                        self._on_obstacle_front()
                    except Exception as cb_err:
                        log.error(f"on_obstacle_front callback error (sim): {cb_err}")

            tel = Telemetry(
                timestamp=time.time(),
                angle=round(self._sim_angle, 2),
                x_pos=round(self._sim_x, 4),
                y_pos=round(self._sim_y, 4),
                metal_detected=False,
                raw="SIM",
            )

            with self._lock:
                self._latest_tel = tel
                self._tel_history.append(tel)
                if len(self._tel_history) > TELEMETRY_HISTORY:
                    self._tel_history.pop(0)

                # FIX: always generate fresh ultrasonic data every tick
                # simulate obstacle when last command was stop (front=45)
                self._latest_us = UltrasonicData(
                    timestamp=time.time(),
                    front=45.0 if self._last_cmd == 's' else 150.0,
                    left=80.0  + random.gauss(0, 2),
                    right=90.0 + random.gauss(0, 2),
                )

            if self._on_telemetry:
                try:
                    self._on_telemetry(tel)
                except Exception:
                    pass

    def _enqueue(self, cmd: str):
        self._last_cmd = cmd
        self._tx_count += 1
        log.debug(f"SIM TX → {repr(cmd)}")


# ──────────────────────────────────────────────
#  FACTORY
# ──────────────────────────────────────────────

def create_bridge(
    port:        str  = "/dev/ttyUSB0",
    simulate:    bool = False,
    auto_detect: bool = True,
    **callbacks,
) -> ArduinoBridge:
    """
    Factory — returns real or simulated bridge.

    Example:
        bridge = create_bridge(port="/dev/ttyUSB0", simulate=False)
        bridge.start()
    """
    if simulate:
        return SimulatedArduinoBridge(**callbacks)
    return ArduinoBridge(port=port, auto_detect=auto_detect, **callbacks)


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    )

    def on_tel(t: Telemetry):
        print(
            f"  TEL | angle={t.angle:6.1f}°  "
            f"x={t.x_pos:.3f}m  y={t.y_pos:.3f}m  "
            f"metal={t.metal_detected}"
        )

    print("=" * 60)
    print("  Arduino Bridge — Simulation Test")
    print("=" * 60)

    bridge = create_bridge(simulate=True, on_telemetry=on_tel)
    bridge.start()

    time.sleep(1)
    print("\n[TEST] Sending FORWARD...")
    bridge.send_forward()
    time.sleep(2)

    print("[TEST] Sending STOP...")
    bridge.send_stop()
    time.sleep(1)

    print("[TEST] Checking ultrasonic freshness...")
    us = bridge.get_ultrasonic()
    assert us.is_fresh(1.0), "Ultrasonic should be fresh!"
    print(f"  ✅ US fresh | F={us.front:.0f} L={us.left:.0f} R={us.right:.0f}")

    print("[TEST] Checking telemetry history...")
    hist = bridge.get_telemetry_history(5)
    assert len(hist) > 0, "History should not be empty!"
    print(f"  ✅ History has {len(hist)} frames")

    print(f"\n[STATS] {bridge}")
    bridge.stop()
    print("\nDone.")