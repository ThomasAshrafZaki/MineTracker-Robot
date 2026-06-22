"""
================================================================================
  watchdog.py  —  Safety Watchdog Thread
================================================================================
  - لو المين لوب وقف أكتر من timeout → يبعت STOP فوراً
  - بيحمي من: crash / freeze / exception غير متوقع
  - Thread-safe تماماً
  - يسجل كل triggered event في الـ log

  Author  : Robot Team
  Version : 1.1 (bug fixes applied)
================================================================================
"""

import threading
import time
import logging
from typing import Callable, Optional

log = logging.getLogger("Watchdog")


class Watchdog:
    """
    Safety watchdog — يراقب المين لوب.

    الاستخدام:
        wd = Watchdog(on_trigger=bridge.send_stop, timeout_s=1.0)
        wd.start()
        # في كل loop iteration:
        wd.ping()
        # عند الإغلاق:
        wd.stop()

    لو ping() ما اتعملتش في خلال timeout_s ثانية:
        → يستدعي on_trigger() تلقائياً
        → يسجل warning في الـ log
        → يكرر كل triggered_repeat_s لو لسه مش شاغل
    """

    def __init__(
        self,
        on_trigger:         Callable,
        timeout_s:          float = 1.0,
        triggered_repeat_s: float = 0.5,
        name:               str   = "Watchdog",
    ):
        self._on_trigger = on_trigger
        self._timeout    = timeout_s
        self._repeat     = triggered_repeat_s
        self._name       = name

        self._last_ping      = time.time()
        self._running        = False
        self._triggered      = False
        self._trigger_count  = 0
        self._lock           = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ── PUBLIC API ──────────────────────────────

    def start(self):
        self._running   = True
        self._last_ping = time.time()
        self._thread    = threading.Thread(
            target=self._loop, daemon=True, name=self._name
        )
        self._thread.start()
        log.info(f"[{self._name}] Started — timeout={self._timeout}s")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info(f"[{self._name}] Stopped — triggers={self._trigger_count}")

    def ping(self):
        """استدعيها في كل iteration من المين لوب."""
        with self._lock:
            self._last_ping = time.time()
            if self._triggered:
                self._triggered = False
                log.info(f"[{self._name}] Loop recovered ✓")

    @property
    def triggered(self) -> bool:
        with self._lock:
            return self._triggered

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    # ── INTERNAL ────────────────────────────────

    def _loop(self):
        """
        FIX: replaced float modulo (age % self._repeat < 0.15) with a real
             timestamp (last_repeat_time) to control repeat interval.

        The old approach was unreliable because:
          - time.sleep(0.1) doesn't wake up at exactly 0.1s
          - float % float on a monotonically growing value skips edges
            unpredictably, causing either missed repeats or bursts of calls
        """
        last_repeat_time = 0.0

        while self._running:
            time.sleep(0.05)   # tighter loop for faster response

            with self._lock:
                age       = time.time() - self._last_ping
                triggered = self._triggered

            if age > self._timeout:
                if not triggered:
                    # first trigger
                    self._trigger_count += 1
                    with self._lock:
                        self._triggered = True
                    log.warning(
                        f"[{self._name}] ⚠️ TRIGGERED "
                        f"(no ping for {age:.2f}s) — calling on_trigger()"
                    )
                    try:
                        self._on_trigger()
                    except Exception as e:
                        log.error(f"[{self._name}] on_trigger() error: {e}")
                    last_repeat_time = time.time()

                else:
                    # still triggered — repeat on real timer, not modulo
                    if time.time() - last_repeat_time >= self._repeat:
                        log.warning(
                            f"[{self._name}] Still triggered "
                            f"({age:.1f}s) — repeating on_trigger()"
                        )
                        try:
                            self._on_trigger()
                        except Exception as e:
                            log.error(
                                f"[{self._name}] on_trigger() repeat error: {e}"
                            )
                        last_repeat_time = time.time()

    def __repr__(self):
        with self._lock:
            age = time.time() - self._last_ping
        return (
            f"<Watchdog timeout={self._timeout}s "
            f"age={age:.2f}s triggered={self._triggered} "
            f"count={self._trigger_count}>"
        )


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    )

    triggered_calls = []

    def fake_stop():
        triggered_calls.append(time.time())
        print(f"  STOP called! (total={len(triggered_calls)})")

    print("TEST 1: Normal operation — ping every 0.2s, timeout=1.0s")
    wd = Watchdog(on_trigger=fake_stop, timeout_s=1.0)
    wd.start()
    for _ in range(20):
        time.sleep(0.2)
        wd.ping()
    assert len(triggered_calls) == 0, f"Should not trigger: {len(triggered_calls)}"
    wd.stop()
    print("  ✅ No false triggers")

    print("TEST 2: Freeze — no ping for 1.0s → must trigger")
    triggered_calls.clear()
    wd2 = Watchdog(on_trigger=fake_stop, timeout_s=0.5)
    wd2.start()
    time.sleep(0.2)
    wd2.ping()
    time.sleep(1.0)
    assert len(triggered_calls) >= 1, "Should have triggered"
    print(f"  ✅ Triggered {len(triggered_calls)} time(s)")
    wd2.stop()

    print("TEST 3: Recovery after freeze")
    triggered_calls.clear()
    wd3 = Watchdog(on_trigger=fake_stop, timeout_s=0.5)
    wd3.start()
    time.sleep(0.8)
    assert len(triggered_calls) >= 1
    wd3.ping()
    time.sleep(0.1)
    assert not wd3.triggered, "Should be recovered"
    print("  ✅ Recovered after ping")
    wd3.stop()

    print("TEST 4: Repeat trigger — must not burst")
    triggered_calls.clear()
    wd4 = Watchdog(on_trigger=fake_stop, timeout_s=0.3, triggered_repeat_s=0.5)
    wd4.start()
    time.sleep(1.5)   # freeze for 1.5s, repeat=0.5s → expect ~2-3 calls
    count = len(triggered_calls)
    assert 2 <= count <= 4, f"Expected 2-4 repeat calls, got {count}"
    print(f"  ✅ Repeat calls={count} (no burst)")
    wd4.stop()

    print("\n✅ All Watchdog tests passed")
