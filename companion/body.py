"""The robot's body: speech, neck, gestures and face display behind one lock.

`Palmimo.look()` only sets a target; `run()` is what moves the servos, and only
one `run()` may be in flight. Every motion therefore goes through `_motion`,
which serializes callers. With `hardware=False` everything prints instead, so
the conversation loop can be developed on a laptop.
"""

from __future__ import annotations

import logging
import threading
import time

from . import logs

log = logging.getLogger(__name__)

#: Proportional gain from face offset (frame fraction) to neck target (normalized).
_LOOK_GAIN = 0.9


class Body:
    def __init__(self, hardware: bool, echo_cancel: bool = True) -> None:
        self.hardware = hardware
        self.robot = None
        self.camera = None
        self.mic = None
        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self._pitch = 0.0
        self._yaw = 0.0
        if hardware:
            from palmimo_sdk import DynamixelDriver, EchoCanceller, FaceDisplay, HeadCamera, MicStream, Palmimo, Speaker, SpeakerConfig

            self.camera = HeadCamera()
            # processors=[] disables the default echo canceller; it needs a loopback channel (ReSpeaker).
            self.mic = MicStream(processors=[EchoCanceller()] if echo_cancel else [])
            self.robot = Palmimo(
                driver=DynamixelDriver(),
                display=FaceDisplay(),
                speaker=Speaker(SpeakerConfig(lang="en")),
                camera=self.camera,
                mic=self.mic,
            )

    # --- lifecycle --------------------------------------------------------------
    def __enter__(self) -> Body:
        if self.robot is not None:
            log.info("connecting to hardware (servos, display, speaker, camera, mic)...")
            t0 = time.perf_counter()
            try:
                self.robot.connect()  # wakes the legs, opens camera/mic/speaker/display
            except Exception:
                log.exception("hardware connect failed; is another app (e.g. palmimo_sdk.mcp) holding the servos?")
                raise
            log.info("hardware connected in %.1fs", time.perf_counter() - t0)
        else:
            log.info("no hardware: speech prints to the console, motions are logged at debug level")
        return self

    def __exit__(self, *exc) -> None:
        if self.robot is not None:
            log.info("disconnecting hardware (parking)")
            try:
                self.robot.disconnect()
            except Exception:
                log.exception("hardware disconnect failed")

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    # --- speech -----------------------------------------------------------------
    def say(self, text: str) -> None:
        """Speak and block until done (so the mic loop can ignore our own voice)."""
        log.info("say %s", logs.text(text))
        if self.robot is None:
            print(f"Palmimo: {text}", flush=True)  # laptop mode: the console is the speaker
            return
        self._speaking.set()
        t0 = time.perf_counter()
        try:
            handle = self.robot.say(text)
            if handle is None:
                log.warning("say: no speaker attached")
                return
            handle.join(timeout=30)
            if handle.is_alive():
                log.warning("say: still speaking after 30s, moving on")
            if getattr(handle, "error", None):
                log.error("say failed: %s", handle.error)
            else:
                log.debug("say done in %.2fs", time.perf_counter() - t0)
        except Exception:
            log.exception("say raised")
        finally:
            self._speaking.clear()

    # --- motion -----------------------------------------------------------------
    def _motion(self, fn, seconds: float, name: str = "motion") -> None:
        if self.robot is None:
            log.debug("motion %s (no hardware)", name)
            return
        if not self._lock.acquire(timeout=5):
            log.warning("motion %s skipped: previous motion still running after 5s", name)
            return
        try:
            accepted = fn()
            if accepted is False:  # look()/nod()/head_shake() return False while the neck is thermally locked
                log.warning("motion %s rejected: neck lock active (temp=%s)", name, self.robot.neck_temperature_c)
            self.robot.run(seconds=seconds)
            log.debug("motion %s done", name)
        except Exception:  # a motion failure must never kill the conversation
            log.exception("motion %s failed", name)
        finally:
            self._lock.release()

    def look_toward(self, center: tuple[float, float] | None) -> None:
        if center is None:
            return
        ex, ey = center[0] - 0.5, center[1] - 0.5
        self._yaw = max(-1.0, min(1.0, self._yaw + ex * _LOOK_GAIN))
        self._pitch = max(-1.0, min(1.0, self._pitch + ey * _LOOK_GAIN))
        log.debug("look pitch=%.2f yaw=%.2f", self._pitch, self._yaw)
        self._motion(lambda: self.robot.look(self._pitch, self._yaw), 0.4, "look_toward")

    def look(self, pitch: float, yaw: float, seconds: float = 0.6) -> None:
        self._pitch, self._yaw = pitch, yaw
        self._motion(lambda: self.robot.look(pitch, yaw), seconds, f"look({pitch:.2f},{yaw:.2f})")

    def look_center(self) -> None:
        self.look(0.0, 0.0, 0.4)

    def nod(self) -> None:
        self._motion(self.robot.nod if self.robot else None, 1.2, "nod")

    def head_shake(self) -> None:
        self._motion(self.robot.head_shake if self.robot else None, 1.4, "head_shake")

    def wave(self) -> None:
        self._motion(self.robot.wave if self.robot else None, 2.5, "wave")

    def expression(self, name: str, hold_ms: int = 0) -> None:
        if self.robot is None:
            return
        try:
            self.robot.set_expression(name, hold_ms)
            log.debug("expression %s hold=%dms", name, hold_ms)
        except Exception:
            log.exception("expression %s failed", name)
