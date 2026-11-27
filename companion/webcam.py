"""Laptop-webcam stand-in for HeadCamera, for testing the face gate off the robot.

Implements just what this app uses: `add_consumer(fn)`, `latest()`, `start()`, `close()`.
"""

from __future__ import annotations

import threading
import time

import cv2


class Webcam:
    def __init__(self, index: int = 0, width: int = 640) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError("could not open webcam (on macOS, allow camera access for your terminal)")
        self._width = width
        self._consumers = []
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="webcam", daemon=True)

    def add_consumer(self, fn) -> None:
        self._consumers.append(fn)

    def start(self) -> Webcam:
        self._thread.start()
        return self

    def latest(self, timeout: float = 2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._frame is not None:
                    return self._frame
            time.sleep(0.05)
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            h, w = frame.shape[:2]
            if w > self._width:
                frame = cv2.resize(frame, (self._width, int(h * self._width / w)))
            with self._lock:
                self._frame = frame
            for fn in self._consumers:
                try:
                    fn(frame, time.monotonic())
                except Exception:
                    pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()
