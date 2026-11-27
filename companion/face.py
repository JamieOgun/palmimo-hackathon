"""Local face recognition: OpenCV YuNet (detect) + SFace (128-d embedding).

Everything here runs on the Pi's CPU; frames and embeddings never leave it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from palmimo_sdk.download import default_model_dir, download_atomic

from .store import Person

log = logging.getLogger(__name__)

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
YUNET = ("face_detection_yunet_2023mar.onnx", f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
         "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")
SFACE = ("face_recognition_sface_2021dec.onnx", f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
         "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79")

#: Cosine similarity at/above which a face counts as a known person. OpenCV's
#: published SFace threshold is 0.363; 0.40 is a little stricter.
DEFAULT_THRESHOLD = 0.40

#: Faces smaller than this fraction of frame width are too far away to trust.
MIN_FACE_FRAC = 0.08


def _model(spec: tuple[str, str, str]) -> str:
    name, url, sha = spec
    path = Path(default_model_dir()) / name
    if not path.exists():
        log.info("downloading %s -> %s", name, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_atomic(url, path, timeout=120.0, sha256=sha)
        except Exception:
            log.exception("could not download %s (no internet? copy it into %s by hand)", name, path.parent)
            raise
    return str(path)


@dataclass
class FaceHit:
    embedding: np.ndarray  # L2-normalized, float32, shape (128,)
    center: tuple[float, float]  # normalized 0..1 (x, y) in the frame
    width_frac: float
    score: float


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


class FaceEngine:
    """Thread-safe wrapper: OpenCV DNN objects are not safe to share across threads."""

    def __init__(self) -> None:
        t0 = time.perf_counter()
        self._det = cv2.FaceDetectorYN.create(_model(YUNET), "", (320, 240), score_threshold=0.8)
        self._rec = cv2.FaceRecognizerSF.create(_model(SFACE), "")
        self._lock = threading.Lock()
        log.info("face models loaded in %.2fs (opencv %s)", time.perf_counter() - t0, cv2.__version__)

    def largest_face(self, frame: np.ndarray) -> FaceHit | None:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        with self._lock:
            self._det.setInputSize((w, h))
            _, faces = self._det.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            row = max(faces, key=lambda f: f[2] * f[3])
            if row[2] / w < MIN_FACE_FRAC:
                log.debug("face too small: %.1f%% of frame width (%d faces)", 100 * row[2] / w, len(faces))
                return None
            aligned = self._rec.alignCrop(frame, row)
            feat = self._rec.feature(aligned)
        x, y, fw, fh = row[:4]
        log.debug("face found: %d in view, largest %.0f%% wide, score %.2f, %.0fms",
                  len(faces), 100 * fw / w, row[-1], 1000 * (time.perf_counter() - t0))
        return FaceHit(normalize(feat), ((x + fw / 2) / w, (y + fh / 2) / h), float(fw / w), float(row[-1]))


def best_match(emb: np.ndarray, people: list[Person], threshold: float) -> tuple[Person | None, float]:
    best, best_sim = None, -1.0
    for p in people:
        if p.face_emb is None:
            continue
        sim = cosine(emb, p.face_emb)
        if sim > best_sim:
            best, best_sim = p, sim
    return (best, best_sim) if best_sim >= threshold else (None, best_sim)


Kind = Literal["owner", "guest", "stranger", "none"]


@dataclass
class Presence:
    kind: Kind
    person: Person | None = None
    center: tuple[float, float] | None = None
    similarity: float = 0.0


class PresenceTracker:
    """Camera consumer that keeps a debounced "who is in front of me" answer.

    Registered with ``HeadCamera.add_consumer``; runs on the camera's drain
    thread, throttled to ``hz``. A recognized person is held for ``hold_s``
    after the last sighting so a blink or a head turn doesn't drop them.
    """

    def __init__(self, engine: FaceEngine, people_fn, threshold: float = DEFAULT_THRESHOLD,
                 hz: float = 5.0, hold_s: float = 2.0) -> None:
        self._engine = engine
        self._people_fn = people_fn  # callable -> list[Person]; re-read so enrollments apply live
        self.threshold = threshold
        self._period = 1.0 / hz
        self._hold = hold_s
        self._lock = threading.Lock()
        self._last_run = 0.0
        self._known: Presence = Presence("none")
        self._known_at = 0.0
        self._raw: Presence = Presence("none")
        self._raw_at = 0.0
        self._last_logged: tuple | None = None
        self._frames = 0
        self._stats_at = time.monotonic()

    def __call__(self, frame: np.ndarray, ts: float) -> None:
        now = time.monotonic()
        if now - self._last_run < self._period:
            return
        self._last_run = now
        try:
            hit = self._engine.largest_face(frame)
        except Exception:
            log.exception("face detection failed on a frame")
            return
        self.observe(hit, now)
        self._frames += 1
        if now - self._stats_at >= 60:
            log.info("face gate heartbeat: %d frames checked in the last %.0fs, now=%s",
                     self._frames, now - self._stats_at, self.current(now).kind)
            self._frames, self._stats_at = 0, now

    def observe(self, hit: FaceHit | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if hit is None:
            p = Presence("none")
        else:
            person, sim = best_match(hit.embedding, self._people_fn(), self.threshold)
            kind: Kind = person.role if person else "stranger"  # type: ignore[assignment]
            p = Presence(kind, person, hit.center, sim)
            log.debug("match: best=%s sim=%.3f threshold=%.2f -> %s",
                      person.name if person else "-", sim, self.threshold, kind)
        key = (p.kind, p.person.id if p.person else None)
        if key != self._last_logged:
            # Only log changes: at 5 checks/s every frame would drown the log.
            log.info("presence: %s -> %s%s (similarity %.2f, threshold %.2f)",
                     self._last_logged[0] if self._last_logged else "start", p.kind,
                     f" {p.person.name}" if p.person else "", p.similarity, self.threshold)
            self._last_logged = key
        with self._lock:
            self._raw, self._raw_at = p, now
            if p.person is not None:
                self._known, self._known_at = p, now

    def current(self, now: float | None = None) -> Presence:
        now = time.monotonic() if now is None else now
        with self._lock:
            # A stranger seen *after* the known person takes over immediately: no hold for impostors.
            if self._known.person is not None and now - self._known_at <= self._hold:
                if not (self._raw.kind == "stranger" and self._raw_at > self._known_at):
                    return self._known
            if now - self._raw_at <= self._hold:
                return self._raw
            return Presence("none")
