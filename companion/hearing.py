"""Turns the mic stream into whole utterances (WAV bytes) on a queue.

Audio is dropped while Palmimo is speaking so it doesn't hear itself.
Nothing here leaves the device: utterances are only sent to OpenAI by the
gate in companion.py, and only for the verified owner.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
from palmimo_sdk.audio import int16_to_wav

from .vad import SileroVad, UtteranceSegmenter

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_UTTERANCE_S = 0.4


class Hearing:
    def __init__(self, mic, is_speaking, out: queue.Queue) -> None:
        self._mic = mic
        self._is_speaking = is_speaking  # callable -> bool
        self._out = out
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="hearing", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Restart on unexpected errors instead of going deaf silently.
        while not self._stop.is_set():
            try:
                self._listen()
            except Exception:
                log.exception("hearing loop crashed; restarting in 1s")
                self._stop.wait(1.0)

    def _listen(self) -> None:
        segmenter = UtteranceSegmenter(SileroVad.load(), sample_rate=SAMPLE_RATE)
        sub = self._mic.subscribe()
        try:
            self._consume(sub, segmenter)
        finally:
            sub.close()  # don't leak a subscription if we restart after a crash

    def _consume(self, sub, segmenter: UtteranceSegmenter) -> None:
        log.info("listening (VAD ready, %d Hz)", SAMPLE_RATE)
        was_speaking = False
        last_chunk = time.monotonic()
        warned_silent = False
        peak = 0
        stats_at = time.monotonic()
        dropped_short = dropped_self = 0
        while not self._stop.is_set():
            chunk = sub.get(timeout=0.5)
            now = time.monotonic()
            if chunk is None:
                if now - last_chunk > 5 and not warned_silent:
                    log.warning("no audio from the mic for %.0fs; is the mic open / declared in devices?", now - last_chunk)
                    warned_silent = True
                continue
            if warned_silent:
                log.info("mic audio is back")
                warned_silent = False
            last_chunk = now
            samples = np.asarray(chunk, dtype=np.int16).reshape(-1)
            if samples.size:
                peak = max(peak, int(np.abs(samples).max()))
            if now - stats_at >= 60:
                # Peak near 0 = muted/wrong device; near 32767 = clipping (too loud / too close).
                log.info("mic heartbeat: peak level %d/32767 in last 60s; dropped %d too-short, %d during own speech",
                         peak, dropped_short, dropped_self)
                peak, stats_at, dropped_short, dropped_self = 0, now, 0, 0

            speaking = self._is_speaking()
            if speaking or was_speaking:
                # Throw away anything captured during (or straddling) our own speech.
                if not was_speaking:
                    log.debug("muted while speaking")
                    dropped_self += 1
                segmenter.reset()
                was_speaking = speaking
                continue
            utt = segmenter.feed(samples)
            if utt is None:
                continue
            seconds = len(utt) / SAMPLE_RATE
            if seconds < MIN_UTTERANCE_S:
                dropped_short += 1
                log.debug("utterance dropped: %.2fs is shorter than %.1fs", seconds, MIN_UTTERANCE_S)
                continue
            log.info("utterance: %.2fs of speech (queue depth %d)", seconds, self._out.qsize())
            self._out.put(("audio", int16_to_wav(utt, SAMPLE_RATE)))
