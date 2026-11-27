"""Logging setup: console (shown in the Portal's log screen) + a rotating file on the device.

Log file: <data dir>/logs/companion.log (2 MB x 5). Over SSH:
    tail -f ~/.local/share/palmimo-hackathon/logs/companion.log

What people say is private, so transcripts and replies are logged only with
`--log-text`; otherwise just their length appears ("<23 chars>").
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_log_text = False

FORMAT = "%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s: %(message)s"


def setup_logging(data_dir: Path, level: str = "info", log_text: bool = False) -> Path:
    global _log_text
    _log_text = log_text
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(log_dir, 0o700)  # logs name people and timings; keep them private to this user
    log_file = log_dir / "companion.log"
    if not log_file.exists():
        log_file.touch(mode=0o600)
    os.chmod(log_file, 0o600)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if level == "debug" else logging.INFO)
    fmt = logging.Formatter(FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file.setFormatter(fmt)
    console._companion = file._companion = True  # type: ignore[attr-defined]
    # Replace only our own handlers (re-setup, tests), never someone else's.
    for h in [h for h in root.handlers if getattr(h, "_companion", False)]:
        root.removeHandler(h)
        h.close()
    root.addHandler(console)
    root.addHandler(file)
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # A crash in a background thread (hearing, camera consumer, memories) would otherwise
    # vanish silently and the robot would just stop listening.
    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("crash").critical(
            "uncaught exception in thread %s", args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_hook
    sys.excepthook = lambda t, v, tb: logging.getLogger("crash").critical("uncaught exception", exc_info=(t, v, tb))
    return log_file


def text(s: str) -> str:
    """Render user/robot speech for a log line, respecting --log-text."""
    return repr(s) if _log_text else f"<{len(s)} chars>"


@contextmanager
def timed(timings: dict[str, float], key: str):
    """Record how long a block took, in seconds, into timings[key]."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - t0


def fmt_timings(timings: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.2f}s" for k, v in timings.items())
