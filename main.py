"""Palmimo personal companion: knows its owner's face, remembers them, chats only with them.

Modes:
  enroll  learn the owner's face (Palmimo sweeps its head while you look at it)
  run     the companion: chats with the owner, politely declines everyone else
  forget  erase a person's face, history and memories

Off the robot, `--no-hardware` runs on a laptop: `--webcam` uses the laptop camera for
the face gate, otherwise you type as the owner (prefix a line with `!stranger` to test
the stranger path).
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
from pathlib import Path

from companion.vault import default_data_dir


def _typed_input(inbox: queue.Queue) -> None:
    print("Type to talk (Ctrl-D to quit). Prefix with !stranger to simulate someone else.", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            inbox.put(("text", line))
    inbox.put(("quit", None))


def main() -> None:
    ap = argparse.ArgumentParser(description="Palmimo personal companion")
    ap.add_argument("--mode", choices=["run", "enroll", "forget"], default="run")
    ap.add_argument("--name", default="", help="owner's name (enroll / forget)")
    ap.add_argument("--role", choices=["owner", "guest"], default="owner", help="enroll as owner or guest")
    ap.add_argument("--chat-model", default="gpt-5-mini")
    ap.add_argument("--stt-model", default="gpt-4o-mini-transcribe")
    ap.add_argument("--face-threshold", type=float, default=0.40)
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--no-hardware", action="store_true", help="run on a laptop without the robot")
    ap.add_argument("--webcam", action="store_true", help="with --no-hardware: use the laptop camera")
    ap.add_argument("--no-echo-cancel", action="store_true", help="for mics without a loopback channel")
    ap.add_argument("--log-level", choices=["info", "debug"], default="info",
                    help="debug adds every face match score, motion and token count")
    ap.add_argument("--log-text", action="store_true",
                    help="include what was said in logs (off by default: conversations are private)")
    ap.add_argument("-v", "--verbose", action="store_true", help="same as --log-level debug")
    args = ap.parse_args()

    from companion.logs import setup_logging

    log_file = setup_logging(args.data_dir, "debug" if args.verbose else args.log_level, args.log_text)
    log = logging.getLogger("companion")
    log.info("=== palmimo companion starting: mode=%s ===", args.mode)
    log.info("app id=%s, data dir=%s, log file=%s", os.environ.get("PALMIMO_APP_ID", "(outside Portal)"), args.data_dir, log_file)
    log.info("config: chat_model=%s stt_model=%s face_threshold=%.2f hardware=%s webcam=%s echo_cancel=%s log_text=%s",
             args.chat_model, args.stt_model, args.face_threshold, not args.no_hardware, args.webcam,
             not args.no_echo_cancel, args.log_text)
    log.info("python %s, palmimo-sdk %s, OPENAI_API_KEY %s", sys.version.split()[0], _version("palmimo-sdk"),
             "set" if os.environ.get("OPENAI_API_KEY") else "NOT set")

    from companion.store import Store

    store = Store(args.data_dir)
    people = store.people()
    log.info("store: %d people enrolled (%s)", len(people),
             ", ".join(f"{p.name}/{p.role}{'' if p.face_emb is not None else ' no-face'}" for p in people) or "none")

    if args.mode == "forget":
        target = store.find(args.name) if args.name else store.owner()
        if target is None:
            print("Nobody to forget.")
            return
        store.delete_person(target.id)
        log.info("forget: erased %s", target.name)
        print(f"Erased {target.name}'s face, conversation history and memories.")
        return

    from companion.body import Body
    from companion.face import FaceEngine, PresenceTracker

    hardware = not args.no_hardware
    body = Body(hardware=hardware, echo_cancel=not args.no_echo_cancel)
    camera = body.camera
    if not hardware and args.webcam:
        from companion.webcam import Webcam

        camera = Webcam().start()

    with body:
        if args.mode == "enroll":
            if camera is None:
                sys.exit("enroll needs a camera: run on the robot, or add --webcam")
            if not args.name:
                sys.exit("enroll needs --name")
            from companion.enroll import enroll

            ok = enroll(body, camera, FaceEngine(), store, args.name.strip(), args.role)
            log.info("enroll %s", "succeeded" if ok else "FAILED (see messages above)")
            sys.exit(0 if ok else 1)

        if not os.environ.get("OPENAI_API_KEY"):
            log.critical("OPENAI_API_KEY is not set")
            sys.exit("OPENAI_API_KEY is not set (set it in the Portal's environment screen, or export it)")
        from companion.brain import Brain
        from companion.companion import Companion

        tracker = PresenceTracker(FaceEngine(), store.people, threshold=args.face_threshold)
        if camera is not None:
            camera.add_consumer(tracker)

        inbox: queue.Queue = queue.Queue()
        if body.mic is not None:
            from companion.hearing import Hearing

            Hearing(body.mic, lambda: body.speaking, inbox).start()
        else:
            threading.Thread(target=_typed_input, args=(inbox,), daemon=True).start()

        companion = Companion(body, store, Brain(args.chat_model, args.stt_model), tracker, inbox,
                              typed_presence=camera is None)
        def _on_sigterm(*_):
            log.info("SIGTERM (Portal stop): saving memories and parking")
            companion.stop()

        signal.signal(signal.SIGTERM, _on_sigterm)
        try:
            companion.run()
        except KeyboardInterrupt:
            log.info("Ctrl-C: stopping")
    if not hardware and args.webcam:
        camera.close()
    log.info("=== stopped ===")


def _version(pkg: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return "?"


if __name__ == "__main__":
    main()
