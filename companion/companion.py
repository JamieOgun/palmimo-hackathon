"""The companion loop: the identity gate, sessions, conversation and memory.

Privacy rule: audio and text go to OpenAI only when the verified owner's face
is in view. Everyone else gets a local reply and no cloud call.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field

from . import logs
from .body import Body
from .brain import Brain
from .face import Presence, PresenceTracker
from .store import Memory, Person, Store

log = logging.getLogger(__name__)

#: Session ends after the owner has been out of view this long; memories are consolidated then.
SESSION_TIMEOUT_S = 60.0
#: How long the owner must be gone before Palmimo greets them again on return.
REGREET_AFTER_S = 300.0
HISTORY_TURNS = 20
ALL_MEMORIES_UP_TO = 50
#: Minimum gap between "I don't know you" reactions, so background chatter doesn't make it spam.
STRANGER_COOLDOWN_S = 8.0

_REMEMBER = re.compile(r"\bwhat do you (remember|know) about me\b", re.I)
_FORGET_LAST = re.compile(r"^\s*(please\s+)?forget (that|what i just said)\b", re.I)
_FORGET_ALL = re.compile(r"\bforget everything( about me)?\b", re.I)
_WORD = re.compile(r"[a-z']{4,}")


def select_memories(memories: list[Memory], text: str) -> list[Memory]:
    """All memories if few; otherwise the 20 most recent plus keyword matches."""
    if len(memories) <= ALL_MEMORIES_UP_TO:
        return memories
    words = set(_WORD.findall(text.lower()))
    recent = memories[-20:]
    matched = [m for m in memories[:-20] if words & set(_WORD.findall(m.fact.lower()))]
    return matched[-15:] + recent


@dataclass
class Session:
    owner: Person
    started: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.monotonic)
    turns: int = 0


class Companion:
    def __init__(self, body: Body, store: Store, brain: Brain, tracker: PresenceTracker,
                 inbox: queue.Queue, typed_presence: bool = False) -> None:
        self.body = body
        self.store = store
        self.brain = brain
        self.tracker = tracker
        self.inbox = inbox
        # Laptop mode with no camera: typed lines count as the owner unless prefixed with "!stranger".
        self.typed_presence = typed_presence
        self.session: Session | None = None
        self._owner_gone_at: float | None = None
        self._last_stranger_reaction = 0.0
        self._turn_id = 0
        self._stop = threading.Event()

    # --- main loop --------------------------------------------------------------
    def run(self) -> None:
        owner = self.store.owner()
        if owner is None:
            self.body.say("I don't have an owner yet. Run me in enroll mode first.")
            return
        self.body.expression("SLEEPY", 2000)
        mems = self.store.memories(owner.id)
        msgs = self.store.recent_messages(owner.id, limit=1000)
        log.info("ready: owner=%s, %d memories, %d stored messages, face enrolled=%s, presence=%s",
                 owner.name, len(mems), len(msgs), owner.face_emb is not None,
                 "typed (always owner)" if self.typed_presence else "camera")
        try:
            while not self._stop.is_set():
                self._tick_presence()
                try:
                    kind, payload = self.inbox.get(timeout=0.5)
                except queue.Empty:
                    continue
                if kind == "quit":
                    log.info("quit requested")
                    break
                try:
                    self._handle(kind, payload)
                except Exception:
                    log.exception("turn %d crashed; continuing", self._turn_id)
        finally:
            self._end_session(blocking=True)

    def stop(self) -> None:
        self._stop.set()

    def _presence(self) -> Presence:
        if self.typed_presence:
            owner = self.store.owner()
            return Presence("owner", owner, None, 1.0)
        return self.tracker.current()

    def _tick_presence(self) -> None:
        p = self._presence()
        now = time.monotonic()
        if p.kind == "owner":
            if self.session is None:
                self._start_session(p)
            else:
                self.session.last_seen = now
            if self._owner_gone_at is not None and self.session is not None:
                log.info("owner back after %.0fs", now - self._owner_gone_at)
            self._owner_gone_at = None
        elif self.session is not None:
            if self._owner_gone_at is None:
                self._owner_gone_at = now
                log.info("owner out of view; session ends in %.0fs unless they come back", SESSION_TIMEOUT_S)
            if now - self.session.last_seen > SESSION_TIMEOUT_S:
                self._end_session()

    # --- sessions ---------------------------------------------------------------
    def _start_session(self, p: Presence) -> None:
        assert p.person is not None
        last = self.store.recent_messages(p.person.id, limit=1)
        away = time.time() - last[0].ts if last else None
        self.session = Session(p.person)
        log.info("session start: %s (similarity %.2f)", p.person.name, p.similarity)
        if self.typed_presence:
            return
        self.body.look_toward(p.center)
        if away is None or away > REGREET_AFTER_S:
            self.body.expression("HAPPY", 2500)
            self.body.wave()
            self.body.say(f"Hi {p.person.name}! " + ("Good to see you." if away is None else "Welcome back."))

    def _end_session(self, blocking: bool = False) -> None:
        s, self.session = self.session, None
        if s is None:
            return
        if s.turns == 0:
            log.info("session end: %s, no conversation, nothing to remember", s.owner.name)
            return
        log.info("session end: %s, %d turns; consolidating memories", s.owner.name, s.turns)
        t = threading.Thread(target=self._consolidate, args=(s,), name="memories", daemon=not blocking)
        t.start()
        if blocking:
            t.join(timeout=60)

    def _consolidate(self, s: Session) -> None:
        try:
            convo = self.store.recent_messages(s.owner.id, limit=200, since=s.started)
            current = self.store.memories(s.owner.id)
            t0 = time.perf_counter()
            updated = self.brain.consolidate_memories(s.owner.name, current, convo)
            log.info("memory consolidation took %.1fs over %d messages", time.perf_counter() - t0, len(convo))
            if updated is None:
                log.warning("memory consolidation returned nothing usable; keeping old memories")
                return
            self.store.replace_memories(s.owner.id, updated)
            log.info("memories: %d -> %d", len(current), len(updated))
        except Exception as e:
            log.error("memory consolidation failed: %s", _explain(e), exc_info=not _is_api_error(e))

    # --- one utterance ----------------------------------------------------------
    def _handle(self, kind: str, payload) -> None:
        self._turn_id += 1
        tid = self._turn_id
        if kind == "text" and isinstance(payload, str) and payload.startswith("!stranger"):
            log.info("turn %d: simulated stranger (typed)", tid)
            self._react_to_stranger(Presence("stranger"))
            return
        p = self._presence()
        if p.kind != "owner" or p.person is None:
            # Nothing is sent to the cloud for anyone but the owner.
            log.info("turn %d: gate=%s (similarity %.2f) -> local reply only, nothing sent to OpenAI",
                     tid, p.kind, p.similarity)
            self._react_to_stranger(p)
            return
        if self.session is None:
            self._start_session(p)
        assert self.session is not None
        owner = self.session.owner
        log.info("turn %d: gate=owner %s (similarity %.2f) -> %s", tid, owner.name, p.similarity,
                 "speech-to-text + chat" if kind == "audio" else "chat (typed)")

        t: dict[str, float] = {}
        self.body.expression("THINKING")
        try:
            with logs.timed(t, "stt"):
                text = self.brain.transcribe(payload) if kind == "audio" else str(payload)
        except Exception as e:
            log.error("turn %d: speech-to-text failed: %s", tid, _explain(e), exc_info=not _is_api_error(e))
            self.body.expression("SAD", 1500)
            return
        if not text or len(text) < 2:
            log.info("turn %d: empty transcript (noise?), ignoring", tid)
            self.body.expression("IDLE")
            return
        log.info("turn %d: heard %s", tid, logs.text(text))
        self.body.look_toward(p.center)

        answer = self._local_command(owner, text)
        if answer is not None:
            log.info("turn %d: handled locally as a memory command", tid)
        else:
            history = self.store.recent_messages(owner.id, limit=HISTORY_TURNS)
            all_mems = self.store.memories(owner.id)
            memories = select_memories(all_mems, text)
            log.debug("turn %d: context = %d/%d memories, %d history messages", tid, len(memories), len(all_mems), len(history))
            try:
                with logs.timed(t, "chat"):
                    answer = self.brain.reply(owner.name, memories, history, text)
            except Exception as e:
                log.error("turn %d: chat failed: %s", tid, _explain(e), exc_info=not _is_api_error(e))
                self.body.expression("SAD", 1500)
                self.body.say("Sorry, my brain hiccuped. Can you say that again?")
                return
            if not answer:
                answer = "Hmm, I'm not sure what to say to that."
            self.store.add_message(owner.id, "user", text)
            self.store.add_message(owner.id, "assistant", answer)
            self.session.turns += 1

        self.body.expression("SMILE")
        with logs.timed(t, "speak"):
            self.body.say(answer)
        if self.session and self.session.turns % 3 == 1:
            self.body.nod()
        self.body.expression("IDLE")
        log.info("turn %d: done, %s (session turns=%d)", tid, logs.fmt_timings(t), self.session.turns if self.session else 0)

    def _react_to_stranger(self, p: Presence) -> None:
        if p.kind == "none":
            log.info("speech ignored: nobody in view (if you were there, check lighting / distance / --log-level debug)")
            return
        now = time.monotonic()
        if now - self._last_stranger_reaction < STRANGER_COOLDOWN_S:
            log.debug("stranger reaction skipped (cooldown %.0fs)", STRANGER_COOLDOWN_S)
            return
        self._last_stranger_reaction = now
        owner = self.store.owner()
        name = owner.name if owner else "my owner"
        if p.kind == "guest" and p.person is not None:
            self.body.expression("HAPPY", 2000)
            self.body.say(f"Hi {p.person.name}! I keep my chats for {name}, but it's nice to see you.")
        else:
            self.body.look_toward(p.center)
            self.body.head_shake()
            self.body.say(f"Hi! I don't know you yet. I only chat with {name}.")
        self.body.expression("IDLE")

    def _local_command(self, owner: Person, text: str) -> str | None:
        """Memory commands handled on-device, before the LLM."""
        if _FORGET_ALL.search(text):
            self.store.replace_memories(owner.id, [])
            return "Done. I've forgotten everything I remembered about you."
        if _FORGET_LAST.search(text):
            # Drop the last exchange from history so it never reaches memory consolidation.
            if self.store.delete_last_exchange(owner.id) == 0:
                return "There's nothing to forget yet."
            if self.session:
                self.session.turns = max(0, self.session.turns - 1)
            return "Okay, I'll forget that."
        if _REMEMBER.search(text):
            mems = self.store.memories(owner.id)
            if not mems:
                return "Not much yet! Tell me about yourself."
            picked = [m.fact for m in mems[-5:]]
            return f"I remember {len(mems)} things. For example: " + " ".join(picked)
        return None


def _is_api_error(e: Exception) -> bool:
    import openai

    return isinstance(e, openai.APIError)


def _explain(e: Exception) -> str:
    """One-line, actionable reason for an OpenAI failure (no traceback needed for these)."""
    import openai

    if isinstance(e, openai.AuthenticationError):
        return "OpenAI rejected the API key (401). Check OPENAI_API_KEY in the Portal env screen."
    if isinstance(e, openai.PermissionDeniedError):
        return f"OpenAI says this key can't use that model (403): {e.message}"
    if isinstance(e, openai.NotFoundError):
        return f"model not found (404). Check the chat_model / stt_model params: {e.message}"
    if isinstance(e, openai.RateLimitError):
        return f"rate limited or out of credits (429): {e.message}"
    if isinstance(e, openai.APITimeoutError):
        return "OpenAI request timed out (slow network?)"
    if isinstance(e, openai.APIConnectionError):
        return "can't reach OpenAI (no internet on the robot?)"
    if isinstance(e, openai.BadRequestError):
        return f"bad request (400): {e.message}"
    if isinstance(e, openai.APIStatusError):
        return f"OpenAI error {e.status_code}: {e.message}"
    return repr(e)
