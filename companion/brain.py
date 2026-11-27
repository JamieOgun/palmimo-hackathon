"""OpenAI calls: speech-to-text, the reply, and memory consolidation.

Only ever called for the verified owner (the gate in companion.py decides).
"""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from . import logs
from .store import Memory, Message

log = logging.getLogger(__name__)

PERSONA = """You are Palmimo, a small, curious six-legged tabletop robot and {name}'s personal companion.
You only talk with {name}; you recognized them by their face just now.
Speak like a warm friend: short, natural spoken sentences, at most {max_words} words, no lists, no emoji, no markdown.
Use what you remember about {name} naturally when it fits — don't recite it. If you don't know something, ask.
Never invent memories."""

MEMORY_PROMPT = """You maintain a robot companion's long-term memory about its owner, {name}.
Given the CURRENT memory list and a NEW conversation, return the UPDATED memory list as JSON:
{{"memories": ["...", "..."]}}
Rules:
- Each memory is one short, self-contained fact about {name} (preferences, people, plans, events, feelings worth following up on).
- Merge duplicates, update facts that changed, drop facts {name} asked to forget.
- Keep at most {limit} memories; drop the least useful ones if needed.
- Do not store small talk, or anything said by the robot unless {name} confirmed it."""


class Brain:
    def __init__(self, chat_model: str, stt_model: str, max_words: int = 40) -> None:
        self._client = OpenAI()  # reads OPENAI_API_KEY
        self.chat_model = chat_model
        self.stt_model = stt_model
        self.max_words = max_words

    def transcribe(self, wav: bytes) -> str:
        r = self._client.audio.transcriptions.create(model=self.stt_model, file=("utterance.wav", wav), language="en")
        text = r.text.strip()
        log.debug("stt %s: %d bytes in -> %s", self.stt_model, len(wav), logs.text(text))
        return text

    def reply(self, name: str, memories: list[Memory], history: list[Message], text: str) -> str:
        system = PERSONA.format(name=name, max_words=self.max_words)
        if memories:
            system += "\n\nWhat you remember about " + name + ":\n" + "\n".join(f"- {m.fact}" for m in memories)
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": m.role, "content": m.text} for m in history]
        msgs.append({"role": "user", "content": text})
        r = self._client.chat.completions.create(model=self.chat_model, messages=msgs)
        _log_usage("reply", r, memories=len(memories), history=len(history))
        out = (r.choices[0].message.content or "").strip()
        if not out:
            log.warning("chat returned an empty reply (finish_reason=%s)", r.choices[0].finish_reason)
        return out

    def consolidate_memories(self, name: str, current: list[Memory], session: list[Message], limit: int = 60) -> list[str] | None:
        """Return the new full memory list, or None if the model's answer was unusable."""
        convo = "\n".join(f"{name if m.role == 'user' else 'Palmimo'}: {m.text}" for m in session)
        user = "CURRENT:\n" + json.dumps([m.fact for m in current]) + "\n\nNEW CONVERSATION:\n" + convo
        r = self._client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": MEMORY_PROMPT.format(name=name, limit=limit)},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        _log_usage("memories", r, current=len(current), session_msgs=len(session))
        raw = r.choices[0].message.content or ""
        parsed = parse_memories(raw)
        if parsed is None:
            log.warning("memory consolidation: unparseable model output (%d chars): %s", len(raw), logs.text(raw))
        return parsed


def parse_memories(raw: str) -> list[str] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    items = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def _log_usage(what: str, r, **ctx) -> None:
    u = getattr(r, "usage", None)
    extra = " ".join(f"{k}={v}" for k, v in ctx.items())
    if u is not None:
        log.debug("%s: model=%s tokens in=%s out=%s %s", what, getattr(r, "model", "?"),
                  u.prompt_tokens, u.completion_tokens, extra)
